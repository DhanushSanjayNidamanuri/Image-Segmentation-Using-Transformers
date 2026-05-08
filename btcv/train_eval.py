"""
train_eval.py — Train and evaluate TransUNet (2D) on the BTCV multi-organ CT dataset.
Dataset       : BTCV — 30 abdominal CT volumes → 2D axial slices
Loss          : Dice + CrossEntropy (Eq. 10 of paper)
Metric        : Average Dice over 8 abdominal organs (excl. background)

Usage:
    python train_eval.py --data_dir BTCV
    python train_eval.py --data_dir BTCV --epochs 30 --batch_size 8
    python train_eval.py --data_dir BTCV --no_pretrained
"""

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, BTCVDataset, N_CLASSES, CLASS_NAMES
from transunet import TransUNet, TransUNetLoss


# ---------------------------------------------------------------------------
# Metrics  — per-class Dice, reported separately for each organ
# ---------------------------------------------------------------------------

def compute_metrics(logits, targets, n_classes=N_CLASSES, smooth=1e-5):
    """
    logits  : (B, C, H, W)
    targets : (B, H, W) long, values 0..n_classes-1

    Returns:
        mean_dice  : float — average Dice across foreground classes (1..C-1)
        class_dice : list[float] — Dice per class (index 0 = background)
    """
    preds  = logits.argmax(dim=1)          # (B, H, W)
    dices  = []

    for c in range(n_classes):
        pred_c   = (preds   == c).float()
        target_c = (targets == c).float()
        inter    = (pred_c * target_c).sum()
        union    = pred_c.sum() + target_c.sum()
        dices.append(((2 * inter + smooth) / (union + smooth)).item())

    # Mean dice over foreground classes only (skip background class 0)
    mean_dice = float(sum(dices[1:]) / (n_classes - 1))
    return mean_dice, dices


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out  = model(images)
                loss = criterion(out, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out  = model(images)
            loss = criterion(out, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    all_logits, all_targets = [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    mean_dice, class_dices = compute_metrics(all_logits, all_targets)
    return total_loss / len(loader), mean_dice, class_dices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train TransUNet (Encoder+Decoder) on BTCV multi-organ CT"
    )
    p.add_argument("--data_dir",       required=True,
                   help="Path to BTCV/ folder (contains Training/img and Training/label)")
    p.add_argument("--epochs",         type=int,   default=30,
                   help="Training epochs (30 ≈ 4-5 hr on RTX Pro 4050)")
    p.add_argument("--batch_size",     type=int,   default=8,
                   help="Batch size (8 fits 32 GB VRAM at 224×224)")
    p.add_argument("--patch_size",     type=int,   default=224,
                   help="Resize target for each 2D axial slice")
    p.add_argument("--n_train",        type=int,   default=2000,
                   help="Virtual patches sampled per training epoch")
    p.add_argument("--n_val",          type=int,   default=500,
                   help="Virtual patches sampled per validation epoch")
    p.add_argument("--lr",             type=float, default=3e-4,
                   help="AdamW LR for new layers (paper Table 1: 3e-4 for BTCV decoder)")
    p.add_argument("--in_channels",    type=int,   default=1,
                   help="1=CT greyscale (recommended), 3=replicated")
    # ViT settings (ViT-B/16 defaults — Table 1 of paper)
    p.add_argument("--embed_dim",      type=int,   default=768)
    p.add_argument("--num_layers",     type=int,   default=12,
                   help="12-layer ViT encoder (best for multi-organ, Table 2)")
    p.add_argument("--num_heads",      type=int,   default=12)
    # Transformer Decoder (paper uses 3 C2F stages for BTCV)
    p.add_argument("--query_dim",      type=int,   default=192,
                   help="Transformer decoder hidden dim d_dec (paper Section 4.2: 192)")
    p.add_argument("--num_queries",    type=int,   default=20,
                   help="Organ queries N >> K to reduce false negatives (paper Table 1: 20)")
    p.add_argument("--num_dec_layers", type=int,   default=3,
                   help="C2F stages (paper Table 1: 3 for BTCV)")
    # Pretrained
    p.add_argument("--pretrained",     action="store_true", default=True)
    p.add_argument("--no_pretrained",  dest="pretrained", action="store_false")
    # AMP
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--no_amp",         dest="amp", action="store_false")
    p.add_argument("--num_workers",    type=int,   default=4,
                   help="DataLoader workers (set 0 on Windows)")
    p.add_argument("--save_dir",       type=str,   default="outputs_btcv")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        args.amp = False
    else:
        device = torch.device("cpu")
        args.amp = False

    print(f"\n{'='*70}")
    print(f"  TransUNet (Encoder+Decoder) | BTCV Multi-Organ CT | 3D→2D slices")
    print(f"  Device: {device}  |  patch_size={args.patch_size}  "
          f"batch={args.batch_size}  epochs={args.epochs}")
    print(f"  n_classes={N_CLASSES}  embed_dim={args.embed_dim}  "
          f"num_dec_layers={args.num_dec_layers} (C2F stages)")
    print(f"  pretrained={args.pretrained}  amp={args.amp}")
    print(f"{'='*70}\n")

    if device.type == "cuda":
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  CUDA: {torch.version.cuda}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        data_dir    = args.data_dir,
        patch_size  = args.patch_size,
        n_train     = args.n_train,
        n_val       = args.n_val,
        batch_size  = args.batch_size,
        in_channels = args.in_channels,
        num_workers = args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    # Key change from DRIVE: n_classes=9 (background + 8 organs)
    model = TransUNet(
        in_channels    = args.in_channels,
        n_classes      = N_CLASSES,          # 9 for BTCV (vs 2 for DRIVE)
        img_size       = args.patch_size,
        embed_dim      = args.embed_dim,
        num_heads      = args.num_heads,
        num_layers     = args.num_layers,
        query_dim      = args.query_dim,     # 192 (paper Section 4.2)
        num_queries    = args.num_queries,   # 20  (paper Table 1)
        num_dec_layers = args.num_dec_layers,
        pretrained     = args.pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {n_params:.1f}M")

    # Warm-up
    if device.type == "cuda":
        dummy = torch.zeros(1, args.in_channels,
                            args.patch_size, args.patch_size).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy)
        alloc = torch.cuda.memory_allocated(0) / 1e9
        resv  = torch.cuda.memory_reserved(0)  / 1e9
        print(f"VRAM after init: {alloc:.2f} GB alloc / {resv:.2f} GB reserved\n")

    # ── Optimiser — two-group LR (paper Table 1) ──────────────────────────────
    # Pretrained backbone gets 10× lower LR (standard fine-tuning practice)
    pretrained_params, new_params = [], []
    pretrained_names = {"cnn_encoder", "vit_encoder"}
    for name, param in model.named_parameters():
        if any(name.startswith(pn) for pn in pretrained_names):
            pretrained_params.append(param)
        else:
            new_params.append(param)

    optimizer = AdamW([
        {"params": pretrained_params, "lr": args.lr * 0.1},
        {"params": new_params,        "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Loss weights per paper Eq. 10: λ₀=0.7 (mask loss), λ₁=0.3 (cls loss)
    # aux_weight=0.4 for deep supervision at intermediate C2F stages
    criterion = TransUNetLoss(lam0=0.7, lam1=0.3, aux_weight=0.4)
    scaler    = (torch.amp.GradScaler("cuda")
                 if args.amp and device.type == "cuda" else None)

    # ── Output dir ────────────────────────────────────────────────────────────
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"

    # ── Training loop ─────────────────────────────────────────────────────────
    history   = []
    best_dice = 0.0
    t_start   = time.time()

    # Print header with per-organ columns
    organ_header = " ".join(f"{n[:5]:>5}" for n in CLASS_NAMES[1:])
    print(f"{'Ep':>4} {'TrLoss':>7} {'VaLoss':>7} {'mDice':>6} | "
          f"{organ_header} | {'Time':>5}")
    print("-" * (35 + len(organ_header) + 10))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer,
                                  criterion, device, scaler)
        va_loss, mean_dice, class_dices = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        organ_str = " ".join(f"{d:>5.3f}" for d in class_dices[1:])
        vram_str  = ""
        if device.type == "cuda":
            vram_str = f" VRAM={torch.cuda.memory_allocated(0)/1e9:.1f}GB"

        print(f"{epoch:>4} {tr_loss:>7.4f} {va_loss:>7.4f} "
              f"{mean_dice:>6.4f} | {organ_str} | {elapsed:>4.0f}s{vram_str}")

        record = {
            "epoch":       epoch,
            "train_loss":  round(tr_loss, 4),
            "val_loss":    round(va_loss, 4),
            "mean_dice":   round(mean_dice, 4),
        }
        for name, d in zip(CLASS_NAMES, class_dices):
            record[f"dice_{name}"] = round(d, 4)
        history.append(record)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "mean_dice":  best_dice,
                "class_dice": class_dices,
                "args":       vars(args),
            }, best_path)

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min  |  Best mean Dice: {best_dice:.4f}")
    print(f"Best model -> {best_path}")

    # Save history
    hist_path = save_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History    -> {hist_path}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n--- Final evaluation (best checkpoint) ---")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, final_mean, final_class = evaluate(model, val_loader, criterion, device)
    print(f"  Mean Dice (8 organs): {final_mean:.4f}")
    for name, d in zip(CLASS_NAMES, final_class):
        print(f"  {name:<15}: {d:.4f}")


if __name__ == "__main__":
    main()
