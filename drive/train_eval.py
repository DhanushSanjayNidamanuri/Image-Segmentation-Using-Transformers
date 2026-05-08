"""
train_eval.py — Train and evaluate 3D-style TransUNet (2D adaptation) on DRIVE.

Architecture: Hybrid CNN-ViT Encoder + CNN Decoder + Transformer Decoder
Reference: Chen et al., "3D TransUNet", arXiv 2310.07781

Usage:
    python train_eval.py --data_dir DRIVE
    python train_eval.py --data_dir DRIVE --epochs 50
    python train_eval.py --data_dir DRIVE --batch_size 4 --pretrained
"""

import argparse
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import get_dataloaders, DRIVEDataset
from transunet import TransUNet, TransUNetLoss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(logits, targets, smooth=1e-5):
    """
    logits  : (B, C, H, W) — raw model output (before softmax)
    targets : (B, H, W)    — long tensor, values 0/1
    Returns dict with dice, sensitivity, specificity, accuracy.
    """
    preds = logits.argmax(dim=1)          # (B, H, W)

    pred_v    = (preds   == 1).float()
    target_v  = (targets == 1).float()
    pred_bg   = (preds   == 0).float()
    target_bg = (targets == 0).float()

    tp = (pred_v  * target_v ).sum()
    fp = (pred_v  * target_bg).sum()
    fn = (pred_bg * target_v ).sum()
    tn = (pred_bg * target_bg).sum()

    dice        = (2*tp + smooth) / (2*tp + fp + fn + smooth)
    sensitivity = (tp  + smooth) / (tp + fn + smooth)
    specificity = (tn  + smooth) / (tn + fp + smooth)
    accuracy    = (tp  + tn)     / (tp + tn + fp + fn + smooth)

    return {
        "dice":        dice.item(),
        "sensitivity": sensitivity.item(),
        "specificity": specificity.item(),
        "accuracy":    accuracy.item(),
    }


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            from torch.cuda.amp import autocast
            with autocast():
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
    total_loss  = 0.0
    all_logits, all_targets = [], []

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)          # eval mode → plain tensor, no aux
        # criterion handles both tuple and tensor
        total_loss += criterion(logits, masks).item()
        all_logits.append(logits.cpu())
        all_targets.append(masks.cpu())

    all_logits  = torch.cat(all_logits,  dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_logits, all_targets)
    return total_loss / len(loader), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train 3D-style TransUNet (2D) on DRIVE retinal vessel dataset"
    )
    p.add_argument("--data_dir",     required=True,
                   help="Path to the DRIVE/ folder (contains training/ and test/)")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=4,
                   help="4-8 fits on RTX 3060 12GB at 224x224; reduce if OOM")
    p.add_argument("--patch_size",   type=int,   default=224)
    p.add_argument("--n_train",      type=int,   default=2000,
                   help="Patches sampled per training epoch")
    p.add_argument("--n_val",        type=int,   default=400,
                   help="Patches sampled per validation epoch")
    p.add_argument("--lr",           type=float, default=1e-4,
                   help="AdamW learning rate (1e-4 recommended for pretrained)")
    p.add_argument("--in_channels",  type=int,   default=1,
                   help="1=green channel (recommended for DRIVE), 3=RGB")
    # ViT-Encoder settings
    p.add_argument("--embed_dim",    type=int,   default=768,
                   help="ViT hidden size (768 = ViT-B)")
    p.add_argument("--num_layers",   type=int,   default=12,
                   help="ViT transformer layers")
    p.add_argument("--num_heads",    type=int,   default=12,
                   help="ViT attention heads")
    # Transformer Decoder settings
    p.add_argument("--query_dim",    type=int,   default=256,
                   help="Transformer decoder hidden dim")
    p.add_argument("--num_dec_layers", type=int, default=4,
                   help="Number of Transformer decoder layers (coarse-to-fine steps)")
    # Pretrained
    p.add_argument("--pretrained",   action="store_true", default=True,
                   help="Use pretrained ResNet-50 (ImageNet) + ViT-B/16 (ImageNet-21k)")
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    # AMP
    p.add_argument("--amp",          action="store_true", default=True,
                   help="Mixed precision training (faster, less VRAM)")
    p.add_argument("--no_amp",       dest="amp", action="store_false")
    p.add_argument("--num_workers",  type=int,   default=0,
                   help="0 required on Windows; increase on Linux/Mac")
    p.add_argument("--save_dir",     type=str,   default="outputs")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        args.amp = False        # AMP not supported on MPS
    else:
        device = torch.device("cpu")
        args.amp = False

    print(f"\n{'='*65}")
    print(f"  3D-style TransUNet (2D)  |  device: {device}")
    print(f"  Architecture: Hybrid CNN-ViT Encoder + CNN Decoder + Transformer Decoder")
    print(f"  img_size={args.patch_size}  batch={args.batch_size}  "
          f"epochs={args.epochs}  embed_dim={args.embed_dim}")
    print(f"  pretrained={args.pretrained}  amp={args.amp}")
    print(f"{'='*65}\n")

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
    model = TransUNet(
        in_channels    = args.in_channels,
        n_classes      = DRIVEDataset.N_CLASSES,
        img_size       = args.patch_size,
        embed_dim      = args.embed_dim,
        num_heads      = args.num_heads,
        num_layers     = args.num_layers,
        query_dim      = args.query_dim,
        num_dec_layers = args.num_dec_layers,
        pretrained     = args.pretrained,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {n_params:.1f}M")

    # Warm-up forward pass
    if device.type == "cuda":
        dummy = torch.zeros(1, args.in_channels, args.patch_size, args.patch_size).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy)
        alloc = torch.cuda.memory_allocated(0) / 1e9
        resv  = torch.cuda.memory_reserved(0)  / 1e9
        print(f"VRAM after init: {alloc:.2f} GB allocated / {resv:.2f} GB reserved\n")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Use separate LR for pretrained backbone vs newly initialised layers
    pretrained_params, new_params = [], []
    pretrained_names = {"cnn_encoder", "vit_encoder"}
    for name, param in model.named_parameters():
        if any(name.startswith(pn) for pn in pretrained_names):
            pretrained_params.append(param)
        else:
            new_params.append(param)

    optimizer = AdamW([
        {"params": pretrained_params, "lr": args.lr * 0.1},  # 10× lower for pretrained
        {"params": new_params,        "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = TransUNetLoss(ce_w=0.5, dice_w=0.5, aux_weight=0.4)
    scaler    = torch.cuda.amp.GradScaler() if args.amp and device.type == "cuda" else None

    # ── Output dir ────────────────────────────────────────────────────────────
    save_dir  = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_model.pth"

    # ── Training loop ─────────────────────────────────────────────────────────
    history   = []
    best_dice = 0.0
    t_start   = time.time()

    print(f"{'Epoch':>6} {'TrLoss':>8} {'VaLoss':>8} "
          f"{'Dice':>7} {'Sens':>7} {'Spec':>7} {'Acc':>7} {'Time':>6}")
    print("-" * 64)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss          = train_one_epoch(model, train_loader, optimizer,
                                           criterion, device, scaler)
        va_loss, metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        vram_str = ""
        if device.type == "cuda":
            vram_str = f" | VRAM {torch.cuda.memory_allocated(0)/1e9:.1f}GB"

        print(f"{epoch:>6} {tr_loss:>8.4f} {va_loss:>8.4f} "
              f"{metrics['dice']:>7.4f} {metrics['sensitivity']:>7.4f} "
              f"{metrics['specificity']:>7.4f} {metrics['accuracy']:>7.4f} "
              f"{elapsed:>5.1f}s{vram_str}")

        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "val_loss": round(va_loss, 4),
                        **{k: round(v, 4) for k, v in metrics.items()}})

        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "dice": best_dice,
                "args": vars(args),
            }, best_path)

    total_min = (time.time() - t_start) / 60
    print(f"\nDone in {total_min:.1f} min  |  Best Dice: {best_dice:.4f}")
    print(f"Best model -> {best_path}")

    # Save history
    hist_path = save_dir / "history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History    -> {hist_path}")

    # ── Final eval ────────────────────────────────────────────────────────────
    print("\n--- Final evaluation (best checkpoint) ---")
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    _, final = evaluate(model, val_loader, criterion, device)
    for k, v in final.items():
        print(f"  {k:<14}: {v:.4f}")


if __name__ == "__main__":
    main()
