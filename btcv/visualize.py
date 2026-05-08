"""
visualize.py — Visualize BTCV CT slices, ground truth organ masks, and predictions.

Usage:
    # Show raw slices + ground truth only (no model needed):
    python visualize.py --data_dir BTCV

    # Show slices + ground truth + model predictions:
    python visualize.py --data_dir BTCV --checkpoint outputs_btcv/best_model.pth

    # Control how many samples to show:
    python visualize.py --data_dir BTCV --checkpoint outputs_btcv/best_model.pth --n 8

Saves a grid image to: visualization_btcv.png

Colour coding:
  Each organ class gets a distinct colour (see ORGAN_COLORS below).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

from dataset import (_extract_slices, N_CLASSES, CLASS_NAMES)
from transunet import TransUNet


def _discover_volume_ids(root):
    """
    Scan BTCV/Training/img/ for img*.nii.gz and return sorted integer IDs
    actually present on disk — works regardless of what the dataset.py
    constants say (which assume IDs 1-20 and 21-32).
    """
    from pathlib import Path
    img_dir = Path(root) / "Training" / "img"
    if not img_dir.exists():
        # Fallback to dataset.py constants if folder not reachable
        from dataset import TRAIN_VOLUMES, VAL_VOLUMES
        return TRAIN_VOLUMES, VAL_VOLUMES

    ids = []
    for f in img_dir.glob("img*.nii.gz"):
        stem = f.name.replace(".nii.gz", "").replace("img", "")
        try:
            ids.append(int(stem))
        except ValueError:
            pass
    ids = sorted(ids)

    # Split: first 60% train, rest val (mirrors typical BTCV practice)
    split_at   = max(1, int(len(ids) * 0.6))
    train_ids  = ids[:split_at]
    val_ids    = ids[split_at:]
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Colour palette — one distinct colour per organ class
# ---------------------------------------------------------------------------

ORGAN_COLORS = np.array([
    [0.00, 0.00, 0.00, 0.0],   # 0 background  — transparent
    [1.00, 0.20, 0.20, 0.7],   # 1 aorta       — red
    [0.20, 0.80, 0.20, 0.7],   # 2 gallbladder — green
    [0.20, 0.60, 1.00, 0.7],   # 3 spleen      — blue
    [1.00, 0.80, 0.10, 0.7],   # 4 left kidney — yellow
    [1.00, 0.50, 0.00, 0.7],   # 5 right kidney— orange
    [0.80, 0.20, 0.80, 0.7],   # 6 liver       — purple
    [0.00, 0.90, 0.80, 0.7],   # 7 stomach     — cyan
    [1.00, 0.40, 0.70, 0.7],   # 8 pancreas    — pink
])

CMAP = ListedColormap(ORGAN_COLORS[:, :3])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlay_seg(ct_slice_norm, seg_mask):
    """
    Overlay a multi-class segmentation mask on a CT slice (grey).

    ct_slice_norm : (H, W) float32 in [0,1]
    seg_mask      : (H, W) int, values 0..N_CLASSES-1
    Returns       : (H, W, 4) RGBA image
    """
    grey = np.stack([ct_slice_norm] * 3 + [np.ones_like(ct_slice_norm)], axis=-1)
    out  = grey.copy()
    for c in range(1, N_CLASSES):
        mask = seg_mask == c
        if mask.any():
            color = ORGAN_COLORS[c]
            for ch in range(3):
                out[:, :, ch] = np.where(
                    mask,
                    (1 - color[3]) * out[:, :, ch] + color[3] * color[ch],
                    out[:, :, ch],
                )
    return np.clip(out, 0, 1)


@torch.no_grad()
def predict_slice(model, ct_tensor, device):
    """
    ct_tensor : (C, H, W) float32  →  prediction (H, W) int32

    FIX: TransUNet with deep supervision returns a tuple
         (main_logits, *aux_logits).  We always take element [0].
    """
    x      = ct_tensor.unsqueeze(0).to(device)   # (1, C, H, W)
    out    = model(x)

    # Handle both plain-tensor and tuple (deep-supervision) outputs
    logits = out[0] if isinstance(out, (tuple, list)) else out  # (1, N_CLASSES, H, W)

    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred.astype(np.int32)


def _sample_slices(ids, root, patch_size, in_channels, n, seed):
    """
    Load volumes one at a time and stop as soon as we have enough slices.
    Volumes that don't exist on disk are skipped with a warning.
    """
    rng      = random.Random(seed)
    ids_list = list(ids)
    rng.shuffle(ids_list)           # shuffle so we get variety across volumes

    collected   = []
    found_vols  = 0

    for vid in ids_list:
        try:
            slices = _extract_slices(vid, root, patch_size,
                                     in_channels=in_channels, skip_empty=True)
        except FileNotFoundError as e:
            print(f"  [skip] vol {vid:04d} — {e}")
            continue
        except Exception as e:
            print(f"  [skip] vol {vid:04d} — unexpected error: {e}")
            continue

        if not slices:
            print(f"  [skip] vol {vid:04d} — 0 foreground slices")
            continue

        found_vols += 1
        rng.shuffle(slices)
        # Take at most ceil(remaining_needed / remaining_volumes) from each vol
        # so samples are spread across different volumes
        remaining_needed  = n - len(collected)
        remaining_volumes = max(1, len(ids_list) - ids_list.index(vid))
        want = max(1, -(-remaining_needed // remaining_volumes))  # ceil div

        for img_t, seg_t in slices[:want]:
            collected.append((img_t, seg_t, vid))
            if len(collected) >= n:
                return collected

    if not collected:
        print(f"\n[ERROR] Could not load any slices. Tried volume IDs: {ids_list}")
        print(f"  root passed to _extract_slices: {root}")
        # Show what actually exists on disk to help diagnose
        img_dir = root / "Training" / "img"
        if img_dir.exists():
            found = sorted(img_dir.glob("img*.nii.gz"))
            print(f"  Files found in {img_dir}: {[f.name for f in found[:6]]}{'...' if len(found)>6 else ''}")
        else:
            print(f"  Directory does not exist: {img_dir}")

    return collected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True, help="Path to BTCV/ folder")
    p.add_argument("--checkpoint", default=None,  help="Path to best_model.pth (optional)")
    p.add_argument("--n",          type=int, default=6, help="Number of slices to show")
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--split",      default="val", choices=["train", "val"])
    p.add_argument("--out",        default="visualization_btcv.png")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_dir)

    # Discover which volume IDs actually exist on disk (ignores dataset.py constants)
    train_ids, val_ids = _discover_volume_ids(root)
    ids = val_ids if args.split == "val" else train_ids
    print(f"Discovered {len(train_ids)} train + {len(val_ids)} val volumes on disk.")
    print(f"  Train IDs : {train_ids}")
    print(f"  Val IDs   : {val_ids}")

    # ── Load model first so we know in_channels ───────────────────────────────
    # FIX: in the original code, slices were always loaded with in_channels=1
    # but the model was built afterward with in_channels from the checkpoint.
    # If the model was trained with in_channels=3 this caused a shape mismatch.
    # We now read in_channels from the checkpoint BEFORE loading slices.
    model       = None
    in_channels = args.__dict__.get("in_channels", 1)   # default 1
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt        = torch.load(args.checkpoint, map_location=device,
                                 weights_only=False)
        cfg         = ckpt.get("args", {})
        in_channels = cfg.get("in_channels", 1)   # honour training setting
        patch_size  = cfg.get("patch_size", args.patch_size)

        model = TransUNet(
            in_channels    = in_channels,
            n_classes      = N_CLASSES,
            img_size       = patch_size,
            embed_dim      = cfg.get("embed_dim",      768),
            num_heads      = cfg.get("num_heads",       12),
            num_layers     = cfg.get("num_layers",      12),
            query_dim      = cfg.get("query_dim",      256),
            num_dec_layers = cfg.get("num_dec_layers",   3),
            pretrained     = False,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"  epoch={ckpt.get('epoch','?')}  "
              f"mean Dice={ckpt.get('mean_dice', 0):.4f}  "
              f"in_channels={in_channels}")
    else:
        patch_size = args.patch_size

    # ── Load slices (lazy — stop once we have enough) ─────────────────────────
    print(f"Sampling {args.n} slices from {args.split} volumes "
          f"(in_channels={in_channels}) ...")
    samples = _sample_slices(ids, root, patch_size, in_channels,
                             args.n, args.seed)

    # Fallback: if the requested split yielded nothing, try the other split.
    if not samples:
        other_ids  = train_ids if args.split == "val" else val_ids
        other_name = "train"   if args.split == "val" else "val"
        print(f"  No slices from '{args.split}' split. "
              f"Trying '{other_name}' volumes as fallback ...")
        samples = _sample_slices(other_ids, root, patch_size, in_channels,
                                 args.n, args.seed)

    if not samples:
        img_dir = root / "Training" / "img"
        if img_dir.exists():
            found = sorted(img_dir.glob("img*.nii.gz"))
            hint  = (f"Files found in {img_dir}: "
                     f"{[f.name for f in found[:6]]}{'...' if len(found)>6 else ''}")
        else:
            hint = f"Directory does not exist: {img_dir}"
        raise FileNotFoundError(
            f"No valid slices found under '{root}'.\n"
            f"  {hint}\n"
            "  Expected layout: BTCV/Training/img/img0001.nii.gz  etc.\n"
            "  See README_BTCV.txt for setup instructions."
        )
    print(f"Loaded {len(samples)} slices from "
          f"{len({v for *_, v in samples})} volume(s).")

    # ── Build figure ──────────────────────────────────────────────────────────
    n_cols     = 3 if model else 2
    col_titles = ["CT slice (windowed)", "Ground truth"]
    if model:
        col_titles.append("Prediction")

    fig, axes = plt.subplots(
        len(samples), n_cols,
        figsize=(n_cols * 3.5, len(samples) * 3.5),
        gridspec_kw={"wspace": 0.05, "hspace": 0.10},
    )
    if len(samples) == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for row, (img_t, seg_t, vol_id) in enumerate(samples):
        # img_t may be (1,H,W) or (3,H,W) — squeeze channel for display
        ct_np  = img_t[0].numpy()               # always use first channel for grey display
        seg_np = seg_t.numpy().astype(np.int32) # (H, W)

        # Column 0: raw CT
        axes[row, 0].imshow(ct_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_ylabel(f"Vol {vol_id:04d}", fontsize=9)

        # Column 1: ground truth overlay
        gt_rgba = _overlay_seg(ct_np, seg_np)
        axes[row, 1].imshow(ct_np, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(gt_rgba)

        # Column 2: prediction overlay
        if model:
            pred_np   = predict_slice(model, img_t, device)
            pred_rgba = _overlay_seg(ct_np, pred_np)
            axes[row, 2].imshow(ct_np, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(pred_rgba)

            # Per-slice mean Dice (foreground only)
            dice_vals = []
            for c in range(1, N_CLASSES):
                p = (pred_np == c).astype(float)
                g = (seg_np  == c).astype(float)
                d = (2 * p * g).sum() / (p.sum() + g.sum() + 1e-5)
                dice_vals.append(d)
            axes[row, 2].set_xlabel(
                f"mean Dice={sum(dice_vals)/len(dice_vals):.3f}", fontsize=8)

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=ORGAN_COLORS[c, :3], label=CLASS_NAMES[c])
        for c in range(1, N_CLASSES)
    ]
    fig.legend(handles=legend_patches, loc="lower center",
               ncol=4, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.0))

    title = "BTCV Multi-Organ Segmentation — TransUNet (Encoder+Decoder)"
    if model and args.checkpoint:
        title += f"  |  Checkpoint mean Dice: {ckpt.get('mean_dice', 0):.4f}"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    plt.savefig(args.out, dpi=120, bbox_inches="tight", facecolor="white")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()