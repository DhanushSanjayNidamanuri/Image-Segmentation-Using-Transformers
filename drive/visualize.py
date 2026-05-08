"""
visualize.py — Visualize DRIVE images, ground truth masks, and model predictions.

Usage:
    # Just show raw images + ground truth (no model needed):
    python visualize.py --data_dir DRIVE

    # Show images + ground truth + model predictions:
    python visualize.py --data_dir DRIVE --checkpoint outputs/best_model.pth

    # Control how many samples to show:
    python visualize.py --data_dir DRIVE --checkpoint outputs/best_model.pth --n 8

Saves a grid image to: visualization.png
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from dataset import DRIVEDataset, _load_image, _load_mask
from transunet import TransUNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_random_patch(img, msk, fov, patch_size=224, min_fov=0.7):
    """Return a patch that is mostly inside the circular FOV."""
    H, W = img.shape[:2]
    P    = patch_size
    for _ in range(50):
        y = random.randint(0, H - P)
        x = random.randint(0, W - P)
        if fov[y:y+P, x:x+P].mean() > min_fov:
            return (img[y:y+P, x:x+P], msk[y:y+P, x:x+P], y, x)
    y = (H - P) // 2
    x = (W - P) // 2
    return img[y:y+P, x:x+P], msk[y:y+P, x:x+P], y, x


def overlay_mask(rgb_patch, mask, color, alpha=0.45):
    """Overlay a binary mask on an RGB patch with a given colour."""
    out = rgb_patch.copy()
    for c, v in enumerate(color):
        out[:, :, c] = np.where(mask, (1-alpha)*out[:, :, c] + alpha*v,
                                 out[:, :, c])
    return np.clip(out, 0, 1)


@torch.no_grad()
def predict_patch(model, img_patch_rgb, device, patch_size=224, in_channels=1):
    """Run model on a single patch, return binary prediction (H, W)."""
    if in_channels == 1:
        inp = img_patch_rgb[:, :, 1:2].transpose(2, 0, 1)   # green (1, H, W)
    else:
        inp = img_patch_rgb.transpose(2, 0, 1)               # RGB   (3, H, W)
    x = torch.from_numpy(inp).unsqueeze(0).to(device)         # (1, C, H, W)
    # eval mode → plain logits tensor (no aux)
    logits = model(x)                                          # (1, 2, H, W)
    pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy()    # (H, W)
    return pred.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True, help="Path to DRIVE/ folder")
    p.add_argument("--checkpoint", default=None,  help="Path to best_model.pth (optional)")
    p.add_argument("--n",          type=int, default=6,  help="Number of samples to visualise")
    p.add_argument("--patch_size", type=int, default=224)
    p.add_argument("--val_ids",    type=int, nargs="+", default=[37, 38, 39, 40],
                   help="Image IDs used as validation set")
    p.add_argument("--split",      default="val", choices=["train", "val"],
                   help="Which split to sample from")
    p.add_argument("--out",        default="visualization.png")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.data_dir)
    P    = args.patch_size

    all_ids = list(range(21, 41))
    if args.split == "val":
        ids = args.val_ids
    else:
        ids = [i for i in all_ids if i not in args.val_ids]

    img_dir  = root / "training" / "images"
    mask_dir = root / "training" / "1st_manual"
    fov_dir  = root / "training" / "mask"

    records = []
    for img_id in ids:
        img_path       = img_dir / f"{img_id}_training.tif"
        msk_candidates = list(mask_dir.glob(f"{img_id}_manual*"))
        fov_candidates = list(fov_dir.glob(f"{img_id}_training_mask*"))
        if not img_path.exists() or not msk_candidates:
            continue
        img = _load_image(img_path)
        msk = _load_mask(msk_candidates[0])
        fov = _load_mask(fov_candidates[0]) if fov_candidates else np.ones_like(msk)
        records.append((img_id, img, msk, fov))

    if not records:
        raise FileNotFoundError(f"No images found for IDs {ids} in {img_dir}")

    # ── Load model ────────────────────────────────────────────────────────────
    model      = None
    in_channels = 1
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        cfg  = ckpt.get("args", {})
        in_channels = cfg.get("in_channels", 1)
        model = TransUNet(
            in_channels    = in_channels,
            n_classes      = 2,
            img_size       = cfg.get("patch_size", P),
            embed_dim      = cfg.get("embed_dim",  768),
            num_heads      = cfg.get("num_heads",  12),
            num_layers     = cfg.get("num_layers", 12),
            query_dim      = cfg.get("query_dim",  256),
            num_dec_layers = cfg.get("num_dec_layers", 4),
            pretrained     = False,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        print(f"Loaded checkpoint: {args.checkpoint}  "
              f"(epoch {ckpt.get('epoch','?')}, Dice={ckpt.get('dice', 0):.4f})")

    # ── Build figure ──────────────────────────────────────────────────────────
    n_cols     = 4 if model else 3
    n_rows     = min(args.n, len(records) * 10)
    col_titles = ["RGB patch", "Green channel", "Ground truth"]
    if model:
        col_titles.append("Model prediction")

    VESSEL_COLOR = np.array([1.0, 0.2, 0.2])
    ERROR_FP     = np.array([1.0, 0.6, 0.0])
    ERROR_FN     = np.array([0.0, 0.4, 1.0])

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.2, n_rows * 3.2),
        gridspec_kw={"wspace": 0.04, "hspace": 0.12},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=8)

    sample_pool = (records * 20)[:n_rows]
    random.shuffle(sample_pool)

    for row, (img_id, img, msk, fov) in enumerate(sample_pool):
        patch_rgb, patch_msk, y, x = get_random_patch(img, msk, fov, P)
        patch_green = patch_rgb[:, :, 1]

        axes[row, 0].imshow(patch_rgb)
        axes[row, 0].set_ylabel(f"ID {img_id}", fontsize=9, labelpad=4)
        axes[row, 1].imshow(patch_green, cmap="gray", vmin=0, vmax=1)

        gt_overlay = overlay_mask(patch_rgb, patch_msk, VESSEL_COLOR)
        axes[row, 2].imshow(gt_overlay)

        if model:
            pred = predict_patch(model, patch_rgb, device, P, in_channels)
            tp = (pred == 1) & (patch_msk == 1)
            fp = (pred == 1) & (patch_msk == 0)
            fn = (pred == 0) & (patch_msk == 1)
            pred_overlay = patch_rgb.copy()
            pred_overlay = overlay_mask(pred_overlay, tp, VESSEL_COLOR, alpha=0.5)
            pred_overlay = overlay_mask(pred_overlay, fp, ERROR_FP,     alpha=0.55)
            pred_overlay = overlay_mask(pred_overlay, fn, ERROR_FN,     alpha=0.55)
            axes[row, 3].imshow(pred_overlay)
            inter = (pred * patch_msk).sum()
            union = pred.sum() + patch_msk.sum()
            dice  = (2*inter + 1e-5) / (union + 1e-5)
            axes[row, 3].set_xlabel(f"patch Dice={dice:.3f}", fontsize=8)

        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)

    legend_items = [mpatches.Patch(color=VESSEL_COLOR, label="Vessel (TP / GT)")]
    if model:
        legend_items += [
            mpatches.Patch(color=ERROR_FP, label="False positive"),
            mpatches.Patch(color=ERROR_FN, label="False negative"),
        ]
    fig.legend(handles=legend_items, loc="lower center", ncol=len(legend_items),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, 0.0))

    title = "DRIVE Retinal Vessel Segmentation — 3D-style TransUNet (2D)"
    if model and "ckpt" in dir():
        title += f"  |  Checkpoint Dice: {ckpt.get('dice', 0):.4f}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    plt.savefig(args.out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
