"""
BTCV Dataset — Multi-Organ Segmentation (3D CT → 2D axial slices)
------------------------------------------------------------------
Reference: Landman et al., "Multi-Atlas Labeling Beyond the Cranial Vault",
           MICCAI 2015 Challenge.

13 Organ Classes (per BTCV challenge):
    0  -> background
    1  -> aorta
    2  -> gallbladder
    3  -> spleen
    4  -> left kidney
    5  -> right kidney
    6  -> liver
    7  -> stomach
    8  -> pancreas
    (+ 5 more vessels/organs in full challenge — use num_classes=9 for the 8
       abdominal organs reported in the paper, background = class 0)

Strategy:
  - Load all 30 volumes, convert each to axial 2D slices (D slices per volume)
  - Each volume has ~85-198 slices at 512×512
  - We resize slices to patch_size×patch_size (default 224×224)
  - CT Windowing: clip HU values to [-125, 275] (soft-tissue window)
    then normalise to [0,1]  — standard practice for abdominal CT
  - Split: 18 volumes for training, 12 for validation
    (follows the paper's split: Fu et al. 2020, hard split)
  - Only slices that contain at least one foreground label are used for
    training (skips ~30% of pure-background slices) — saves time, improves
    class balance

"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

try:
    import nibabel as nib
    _NIB_OK = True
except ImportError:
    _NIB_OK = False
    print("[WARNING] nibabel not found. Install with: pip install nibabel")


# ---------------------------------------------------------------------------
# BTCV split — adjusted for actual file layout: indices 01-10 and 21-40
# (30 volumes total; indices 11-20 do not exist on disk)
#
# Paper split (Fu et al. 2020): 18 train / 12 val
#   Available indices: 1-10 (10 vols) + 21-40 (20 vols) = 30 total
#   Train (18): 01-10 + 21-28
#   Val   (12): 29-40
# ---------------------------------------------------------------------------

TRAIN_VOLUMES = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10,   # indices 01-10
                 21, 22, 23, 24, 25, 26, 27, 28]              # 18 volumes total
VAL_VOLUMES   = [29, 30, 31, 32, 33, 34,
                 35, 36, 37, 38, 39, 40]                       # 12 volumes total

# BTCV 8-organ label mapping used in the TransUNet paper (Table 8)
# Raw label → class index (0 = background)
# Full BTCV has 13 classes; the paper evaluates 8 abdominal organs
LABEL_MAP = {
    0: 0,   # background
    1: 1,   # aorta
    2: 2,   # gallbladder
    3: 3,   # spleen
    4: 4,   # left kidney
    5: 5,   # right kidney
    6: 6,   # liver
    7: 7,   # stomach
    11: 8,  # pancreas  (raw label 11 in BTCV)
}
N_CLASSES    = 9   # background + 8 organs
CLASS_NAMES  = ["background", "aorta", "gallbladder", "spleen",
                "left_kidney", "right_kidney", "liver", "stomach", "pancreas"]

# CT HU windowing for abdominal soft tissue
HU_MIN, HU_MAX = -125.0, 275.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _load_nifti(path: Path):
    """Load a NIfTI volume, return numpy array (H, W, D) and affine."""
    if not _NIB_OK:
        raise ImportError("nibabel is required: pip install nibabel")
    vol = nib.load(str(path))
    return vol.get_fdata(dtype=np.float32), vol.affine


def _window_ct(vol: np.ndarray) -> np.ndarray:
    """Clip HU values to soft-tissue window and normalise to [0, 1]."""
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol.astype(np.float32)


def _remap_labels(seg: np.ndarray) -> np.ndarray:
    """Map raw BTCV label indices to 0..N_CLASSES-1 (ignore unlisted labels)."""
    out = np.zeros_like(seg, dtype=np.uint8)
    for raw, mapped in LABEL_MAP.items():
        out[seg == raw] = mapped
    return out


def _resize_slice(arr: np.ndarray, size: int, is_mask: bool) -> np.ndarray:
    """Resize a 2D array to (size, size) using PIL."""
    img = Image.fromarray(arr)
    interp = Image.NEAREST if is_mask else Image.BILINEAR
    img = img.resize((size, size), interp)
    return np.array(img)


# ---------------------------------------------------------------------------
# Pre-process and cache all slices at startup
# ---------------------------------------------------------------------------

def _extract_slices(vol_id: int, root: Path, patch_size: int,
                    in_channels: int, skip_empty: bool = True):
    """
    Load one CT volume, window it, and return all 2D axial slices.

    Returns list of (img_slice, mask_slice) tuples:
        img_slice  : (C, patch_size, patch_size) float32
        mask_slice : (patch_size, patch_size) uint8
    """
    # Build paths — BTCV naming: img0001.nii.gz, label0001.nii.gz
    img_path = root / "Training" / "img"   / f"img{vol_id:04d}.nii.gz"
    seg_path = root / "Training" / "label" / f"label{vol_id:04d}.nii.gz"

    if not img_path.exists():
        raise FileNotFoundError(
            f"CT volume not found: {img_path}\n"
            "Check BTCV folder layout — see README_BTCV.txt."
        )

    img_vol, _ = _load_nifti(img_path)   # (H, W, D) or (W, H, D)
    seg_vol, _ = _load_nifti(seg_path)

    # Ensure (H, W, D) orientation
    if img_vol.ndim == 3:
        H, W, D = img_vol.shape
    else:
        raise ValueError(f"Unexpected volume shape: {img_vol.shape}")

    img_vol = _window_ct(img_vol)        # normalise to [0,1]
    seg_vol = _remap_labels(seg_vol)     # remap to 0..N_CLASSES-1

    slices = []
    for d in range(D):
        img_slice = img_vol[:, :, d]     # (H, W) float32  [0,1]
        seg_slice = seg_vol[:, :, d]     # (H, W) uint8

        # Skip background-only slices during training to save time
        if skip_empty and seg_slice.max() == 0:
            continue

        # Resize to model input size
        img_slice = _resize_slice(img_slice.astype(np.float32), patch_size, is_mask=False)
        seg_slice = _resize_slice(seg_slice, patch_size, is_mask=True)

        # Build channel dimension
        if in_channels == 1:
            img_t = torch.from_numpy(img_slice[None])          # (1, H, W)
        else:
            # Replicate single CT channel to 3 channels (common practice)
            img_t = torch.from_numpy(
                np.stack([img_slice] * in_channels, axis=0)    # (C, H, W)
            )

        seg_t = torch.from_numpy(seg_slice.astype(np.int64))  # (H, W)
        slices.append((img_t, seg_t))

    return slices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BTCVDataset(Dataset):
    """
    BTCV multi-organ CT dataset — 2D axial slices extracted from 3D volumes.

    Args:
        root         : path to the BTCV/ folder (contains Training/)
        split        : 'train' or 'val'
        patch_size   : resize target for each slice (default 224)
        n_patches    : virtual epoch length (random sampling with replacement)
        augment      : random flips + rotation (train only)
        in_channels  : 1 (CT greyscale, recommended) or 3 (replicated)
        train_ids    : list of volume IDs for training (default: TRAIN_VOLUMES)
        val_ids      : list of volume IDs for validation (default: VAL_VOLUMES)
    """

    N_CLASSES   = N_CLASSES
    CLASS_NAMES = CLASS_NAMES

    def __init__(
        self,
        root,
        split        = "train",
        patch_size   = 224,
        n_patches    = 2000,
        augment      = False,
        in_channels  = 1,
        train_ids    = None,
        val_ids      = None,
    ):
        self.patch_size  = patch_size
        self.n_patches   = n_patches
        self.augment     = augment
        self.in_channels = in_channels

        root = Path(root)

        if train_ids is None:
            train_ids = TRAIN_VOLUMES
        if val_ids is None:
            val_ids = VAL_VOLUMES

        ids = train_ids if split == "train" else val_ids
        skip_empty = (split == "train")   # val: keep all slices for fair eval

        print(f"[BTCVDataset] {split:5s}: loading {len(ids)} volumes "
              f"(IDs {ids[:4]}...)")

        self._slices = []
        for vid in ids:
            try:
                s = _extract_slices(vid, root, patch_size, in_channels,
                                    skip_empty=skip_empty)
                self._slices.extend(s)
                print(f"  vol {vid:04d} → {len(s)} slices")
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")

        if not self._slices:
            raise RuntimeError(
                f"No slices loaded for split='{split}'. "
                "Check your BTCV folder layout."
            )

        print(f"[BTCVDataset] {split:5s}: {len(self._slices)} slices total "
              f"→ {n_patches} samples/epoch\n")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # Random sampling with replacement for virtual epoch
        img_t, seg_t = self._slices[random.randrange(len(self._slices))]

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
                seg_t = TF.hflip(seg_t.unsqueeze(0)).squeeze(0)
            # Random vertical flip
            if random.random() > 0.5:
                img_t = TF.vflip(img_t)
                seg_t = TF.vflip(seg_t.unsqueeze(0)).squeeze(0)
            # Random rotation ±30°
            angle = random.uniform(-30, 30)
            img_t = TF.rotate(img_t, angle)
            seg_t = TF.rotate(seg_t.unsqueeze(0), angle,
                              interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        return img_t.float(), seg_t.long()

def get_dataloaders(
    data_dir,
    patch_size   = 224,
    n_train      = 2000,
    n_val        = 500,
    batch_size   = 8,
    in_channels  = 1,
    num_workers  = 4,
    train_ids    = None,
    val_ids      = None,
):
    train_ds = BTCVDataset(data_dir, "train", patch_size, n_train,
                           augment=True,  in_channels=in_channels,
                           train_ids=train_ids, val_ids=val_ids)
    val_ds   = BTCVDataset(data_dir, "val",   patch_size, n_val,
                           augment=False, in_channels=in_channels,
                           train_ids=train_ids, val_ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=(num_workers > 0))
    return train_loader, val_loader
