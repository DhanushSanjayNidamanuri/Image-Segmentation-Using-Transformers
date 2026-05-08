"""
DRIVE Dataset — Retinal Vessel Segmentation
--------------------------------------------
Kaggle download:
  https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction

The Kaggle mirror does NOT include test/1st_manual (masks are withheld by
the benchmark organisers). So we split the 20 labelled training images
into 16 train / 4 val ourselves — standard practice in DRIVE papers.

Expected folder layout after unzipping:
  DRIVE/
    training/
      images/        <- 21_training.tif ... 40_training.tif
      1st_manual/    <- 21_manual1.gif  ... 40_manual1.gif
      mask/          <- 21_training_mask.gif ... 40_training_mask.gif
    test/
      images/        <- 01_test.tif ... 20_test.tif  (no masks, not used)
      mask/

Classes:
    0 -> background / outside field of view
    1 -> retinal blood vessel
"""

import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ---------------------------------------------------------------------------
# Low-level loaders
# ---------------------------------------------------------------------------

def _load_image(path: Path) -> np.ndarray:
    """RGB image as float32 (H, W, 3) in [0, 1]."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _load_mask(path: Path) -> np.ndarray:
    """Binary mask as uint8 (H, W), values 0/1."""
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DRIVEDataset(Dataset):
    """
    Args:
        root        : path to the DRIVE/ folder
        split       : 'train' or 'val'
        patch_size  : random crop size fed to the model (default 224)
        n_patches   : virtual epoch length
        augment     : random flips + rotation (train only)
        in_channels : 1 = green channel (best contrast), 3 = RGB
        val_ids     : which image numbers to hold out for val
                      default = last 4 of the 20 training images
    """

    N_CLASSES   = 2
    CLASS_NAMES = ["background", "vessel"]

    # All 20 training image IDs in the DRIVE dataset
    ALL_IDS = list(range(21, 41))          # 21 .. 40

    def __init__(
        self,
        root,
        split       = "train",
        patch_size  = 224,
        n_patches   = 1000,
        augment     = False,
        in_channels = 1,
        val_ids     = None,
    ):
        self.root        = Path(root)
        self.patch_size  = patch_size
        self.n_patches   = n_patches
        self.augment     = augment
        self.in_channels = in_channels

        if val_ids is None:
            val_ids = [37, 38, 39, 40]     # last 4 images -> val

        if split == "train":
            ids = [i for i in self.ALL_IDS if i not in val_ids]
        else:
            ids = val_ids

        img_dir  = self.root / "training" / "images"
        mask_dir = self.root / "training" / "1st_manual"
        fov_dir  = self.root / "training" / "mask"

        # Build file triplets
        self.samples = []
        for img_id in ids:
            img_path = img_dir  / f"{img_id}_training.tif"
            # mask filename varies slightly between dataset versions
            msk_candidates = list(mask_dir.glob(f"{img_id}_manual*"))
            fov_candidates = list(fov_dir.glob(f"{img_id}_training_mask*"))
            if not img_path.exists() or not msk_candidates:
                raise FileNotFoundError(
                    f"Missing file for image ID {img_id}.\n"
                    f"  Expected image : {img_path}\n"
                    f"  Expected mask  : {mask_dir}/{img_id}_manual1.gif\n"
                    "Check your DRIVE folder layout — see README.txt."
                )
            self.samples.append((
                img_path,
                msk_candidates[0],
                fov_candidates[0] if fov_candidates else None,
            ))

        # Pre-load everything into RAM (~50 MB for all 20 images)
        self._images, self._vmasks, self._fovs = [], [], []
        for img_path, msk_path, fov_path in self.samples:
            self._images.append(_load_image(img_path))
            self._vmasks.append(_load_mask(msk_path))
            fov = _load_mask(fov_path) if fov_path else \
                  np.ones(self._vmasks[-1].shape, dtype=np.uint8)
            self._fovs.append(fov)

        print(f"[DRIVEDataset] {split:5s}: {len(self.samples)} images "
              f"(IDs {ids}) -> {n_patches} patches/epoch")

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        i   = random.randrange(len(self.samples))
        img = self._images[i]
        msk = self._vmasks[i]
        fov = self._fovs[i]

        H, W = img.shape[:2]
        P    = self.patch_size

        # Random crop — prefer patches that fall inside the circular FOV
        for _ in range(20):
            y = random.randint(0, H - P)
            x = random.randint(0, W - P)
            if fov[y:y+P, x:x+P].mean() > 0.5:
                break

        img_patch = img[y:y+P, x:x+P]
        msk_patch = msk[y:y+P, x:x+P]

        if self.in_channels == 1:
            img_patch = img_patch[:, :, 1:2]   # green channel

        img_t = torch.from_numpy(img_patch.transpose(2, 0, 1))   # float32
        msk_t = torch.from_numpy(msk_patch.astype(np.int64))      # long

        if self.augment:
            if random.random() > 0.5:
                img_t = TF.hflip(img_t)
                msk_t = TF.hflip(msk_t.unsqueeze(0)).squeeze(0)
            if random.random() > 0.5:
                img_t = TF.vflip(img_t)
                msk_t = TF.vflip(msk_t.unsqueeze(0)).squeeze(0)
            angle = random.uniform(-30, 30)
            img_t = TF.rotate(img_t, angle)
            msk_t = TF.rotate(msk_t.unsqueeze(0), angle).squeeze(0)

        return img_t, msk_t


def get_dataloaders(
    data_dir,
    patch_size  = 224,
    n_train     = 2000,
    n_val       = 400,
    batch_size  = 8,
    in_channels = 1,
    num_workers = 4,
    val_ids     = None,
):
    train_ds = DRIVEDataset(data_dir, "train", patch_size, n_train,
                            augment=True,  in_channels=in_channels, val_ids=val_ids)
    val_ds   = DRIVEDataset(data_dir, "val",   patch_size, n_val,
                            augment=False, in_channels=in_channels, val_ids=val_ids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
