TransUNet (Encoder+Decoder) — BTCV Multi-Organ CT Segmentation
===============================================================
Reference : Chen et al., "TransUNet: Rethinking the U-Net architecture
            design for medical image segmentation through the lens of
            transformers", Medical Image Analysis 2024.
Dataset   : BTCV (Beyond the Cranial Vault) — 30 abdominal CT volumes
            Landman et al., MICCAI 2015 Challenge

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Downloading the BTCV dataset
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. Register at: https://www.synapse.org/#!Synapse:syn3193805/wiki/217789
  2. Download "RawData.zip" (~3.5 GB) from the Files tab
  3. Unzip it

After unzipping you should have:

  RawData/
    Training/
      img/       <- img0001.nii.gz ... img0030.nii.gz   (30 CT volumes)
      label/     <- label0001.nii.gz ... label0030.nii.gz

  Rename/move RawData/ → BTCV/ :
    mv RawData BTCV

Final layout:
  BTCV/
    Training/
      img/
      label/


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Installing dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # PyTorch with CUDA 12:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  # NIfTI reader (required for 3D CT volumes):
  pip install nibabel

  # ViT pretrained weights:
  pip install timm

  # Other:
  pip install Pillow numpy matplotlib

  # Verify:
  python -c "import torch, nibabel, timm; print(torch.cuda.is_available())"
  # should print: True


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Folder layout
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  my_project/
    transunet.py       
    dataset.py         <- BTCV NIfTI → 2D slice loader
    train_eval.py      <- 9-class training loop
    visualize.py       <- multi-organ colour overlay
    README_BTCV.txt    <- this file
    BTCV/              <- dataset folder from Step 1


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd my_project

  # Recommended for RTX Pro 4050 (~4-5 hrs, 30 epochs):
  python train_eval.py --data_dir BTCV

  # Shorter run for quick validation (~1-1.5 hrs, 10 epochs):
  python train_eval.py --data_dir BTCV --epochs 10

  # Larger batch (if VRAM allows):
  python train_eval.py --data_dir BTCV --batch_size 16

  # Without pretrained ViT (~2pp lower Dice):
  python train_eval.py --data_dir BTCV --no_pretrained

  # RGB (replicated channels):
  python train_eval.py --data_dir BTCV --in_channels 3

  All train_eval.py command-line options:

    --data_dir       Path to BTCV/ folder              (required)
    --epochs         Training epochs                    (default: 30)
    --batch_size     Images per batch                   (default: 8)
    --patch_size     Resize target for each slice       (default: 224)
    --n_train        Patches sampled per train epoch    (default: 2000)
    --n_val          Patches sampled per val epoch      (default: 500)
    --lr             AdamW LR for new layers            (default: 3e-4)
    --in_channels    1=CT grey, 3=replicated            (default: 1)
    --embed_dim      ViT hidden size                    (default: 768)
    --num_layers     ViT transformer layers             (default: 12)
    --num_heads      ViT attention heads                (default: 12)
    --query_dim      Transformer decoder hidden dim     (default: 256)
    --num_dec_layers C2F stages (paper: 3 for BTCV)    (default: 3)
    --pretrained     Use pretrained ResNet+ViT          (default: True)
    --no_pretrained  Disable pretrained weights
    --amp            Mixed precision training           (default: True)
    --no_amp         Disable AMP
    --num_workers    DataLoader workers                 (default: 4)
    --save_dir       Output directory                   (default: outputs_btcv/)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Expected Output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Ep TrLoss VaLoss mDice | aorta galbl splee lkidn rkidn liver stomc pancr | Time
 ─────────────────────────────────────────────────────────────────────────────────
  1 0.7812 0.7401 0.4201| 0.612 0.312 0.531 0.481 0.476 0.821 0.391 0.201 |  78s
  5 0.4912 0.4601 0.6801| 0.801 0.521 0.712 0.681 0.675 0.921 0.631 0.401 |  76s
 30 0.2101 0.2301 0.8839| 0.930 0.820 0.857 0.889 0.887 0.972 0.851 0.829 |  75s

Outputs saved:
  outputs_btcv/best_model.pth    <- best checkpoint (highest mean val Dice)
  outputs_btcv/history.json      <- per-epoch metrics for all organs


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 6 — Visualize
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Before training (raw slices + ground truth only):
  python visualize.py --data_dir BTCV

  # After training (with model predictions):
  python visualize.py --data_dir BTCV --checkpoint outputs_btcv/best_model.pth

  # Show more samples:
  python visualize.py --data_dir BTCV --checkpoint outputs_btcv/best_model.pth --n 12

Colour coding:
  Red       — aorta
  Green     — gallbladder
  Blue      — spleen
  Yellow    — left kidney
  Orange    — right kidney
  Purple    — liver
  Cyan      — stomach
  Pink      — pancreas




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture details relevant to BTCV
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  3D → 2D conversion:
    Each NIfTI volume (H×W×D) is split into D axial slices.
    Slices are resized to patch_size×patch_size (224×224).
    Background-only slices are skipped during training.
    HU windowing: clip to [-125, 275] → [0, 1] (soft-tissue window).

  TransformerDecoder changes vs DRIVE:
    DRIVE:  num_queries=2   (background + vessel)
    BTCV:   num_queries=9   (background + 8 organs)
    Paper uses num_queries >> n_classes to reduce false negatives, but
    we use num_queries=n_classes for simplicity with fixed matching.
    num_dec_layers=3 (C2F stages) per paper Table 1 for BTCV/Synapse.

  Training split (hard split, Fu et al. 2020):
    Train: volumes 0001-0020 (18 volumes, ~1800 foreground slices)
    Val:   volumes 0021-0032 (12 volumes, ~1200 slices)
