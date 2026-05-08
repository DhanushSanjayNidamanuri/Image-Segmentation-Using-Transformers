TransUNet (3D-style, 2D adaptation) — DRIVE Retinal Vessel Segmentation
========================================================================
Reference: Chen et al., "3D TransUNet: Advancing Medical Image Segmentation
           through Vision Transformers", arXiv 2310.07781

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Download the DRIVE dataset (~27 MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction

After unzipping you should have:

  DRIVE/
    training/
      images/        <- 21_training.tif ... 40_training.tif   (20 files)
      1st_manual/    <- 21_manual1.gif  ... 40_manual1.gif    (20 files)
      mask/          <- 21_training_mask.gif ... (20 files)
    test/
      images/        <- not used (no labels available)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. Installing dependencies
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # PyTorch with CUDA 12 (RTX 3060):
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

  # Required: timm for pretrained ViT-B/16 weights
  pip install timm

  # Other deps:
  pip install Pillow numpy matplotlib

  # Verify GPU + timm:
  python -c "import torch; import timm; print(torch.cuda.is_available())"
  # should print: True

NOTE: timm is required to load pretrained ViT-B/16 (ImageNet-21k) weights.
      Without it, training still works but starts from random ViT initialisation.
      The ResNet-50 backbone always loads ImageNet weights via torchvision.


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. Folder layout
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  my_project/
    transunet.py
    dataset.py
    train_eval.py
    visualize.py
    README.txt
    DRIVE/             <- dataset folder


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  cd my_project

  # Recommended for RTX 3060 (~30-45 min):
  python train_eval.py --data_dir DRIVE

  # Reduce batch if you see OOM errors:
  python train_eval.py --data_dir DRIVE --batch_size 2

  # Train longer for higher accuracy (~1.5 hr):
  python train_eval.py --data_dir DRIVE --epochs 60

  # Without pretrained ViT (faster start, ~2pp lower Dice):
  python train_eval.py --data_dir DRIVE --no_pretrained

  # RGB input (3 channels) instead of green channel:
  python train_eval.py --data_dir DRIVE --in_channels 3

MEMORY NOTE:
  The full ViT-B/16 encoder + Transformer decoder requires ~8-10 GB VRAM
  at batch_size=4, patch_size=224.
  If you get OOM errors:
    - Reduce batch_size to 2
    - Reduce num_layers to 6 (--num_layers 6)
    - Reduce embed_dim to 384 (--embed_dim 384 --num_heads 6) — ViT-Small

All train_eval.py command-line options:

  --data_dir       Path to DRIVE/ folder               (required)
  --epochs         Number of training epochs           (default: 30)
  --batch_size     Images per batch                    (default: 4)
  --patch_size     Crop size fed to model              (default: 224)
  --n_train        Patches per training epoch          (default: 2000)
  --n_val          Patches per validation epoch        (default: 400)
  --lr             AdamW learning rate                 (default: 1e-4)
  --in_channels    1=green channel, 3=RGB              (default: 1)
  --embed_dim      ViT hidden size                     (default: 768)
  --num_layers     ViT transformer layers              (default: 12)
  --num_heads      ViT attention heads                 (default: 12)
  --query_dim      Transformer decoder hidden dim      (default: 256)
  --num_dec_layers Transformer decoder layers          (default: 4)
  --pretrained     Use pretrained ResNet-50 + ViT-B/16 (default: True)
  --no_pretrained  Disable pretrained weights
  --amp            Mixed precision training            (default: True)
  --no_amp         Disable AMP
  --num_workers    DataLoader workers (0 on Windows)   (default: 0)
  --save_dir       Where to save outputs               (default: outputs/)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5. Expected Output
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Epoch  TrLoss   VaLoss    Dice    Sens    Spec     Acc   Time
 ---------------------------------------------------------------
     1  0.5123   0.4901  0.5814  0.7012  0.9501  0.9301  55.2s
     5  0.3401   0.3201  0.7123  0.7601  0.9701  0.9521  54.0s
    30  0.1801   0.1923  0.8101  0.8301  0.9831  0.9701  54.0s

Expected results after 30 epochs (using some pretrained weights):
  Dice        ~0.80–0.82
  Sensitivity ~0.82
  Specificity ~0.98
  Accuracy    ~0.97

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Outputs saved automatically
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  outputs/best_model.pth   <- best checkpoint (highest val Dice)
  outputs/history.json     <- loss + metrics for every epoch


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
6. Visualization of the Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
visualize.py — Visualize DRIVE images, ground truth masks, and model predictions.

Usage:
    # Just show raw images + ground truth (no model needed):
    python visualize.py --data_dir DRIVE

    # Show images + ground truth + model predictions:
    python visualize.py --data_dir DRIVE --checkpoint outputs/best_model.pth

    # Control how many samples to show:
    python visualize.py --data_dir DRIVE --checkpoint outputs/best_model.pth --n 8

Colour coding in prediction column:
  Red    — correctly detected vessels  (true positive)
  Orange — falsely predicted vessels   (false positive)
  Blue   — vessels the model missed    (false negative)




━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Architecture details (transunet.py)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  HybridCNNEncoder   ResNet-50 truncated after layer3 → (B,1024,H/16,W/16)
                     Skip connections: s0(H/2,64), s1(H/4,256), s2(H/8,512)

  ViTEncoder         PatchEmbedding (patch_size=1, each spatial loc = token)
                     + 12 Transformer layers (ViT-B/16)
                     → (B, 768, H/16, W/16)

  CNNDecoder         4× UpBlock with skip connections
                     → [f1(H/8,256), f2(H/4,128), f3(H/2,64), f4(H,32)]

  TransformerDecoder  n_classes=2 learnable organ queries
                      Per layer: self-attention → masked cross-attention
                      → update coarse mask Z^t → new attention mask
                      Final output: (B, 2, H, W) logits


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Files in this project
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  transunet.py   — Model (HybridCNNEncoder + ViTEncoder + CNNDecoder + TransformerDecoder)
  dataset.py     — DRIVE loader (16 train / 4 val split from training/)
  train_eval.py  — Training loop, metrics, checkpointing
  visualize.py   — Grid visualizer for images, ground truth and predictions
  README.txt     — This file
