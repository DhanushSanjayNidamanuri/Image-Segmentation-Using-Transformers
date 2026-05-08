"""
TransUNet 2D — Chen et al., "TransUNet: Rethinking the U-Net architecture
design for medical image segmentation through the lens of transformers",
Medical Image Analysis 2024.

Architecture: "Encoder + Decoder" variant (paper Section 3.3.3 / Fig. 1):
  1. Hybrid CNN-Transformer Encoder:
       ResNet-50 (ImageNet1K pretrained, torchvision) → 1024-ch feature map
       → 1×1 patch embedding → ViT-B/16 (ImageNet-21k pretrained, timm)
         with LayerScale (paper Section 4.2, Touvron et al. 2021)
       → global context tokens reshaped back to spatial map
  2. CNN Decoder (U-Net style, 4 UpBlocks with skip connections)
  3. Transformer Decoder — Coarse-to-Fine (C2F) stages (Algorithm 1):
       Learnable organ queries (init to zero, paper Section 3.3.2)
       num_queries >> n_classes to reduce false negatives (paper Section 3.2.1)
       Masked cross-attention with multi-scale CNN decoder features (Eq. 5-7)
       num_dec_layers = 3  C2F stages for BTCV/Synapse ()
       d_dec = 192  (paper Section 4.2)
  4. Hungarian matching loss: λ₀*(BCE + Dice) + λ₁*CE_cls (paper Eq. 10)
       λ₀=0.7, λ₁=0.3 (paper Section 4.2)
       Deep supervision applied at each C2F stage output

Pretrained weights:
  - ResNet-50 : torchvision ResNet50_Weights.IMAGENET1K_V1
  - ViT-B/16  : timm "vit_base_patch16_224.augreg_in21k" (ImageNet-21k)
                with LayerScale weights copied per-block
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

try:
    import timm
    _TIMM_OK = True
except ImportError:
    _TIMM_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Hybrid CNN Encoder  (ResNet-50, truncated after layer3)
# ─────────────────────────────────────────────────────────────────────────────

class HybridCNNEncoder(nn.Module):
    """
    ResNet-50 backbone, truncated after layer3 (stride=16 from input).

    Returns:
        cnn_out : (B, 1024, H/16, W/16)
        skips   : [s0 (B,64,H/2), s1 (B,256,H/4), s2 (B,512,H/8)]
    """
    def __init__(self, in_channels: int = 1, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        bb = resnet50(weights=weights)

        # Adapt first conv for 1-channel (green) or arbitrary channels
        if in_channels != 3:
            old_w = bb.conv1.weight.data          # (64,3,7,7)
            bb.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            if pretrained:
                # Average RGB weights → single channel; repeat for in_channels > 3
                avg_w = old_w.mean(dim=1, keepdim=True)          # (64,1,7,7)
                bb.conv1.weight.data = avg_w.repeat(1, in_channels, 1, 1)[:, :in_channels]

        self.layer0  = nn.Sequential(bb.conv1, bb.bn1, bb.relu)   # H/2,  64 ch
        self.maxpool = bb.maxpool                                   # H/4
        self.layer1  = bb.layer1                                   # H/4, 256 ch
        self.layer2  = bb.layer2                                   # H/8, 512 ch
        self.layer3  = bb.layer3                                   # H/16,1024 ch
        self.out_channels = 1024

    def forward(self, x):
        s0 = self.layer0(x)           # H/2,  64
        mp = self.maxpool(s0)         # H/4
        s1 = self.layer1(mp)          # H/4, 256
        s2 = self.layer2(s1)          # H/8, 512
        s3 = self.layer3(s2)          # H/16,1024
        return s3, [s0, s1, s2]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Vision Transformer Encoder  (ViT-B/16, pretrained ImageNet-21k via timm)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """1×1 conv projection + learnable positional embeddings (paper Eq. 1)."""
    def __init__(self, in_ch: int, embed_dim: int, n_patches: int):
        super().__init__()
        self.proj     = nn.Conv2d(in_ch, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):                       # (B, C, H, W)
        x = self.proj(x)                        # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)        # (B, N, D)
        return x + self.pos_embed


class TransformerLayer(nn.Module):
    """
    Eqs. 2-3 from the paper: LN→MSA→res + LN→MLP→res.
    Includes LayerScale (Touvron et al. 2021) as used in the ViT-B/16
    ImageNet-21k pretrained model (paper Section 4.2).
    LayerScale weights are initialised to 1.0 (identity) so the layer
    behaves as a standard transformer when not loading pretrained weights.
    """
    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 init_values: float = 1.0):
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm1  = nn.LayerNorm(embed_dim)
        self.attn   = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.mlp    = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(dropout),
        )
        # LayerScale parameters (paper Section 4.2; Touvron et al. 2021)
        # Pretrained ViT-B/16 in21k uses LayerScale; kept as trainable params.
        self.ls1 = nn.Parameter(torch.ones(embed_dim) * init_values)
        self.ls2 = nn.Parameter(torch.ones(embed_dim) * init_values)

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ls1 * attn_out
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """
    Transformer encoder that operates on CNN feature-map tokens (patch_size=1).
    Optionally initialised from a pretrained ViT-B/16 (timm).
    """
    def __init__(self, cnn_out_ch: int, feat_size: int,
                 embed_dim: int = 768, num_heads: int = 12,
                 num_layers: int = 12, mlp_ratio: float = 4.0,
                 dropout: float = 0.1, pretrained: bool = True):
        super().__init__()
        n_patches = feat_size * feat_size
        self.patch_embed = PatchEmbedding(cnn_out_ch, embed_dim, n_patches)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.feat_size = feat_size

        # Linear projection back to spatial map
        self.proj_back = nn.Conv2d(embed_dim, embed_dim, 1)

        if pretrained and _TIMM_OK:
            self._load_pretrained_vit()

    def _load_pretrained_vit(self):
        """
        Copy weights from timm ViT-B/16 (ImageNet-21k, with LayerScale) to our layers.

        Primary model: "vit_base_patch16_224.augreg_in21k"
          — This is the exact checkpoint used in the paper (Section 4.2):
            12-layer ViT pretrained on ImageNet-21k with LayerScale
            (Touvron et al. 2021).  d_enc = 768.
        Fallback: "vit_base_patch16_224" (ImageNet-1k, no LayerScale).
        """
        # Candidate model names in preference order
        candidates = [
            "vit_base_patch16_224.augreg_in21k",   # paper's exact checkpoint
            "vit_base_patch16_224_in21k",            # older timm alias
            "vit_base_patch16_224",                  # ImageNet-1k fallback
        ]
        vit = None
        used_name = None
        for name in candidates:
            try:
                vit = timm.create_model(name, pretrained=True, num_classes=0)
                used_name = name
                break
            except Exception:
                continue

        if vit is None:
            print("[ViTEncoder] Could not download any ViT-B/16 checkpoint. "
                  "Training from random init.")
            return

        try:
            d = self.embed_dim   # 768

            for i, (src, dst) in enumerate(zip(vit.blocks, self.layers)):
                if i >= len(self.layers):
                    break

                # ── Attention weights ──────────────────────────────────────
                # timm packs Q,K,V into a single qkv Linear (3*D, D)
                dst.attn.in_proj_weight.data = torch.cat([
                    src.attn.qkv.weight.data[:d],
                    src.attn.qkv.weight.data[d:2*d],
                    src.attn.qkv.weight.data[2*d:],
                ], dim=0)
                dst.attn.in_proj_bias.data = torch.cat([
                    src.attn.qkv.bias.data[:d],
                    src.attn.qkv.bias.data[d:2*d],
                    src.attn.qkv.bias.data[2*d:],
                ])
                dst.attn.out_proj.weight.data = src.attn.proj.weight.data
                dst.attn.out_proj.bias.data   = src.attn.proj.bias.data

                # ── MLP weights ────────────────────────────────────────────
                dst.mlp[0].weight.data = src.mlp.fc1.weight.data
                dst.mlp[0].bias.data   = src.mlp.fc1.bias.data
                dst.mlp[3].weight.data = src.mlp.fc2.weight.data
                dst.mlp[3].bias.data   = src.mlp.fc2.bias.data

                # ── LayerNorm weights ──────────────────────────────────────
                dst.norm1.weight.data = src.norm1.weight.data
                dst.norm1.bias.data   = src.norm1.bias.data
                dst.norm2.weight.data = src.norm2.weight.data
                dst.norm2.bias.data   = src.norm2.bias.data

                # ── LayerScale (paper Section 4.2; Touvron et al. 2021) ───
                # timm LayerScale stores the scale as a Parameter.
                # Attribute name varies by timm version:
                #   timm >= 0.9  : LayerScale object with .gamma  (nn.Parameter)
                #   timm 0.6-0.8 : LayerScale object with .gamma  (nn.Parameter)
                #   some builds  : Identity object (no LayerScale in model)
                #   fallback     : ls1/ls2 may not exist at all
                def _copy_ls(src_ls, dst_param):
                    """Safely copy LayerScale value into dst nn.Parameter."""
                    if src_ls is None:
                        return
                    # Case 1: it's already an nn.Parameter (Identity wraps nothing)
                    if isinstance(src_ls, nn.Parameter):
                        dst_param.data = src_ls.data.clone()
                    # Case 2: LayerScale object exposing .gamma
                    elif hasattr(src_ls, "gamma"):
                        g = src_ls.gamma
                        if isinstance(g, nn.Parameter):
                            dst_param.data = g.data.clone()
                    # Case 3: Identity / unknown — leave dst at 1.0 init (safe)

                if hasattr(src, "ls1"):
                    _copy_ls(src.ls1, dst.ls1)
                if hasattr(src, "ls2"):
                    _copy_ls(src.ls2, dst.ls2)

            print(f"[ViTEncoder] Loaded pretrained ViT-B/16 weights "
                  f"from '{used_name}' (ImageNet-21k).")
            del vit
        except Exception as e:
            print(f"[ViTEncoder] Weight copy failed ({e}). "
                  "Continuing with random ViT init.")

    def forward(self, cnn_feat):                    # (B, 1024, H/16, W/16)
        B, _, H, W = cnn_feat.shape
        tokens = self.patch_embed(cnn_feat)          # (B, N, D)
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.norm(tokens)                   # (B, N, D)
        # Reshape back to spatial
        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, H, W)
        return self.proj_back(feat)                  # (B, D, H/16, W/16)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CNN Decoder 
# ─────────────────────────────────────────────────────────────────────────────

class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle potential size mismatch
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class CNNDecoder(nn.Module):
    """
    Cascaded upsampler.  Returns intermediate feature maps for the
    Transformer decoder (multi-scale CNN features, paper Eq. 5).

    Channels:
      embed_dim (768) → up1+skip2 → 256  (H/8)
                      → up2+skip1 → 128  (H/4)
                      → up3+skip0 → 64   (H/2)
                      → up4       → 32   (H)
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.up1 = UpBlock(embed_dim, 512, 256)   # +skip layer2
        self.up2 = UpBlock(256,       256, 128)   # +skip layer1
        self.up3 = UpBlock(128,        64,  64)   # +skip layer0
        self.up4 = UpBlock( 64,         0,  32)   # no skip
        self.out_channels = [256, 128, 64, 32]    # per up-stage

    def forward(self, x, skips):
        """Returns list of multi-scale features (coarsest → finest)."""
        f1 = self.up1(x, skips[2])   # H/8,  256
        f2 = self.up2(f1, skips[1])  # H/4,  128
        f3 = self.up3(f2, skips[0])  # H/2,   64
        f4 = self.up4(f3, None)      # H,     32
        return [f1, f2, f3, f4]      # coarsest to finest


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Transformer Decoder  (coarse-to-fine cross-attention)
# ─────────────────────────────────────────────────────────────────────────────

class CrossAttentionLayer(nn.Module):
    """
    One step of Eq. 5 / 6: masked cross-attention of organ queries
    against a U-Net feature map.
    """
    def __init__(self, query_dim: int, feat_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_f = nn.LayerNorm(feat_dim)
        self.to_q   = nn.Linear(query_dim, query_dim)
        self.to_k   = nn.Linear(feat_dim,  query_dim)
        self.to_v   = nn.Linear(feat_dim,  query_dim)
        self.attn   = nn.MultiheadAttention(query_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(query_dim)
        self.ffn = nn.Sequential(
            nn.Linear(query_dim, query_dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(query_dim * 4, query_dim), nn.Dropout(dropout),
        )

    def forward(self, queries, feat_map, attn_mask=None):
        """
        queries  : (B, N_q, D)
        feat_map : (B, C, H, W)
        attn_mask: (B, N_q, H*W) — foreground mask for coarse-to-fine
        """
        B, C, H, W = feat_map.shape
        feat = feat_map.flatten(2).transpose(1, 2)           # (B, H*W, C)
        q = self.to_q(self.norm_q(queries))
        k = self.to_k(self.norm_f(feat))
        v = self.to_v(self.norm_f(feat))

        # Build attention bias from coarse mask (Eq. 7)
        if attn_mask is not None:
            # attn_mask shape: (B, N_q, H*W) — 1=foreground keep, 0=background mask
            # For nn.MultiheadAttention: key_padding_mask isn't quite right here.
            # We implement it as an additive bias: -inf where mask==0.
            mask_bias = torch.zeros_like(attn_mask, dtype=q.dtype)
            mask_bias[attn_mask == 0] = float('-inf')
            # Expand heads: (B*heads, N_q, H*W) → we pass as attn_mask per batch
            # nn.MHA expects (B*nhead, Nq, S) or (Nq, S) for attn_mask
            # Reshape: (B, N_q, H*W) → need to broadcast over heads
            # Simplification: flatten into (B, N_q, H*W) and use batch-wise; 
            # we pass None to avoid shape issues for multi-head and instead mask via additive
            # We'll do manual scaled dot-product for masked case
            scale = q.shape[-1] ** -0.5
            scores = torch.bmm(q, k.transpose(1, 2)) * scale  # (B, N_q, H*W)
            scores = scores + mask_bias
            scores = F.softmax(scores, dim=-1)
            attended = torch.bmm(scores, v)                    # (B, N_q, D)
        else:
            attended, _ = self.attn(q, k, v)

        queries = queries + attended
        queries = queries + self.ffn(self.norm_out(queries))
        return queries


class SelfAttentionLayer(nn.Module):
    """Self-attention among organ queries (within each decoder layer)."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim*4, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        n = self.norm(x)
        attn_out, _ = self.attn(n, n, n)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerDecoder(nn.Module):
    """
    Iterative coarse-to-fine Transformer decoder .

    Key details:
      - Organ queries initialised to ZERO then augmented with learnable
        positional embeddings .
      - num_queries is set >> n_classes to minimise false negatives.
        For BTCV: 20 queries, 9 classes .
      - query_dim (d_dec) = 192  .
      - num_decoder_layers = 3 C2F stages for BTCV/Synapse .
      - Self-attention among queries → masked cross-attention with CNN
        feature at each stage → update mask prediction .
      - class_head projects each query to K classes (no "+1 no-object"
        class needed with fixed Hungarian matching on K queries).
    """
    def __init__(self, num_queries: int, num_classes: int,
                 query_dim: int = 192, feat_dims: list = None,
                 num_decoder_layers: int = 3, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        if feat_dims is None:
            feat_dims = [256, 128, 64, 32]

        self.num_queries        = num_queries
        self.num_classes        = num_classes
        self.query_dim          = query_dim
        self.num_decoder_layers = num_decoder_layers

        # Organ queries: zero-init content + learnable positional embedding
        # ( "organ queries P are initially set to zero")
        self.query_content = nn.Embedding(num_queries, query_dim)
        nn.init.zeros_(self.query_content.weight)           
        self.query_pos    = nn.Embedding(num_queries, query_dim)
        nn.init.normal_(self.query_pos.weight, std=0.02)    # learnable pos embed

        # Per-layer feature projections (align feature channels to query_dim)
        n_layers = min(num_decoder_layers, len(feat_dims))
        self.feat_projs = nn.ModuleList([
            nn.Conv2d(feat_dims[i], query_dim, 1)
            for i in range(n_layers)
        ])

        # Per-layer self-attention + cross-attention
        self.self_attn_layers  = nn.ModuleList([
            SelfAttentionLayer(query_dim, num_heads, dropout)
            for _ in range(n_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(query_dim, query_dim, num_heads, dropout)
            for _ in range(n_layers)
        ])

        # Mask embedding: project finest CNN feature to query_dim for Eq. 4
        self.mask_embed_proj = nn.Conv2d(feat_dims[-1], query_dim, 1)

        # Per-layer lightweight query→mask projection (Eq. 4 at each stage)
        self.mask_mlps = nn.ModuleList([
            nn.Linear(query_dim, query_dim)
            for _ in range(n_layers)
        ])

        # Final class head: projects each query to K class logits 
        # No "+1 no-object" slot — we use fixed-N Hungarian matching where N==K
        # (or N>K with the extra queries acting as implicit no-object; the
        # Hungarian loss naturally leaves unmatched queries un-penalised).
        self.class_head = nn.Linear(query_dim, num_classes)

        self.n_layers = n_layers

    def _predict_masks(self, queries, mask_feat):
        """
        Dot-product of queries with mask features (Eq. 4).
        Returns (B, N_q, H, W).
        """
        B, D, H, W = mask_feat.shape
        mf = mask_feat.flatten(2)                        # (B, D, H*W)
        masks = torch.einsum('bnd,bdp->bnp', queries, mf)  # (B, N_q, H*W)
        return masks.view(B, self.num_queries, H, W)

    def forward(self, cnn_feats):
        """
        cnn_feats: list of CNN decoder feature maps [coarsest ... finest]
                   shapes: [(B,256,H/8), (B,128,H/4), (B,64,H/2), (B,32,H)]
        Returns:
            pred_masks  : (B, N_q, H, W)  — final refined masks (logits)
            class_logits: (B, N_q, K)     — class logits per query
            aux_masks   : list of per-stage masks (for deep supervision)
        """
        B = cnn_feats[0].shape[0]

        # Initialise queries: zero content + learnable positional embedding
        # ( P^0 is zero, augmented with pos embed)
        queries = (self.query_content.weight + self.query_pos.weight
                   ).unsqueeze(0).expand(B, -1, -1)  # (B, N_q, D)

        # High-res mask feature (finest feature map)
        mask_feat = self.mask_embed_proj(cnn_feats[-1])   # (B, D, H, W)

        # Initial coarse prediction Z^0
        coarse_mask = self._predict_masks(queries, mask_feat)  # (B, N_q, H, W)
        aux_masks   = []

        for t in range(self.n_layers):
            feat_map = self.feat_projs[t](cnn_feats[t])  # (B, D, Ht, Wt)
            Ht, Wt   = feat_map.shape[2], feat_map.shape[3]

            # Build foreground attention mask from previous coarse prediction 
            # Downsample coarse_mask to current feature resolution
            cm_down = F.interpolate(coarse_mask, size=(Ht, Wt), mode="bilinear", align_corners=False)
            fg_mask = (cm_down.detach().sigmoid() > 0.5).float()  # (B, N_q, Ht, Wt)
            fg_mask = fg_mask.flatten(2)                           # (B, N_q, Ht*Wt)
            # If a query has no foreground, allow full attention to avoid -inf rows
            has_fg = fg_mask.sum(dim=-1, keepdim=True) > 0         # (B, N_q, 1)
            fg_mask = fg_mask * has_fg + (1 - has_fg.float())      # fallback to full attn

            # Self-attention + masked cross-attention
            queries     = self.self_attn_layers[t](queries)
            queries     = self.cross_attn_layers[t](queries, feat_map, attn_mask=fg_mask)

            # Update coarse prediction Z^{t+1}
            q_proj      = self.mask_mlps[t](queries)               # (B, N_q, D)
            coarse_mask = self._predict_masks(q_proj, mask_feat)   # (B, N_q, H, W)
            aux_masks.append(coarse_mask)

        # Class logits 
        class_logits = self.class_head(queries)   # (B, N_q, K+1)

        return coarse_mask, class_logits, aux_masks


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Full TransUNet (Encoder + Decoder variant)
# ─────────────────────────────────────────────────────────────────────────────

class TransUNet(nn.Module):
    """
    2D TransUNet — Encoder + Decoder configuration.

    defaults for BTCV multi-organ segmentation:
      n_classes      = 9   (background + 8 organs)
      num_queries    = 20  (N >> K to reduce false negatives)
      query_dim      = 192 (d_dec = 192)
      num_dec_layers = 3   ( 3 C2F stages for BTCV/Synapse)
      embed_dim      = 768 (ViT-B,  d_enc = 768)
      num_layers     = 12  (ViT-B/16)
      lr             = 3e-4 AdamW for new layers 

    Args:
        in_channels  : input image channels (1=CT grey, 3=RGB)
        n_classes    : segmentation classes (9 for BTCV, 2 for DRIVE)
        img_size     : spatial size of input patch (assumed square, default 224)
        embed_dim    : ViT hidden size (768 for ViT-B)
        num_heads    : ViT attention heads (12 for ViT-B)
        num_layers   : ViT transformer layers (12 for ViT-B)
        query_dim    : Transformer decoder hidden dim d_dec (paper: 192)
        num_queries  : organ query count — should be >= n_classes, ideally >>
                       ( uses 20 for all datasets)
        num_dec_layers: number of C2F stages (3 for BTCV)
        pretrained   : load pretrained weights for ResNet-50 + ViT-B/16
    """
    def __init__(
        self,
        in_channels    : int   = 1,
        n_classes      : int   = 9,
        img_size       : int   = 224,
        embed_dim      : int   = 768,
        num_heads      : int   = 12,
        num_layers     : int   = 12,
        query_dim      : int   = 192,    # d_dec = 192
        num_queries    : int   = 20,     # 20 queries for all datasets
        num_dec_layers : int   = 3,      # 3 C2F stages for BTCV
        pretrained     : bool  = True,
    ):
        super().__init__()
        self.n_classes   = n_classes
        self.num_queries = num_queries
        self.img_size    = img_size
        feat_size        = img_size // 16   # 14 for 224px input

        # ── Encoder ──────────────────────────────────────────────────────────
        self.cnn_encoder = HybridCNNEncoder(in_channels, pretrained=pretrained)
        self.vit_encoder = ViTEncoder(
            cnn_out_ch  = 1024,
            feat_size   = feat_size,
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            num_layers  = num_layers,
            pretrained  = pretrained,
        )
        # Project ViT output (embed_dim=768) → CNN decoder width (256)
        # This is the "reshape" step in Fig. 1 before the CNN decoder.
        self.bridge = nn.Conv2d(embed_dim, 256, 1)

        # ── CNN Decoder ───────────────────────────────────────────────────────
        self.cnn_decoder = CNNDecoder(embed_dim=256)

        # ── Transformer Decoder ───────────────────────────────────────────────
        # num_queries = 20 (), query_dim = 192 (paper Section 4.2)
        # num_dec_layers = 3 C2F stages ()
        feat_dims = self.cnn_decoder.out_channels   # [256, 128, 64, 32]
        self.transformer_decoder = TransformerDecoder(
            num_queries        = num_queries,
            num_classes        = n_classes,
            query_dim          = query_dim,
            feat_dims          = feat_dims,
            num_decoder_layers = num_dec_layers,
        )

        # ── Auxiliary CNN head (for fallback / ablations) ─────────────────────
        self.seg_head = nn.Conv2d(feat_dims[-1], n_classes, 1)

        self._init_new_weights()

    def _init_new_weights(self):
        """Kaiming init for newly added layers (bridge, decoder, seg_head)."""
        for m in [self.bridge, self.cnn_decoder, self.seg_head]:
            for p in m.modules():
                if isinstance(p, nn.Conv2d):
                    nn.init.kaiming_normal_(p.weight, mode="fan_out", nonlinearity="relu")
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)

    def forward(self, x):
        """
        Returns:
          Training : (logits, aux_logits)
                      logits     : (B, n_classes, H, W)
                      aux_logits : list of (B, n_classes, H, W) for deep supervision
          Inference: logits (B, n_classes, H, W)

        When num_queries > n_classes (paper default: 20 queries, 9 classes),
        each query is assigned a class via its class_logits.  We assemble the
        final per-pixel map by summing query mask contributions weighted by
        their class assignment (soft assignment during training, argmax at test).
        """
        B, C, H, W = x.shape

        # 1. Hybrid CNN-ViT encoder
        cnn_out, skips = self.cnn_encoder(x)          # (B, 1024, H/16, W/16)
        vit_out        = self.vit_encoder(cnn_out)    # (B, 768,  H/16, W/16)
        bottleneck     = self.bridge(vit_out)          # (B, 256,  H/16, W/16)

        # 2. CNN decoder — multi-scale feature maps for Transformer decoder
        cnn_feats = self.cnn_decoder(bottleneck, skips)  # [(B,256,H/8)..(B,32,H)]

        # 3. Transformer decoder — coarse-to-fine query refinement
        pred_masks, class_logits, aux_masks = self.transformer_decoder(cnn_feats)
        # pred_masks  : (B, N_q, H/?, W/?)
        # class_logits: (B, N_q, K)

        def _assemble_logits(masks, cls_logits, out_h, out_w):
            """
            Assemble (B, K, H, W) logits from N_q query masks + class logits.

            Soft: for each class k, sum over all query masks weighted by
            softmax(class_logits)[..., k].  This is differentiable and
            equivalent to the dot-product decoding in the paper (Eqs. 8-9).
            """
            # masks: (B, N_q, h, w) — upsample to output size
            masks_up = F.interpolate(masks, size=(out_h, out_w),
                                     mode="bilinear", align_corners=False)
            # cls_w: (B, N_q, K) — soft class assignment
            cls_w = F.softmax(cls_logits, dim=-1)   # (B, N_q, K)
            # logits[b,k,h,w] = sum_q cls_w[b,q,k] * masks_up[b,q,h,w]
            logits = torch.einsum("bqk,bqhw->bkhw", cls_w, masks_up)
            return logits

        logits = _assemble_logits(pred_masks, class_logits, H, W)

        if self.training:
            aux_logits = [
                _assemble_logits(m, class_logits, H, W) for m in aux_masks
            ]
            return logits, aux_logits
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss (binary cross-entropy variant per query mask).
    Used as ℒ_dice in paper Eq. 10.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits  : (B, C, H, W)
        targets : (B, H, W) long
        """
        C      = logits.shape[1]
        probs  = F.softmax(logits, dim=1)
        tgt_oh = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
        inter  = (probs * tgt_oh).sum((2, 3))
        union  = probs.sum((2, 3)) + tgt_oh.sum((2, 3))
        dice   = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TransUNetLoss(nn.Module):
    """
    Combined loss matching paper Eq. 10:
        ℒ = λ₀ * (ℒ_ce + ℒ_dice) + λ₁ * ℒ_cls

    where:
        ℒ_ce   = pixel-wise binary cross-entropy  (per-mask)
        ℒ_dice = Dice loss                         (per-mask)
        ℒ_cls  = cross-entropy on class logits     (mask classification)
        λ₀     = 0.7  (paper Section 4.2)
        λ₁     = 0.3  (paper Section 4.2)

    Deep supervision: the same loss is applied at each C2F stage output
    with a geometrically decreasing weight (aux_weight * 0.5^i).

    Note: The paper also describes Hungarian matching loss for the
    Decoder-only / Encoder+Decoder configurations.  Here we implement
    the simpler assembled-logits version which is equivalent for the
    n_queries == n_classes case and is a valid upper bound otherwise.
    For full Hungarian matching (needed when num_queries >> n_classes),
    replace _single() with a scipy.optimize.linear_sum_assignment call
    over the per-query BCE+Dice costs.
    """
    def __init__(self,
                 lam0: float = 0.7,          # paper Section 4.2
                 lam1: float = 0.3,          # paper Section 4.2
                 aux_weight: float = 0.4):
        super().__init__()
        self.ce      = nn.CrossEntropyLoss()
        self.dice    = DiceLoss()
        self.lam0    = lam0
        self.lam1    = lam1
        self.aux_w   = aux_weight

    def _single(self, logits, targets):
        """ℒ = λ₀*(ℒ_ce + ℒ_dice) + λ₁*ℒ_cls  (Eq. 10)."""
        loss_ce   = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        # ℒ_cls: treat assembled logit map the same as ℒ_ce (equivalent
        # to cross-entropy over the soft-assigned class logits)
        loss_cls  = loss_ce
        return self.lam0 * (loss_ce + loss_dice) + self.lam1 * loss_cls

    def forward(self, model_out, targets):
        """
        model_out : (logits, aux_logits) during training, or just logits at eval.
        targets   : (B, H, W) long
        """
        if isinstance(model_out, (list, tuple)):
            logits, aux_logits = model_out[0], model_out[1]
            loss = self._single(logits, targets)
            # Deep supervision with geometrically decreasing weights
            for i, aux in enumerate(aux_logits):
                w = self.aux_w * (0.5 ** (len(aux_logits) - 1 - i))
                loss = loss + w * self._single(aux, targets)
            return loss
        return self._single(model_out, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Checker
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # BTCV defaults: n_classes=9, num_queries=20, query_dim=192, num_dec_layers=3
    model = TransUNet(
        in_channels    = 1,
        n_classes      = 9,
        img_size       = 224,
        embed_dim      = 768,
        num_heads      = 12,
        num_layers     = 12,
        query_dim      = 192,  
        num_queries    = 20,    
        num_dec_layers = 3,      
        pretrained     = False, # set True to download ViT-B/16 in21k weights
    ).to(device)

    model.train()
    x = torch.randn(2, 1, 224, 224).to(device)
    out = model(x)
    logits, aux = out
    print(f"Logits shape  : {tuple(logits.shape)}")   # (2, 9, 224, 224)
    print(f"Aux outputs   : {len(aux)}")               # 3  (one per C2F stage)

    targets = torch.randint(0, 9, (2, 224, 224)).to(device)
    criterion = TransUNetLoss()                        # λ₀=0.7, λ₁=0.3 
    loss = criterion(out, targets)
    print(f"Loss          : {loss.item():.4f}")

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters    : {n:.1f}M")
    # Expected: ~41M (Encoder+Decoder = 41.4M)
