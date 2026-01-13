# ─────────────────────────────────────────────
# decoder.py — Multi-Task Decoder (Denoising + Inpainting)
# For vit_small_patch16_dinov3 (embed_dim=384, patch_size=16)
# - Residual denoiser with noisy skip + noise-level conditioning
# - Optional cross-attention (token / spatial / global)
# ─────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Reshape helpers
# ─────────────────────────────────────────────
def reshape_vit_features_to_map(features, image_size, patch_size):
    """
    Convert ViT patch embeddings [B, N, C] → feature map [B, C, H', W'].
    Handles DINOv3 outputs (with CLS + register tokens).
    """
    B, N, C = features.shape

    grid_h = grid_w = image_size // patch_size
    N_expected = grid_h * grid_w

    # Drop extra tokens (CLS + registers) if present: keep last N_expected patch tokens
    if N > N_expected:
        features = features[:, -N_expected:, :]  # [B, N_expected, C]

    feat_map = features.permute(0, 2, 1).reshape(B, C, grid_h, grid_w)
    return feat_map


def map_to_tokens(feat_map):
    """
    Convert feature map [B, C, H, W] → tokens [B, N, C],
    where N = H * W.
    """
    B, C, H, W = feat_map.shape
    tokens = feat_map.flatten(2).permute(0, 2, 1)  # [B, HW, C]
    return tokens


# ─────────────────────────────────────────────
# Core upsampling block
# ─────────────────────────────────────────────
class ConvUpsampleBlock(nn.Module):
    """Upsample + ConvTranspose2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
# Cross-attention modules
# ─────────────────────────────────────────────
class CrossAttentionTokens(nn.Module):
    """
    Token-level cross attention:
    Q comes from one task, K/V from the other.
    All inputs: [B, N, C] (batch_first=True).
    """
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, q_tokens, kv_tokens):
        out, _ = self.attn(q_tokens, kv_tokens, kv_tokens)
        return out


class CrossAttention2D(nn.Module):
    """
    Spatial cross attention over feature maps [B, C, H, W].
    Internally flattens to [B, HW, C], applies MultiheadAttention,
    then reshapes back.
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x_q, x_kv):
        # x_q, x_kv: [B, C, H, W]
        B, C, H, W = x_q.shape
        q = x_q.flatten(2).transpose(1, 2)    # [B, HW, C]
        kv = x_kv.flatten(2).transpose(1, 2)  # [B, HW, C]
        out, _ = self.mha(q, kv, kv)          # [B, HW, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return out


# ─────────────────────────────────────────────
# Denoising Decoder (Residual + noisy skip + noise conditioning)
# ─────────────────────────────────────────────
class DenoisingDecoder(nn.Module):
    """
    Residual denoiser with:
    - Input: ViT patch features [B, N, C]
    - Skip: optional skip from x_noisy in image space
    - Noise conditioning: scalar noise_level -> channel-wise modulation

    Predicts residual ε ≈ (x_noisy - x_clean), so that:
        x_denoised = x_noisy - ε
    No activation on output (handled in loss/forward wrapper).
    """
    def __init__(
        self,
        embed_dim=384,
        image_size=128,
        patch_size=16,
        use_noisy_skip: bool = True,
        use_noise_cond: bool = True,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.use_noisy_skip = use_noisy_skip
        self.use_noise_cond = use_noise_cond

        # Upsampling path from patch grid to full resolution
        self.up_path = nn.Sequential(
            ConvUpsampleBlock(embed_dim, 512),   # 8 -> 16
            ConvUpsampleBlock(512, 256),         # 16 -> 32
            ConvUpsampleBlock(256, 128),         # 32 -> 64
            ConvUpsampleBlock(128, 64),          # 64 -> 128
        )

        # If we concat noisy image at the final resolution
        in_last = 64 + (3 if use_noisy_skip else 0)

        # Optional noise conditioning: scalar -> per-channel gamma
        if self.use_noise_cond:
            self.noise_mlp = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, in_last),  # one gamma per channel
            )

        # Final conv that predicts residual (no activation)
        self.final_conv = nn.Conv2d(in_last, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, features, x_noisy=None, noise_level=None):
        """
        Args:
            features:    [B, N, C] ViT patch embeddings
            x_noisy:     [B, 3, H, W] noisy image (for skip), optional
            noise_level: [B, 1] or [B] scalar per image, optional

        Returns:
            residual: [B, 3, H, W]
        """
        B = features.size(0)

        # ViT tokens -> low-res feature map
        feat_map = reshape_vit_features_to_map(
            features, self.image_size, self.patch_size
        )                                      # [B, C=384, 8, 8] for 128x128
        h = self.up_path(feat_map)             # [B, 64, H, W]

        # ── 1) Noisy-image skip connection at full resolution ──
        if self.use_noisy_skip and x_noisy is not None:
            noisy_resized = F.interpolate(
                x_noisy,
                size=h.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )                                   # [B, 3, H, W]
            h = torch.cat([h, noisy_resized], dim=1)  # [B, 64+3, H, W]

        # ── 2) Noise-level conditioning via FiLM-like modulation ──
        if self.use_noise_cond and (noise_level is not None):
            # Ensure shape [B, 1]
            if noise_level.dim() == 1:
                noise_level = noise_level.unsqueeze(1)  # [B,1]

            gamma = self.noise_mlp(noise_level)         # [B, in_last]
            gamma = gamma.view(B, -1, 1, 1)             # [B, in_last,1,1]

            # Simple affine-style modulation: h * (1 + gamma)
            h = h * (1.0 + gamma)

        residual = self.final_conv(h)                   # [B, 3, H, W]
        return residual


# ─────────────────────────────────────────────
# Inpainting Decoder
# ─────────────────────────────────────────────
class InpaintingDecoder(nn.Module):
    """
    Reconstructs the full image, conditioned on the inpainting mask.
    Output is in [0,1] assuming images are normalized to [0,1].
    """
    def __init__(self, embed_dim=384, image_size=128, patch_size=16):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # Upsampling path
        self.up_path = nn.Sequential(
            ConvUpsampleBlock(embed_dim, 512),
            ConvUpsampleBlock(512, 256),
            ConvUpsampleBlock(256, 128),
            ConvUpsampleBlock(128, 64),
        )

        # Mask-guided reconstruction
        self.merge = nn.Sequential(
            nn.Conv2d(64 + 1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),  # outputs in [0, 1]
        )

    def forward(self, features, mask):
        feat_map = reshape_vit_features_to_map(
            features, self.image_size, self.patch_size
        )
        feat = self.up_path(feat_map)  # [B, 64, H, W]

        if mask.shape[1] != 1:
            mask = mask[:, 0:1, :, :]  # keep only one mask channel

        mask_resized = F.interpolate(
            mask,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        merged = torch.cat([feat, mask_resized], dim=1)  # [B, 65, H, W]
        return self.merge(merged)  # [B, 3, H, W]


# ─────────────────────────────────────────────
# Dual-task Decoder (Combines both tasks + optional cross-attention)
# ─────────────────────────────────────────────
class DualTaskDecoder(nn.Module):
    """
    Multi-task decoder with:
    - Residual denoiser (noisy skip + noise-level conditioning)
    - Inpainting branch
    - Optional cross-attention mechanisms:

    Args:
        use_noisy_skip:    enable x_noisy skip in denoiser
        use_noise_cond:    enable noise-level conditioning in denoiser
        use_token_xattn:   if True, apply token-level cross-attention
        use_spatial_xattn: if True, apply spatial cross-attention on feature maps
        use_global_xattn:  if True, apply global task-token attention
        num_heads:         number of attention heads for all attentions
    """
    def __init__(
        self,
        embed_dim=384,
        image_size=128,
        patch_size=16,
        use_noisy_skip: bool = True,
        use_noise_cond: bool = True,
        use_token_xattn: bool = False,
        use_spatial_xattn: bool = False,
        use_global_xattn: bool = False,
        num_heads: int = 4,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.use_token_xattn = use_token_xattn
        self.use_spatial_xattn = use_spatial_xattn
        self.use_global_xattn = use_global_xattn

        # --- Token-level cross-attention modules ---
        if self.use_token_xattn:
            self.token_proj_d = nn.Linear(embed_dim, embed_dim)
            self.token_proj_i = nn.Linear(embed_dim, embed_dim)
            self.token_xattn = CrossAttentionTokens(embed_dim, num_heads=num_heads)

        # --- Spatial cross-attention module ---
        if self.use_spatial_xattn:
            self.spatial_xattn = CrossAttention2D(channels=embed_dim, num_heads=num_heads)

        # --- Global task-token attention ---
        if self.use_global_xattn:
            self.global_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.global_ln = nn.LayerNorm(embed_dim)

        # Task-specific decoders
        self.denoise_decoder = DenoisingDecoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            use_noisy_skip=use_noisy_skip,
            use_noise_cond=use_noise_cond,
        )
        self.inpaint_decoder = InpaintingDecoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
        )

    def forward(self, features, mask=None, x_noisy=None, noise_level=None):
        """
        Args:
            features:    [B, N, C] encoder tokens
            mask:        [B, 1, H, W] or None
            x_noisy:     [B, 3, H, W] (for skip in denoiser)
            noise_level: [B, 1] or [B] (for conditioning)

        Returns:
            residual:  [B, 3, H, W] predicted noise/residual
            inpainted: [B, 3, H, W] or None
        """
        # Start from shared encoder tokens
        f_d = features  # for denoising branch
        f_i = features  # for inpainting branch

        # -------------------------------------------------
        # 1) Token-level cross-attention (optional)
        # -------------------------------------------------
        if self.use_token_xattn:
            q_d = self.token_proj_d(f_d)
            q_i = self.token_proj_i(f_i)

            # Denoiser queries inpainting tokens
            f_d = self.token_xattn(q_d, q_i)
            # Inpainter queries denoising tokens
            f_i = self.token_xattn(q_i, q_d)

        # -------------------------------------------------
        # 2) Spatial cross-attention (optional)
        # -------------------------------------------------
        if self.use_spatial_xattn:
            map_d = reshape_vit_features_to_map(f_d, self.image_size, self.patch_size)
            map_i = reshape_vit_features_to_map(f_i, self.image_size, self.patch_size)

            map_d = self.spatial_xattn(map_d, map_i)
            map_i = self.spatial_xattn(map_i, map_d)

            f_d = map_to_tokens(map_d)
            f_i = map_to_tokens(map_i)

        # -------------------------------------------------
        # 3) Global task-token attention (optional)
        # -------------------------------------------------
        if self.use_global_xattn:
            g_d = f_d.mean(dim=1)  # [B, C]
            g_i = f_i.mean(dim=1)  # [B, C]

            tokens = torch.stack([g_d, g_i], dim=1)  # [B, 2, C]
            tokens = self.global_ln(tokens)

            tokens_out, _ = self.global_attn(tokens, tokens, tokens)
            g_d_new, g_i_new = tokens_out[:, 0, :], tokens_out[:, 1, :]  # [B, C] each

            f_d = f_d + g_d_new.unsqueeze(1)
            f_i = f_i + g_i_new.unsqueeze(1)

        # -------------------------------------------------
        # Decode each task
        # -------------------------------------------------
        residual = self.denoise_decoder(f_d, x_noisy=x_noisy, noise_level=noise_level)
        inpainted = self.inpaint_decoder(f_i, mask) if mask is not None else None

        return residual, inpainted
