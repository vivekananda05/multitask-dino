# ─────────────────────────────────────────────
# model.py — Joint Denoising + Inpainting (Fused Input, Residual Denoising)
# Encoder receives a fused input image
# Decoder:
#   - predicts residual for denoising
#   - reconstructs masked regions for inpainting
# ─────────────────────────────────────────────

import torch
import torch.nn as nn

from encoder import DinoV3SmallEncoder
from decoder import DualTaskDecoder


class DenoiseInpaintModel(nn.Module):
    """
    End-to-end model combining:
    - Pretrained DINOv3-Small encoder (with LoRA adapters)
    - Dual-task decoder for denoising (residual) & inpainting

    The encoder processes a fused input (noisy + masked projected to 3 channels).
    For denoising, the decoder predicts a residual r ≈ (x_noisy - x_clean),
    and we reconstruct clean as x_denoised = x_noisy - r.
    """

    def __init__(
        self,
        model_name="vit_small_patch16_dinov3",  # DINOv3-small
        image_size=128,
        patch_size=16,                          # DINOv3-small patch size
        embed_dim=384,                          # DINOv3-small output dim
        lora_r=8,
        lora_alpha=16,
        pretrained=True,
    ):
        super().__init__()

        # Shared DINOv3-Small Encoder with LoRA adapters
        self.encoder = DinoV3SmallEncoder(
            model_name=model_name,
            pretrained=pretrained,
            image_size=image_size,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # Dual-task Decoder (for denoising + inpainting)
        self.decoder = DualTaskDecoder(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            use_noisy_skip=True,
            use_noise_cond=True,
            use_token_xattn=True,
            use_spatial_xattn=True,
            use_global_xattn=True,
        )


    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────
    def forward(self, x_fused, x_noisy=None, mask=None, clamp_output=True, noise_level=None):
        """
        Args:
            x_fused: [B, 3, H, W]
                Fused representation (e.g., noisy + masked + possibly channels projected).
            x_noisy: [B, 3, H, W] or None
                The original noisy image. If provided, we return a denoised image
                as x_noisy - residual. If None, 'denoised' will just be the raw
                residual.
            mask: [B, 1, H, W] or None
                Binary inpainting mask (1 = region to inpaint).
            clamp_output: bool
                If True and x_noisy is provided, clamp denoised output to [0, 1].

        Returns:
            dict with:
                "features":         encoder patch embeddings [B, N, C]
                "residual":         predicted residual/noise [B, 3, H, W]
                "denoised":         reconstructed clean image (if x_noisy is not None)
                "inpainted":        reconstructed inpainted image (if mask not None)
        """
        features = self.encoder(x_fused)  # [B, N, 384]
        
        residual, inpainted = self.decoder(
            features, mask=mask, x_noisy=x_noisy, noise_level=noise_level
        )
        if x_noisy is not None:
            denoised = x_noisy - residual
            if clamp_output:
                denoised = denoised.clamp(0.0, 1.0)
        else:
            denoised = residual  # raw residual if we don't have x_noisy

        return {
            "features": features,
            "residual": residual,
            "denoised": denoised,
            "inpainted": inpainted,
        }

    # ─────────────────────────────────────────────
    # Training Utilities
    # ─────────────────────────────────────────────
    def enable_lora_training(self):
        """
        Enable training for LoRA adapters + decoder, freeze the rest.
        Assumes LoRA parameters contain 'lora' in their names.
        """
        for name, p in self.encoder.named_parameters():
            p.requires_grad = "lora" in name.lower()
        for p in self.decoder.parameters():
            p.requires_grad = True

    def get_trainable_params(self):
        """Return trainable parameters (LoRA + decoder)."""
        return [p for p in self.parameters() if p.requires_grad]
