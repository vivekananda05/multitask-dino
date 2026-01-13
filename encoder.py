
# encoder.py — DINOv3 Small

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from lora import inject_lora_to_last_block
import math


class DinoV3SmallEncoder(nn.Module):
    """
    Universal encoder built from pretrained DINOv1/v3 ViTs (via timm).
    Compatible with vit_small_patch16_dinov3 (embed_dim=384)
    - Automatically resizes positional embeddings for arbitrary image sizes.
    - Injects LoRA adapters into the last transformer block.
    - Supports both DINOv1 ('vit_base_patch16_dino') and DINOv3 ('vit_small_patch16_dinov3').
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3",
        pretrained: bool = True,
        image_size: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.image_size = image_size

        # --- Load pretrained backbone ---
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        print(f"[Encoder] Loaded pretrained backbone: {model_name}")

        # --- Patch embed fix: override img_size for new resolution ---
        if hasattr(self.backbone, "patch_embed"):
            patch_embed = self.backbone.patch_embed
            if hasattr(patch_embed, "img_size"):
                patch_embed.img_size = (image_size, image_size)
            print(f"[Encoder] Adjusted patch_embed for {image_size}×{image_size} inputs.")

        # --- Freeze backbone weights ---
        for p in self.backbone.parameters():
            p.requires_grad = False

        # --- Inject LoRA adapters into last block ---
        wrapped = inject_lora_to_last_block(self.backbone, r=lora_r, alpha=lora_alpha)
        print(f"[LoRA] Applied to layers: {wrapped}")

        # --- Cache patch size & positional embedding ---
        self.patch_size = getattr(self.backbone.patch_embed, "patch_size", 16)
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]

        self.pos_embed = getattr(self.backbone, "pos_embed", None)
        self.embed_dim = 384  # ✅ DINOv3-small feature dimension

    # ---------------------------------------------------
    # Helper: interpolate positional embeddings dynamically
    # ---------------------------------------------------
    def _resize_pos_embed(self, pos_embed, target_hw):
        """
        Interpolates positional embeddings to target patch grid.
        Args:
            pos_embed: [1, N+1, C] or [1, N, C]
            target_hw: (H_new, W_new)
        Returns:
            resized pos_embed tensor [1, N_new(+1), C]
        """
        if pos_embed is None:
            return None

        # Separate class token if present
        if pos_embed.shape[1] in [197, 257]:
            cls_token = pos_embed[:, 0:1, :]
            pos_tokens = pos_embed[:, 1:, :]
        else:
            cls_token, pos_tokens = None, pos_embed

        dim = pos_tokens.shape[-1]
        num_patches = pos_tokens.shape[1]
        h_old = w_old = int(math.sqrt(num_patches))

        pos_tokens = pos_tokens.reshape(1, h_old, w_old, dim).permute(0, 3, 1, 2)
        h_new, w_new = target_hw

        pos_tokens = F.interpolate(pos_tokens, size=(h_new, w_new), mode="bicubic", align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, h_new * w_new, dim)

        if cls_token is not None:
            pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
        else:
            pos_embed = pos_tokens

        return pos_embed

    # ---------------------------------------------------
    # Forward pass
    # ---------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            features: [B, N, 384] sequence embeddings
        """
        H, W = x.shape[-2:]
        grid_h, grid_w = H // self.patch_size, W // self.patch_size

        # Resize positional embeddings dynamically
        if hasattr(self.backbone, "pos_embed") and self.pos_embed is not None:
            with torch.no_grad():
                resized = self._resize_pos_embed(self.pos_embed, (grid_h, grid_w))
                self.backbone.pos_embed = nn.Parameter(resized)

        feats = self.backbone.forward_features(x)

        # Convert to [B, N, C] if spatial
        if feats.ndim == 4:
            B, C, H, W = feats.shape
            feats = feats.flatten(2).transpose(1, 2)  # [B, N, C]

        return feats

    # ---------------------------------------------------
    # Utility methods
    # ---------------------------------------------------
    def lora_parameters(self):
        """Return only LoRA adapter parameters."""
        for name, param in self.named_parameters():
            if "lora" in name.lower():
                yield param

    def enable_lora_training(self):
        """Enable training only for LoRA adapters."""
        for name, param in self.named_parameters():
            param.requires_grad = "lora" in name.lower()
