# ─────────────────────────────────────────────
# train.py — Joint Denoising + Inpainting (Residual Denoising + GAN)
# Debug mask+masked saving integrated + Perceptual loss
# ─────────────────────────────────────────────

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
# >>> NEW: for perceptual (VGG) loss
import torchvision.models as models

from model import DenoiseInpaintModel
from dataloader import get_dataloader
from config import *

# ─────────────────────────────────────────────
# DEBUG FUNCTION
# ─────────────────────────────────────────────
def save_debug_images(epoch, global_step, noisy, masked, mask, inpainted,
                      save_root, max_samples=4, save_composite=False):
    """
    Saves masked, mask, and optional composite images into:
        <save_root>/debug_masks/
    """
    debug_dir = os.path.join(save_root, "debug_masks")
    os.makedirs(debug_dir, exist_ok=True)

    B = noisy.size(0)
    n = min(max_samples, B)

    for i in range(n):
        # Save masked
        masked_path = os.path.join(
            debug_dir, f"epoch{epoch:03d}_step{global_step:06d}_sample{i}_masked.png"
        )
        save_image(masked[i].cpu().clamp(0,1), masked_path)

        # Save mask
        if mask.shape[1] != 1:
            mask_use = mask[i:i+1, 0:1]
        else:
            mask_use = mask[i:i+1]
        mask_vis = mask_use.repeat(1, 3, 1, 1)
        mask_path = os.path.join(
            debug_dir, f"epoch{epoch:03d}_step{global_step:06d}_sample{i}_mask.png"
        )
        save_image(mask_vis.cpu(), mask_path)

        # Save composite if requested and inpainted available
        if save_composite and inpainted is not None:
            mask_3 = mask_use.repeat(1, 3, 1, 1)
            # mask == 1 → region to inpaint
            composite = inpainted[i:i+1] * mask_3 + masked[i:i+1] * (1 - mask_3)
            composite_path = os.path.join(
                debug_dir, f"epoch{epoch:03d}_step{global_step:06d}_sample{i}_composite.png"
            )
            save_image(composite.cpu().clamp(0,1), composite_path)

# ─────────────────────────────────────────────
# GAN & PERCEPTUAL training knobs
# ─────────────────────────────────────────────
WARMUP_EPOCHS = 2
ADV_SCALE = 0.25
D_UPDATE_INTERVAL = 5

# >>> NEW: perceptual loss weight (you can tweak)
LAMBDA_PERCEPTUAL = 0.1

# ─────────────────────────────────────────────
# VGG Perceptual Loss
# ─────────────────────────────────────────────
class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features.
    - Expects inputs in [0,1]
    - Internally normalizes with ImageNet mean/std
    - Uses early conv blocks (conv1_2 .. conv3_3) for structure/texture
    """
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # Use layers up to conv3_3 (index 16) – standard choice
        self.features = vgg[:16].to(device)
        for p in self.features.parameters():
            p.requires_grad = False

        # Register mean/std buffers (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        # x,y in [0,1]; normalize to ImageNet stats
        x_norm = (x - self.mean) / self.std
        y_norm = (y - self.mean) / self.std
        fx = self.features(x_norm)
        fy = self.features(y_norm)
        return self.criterion(fx, fy)

# ─────────────────────────────────────────────
# PatchGAN Discriminator
# ─────────────────────────────────────────────
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# Adversarial loss helper
# ─────────────────────────────────────────────
def compute_adversarial_loss(disc, real, fake, adv_criterion):
    """
    real, fake are already:
      - restricted to masked region
      - normalized to [-1, 1]
    """
    pred_real = disc(real)
    loss_D_real = adv_criterion(pred_real, torch.ones_like(pred_real))

    pred_fake = disc(fake.detach())
    loss_D_fake = adv_criterion(pred_fake, torch.zeros_like(pred_fake))

    loss_D = 0.5 * (loss_D_real + loss_D_fake)

    pred_fake_for_G = disc(fake)
    loss_G_adv = adv_criterion(pred_fake_for_G, torch.ones_like(pred_fake_for_G))

    return loss_D, loss_G_adv

# ─────────────────────────────────────────────
# Validation
# (keeps only L1 inpainting loss – no GAN, no perceptual)
# ─────────────────────────────────────────────
def validate(model, fusion_conv, discriminator, dataloader, device,
             mse_loss, l1_loss, adv_criterion,
             use_gan=False, effective_lambda_adv=0.0):

    model.eval(); fusion_conv.eval()
    val_loss_denoise, val_loss_inpaint = 0.0, 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            noisy  = batch["noisy"].to(device)
            masked = batch["masked"].to(device)
            mask   = batch["mask"].to(device)
            clean  = batch["clean"].to(device)

            # Fuse
            x_fused = torch.cat([noisy, masked, mask], dim=1)
            fusion_out = fusion_conv(x_fused)

            # Forward
            B = noisy.size(0)
            noise_level = torch.ones(B, 1, device=device)

            outputs = model(
                x_fused=fusion_out,
                x_noisy=noisy,
                mask=mask,
                clamp_output=True,
                noise_level=noise_level,
            )

            denoised  = outputs["denoised"]
            inpainted = outputs["inpainted"]

            # -------- DENOISING LOSS --------
            loss_denoise = mse_loss(denoised, clean)

            # -------- INPAINTING LOSS (L1 only in val) --------
            if inpainted is not None:
                mask_use = mask[:, 0:1] if mask.shape[1] != 1 else mask
                mask_3 = mask_use.expand_as(inpainted)

                loss_l1 = l1_loss(inpainted * mask_3, clean * mask_3)
                loss_inpaint = LAMBDA_L1 * loss_l1
            else:
                loss_inpaint = torch.tensor(0.0, device=device)

            bs = noisy.size(0)
            val_loss_denoise += loss_denoise.item() * bs
            val_loss_inpaint += loss_inpaint.item() * bs

    n = len(dataloader.dataset)
    return val_loss_denoise / n, val_loss_inpaint / n

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():

    model = DenoiseInpaintModel(
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        pretrained=True
    ).to(DEVICE)
    model.enable_lora_training()

    discriminator = PatchDiscriminator().to(DEVICE)
    # noisy (3) + masked (3) + mask (1) = 7
    fusion_conv = nn.Conv2d(7, 3, kernel_size=1).to(DEVICE)

    mse_loss_fn, l1_loss_fn = nn.MSELoss(), nn.L1Loss()
    adv_criterion = nn.BCEWithLogitsLoss()

    # >>> NEW: perceptual loss module
    perceptual_loss_fn = VGGPerceptualLoss(device=DEVICE)

    optimizer_G = Adam(
        list(model.get_trainable_params()) + list(fusion_conv.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    optimizer_D = Adam(
        discriminator.parameters(),
        lr=LR_DISCRIMINATOR,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch, best_val_loss = 1, float("inf")

    train_loader = get_dataloader(DATA_ROOT, "train", BATCH_SIZE_TRAIN, IMAGE_SIZE, NUM_WORKERS)
    val_loader = get_dataloader(DATA_ROOT, "validation", BATCH_SIZE_VAL, IMAGE_SIZE, NUM_WORKERS)

    train_losses_denoise, train_losses_inpaint = [], []
    val_losses_denoise, val_losses_inpaint = [], []
    total_train_loss, total_val_loss = [], []
    train_losses_D, train_losses_G_adv = [], []

    effective_lambda_adv = LAMBDA_ADV * ADV_SCALE
    global_step = 0

    os.makedirs(SAVE_ROOT, exist_ok=True)
    csv_path = os.path.join(SAVE_ROOT, "training_logs.csv")

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch","use_gan","train_denoise","val_denoise",
            "train_inpaint","val_inpaint","train_total","val_total",
            "train_D","train_G_adv"
        ])

    # ────────── TRAINING LOOP ──────────
    for epoch in range(start_epoch, EPOCHS + 1):

        model.train(); fusion_conv.train(); discriminator.train()
        total_loss_denoise, total_loss_inpaint = 0.0, 0.0
        total_loss_D, total_loss_G_adv = 0.0, 0.0

        use_gan = epoch > WARMUP_EPOCHS

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):

            global_step += 1

            noisy  = batch["noisy"].to(DEVICE)
            masked = batch["masked"].to(DEVICE)
            mask   = batch["mask"].to(DEVICE)
            clean  = batch["clean"].to(DEVICE)

            # ----- DEBUG BEFORE FORWARD -----
            if global_step == 1 or global_step % len(train_loader) == 1:
                save_debug_images(
                    epoch, global_step,
                    noisy=noisy, masked=masked, mask=mask,
                    inpainted=None,
                    save_root=SAVE_ROOT,
                    max_samples=4,
                    save_composite=False
                )

            # ----- Fuse inputs -----
            x_fused = torch.cat([noisy, masked, mask], dim=1)
            fusion_out = fusion_conv(x_fused)

            # Noise level
            B = noisy.size(0)
            noise_level = torch.ones(B, 1, device=DEVICE)

            # Forward
            out = model(
                x_fused=fusion_out,
                x_noisy=noisy,
                mask=mask,
                clamp_output=True,
                noise_level=noise_level,
            )
            denoised, inpainted = out["denoised"], out["inpainted"]

            # ----- DEBUG AFTER FORWARD -----
            if global_step == 1 or global_step % len(train_loader) == 1:
                save_debug_images(
                    epoch, global_step,
                    noisy=noisy, masked=masked, mask=mask,
                    inpainted=inpainted,
                    save_root=SAVE_ROOT,
                    max_samples=4,
                    save_composite=True
                )

            # ----- Denoising loss (full image) -----
            loss_denoise = mse_loss_fn(denoised, clean)

            # ----- Inpainting loss (masked L1 + perceptual + GAN) -----
            if inpainted is not None:
                mask_use = mask[:, 0:1] if mask.shape[1] != 1 else mask
                mask_3 = mask_use.expand_as(inpainted)  # mask==1 → region to inpaint

                # Masked regions only
                inpaint_region = inpainted * mask_3
                clean_region   = clean * mask_3

                # L1 only in masked region
                loss_l1 = l1_loss_fn(inpaint_region, clean_region)

                # >>> NEW: perceptual loss on masked region
                loss_perc = perceptual_loss_fn(inpaint_region, clean_region)

                if use_gan:
                    # --- masked-region GAN ---
                    real_region = clean_region
                    fake_region = inpaint_region

                    # normalize to [-1, 1] for D
                    real_d = real_region * 2.0 - 1.0
                    fake_d = fake_region * 2.0 - 1.0

                    loss_D, loss_G_adv = compute_adversarial_loss(
                        discriminator, real_d, fake_d, adv_criterion
                    )
                    loss_inpaint = (
                        LAMBDA_L1 * loss_l1
                        + LAMBDA_PERCEPTUAL * loss_perc
                        + effective_lambda_adv * loss_G_adv
                    )
                else:
                    loss_D = torch.tensor(0.0, device=DEVICE)
                    loss_G_adv = torch.tensor(0.0, device=DEVICE)
                    loss_inpaint = (
                        LAMBDA_L1 * loss_l1
                        + LAMBDA_PERCEPTUAL * loss_perc
                    )
            else:
                loss_D = torch.tensor(0.0, device=DEVICE)
                loss_G_adv = torch.tensor(0.0, device=DEVICE)
                loss_inpaint = torch.tensor(0.0, device=DEVICE)

            # ----- Total generator loss -----
            total_G_loss = loss_denoise + loss_inpaint

            # G update
            optimizer_G.zero_grad()
            total_G_loss.backward()
            optimizer_G.step()

            # D update (slower & only when GAN is active)
            if use_gan and (global_step % D_UPDATE_INTERVAL == 0):
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Accumulate
            bs = noisy.size(0)
            total_loss_denoise += loss_denoise.item() * bs
            total_loss_inpaint += loss_inpaint.item() * bs
            total_loss_D += loss_D.item() * bs
            total_loss_G_adv += loss_G_adv.item() * bs

        # ────────── VALIDATION ──────────
        val_loss_denoise, val_loss_inpaint = validate(
            model, fusion_conv, discriminator, val_loader,
            DEVICE, mse_loss_fn, l1_loss_fn, adv_criterion,
            use_gan, effective_lambda_adv
        )

        n_train = len(train_loader.dataset)
        avg_train_denoise = total_loss_denoise / n_train
        avg_train_inpaint = total_loss_inpaint / n_train
        avg_train_total = avg_train_denoise + avg_train_inpaint

        avg_train_D = total_loss_D / n_train
        avg_train_G_adv = total_loss_G_adv / n_train

        avg_val_total = val_loss_denoise + val_loss_inpaint

        # Log to memory
        train_losses_denoise.append(avg_train_denoise)
        val_losses_denoise.append(val_loss_denoise)
        train_losses_inpaint.append(avg_train_inpaint)
        val_losses_inpaint.append(val_loss_inpaint)
        total_train_loss.append(avg_train_total)
        total_val_loss.append(avg_val_total)
        train_losses_D.append(avg_train_D)
        train_losses_G_adv.append(avg_train_G_adv)

        print(f"\nEpoch [{epoch}/{EPOCHS}] (use_gan={use_gan})")
        print(f"  Denoise → Train: {avg_train_denoise:.6f}, Val: {val_loss_denoise:.6f}")
        print(f"  Inpaint → Train: {avg_train_inpaint:.6f}, Val: {val_loss_inpaint:.6f}")
        print(f"  Total → Train: {avg_train_total:.6f}, Val: {avg_val_total:.6f}")
        print(f"  D loss (train avg): {avg_train_D:.6f}")
        print(f"  G_adv loss (train avg): {avg_train_G_adv:.6f}")

        # Write to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, int(use_gan),
                avg_train_denoise, val_loss_denoise,
                avg_train_inpaint, val_loss_inpaint,
                avg_train_total, avg_val_total,
                avg_train_D, avg_train_G_adv
            ])

        # Save checkpoint
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            ckpt_path = os.path.join(SAVE_ROOT, CKPT_PATH)
            torch.save(
                {"model": model.state_dict(),
                 "fusion_conv": fusion_conv.state_dict()},
                ckpt_path,
            )
            print(f"Best model updated at epoch {epoch} → {ckpt_path}")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
