# ─────────────────────────────────────────────
# test.py — Matching updated GAN training pipeline
# with masked-region GAN and 7-channel fusion input
# ─────────────────────────────────────────────

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from math import log10

from model import DenoiseInpaintModel
from dataloader import get_dataloader
from config import *


# ─────────────────────────────────────────────
# Metric Functions
# ─────────────────────────────────────────────
def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2).item()

def mae_loss(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse.item())

def ssim_metric(pred, target):
    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().permute(1, 2, 0).numpy()
    return ssim(pred_np, target_np, channel_axis=2, data_range=1.0)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────
def visualize_results(clean, noisy, masked, mask, denoised, composite, save_dir, idx):
    """
    Save side-by-side comparison of:
    Clean | Noisy | Masked | Mask | Denoised | Composite (blended inpainting)
    """
    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    titles = ["Clean", "Noisy", "Masked", "Mask", "Denoised", "Inpainted"]
    imgs = [clean, noisy, masked, mask, denoised, composite]

    for ax, img, title in zip(axes, imgs, titles):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        if img_np.shape[-1] == 1:  
            img_np = np.repeat(img_np, 3, axis=-1)
        img_np = np.clip(img_np, 0.0, 1.0)
        ax.imshow(img_np)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"sample_{idx}.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main Testing Routine
# ─────────────────────────────────────────────
def main():
    print("\n Loading model for testing...")
    model = DenoiseInpaintModel(
        model_name=MODEL_NAME,
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        pretrained=True,
    ).to(DEVICE)

    # <<< FIXED: 7-channel Fusion Conv, same as training
    fusion_conv = nn.Conv2d(7, 3, kernel_size=1).to(DEVICE)

    print(f" Loading checkpoint from: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    # <<< FIXED: Support correct checkpoint format
    if isinstance(ckpt, dict) and "model" in ckpt:
        print("  Detected checkpoint with model+fusion_conv keys.")
        model_state = ckpt["model"]
        fusion_state = ckpt.get("fusion_conv")
    else:
        print("  Detected plain state_dict (legacy).")
        model_state = ckpt
        fusion_state = None

    # Load model weights with shape check
    current_state = model.state_dict()
    filtered = {k: v for k, v in model_state.items()
                if k in current_state and v.shape == current_state[k].shape}
    model.load_state_dict(filtered, strict=False)

    if fusion_state is not None:
        print("  Loading fusion_conv weights...")
        fc_state = fusion_conv.state_dict()
        filtered_fc = {k: v for k, v in fusion_state.items()
                       if k in fc_state and v.shape == fc_state[k].shape}
        fusion_conv.load_state_dict(filtered_fc, strict=False)
    else:
        print("⚠️ Using randomly initialized fusion_conv")

    model.eval()
    fusion_conv.eval()

    # Loader
    test_loader = get_dataloader(
        DATA_ROOT, "test",
        batch_size=BATCH_SIZE_TEST,
        image_size=IMAGE_SIZE,
        num_workers=NUM_WORKERS,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics = {
        "denoise": {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0},
        "inpaint": {"mse": 0.0, "mae": 0.0, "psnr": 0.0, "ssim": 0.0},
    }

    count = 0
    visualized = 0

    print(" Running inference on test set...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing Progress")):

            noisy = batch["noisy"].to(DEVICE)
            masked = batch["masked"].to(DEVICE)
            mask = (batch["mask"] > 0.5).float().to(DEVICE)   # <<< FIXED
            clean = batch["clean"].to(DEVICE)

            # <<< FIXED: match train fusion EXACTLY (noisy, masked, mask)
            x_fused = torch.cat([noisy, masked, mask], dim=1)
            x_fused = fusion_conv(x_fused)

            B = noisy.size(0)
            noise_level = torch.ones(B, 1, device=DEVICE)

            # Forward
            outputs = model(
                x_fused=x_fused,
                x_noisy=noisy,
                mask=mask,
                clamp_output=True,
                noise_level=noise_level,
            )
            denoised = torch.clamp(outputs["denoised"], 0, 1)
            inpainted = torch.clamp(outputs["inpainted"], 0, 1)

            # <<< FIXED: blended output (same as train debugging)
            mask_3 = mask.expand_as(inpainted)
            composite = inpainted * mask_3 + masked * (1 - mask_3)

            batch_size = noisy.size(0)
            count += batch_size

            for i in range(batch_size):

                # Denoising metrics
                metrics["denoise"]["mse"]  += mse_loss(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["mae"]  += mae_loss(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["psnr"] += psnr(denoised[i:i+1], clean[i:i+1])
                metrics["denoise"]["ssim"] += ssim_metric(denoised[i], clean[i])

                # Inpainting metrics (on blended)
                metrics["inpaint"]["mse"]  += mse_loss(composite[i:i+1], clean[i:i+1])
                metrics["inpaint"]["mae"]  += mae_loss(composite[i:i+1], clean[i:i+1])
                metrics["inpaint"]["psnr"] += psnr(composite[i:i+1], clean[i:i+1])
                metrics["inpaint"]["ssim"] += ssim_metric(composite[i], clean[i])

                # Visualization
                if visualized < NUM_VISUAL_SAMPLES:
                    visualize_results(
                        clean[i], noisy[i], masked[i], mask[i],
                        denoised[i], composite[i],
                        OUTPUT_DIR, visualized
                    )
                    visualized += 1

    # Average metrics
    for task in ["denoise", "inpaint"]:
        for key in metrics[task]:
            metrics[task][key] /= float(count)

    # Final output
    print("\n ===== TEST SUMMARY =====")
    print(" DENOISING:")
    for k, v in metrics["denoise"].items():
        print(f"  {k.upper()}: {v:.6f}")

    print(" INPAINTING:")
    for k, v in metrics["inpaint"].items():
        print(f"  {k.upper()}: {v:.6f}")

    # Save metrics
    metrics_path = os.path.join(OUTPUT_DIR, METRICS_PATH_FILE)
    with open(metrics_path, "w") as f:
        f.write("===== TEST PERFORMANCE SUMMARY =====\n\n")
        for task in metrics:
            f.write(f"[{task.upper()}]\n")
            for k, v in metrics[task].items():
                f.write(f"{k.upper()}: {v:.6f}\n")
            f.write("\n")

    print(f" Metrics saved to: {metrics_path}")
    print(f" Visual samples saved to: {OUTPUT_DIR}")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
