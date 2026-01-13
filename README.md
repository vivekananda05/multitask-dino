# Multi-Task Image Restoration Using LoRA-Adapted DINOv3
This repository contains a unified multi-task image reconstruction framework capable of performing **image denoising** and **image inpainting** within a single neural network architecture.

The model utilizes a **frozen pretrained DINOv3-Small Vision Transformer** as a feature extractor and introduces **LoRA (Low-Rank Adaptation)** modules for lightweight fine-tuning. Task-specific decoders handle denoising and inpainting independently, while sharing a common learned latent representation.

---

## Key Features

- ✔ **Multi-task learning:** Single model performs denoising and inpainting  
- ✔ **LoRA-based fine-tuning:** Efficient parameter training with a frozen ViT backbone  
- ✔ **7-channel fusion input:** Noisy, masked, and binary mask are fused for context  
- ✔ **Residual denoising + Mask-aware inpainting**
- ✔ **GAN + Perceptual loss refinement**
- ✔ **Quantitative and qualitative evaluation support**

---

## Project Structure

├── model.py # Main architecture (denoise + inpaint)
├── encoder.py # DINOv3 + LoRA integration
├── decoder.py # Dual-task decoder
├── dataloader.py # Dataset loader with noise & masking
├── lora.py # Low-Rank Adapter implementation

├── train.py # Training pipeline
├── test.py # Evaluation script (PSNR/SSIM/MSE/MAE)

├── config.py # Hyperparameters and I/O configuration

└── outputs/ # Checkpoints, metrics, visualization


---

## Dataset & Preprocessing

The model is trained using **miniImageNet** resized to **128×128**.

Two corruption variants are generated:

| Type | Operation | Details |
|------|-----------|---------|
| Noise | Gaussian | σ ∈ {0.1} |
| Masking | Rectangular missing region | Mask ratio ≈ 30% |

Inputs are concatenated as:

Inputs → (Fusion Conv) → DINOv3 Encoder + LoRA → Shared Latent Tokens
├─────────────── Denoising Decoder (Residual + Noise Conditioning)
└─────────────── Inpainting Decoder (Mask-Aware + GAN Refinement)

Final inpainting is computed as:

I_final = I_inpainted ⊙ Mask + I_masked ⊙ (1 - Mask)

---

## Training Configuration

| Parameter | Value |
|----------|-------|
| Epochs | 100 |
| Batch Size | 64 |
| Learning Rate (G / D) | 1e-4 / 0.5e-4 |
| LoRA Rank | 8 |
| LoRA α | 16 |
| Noise Level | σ = 0.1 |

### Loss Functions Used

| Component | Loss |
|----------|------|
| Denoising | MSE |
| Inpainting | Masked L1 + GAN |

---

## Evaluation Metrics

Evaluation runs via `test.py` and computes:

| Metric | Purpose |
|--------|---------|
| MSE | Pixel-wise error |
| MAE | Absolute reconstruction deviation |
| PSNR | Signal fidelity |
| SSIM | Structural similarity |

Metrics are reported **separately for**:

- **Denoised output**
- **Inpainting composite**

The script also saves **visual comparison grids**:

Clean | Noisy | Masked | Mask | Denoised | Inpainted


---

## Qualitative Results Example

Each row contains:

> Clean → Noisy → Masked → Mask → Denoised → Inpainted

These samples demonstrate both noise removal quality and semantic inpainting performance.
## Model weight
https://drive.google.com/drive/folders/1t68BPPN9yyUAYzl7nnPuYQ1iKHk1PJoM?usp=drive_link
---


