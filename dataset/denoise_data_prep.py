# ─────────────────────────────────────────────────────────────
# mini_imagenet_denoising_generator_hierarchical.py
# Create Gaussian-noisy & clean Mini-ImageNet datasets with
# structure: mini_imagenet_denoising/noisy|clean/train|val|test/class/
# ─────────────────────────────────────────────────────────────

import os, torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --- Config ---
BASE_ROOT = "/mnt/DATA1/pankhi/DATA/mini_imagenet"      # contains train/validation/test
OUT_ROOT  = "/mnt/DATA1/pankhi/gnr650/multitask_dino_n1/DATA"
NOISE_STD = 0.1                                        # noise strength (0.05–0.2 recommended)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --- Noise function ---
def add_gaussian_noise(img, sigma):
    noise = torch.randn_like(img) * sigma
    noisy = torch.clamp(img + noise, 0., 1.)
    return noisy

# --- Prepare main folders ---
for mode in ["noisy", "clean"]:
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, mode, split), exist_ok=True)

# --- Process each split ---
for split in ["train", "validation", "test"]:
    src_root = os.path.join(BASE_ROOT, split)
    dataset = datasets.ImageFolder(root=src_root, transform=transform)
    classes = dataset.classes

    print(f"\nProcessing {split}: {len(dataset)} images across {len(classes)} classes...")

    # Create subfolders for each class inside noisy/clean
    for cls in classes:
        os.makedirs(os.path.join(OUT_ROOT, "noisy", split, cls), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "clean", split, cls), exist_ok=True)

    for i, (img, label) in enumerate(tqdm(dataset, desc=f"{split}")):
        cls = classes[label]
        noisy = add_gaussian_noise(img, NOISE_STD)

        TF.to_pil_image(noisy).save(os.path.join(OUT_ROOT, "noisy", split, cls, f"{i:06d}.png"))
        TF.to_pil_image(img).save(os.path.join(OUT_ROOT, "clean", split, cls, f"{i:06d}.png"))

    print(f"✅ Completed: {split}")

print(f"\nAll denoising datasets created successfully under: {OUT_ROOT}")
