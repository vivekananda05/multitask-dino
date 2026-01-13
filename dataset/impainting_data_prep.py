# ─────────────────────────────────────────────────────────────
# mini_imagenet_inpainting_generator_hierarchical.py (FIXED)
# Creates clean / mask / masked triplets with correct semantics:
#   mask WHITE (1) = HOLE to inpaint
#   mask BLACK (0) = CONTEXT to keep
# Masked image = clean * (1 - mask)
# ─────────────────────────────────────────────────────────────

import os, numpy as np
from PIL import Image, ImageDraw
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

# --- Config ---
BASE_ROOT = "/mnt/DATA1/pankhi/DATA/mini_imagenet"
OUT_ROOT  = "/mnt/DATA1/pankhi/gnr650/multitask_dino_n1/DATA"
MASK_RATIO = 0.3
IMG_SIZE = (128, 128)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
])

# ------------------------------------------------------------
# CORRECT MASK GENERATOR (single white rectangle)
# ------------------------------------------------------------
def random_mask(size, ratio=0.3):
    w, h = size

    # mask BLACK = 0 (context)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    hole_w = int(w * ratio)
    hole_h = int(h * ratio)

    x1 = np.random.randint(0, w - hole_w)
    y1 = np.random.randint(0, h - hole_h)
    x2 = x1 + hole_w
    y2 = y1 + hole_h

    # WHITE = 255 → hole
    draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask

# ------------------------------------------------------------
# PREPARE DIRECTORY STRUCTURE
# ------------------------------------------------------------
for mode in ["masked", "mask", "clean"]:
    for split in ["train", "validation", "test"]:
        os.makedirs(os.path.join(OUT_ROOT, mode, split), exist_ok=True)

# ------------------------------------------------------------
# PROCESS EACH SPLIT
# ------------------------------------------------------------
for split in ["train", "validation", "test"]:
    src_root = os.path.join(BASE_ROOT, split)
    dataset = datasets.ImageFolder(root=src_root, transform=transform)
    classes = dataset.classes

    print(f"\nProcessing {split}: {len(dataset)} images across {len(classes)} classes...")

    # Create class subfolders
    for cls in classes:
        for mode in ["masked", "mask", "clean"]:
            os.makedirs(os.path.join(OUT_ROOT, mode, split, cls), exist_ok=True)

    # Generate triplets
    for i, (img_tensor, label) in enumerate(tqdm(dataset, desc=f"{split}")):
        cls = classes[label]

        # Convert to PIL
        img = TF.to_pil_image(img_tensor)

        # Create mask
        mask = random_mask(img.size, MASK_RATIO)

        # ----- Correct masked image computation -----
        img_np = np.array(img).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32) / 255.0  # 1=hole, 0=context
        mask_3 = np.stack([mask_np]*3, axis=-1)

        masked_np = img_np * (1 - mask_3)
        masked = Image.fromarray((masked_np * 255).astype(np.uint8))

        # Save all three
        masked.save(os.path.join(OUT_ROOT, "masked", split, cls, f"{i:06d}.png"))
        mask.save(os.path.join(OUT_ROOT, "mask", split, cls, f"{i:06d}.png"))
        img.save(os.path.join(OUT_ROOT, "clean", split, cls, f"{i:06d}.png"))

    print(f"✅ Completed {split}")

print(f"\nAll inpainting datasets created successfully under: {OUT_ROOT}")
