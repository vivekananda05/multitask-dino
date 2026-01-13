# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class JointDenoiseInpaintDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=128):
        """
        Expected structure:
        root_dir/
         ├── clean/
         ├── noisy/
         ├── masked/
         └── mask/
        Each contains subfolders: train / validation / test / class_name
        """
        self.clean_dir = os.path.join(root_dir, "clean", split)
        self.noisy_dir = os.path.join(root_dir, "noisy", split)
        self.masked_dir = os.path.join(root_dir, "masked", split)
        self.mask_dir = os.path.join(root_dir, "mask", split)

        self.classes = sorted(os.listdir(self.clean_dir))
        self.samples = []

        for cls in self.classes:
            clean_cls = os.path.join(self.clean_dir, cls)
            noisy_cls = os.path.join(self.noisy_dir, cls)
            masked_cls = os.path.join(self.masked_dir, cls)
            mask_cls = os.path.join(self.mask_dir, cls)

            for img_name in os.listdir(clean_cls):
                clean_path = os.path.join(clean_cls, img_name)
                noisy_path = os.path.join(noisy_cls, img_name)
                masked_path = os.path.join(masked_cls, img_name)
                mask_path = os.path.join(mask_cls, img_name)

                if all(os.path.exists(p) for p in [clean_path, noisy_path, masked_path, mask_path]):
                    self.samples.append((clean_path, noisy_path, masked_path, mask_path))

        self.transform_rgb = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
            # transforms.Lambda(lambda m: 1 - m)   # invert mask values
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clean_path, noisy_path, masked_path, mask_path = self.samples[idx]

        clean = self.transform_rgb(Image.open(clean_path).convert("RGB"))
        noisy = self.transform_rgb(Image.open(noisy_path).convert("RGB"))
        masked = self.transform_rgb(Image.open(masked_path).convert("RGB"))
        mask = self.transform_mask(Image.open(mask_path).convert("L"))

        return {
            "clean": clean,
            "noisy": noisy,
            "masked": masked,  # ✅ required for train.py
            "mask": mask
        }


def get_dataloader(root_dir, split="train", batch_size=16, image_size=128, num_workers=4):
    dataset = JointDenoiseInpaintDataset(root_dir=root_dir, split=split, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


# dl = get_dataloader("/mnt/DATA1/pankhi/gnr650/multitask_dino/DATA", split="train", batch_size=1)
# batch = next(iter(dl))
# print(batch.keys())
