from datasets import load_dataset
import os

from tqdm import tqdm

dataset_path = "/mnt/DATA1/pankhi/DATA/mini_imagenet"
os.makedirs(dataset_path, exist_ok=True)

ds = load_dataset("timm/mini-imagenet", cache_dir=dataset_path)

# export split
def export_split(split_name):
    split_dir = os.path.join(dataset_path, split_name)
    os.makedirs(split_dir, exist_ok=True)

    for i, sample in enumerate(tqdm(ds[split_name], desc=f"Exporting {split_name}")):
        img = sample["image"]
        label = sample["label"]
        label_name = ds[split_name].features["label"].int2str(label)

        class_dir = os.path.join(split_dir, label_name)
        os.makedirs(class_dir, exist_ok=True)
        img.save(os.path.join(class_dir, f"{i:06d}.jpg"))


for split in ["train", "validation", "test"]:
    export_split(split)
