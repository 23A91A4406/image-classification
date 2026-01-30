import os
import random
import shutil
from torchvision.datasets import Caltech101
from torchvision import transforms
from pathlib import Path


DATA_DIR = "data"
TRAIN_SPLIT = 0.8
MIN_CLASSES = 10


def prepare_folders():
    for split in ["train", "val"]:
        split_path = os.path.join(DATA_DIR, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path, ignore_errors=True)
        os.makedirs(split_path, exist_ok=True)


def main():
    print("Downloading Caltech-101 using torchvision...")

    dataset = Caltech101(
        root="data/raw",
        download=True,
        transform=transforms.ToTensor()
    )

    prepare_folders()

    class_names = dataset.categories[:MIN_CLASSES]
    print("Using classes:", class_names)

    class_to_images = {cls: [] for cls in class_names}

    for img, label in dataset:
        class_name = dataset.categories[label]
        if class_name in class_to_images:
            class_to_images[class_name].append(img)

    for class_name, images in class_to_images.items():
        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_SPLIT)

        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        train_dir = Path(DATA_DIR) / "train" / class_name
        val_dir = Path(DATA_DIR) / "val" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for i, img in enumerate(train_imgs):
            img_path = train_dir / f"{i}.png"
            transforms.ToPILImage()(img).save(img_path)

        for i, img in enumerate(val_imgs):
            img_path = val_dir / f"{i}.png"
            transforms.ToPILImage()(img).save(img_path)

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
