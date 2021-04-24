import torch
import albumentations
from PIL import Image
import numpy as np


class Shopee_Dataset:
    def __init__(self, img_paths, labels=None, augmentations=None):
        self.img_paths = img_paths
        self.labels = labels
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if self.labels is not None:
            label = self.labels[idx]

        image = Image.open(img_path)
        image = np.array(image)

        # Image augmentations
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return (
            {
                "image": torch.tensor(image, dtype=torch.float),
                "label": torch.tensor(label, dtype=torch.long),
            }
            if label is not None
            else {
                "image": torch.tensor(image, dtype=torch.float),
                "label": torch.tensor(-1),
            }
        )
