"""Dataset and augmentations.

Task: binary foreground segmentation (pet vs background).

Oxford-IIIT Pet trimaps use 1 = pet, 2 = background, 3 = "border/unclassified"
(a thin band around the outline where the annotators were unsure). How to
treat the border is a design choice, exposed as `border`:
  * "ignore"     -> label 255, excluded from the loss
  * "foreground" -> label 1
Evaluation always uses border = foreground (see metrics.py), so models trained
either way are scored against the same target.
"""
import csv

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from .utils import MEAN, ROOT, STD

DATA_DIR = ROOT / "data" / "processed"
IGNORE = 255


def trimap_to_label(trimap: np.ndarray, border: str = "ignore") -> np.ndarray:
    label = np.zeros_like(trimap, dtype=np.uint8)
    label[trimap == 1] = 1
    label[trimap == 3] = IGNORE if border == "ignore" else 1
    return label


def read_meta(split=None):
    with open(DATA_DIR / "meta.csv") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if split is None or r["split"] == split]


def build_transforms(size: int, train: bool, augment: str = "standard"):
    """Joint image/mask transforms (v2 applies nearest interpolation to Mask)."""
    if train and augment == "standard":
        ops = [
            T.RandomResizedCrop(size, scale=(0.5, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.3, 0.3, 0.3, 0.05)], p=0.8),
            T.RandomGrayscale(p=0.05),
        ]
    else:
        # "none" augmentation for training is the same deterministic resize as
        # validation; used in the augmentation ablation.
        ops = [T.Resize((size, size), antialias=True)]
    return T.Compose(ops + [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)])


class PetSegDataset(Dataset):
    def __init__(self, split, size=224, train=False, augment="standard", border="ignore",
                 limit=None):
        self.rows = read_meta(split)
        if limit:
            self.rows = self.rows[:limit]
        self.border = border
        self.tf = build_transforms(size, train, augment)

    def __len__(self):
        return len(self.rows)

    def load(self, i):
        name = self.rows[i]["name"]
        img = Image.open(DATA_DIR / "images" / f"{name}.jpg").convert("RGB")
        trimap = np.array(Image.open(DATA_DIR / "masks" / f"{name}.png"))
        return img, trimap

    def __getitem__(self, i):
        img, trimap = self.load(i)
        img = tv_tensors.Image(torch.from_numpy(np.array(img)).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(trimap_to_label(trimap, self.border)))
        img, mask = self.tf(img, mask)
        return img, mask.long()
