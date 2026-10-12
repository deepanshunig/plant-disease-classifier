from __future__ import annotations

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .common import ROOT
from .transforms import image_transform


class LeafDataset(Dataset):
    def __init__(self, frame, training: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(ROOT / row.image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        return self.transform(image), int(row.class_id)
