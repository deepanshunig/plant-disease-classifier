"""Shared, deterministic inference for the CLI and Streamlit interface."""

from __future__ import annotations

import argparse
import io
import json
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import ROOT, class_names, readable_label
from .models import build_model
from .transforms import image_transform

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def decode_image(content: bytes) -> Image.Image:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Please upload a JPEG or PNG image smaller than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as opened:
                if opened.format not in {"JPEG", "PNG"}:
                    raise ValueError("Please upload a JPEG or PNG image.")
                if opened.width * opened.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Please resize this image to fewer than 20 million pixels.")
                opened.load()
                return ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("This file could not be read as a valid JPEG or PNG image.") from error


class Predictor:
    def __init__(self, checkpoint_path: str | Path):
        torch.set_num_threads(4)
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.names = self.checkpoint["class_names"]
        if self.names != class_names():
            raise ValueError("Checkpoint label mapping does not match this project")
        expected = {"resize": 256, "crop": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        if self.checkpoint["preprocessing"] != expected:
            raise ValueError("Checkpoint preprocessing is unsupported")
        self.model = build_model(self.checkpoint["architecture"], len(self.names))
        self.model.load_state_dict(self.checkpoint["model_state"], strict=True)
        self.model.eval()
        self.transform = image_transform()

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(tensor)
            scores = torch.softmax(logits, dim=1)[0]
            values, indices = scores.topk(3)
        alternatives = []
        for value, index in zip(values.tolist(), indices.tolist()):
            crop, condition = readable_label(self.names[index])
            alternatives.append({"class_id": index, "label": self.names[index], "crop": crop, "condition": condition, "score": value})
        threshold = self.checkpoint.get("uncertainty_threshold")
        return {
            "prediction": alternatives[0], "top_three": alternatives,
            "uncertain": None if threshold is None else alternatives[0]["score"] < threshold,
            "uncertainty_threshold": threshold,
            "model_version": self.checkpoint["version"],
            "training_profile": self.checkpoint["training_profile"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/model_bundle/model.pt")
    args = parser.parse_args()
    if args.image.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Image exceeds the 10 MB limit")
    image = decode_image(args.image.read_bytes())
    print(json.dumps(Predictor(args.checkpoint).predict(image), indent=2))


if __name__ == "__main__":
    main()
