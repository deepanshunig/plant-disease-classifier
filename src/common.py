from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def class_names() -> list[str]:
    return json.loads((ROOT / "configs/supported_classes.json").read_text())["class_names"]


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def save_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_everything(seed: int, threads: int = 4) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def readable_label(raw: str) -> tuple[str, str]:
    crop, condition = raw.split("___", 1)
    crop = {"Cherry_(including_sour)": "Cherry", "Corn_(maize)": "Corn / maize", "Pepper,_bell": "Bell pepper"}.get(crop, crop)
    condition = condition.replace("Haunglongbing", "Huanglongbing").replace("_", " ").strip()
    return crop, condition[0].upper() + condition[1:]
