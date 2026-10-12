"""Verify dependencies and an actual image-to-model forward pass."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional real sample photo")
    args = parser.parse_args()

    packages = {
        "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
        "pillow": "PIL", "pandas": "pandas", "scikit-learn": "sklearn",
        "matplotlib": "matplotlib", "PyYAML": "yaml", "tqdm": "tqdm",
        "streamlit": "streamlit", "requests": "requests", "psutil": "psutil",
    }
    versions = {}
    for distribution, module in packages.items():
        importlib.import_module(module)
        versions[distribution] = importlib.metadata.version(distribution)

    import psutil
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    torch.set_num_threads(min(4, psutil.cpu_count(logical=False) or 2))
    contract = json.loads((ROOT / "configs/supported_classes.json").read_text())
    classes = contract["class_names"]
    if len(classes) != 38 or len(set(classes)) != 38:
        raise ValueError("Expected 38 unique classes")

    if args.image:
        with Image.open(args.image) as opened:
            image = opened.convert("RGB")
        sample_description = str(args.image.resolve())
    else:
        image = Image.new("RGB", (256, 256), color=(80, 150, 70))
        sample_description = "Synthetic RGB image: environment check only"
    tensor = ResNet18_Weights.IMAGENET1K_V1.transforms()(image).unsqueeze(0)
    model = resnet18(weights=None)  # This check does not download or train weights.
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.eval()
    with torch.inference_mode():
        output = model(tensor)
    if tuple(tensor.shape) != (1, 3, 224, 224):
        raise ValueError(f"Unexpected input shape: {tuple(tensor.shape)}")
    if tuple(output.shape) != (1, 38) or not torch.isfinite(output).all():
        raise ValueError("Invalid model output")

    report = {
        "status": "passed", "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
        "sample": sample_description,
        "input_shape": list(tensor.shape), "output_shape": list(output.shape),
        "note": "Random weights verify execution only; no accuracy claim.",
    }
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    (report_dir / "environment.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
