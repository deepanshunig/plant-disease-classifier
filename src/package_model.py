"""Choose between trained runs on validation results and package shared inference."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import ROOT, save_json
from .dataset import LeafDataset
from .predict import Predictor
from .train import subsample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.9)
    args = parser.parse_args()
    candidates = [(run, torch.load(run / "best.pt", map_location="cpu", weights_only=True)) for run in args.runs]
    fingerprints = {(checkpoint["manifest_sha256"], checkpoint["training_profile"], checkpoint["config"]["max_validation_per_class"], checkpoint["config"]["seed"]) for _, checkpoint in candidates}
    if len(fingerprints) != 1:
        raise ValueError("Candidate runs must use the same validation split and sampling")
    run, checkpoint = max(candidates, key=lambda item: item[1]["validation_macro_f1"])
    predictor = Predictor(run / "best.pt")
    manifest = ROOT / checkpoint["config"]["manifest"]
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint["manifest_sha256"]:
        raise ValueError("Training manifest changed")
    frame = pd.read_csv(manifest)
    validation = subsample(frame[frame.split == "validation"], checkpoint["config"]["max_validation_per_class"], checkpoint["config"]["seed"])
    loader = DataLoader(LeafDataset(validation), batch_size=32, shuffle=False)
    scores, correct = [], []
    last_report = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            values, indices = torch.softmax(predictor.model(images), dim=1).max(dim=1)
            scores.extend(values.tolist())
            correct.extend((indices == labels).tolist())
            if time.perf_counter() - last_report > 25:
                print(f"Uncertainty validation batch {batch_index}/{len(loader)}", flush=True)
                last_report = time.perf_counter()
    scores, correct = np.array(scores), np.array(correct)
    threshold, filtering = None, {"status": "target_not_met", "validation_images": len(validation)}
    for candidate in np.unique(scores):
        accepted = scores >= candidate
        count = int(accepted.sum())
        if count >= 30 and float(correct[accepted].mean()) >= args.target_accepted_accuracy:
            threshold = float(candidate)
            filtering = {"status": "validation_estimate", "threshold": threshold, "accepted_images": count, "validation_images": len(validation), "coverage": float(accepted.mean()), "accepted_accuracy": float(correct[accepted].mean()), "target_accepted_accuracy": args.target_accepted_accuracy, "note": "Validation estimate only; not an unknown-image detector or probability calibration."}
            break
    checkpoint["uncertainty_threshold"] = threshold
    bundle = ROOT / "artifacts/model_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    temporary = bundle / "model.pt.tmp"
    torch.save(checkpoint, temporary)
    temporary.replace(bundle / "model.pt")
    save_json(bundle / "class_mapping.json", {str(index): label for index, label in enumerate(checkpoint["class_names"])})
    save_json(bundle / "preprocessing.json", checkpoint["preprocessing"])
    save_json(bundle / "model_card.json", {
        "model_version": checkpoint["version"], "architecture": checkpoint["architecture"],
        "training_profile": checkpoint["training_profile"], "training_images": checkpoint["training_images"],
        "selected_run": str(run), "selection_metric": "validation macro F1", "validation_macro_f1": checkpoint["validation_macro_f1"],
        "candidate_scores": {str(path): item["validation_macro_f1"] for path, item in candidates},
        "uncertainty_filter": filtering,
        "external_photo_evaluation": "not measured; provide labeled independent photos to evaluate separately",
        "scope": "Single-leaf classification among the 38 supported PlantVillage crop-condition labels",
        "limitations": ["PlantVillage backgrounds differ from field conditions", "Unsupported images can receive high model scores", "Pilot training uses a small sample when training_profile is pilot"],
    })
    print(f"Packaged {checkpoint['training_profile']} model from {run}; validation macro F1={checkpoint['validation_macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
