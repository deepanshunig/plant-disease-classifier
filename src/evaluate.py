"""Evaluate a selected checkpoint without training or choosing model settings."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .common import ROOT, save_json
from .dataset import LeafDataset
from .predict import Predictor


def plot_confusion_matrix(matrix, names, title, output_path):
    labels = list(range(len(names)))
    fig, ax = plt.subplots(figsize=(14, 12))
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    display = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels, names, rotation=90, fontsize=6)
    ax.set_yticks(labels, names, fontsize=6)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(title)
    fig.colorbar(display, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test", "external"], default="test")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/images.csv")
    parser.add_argument("--max-per-class", type=int)
    args = parser.parse_args()
    checkpoint_sha256 = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    predictor = Predictor(args.checkpoint)
    if args.split != "external" and hashlib.sha256(args.manifest.read_bytes()).hexdigest() != predictor.checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from the training manifest")
    full_frame = pd.read_csv(args.manifest)
    frame = full_frame[full_frame.split == args.split]
    if args.max_per_class:
        frame = pd.concat([part.sample(min(len(part), args.max_per_class), random_state=42) for _, part in frame.groupby("class_id")])
    if frame.empty:
        raise ValueError("No images in the selected evaluation split")
    if frame.class_id.isna().any() or not frame.class_id.isin(range(len(predictor.names))).all():
        raise ValueError("Evaluation labels must be valid class IDs from the saved mapping")
    loader = DataLoader(LeafDataset(frame), batch_size=32, shuffle=False, num_workers=0)
    targets, predicted, scores = [], [], []
    started = time.perf_counter()
    last_report = started
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader, start=1):
            probabilities = torch.softmax(predictor.model(images), dim=1)
            confidence, indices = probabilities.max(dim=1)
            targets.extend(labels.tolist())
            predicted.extend(indices.tolist())
            scores.extend(confidence.tolist())
            if time.perf_counter() - last_report > 25:
                print(f"Evaluated batch {index}/{len(loader)}", flush=True)
                last_report = time.perf_counter()
    elapsed = time.perf_counter() - started
    labels = list(range(len(predictor.names)))
    metric_labels = sorted(set(targets)) if args.split == "external" else labels
    report = classification_report(targets, predicted, labels=labels, target_names=predictor.names, output_dict=True, zero_division=0)
    metrics = {
        "split": args.split, "images": len(targets), "sampled": args.max_per_class is not None,
        "training_profile": predictor.checkpoint["training_profile"], "model_version": predictor.checkpoint["version"],
        "checkpoint_sha256": checkpoint_sha256,
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, labels=metric_labels, average="macro", zero_division=0)),
        "macro_f1_class_ids": metric_labels,
        "classes_with_examples": len(set(targets)),
        "elapsed_seconds": elapsed, "pipeline_seconds_per_image": elapsed / len(targets),
        "external_evaluation": "measured" if args.split == "external" else "not measured by this run",
        "class_report": report,
    }
    save_json(args.output_dir / "metrics.json", metrics)
    predictions = frame.reset_index(drop=True).copy()
    predictions["predicted_class_id"] = predicted
    predictions["model_score"] = scores
    predictions["correct"] = np.array(targets) == np.array(predicted)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    matrix = confusion_matrix(targets, predicted, labels=labels)
    pd.DataFrame(matrix, index=predictor.names, columns=predictor.names).to_csv(args.output_dir / "confusion_matrix.csv")
    plot_confusion_matrix(matrix, predictor.names, f"{predictor.checkpoint['training_profile']} {args.split}: row-normalized confusion matrix", args.output_dir / "confusion_matrix.png")
    print(json.dumps({key: value for key, value in metrics.items() if key != "class_report"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
