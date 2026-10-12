"""Train a baseline or two-stage ResNet18; choose checkpoints on validation only."""

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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .common import ROOT, class_names, load_config, save_json, seed_everything
from .dataset import LeafDataset
from .models import build_model, configure_resnet_stage, set_training_mode


def subsample(frame, limit, seed):
    if limit is None:
        return frame.reset_index(drop=True)
    pieces = [part.sample(n=min(int(limit), len(part)), random_state=seed) for _, part in frame.groupby("class_id")]
    return pd.concat(pieces).reset_index(drop=True)


def completed_training_stages(history, stages, patience):
    completed = set()
    for stage, epochs in stages:
        records = [row for row in history if row["stage"] == stage]
        if not records:
            continue
        running_best, trailing_stale, entered_stage = -1.0, 0, False
        for row in history:
            if entered_stage and row["stage"] != stage:
                break
            entered_stage = entered_stage or row["stage"] == stage
            score = row["validation"]["macro_f1"]
            if score > running_best:
                running_best, trailing_stale = score, 0
            elif row["stage"] == stage:
                trailing_stale += 1
        if len(records) < epochs and trailing_stale < patience:
            raise ValueError("Resume requires a completed stage; optimizer state for a partial stage is unavailable")
        completed.add(stage)
    return completed


def plot_learning_curves(history, output_path):
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for split in ("train", "validation"):
        axes[0].plot(epochs, [row[split]["loss"] for row in history], marker="o", label=split)
        axes[1].plot(epochs, [row[split]["macro_f1"] for row in history], marker="o", label=split)
    axes[0].set_title("Loss (training may be class weighted)")
    axes[1].set_title("Macro F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_xticks(epochs)
        axis.set_xlim(min(epochs) - 0.5, max(epochs) + 0.5)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=130)
    plt.close(fig)


def run_epoch(model, loader, loss_function, device, optimizer=None, architecture="resnet18", fine_tuning=False):
    if optimizer is None:
        model.eval()
    else:
        set_training_mode(model, architecture, fine_tuning)
    total_loss, actual, predicted = 0.0, [], []
    started = time.perf_counter()
    last_report = started
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(device), targets.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = loss_function(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(targets)
        actual.extend(targets.cpu().tolist())
        predicted.extend(logits.detach().argmax(1).cpu().tolist())
        if time.perf_counter() - last_report > 25:
            print(f"  {'train' if optimizer else 'validation'} batch {batch_index}/{len(loader)}", flush=True)
            last_report = time.perf_counter()
    return {
        "loss": total_loss / len(actual), "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=list(range(38)), average="macro", zero_division=0)),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--architecture", choices=["small_cnn", "resnet18"], default="resnet18")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="Continue from a saved checkpoint at a completed training-stage boundary")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"], config["threads"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else torch.device(config["device"])
    manifest = ROOT / config["manifest"]
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    frame = pd.read_csv(manifest)
    training = subsample(frame[frame.split == "train"], config["max_train_per_class"], config["seed"])
    validation = subsample(frame[frame.split == "validation"], config["max_validation_per_class"], config["seed"])
    names = class_names()
    if training.class_id.nunique() != len(names) or validation.class_id.nunique() != len(names):
        raise ValueError("Every class must be represented in training and validation")
    run_dir = ROOT / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "best.pt").exists() and not args.resume:
        raise FileExistsError(f"Use a new run directory to preserve prior results: {run_dir}")
    torch.hub.set_dir(str(ROOT / ".runtime/torch"))
    model = build_model(args.architecture, len(names), pretrained=args.architecture == "resnet18" and not args.resume).to(device)
    train_loader = DataLoader(LeafDataset(training, True), batch_size=config["batch_size"], shuffle=True, num_workers=config["workers"], pin_memory=device.type == "cuda")
    validation_loader = DataLoader(LeafDataset(validation), batch_size=config["batch_size"], shuffle=False, num_workers=config["workers"], pin_memory=device.type == "cuda")
    weights = None
    if config["class_weighted_loss"]:
        frequencies = training.class_id.value_counts().reindex(range(len(names))).to_numpy()
        weights = torch.tensor(len(training) / (len(names) * frequencies), dtype=torch.float32, device=device)
    train_loss = torch.nn.CrossEntropyLoss(weight=weights)
    validation_loss = torch.nn.CrossEntropyLoss()
    profile = "pilot" if config["max_train_per_class"] is not None else "full_data"
    summary = {"architecture": args.architecture, "training_profile": profile, "train_images": len(training), "validation_images": len(validation), "config": config, "manifest_sha256": manifest_hash, "device": str(device)}
    stages = [("baseline", config["baseline_epochs"])] if args.architecture == "small_cnn" else [("head", config["head_epochs"]), ("fine_tune", config["fine_tune_epochs"])]
    history, best_score, stale, epoch = [], -1.0, 0, 0
    completed_stages = set()
    if args.resume:
        saved_summary = json.loads((run_dir / "run.json").read_text())
        for key in ("architecture", "config", "manifest_sha256", "train_images", "validation_images"):
            if saved_summary[key] != summary[key]:
                raise ValueError(f"Cannot resume: {key} differs from the saved run")
        checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        if checkpoint["class_names"] != names or checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Cannot resume: checkpoint class mapping or dataset differs")
        history = json.loads((run_dir / "history.json").read_text())["epochs"]
        if not history:
            raise ValueError("Cannot resume a run without a completed epoch")
        best_score = checkpoint["validation_macro_f1"]
        epoch = history[-1]["epoch"]
        # Optimizer state is not saved, so only completed stages can be skipped.
        completed_stages = completed_training_stages(history, stages, config["patience"])
        model.load_state_dict(checkpoint["model_state"])
        print(f"Resuming after epoch {epoch}; completed stages: {sorted(completed_stages)}", flush=True)
    else:
        save_json(run_dir / "run.json", summary)
    print(f"Starting {profile} {args.architecture}: {len(training)} train, {len(validation)} validation, device={device}", flush=True)
    for stage, epochs in stages:
        if epochs <= 0 or stage in completed_stages:
            continue
        fine_tuning = stage == "fine_tune"
        if args.architecture == "resnet18":
            if fine_tuning and (run_dir / "best.pt").exists():
                model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=True)["model_state"])
            configure_resnet_stage(model, fine_tuning)
        if fine_tuning:
            parameters = [{"params": model.layer4.parameters(), "lr": config["backbone_lr"]}, {"params": model.fc.parameters(), "lr": config["fine_tune_head_lr"]}]
        else:
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=config["head_lr"], weight_decay=config["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=1, factor=0.5)
        stale = 0
        for _ in range(epochs):
            epoch += 1
            train_metrics = run_epoch(model, train_loader, train_loss, device, optimizer, args.architecture, fine_tuning)
            validation_metrics = run_epoch(model, validation_loader, validation_loss, device)
            record = {"epoch": epoch, "stage": stage, "train": train_metrics, "validation": validation_metrics}
            history.append(record)
            score = validation_metrics["macro_f1"]
            scheduler.step(score)
            if score > best_score:
                best_score, stale = score, 0
                checkpoint = {"model_state": model.state_dict(), "architecture": args.architecture, "class_names": names, "epoch": epoch, "validation_macro_f1": score, "training_profile": profile, "training_images": len(training), "config": config, "manifest_sha256": manifest_hash, "preprocessing": {"resize": 256, "crop": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "uncertainty_threshold": None, "version": f"{profile}-{args.architecture}-epoch-{epoch}"}
                temporary = run_dir / "best.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(run_dir / "best.pt")
            else:
                stale += 1
            save_json(run_dir / "history.json", {"epochs": history, "best_validation_macro_f1": best_score})
            print(f"Epoch {epoch} ({stage}): train accuracy={train_metrics['accuracy']:.3f}; validation accuracy={validation_metrics['accuracy']:.3f}, macro F1={score:.3f}; elapsed={train_metrics['seconds']+validation_metrics['seconds']:.1f}s", flush=True)
            if stale >= config["patience"]:
                break
    plot_learning_curves(history, run_dir / "learning_curves.png")
    print(f"Training complete. Best validation macro F1={best_score:.4f}; checkpoint={run_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
