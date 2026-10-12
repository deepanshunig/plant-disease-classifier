"""Audit original color images and create reproducible leaf-group split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import ROOT, class_names, save_json


def physical_leaf_id(filename: str, label: str, mapping: dict) -> tuple[str, str]:
    identifier = filename.replace("_final_masked", "").split("___")[-1].split("copy")[0]
    identifier = Path(identifier).stem.lower().strip()
    matches = [item for item in mapping.get(identifier, []) if item.startswith(label + ":::")]
    if len(matches) == 1:
        return matches[0], "metadata"
    return f"{label}:::unresolved::{identifier}", "filename_fallback"


def inspect_image(item):
    path, split, mapping, label_ids = item
    label = path.parent.name
    if label not in label_ids:
        raise ValueError(f"Unsupported dataset label: {label}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        width, height = image.size
        # Include dimensions to distinguish equal bytes interpreted as different shapes.
        digest = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
    leaf_id, resolution = physical_leaf_id(path.name, label, mapping)
    return {
        "image_path": path.relative_to(ROOT).as_posix(), "class_name": label,
        "class_id": label_ids[label], "leaf_id": leaf_id, "leaf_id_source": resolution,
        "source_split": split, "pixel_sha256": digest, "width": width, "height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("Validation fraction must be between zero and 0.5")
    names = class_names()
    label_ids = {name: index for index, name in enumerate(names)}
    source = ROOT / "data/source"
    mapping = json.loads((source / "leaf_grouping/leaf-map.json").read_text())
    assignments = {}
    for split in ("train", "test"):
        for relative in (source / f"splits/color_{split}.txt").read_text().splitlines():
            if not relative.strip():
                continue
            path = ROOT / "data" / relative.strip()
            if path in assignments:
                raise ValueError(f"Image occurs twice in supplied splits: {relative}")
            if not path.is_file():
                raise FileNotFoundError(path)
            assignments[path] = split
    raw = ROOT / "data/raw/color"
    paths = sorted(path for path in raw.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    actual_names = {path.parent.name for path in paths}
    if actual_names != set(names):
        raise ValueError(f"Class mismatch: missing={set(names)-actual_names}, extra={actual_names-set(names)}")
    rows, invalid = [], []
    items = [(path, assignments.get(path, "unassigned"), mapping, label_ids) for path in paths]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(inspect_image, item) for item in items]
        for index, (item, future) in enumerate(zip(items, futures), start=1):
            try:
                rows.append(future.result())
            except Exception as error:
                invalid.append({"path": item[0].relative_to(ROOT).as_posix(), "error": str(error)})
            if index % 5000 == 0:
                print(f"Audited {index}/{len(paths)} images", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No readable images")
    parents = {leaf: leaf for leaf in frame.leaf_id.unique()}

    def find(leaf):
        while parents[leaf] != leaf:
            parents[leaf] = parents[parents[leaf]]
            leaf = parents[leaf]
        return leaf

    conflicts = set()
    duplicates = frame[frame.pixel_sha256.duplicated(keep=False)]
    for _, repeated in duplicates.groupby("pixel_sha256"):
        if repeated.class_name.nunique() > 1:
            conflicts.update(repeated.leaf_id)
            continue
        members = sorted(set(repeated.leaf_id))
        for member in members[1:]:
            first, second = find(members[0]), find(member)
            parents[max(first, second)] = min(first, second)
    frame["group_id"] = frame.leaf_id.map(find)
    conflict_groups = {find(leaf) for leaf in conflicts}
    frame["split"] = "train"
    frame.loc[frame.source_split == "unassigned", "split"] = "quarantine"
    frame.loc[frame.group_id.isin(conflict_groups), "split"] = "quarantine"
    test_groups = set(frame.loc[(frame.source_split == "test") & (frame.split != "quarantine"), "group_id"])
    frame.loc[frame.group_id.isin(test_groups), "split"] = "test"
    # Keep source test examples; exclude overlapping training photos from development.
    overlap = (frame.source_split == "train") & frame.group_id.isin(test_groups)
    frame.loc[overlap, "split"] = "quarantine"
    rng = np.random.default_rng(args.seed)
    for label in names:
        eligible = frame[(frame.class_name == label) & (frame.split == "train")]
        sizes = eligible.groupby("group_id").size()
        if len(sizes) < 2:
            raise ValueError(f"Too few training leaf groups for {label}")
        groups = rng.permutation(sizes.index.to_numpy())
        count, validation_groups = 0, []
        target = max(1, round(len(eligible) * args.validation_fraction))
        for group in groups[:-1]:
            validation_groups.append(group)
            count += int(sizes[group])
            if count >= target:
                break
        frame.loc[frame.group_id.isin(validation_groups) & (frame.split == "train"), "split"] = "validation"
    used = frame[frame.split != "quarantine"]
    for column in ("group_id", "pixel_sha256"):
        if used.groupby(column).split.nunique().max() != 1:
            raise ValueError(f"Cross-split leakage in {column}")
    counts = used.groupby(["class_name", "split"]).size().unstack(fill_value=0).reindex(names)
    if any(column not in counts or (counts[column] == 0).any() for column in ("train", "validation", "test")):
        raise ValueError("At least one class is absent from a required split")
    manifest_dir = ROOT / "data/manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    frame.sort_values("image_path").to_csv(manifest_dir / "images.csv", index=False)
    counts.to_csv(manifest_dir / "class_counts.csv")
    report = {
        "seed": args.seed, "total_files": len(paths), "readable_images": len(frame),
        "invalid_images": invalid, "classes": len(names), "crops": len({label.split("___")[0] for label in names}),
        "splits": {str(key): int(value) for key, value in frame.split.value_counts().items()},
        "leaf_id_sources": {str(key): int(value) for key, value in frame.leaf_id_source.value_counts().items()},
        "duplicate_pixel_images": len(duplicates), "conflicting_duplicate_groups": len(conflict_groups),
        "source_training_images_quarantined_for_test_overlap": int(overlap.sum()),
        "group_overlap": 0, "pixel_duplicate_overlap": 0,
        "grouping_limit": "Physical-leaf identity is verified where metadata resolves it; filename fallback cannot establish identity of otherwise unlinked photos.",
        "dataset_revision": json.loads((ROOT / "configs/dataset.json").read_text())["revision"],
    }
    save_json(ROOT / "reports/data_audit.json", report)
    figures = ROOT / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 13))
    counts[["train", "validation", "test"]].plot.barh(stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("Images")
    ax.set_title("PlantVillage class distribution after grouped splitting")
    fig.tight_layout()
    fig.savefig(figures / "class_distribution.png", dpi=130)
    plt.close(fig)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
