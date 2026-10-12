"""Verify the final full-data deliverables and write the completion report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main():
    import pandas as pd
    from src.common import class_names, save_json
    from src.predict import Predictor, decode_image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", default="reports/full-cpu-progress.json")
    args = parser.parse_args()
    progress = read_json(args.progress)
    if progress["status"] != "complete":
        raise ValueError("The full-data pipeline must finish before final verification")
    latest = read_json("reports/latest_run.json")
    if any(latest[key] != progress[key] for key in ("baseline", "resnet18", "test_evaluation")):
        raise ValueError("Latest run does not match the completed full-data pipeline")
    card = read_json("artifacts/model_bundle/model_card.json")
    metrics = read_json(f"{latest['test_evaluation']}/metrics.json")
    model_sha256 = hashlib.sha256((ROOT / "artifacts/model_bundle/model.pt").read_bytes()).hexdigest()
    if metrics.get("checkpoint_sha256") != model_sha256:
        raise ValueError("The evaluated checkpoint differs from the packaged model")
    frame = pd.read_csv(ROOT / "data/manifests/images.csv")
    expected_test_images = int((frame.split == "test").sum())
    if card["training_images"] != int((frame.split == "train").sum()):
        raise ValueError("The packaged model was not trained on the entire training split")
    if metrics["training_profile"] != "full_data" or metrics["sampled"] or metrics["images"] != expected_test_images:
        raise ValueError("The evaluation must cover the entire held-out test split")
    if metrics["classes_with_examples"] != len(class_names()) or metrics["model_version"] != card["model_version"]:
        raise ValueError("Evaluation class coverage or model version differs from the bundle")
    predictions = pd.read_csv(ROOT / latest["test_evaluation"] / "predictions.csv")
    if len(predictions) != expected_test_images or predictions.image_path.duplicated().any() or set(predictions.image_path) != set(frame.loc[frame.split == "test", "image_path"]):
        raise ValueError("The evaluation predictions do not cover the exact test manifest")
    measured_accuracy = float((predictions.class_id == predictions.predicted_class_id).mean())
    if abs(measured_accuracy - metrics["accuracy"]) > 1e-12:
        raise ValueError("Reported accuracy does not match the saved predictions")

    dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, check=True)
    print(dependency_check.stdout.strip(), flush=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    verification = {
        "status": "passed" if result.wasSuccessful() and not result.skipped else "failed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "dependency_check": dependency_check.stdout.strip(),
        "model_version": card["model_version"],
        "model_sha256": model_sha256,
        "evaluation_checkpoint_sha256_matches_bundle": True,
        "test_images": expected_test_images,
        "full_test_predictions_match_manifest": True,
        "reported_accuracy_matches_predictions": True,
        "external_photo_evaluation": "not measured",
    }
    if verification["status"] != "passed":
        save_json(ROOT / "reports/project_verification.json", verification)
        raise RuntimeError("Final regression checks did not all pass")

    predictor = Predictor(ROOT / "artifacts/model_bundle/model.pt")
    sample = decode_image((ROOT / predictions.iloc[0].image_path).read_bytes())
    predictor.predict(sample)  # Warm-up; model loading is excluded.
    latencies = []
    for _ in range(5):
        started = time.perf_counter()
        predictor.predict(sample)
        latencies.append(time.perf_counter() - started)
    verification["warm_single_image_median_seconds"] = statistics.median(latencies)
    verification["latency_measurement"] = "Five runs of shared RGB-image preprocessing and CPU inference; excludes file upload, decoding and initial model loading"
    browser_record_path = ROOT / "reports/browser_verification.json"
    browser_record = read_json("reports/browser_verification.json") if browser_record_path.is_file() else {}
    browser_verified = browser_record.get("status") == "passed" and browser_record.get("model_sha256") == model_sha256 and (ROOT / "reports/demo.jpg").is_file()
    if browser_verified:
        verification["browser_upload_and_classification"] = "passed; see reports/browser_verification.json"
    save_json(ROOT / "reports/project_verification.json", verification)

    audit = read_json("reports/data_audit.json")
    baseline = read_json(f"{latest['baseline']}/history.json")
    resnet = read_json(f"{latest['resnet18']}/history.json")
    from src.train import plot_learning_curves
    for run_dir, history in ((latest["baseline"], baseline), (latest["resnet18"], resnet)):
        plot_learning_curves(history["epochs"], ROOT / run_dir / "learning_curves.png")
    baseline_best = max(baseline["epochs"], key=lambda row: row["validation"]["macro_f1"])
    resnet_best = max(resnet["epochs"], key=lambda row: row["validation"]["macro_f1"])
    head_epochs = sum(row["stage"] == "head" for row in resnet["epochs"])
    fine_epochs = sum(row["stage"] == "fine_tune" for row in resnet["epochs"])
    report = [
        "# Plant disease classifier: final evaluation", "",
        f"Completed and verified on {verification['checked_at_utc']}. This is a local educational classifier for 38 supported crop-condition classes across 14 crops.", "",
        "## Data and isolation", "",
        f"The audit decoded {audit['readable_images']:,} PlantVillage color images. The usable splits contain {audit['splits']['train']:,} training, {audit['splits']['validation']:,} validation and {audit['splits']['test']:,} test images. {audit['splits']['quarantine']:,} images were quarantined to preserve source test groups.", "",
        "There is no resolved-group or identical-pixel overlap between usable splits. Physical-leaf metadata resolves 40,490 images; 13,815 require filename fallback. That fallback cannot identify otherwise unlinked photographs of the same physical leaf.", "",
        "## Training and model selection", "",
        f"The CNN trained for {len(baseline['epochs'])} epoch(s) on all eligible training images. The ImageNet-pretrained ResNet18 trained its new 38-class classifier for {head_epochs} epoch(s), then fine-tuned its final residual stage for {fine_epochs} epoch(s). Model selection and the uncertainty rule used validation data only.", "",
        "| Candidate | Best validation accuracy | Best validation macro F1 |", "|---|---:|---:|",
        f"| Small CNN | {baseline_best['validation']['accuracy']:.2%} | {baseline_best['validation']['macro_f1']:.4f} |",
        f"| ResNet18 | {resnet_best['validation']['accuracy']:.2%} | {resnet_best['validation']['macro_f1']:.4f} |", "",
        f"Selected model: **{card['architecture']}**, version `{card['model_version']}`, trained on {card['training_images']:,} images.", "",
        f"[CNN training history]({latest['baseline']}/history.json) · [CNN learning curves]({latest['baseline']}/learning_curves.png) · [ResNet18 training history]({latest['resnet18']}/history.json) · [ResNet18 learning curves]({latest['resnet18']}/learning_curves.png)", "",
        "## Complete held-out test results", "",
        f"- Images: **{metrics['images']:,}**, all 38 classes; the test split was not sampled.",
        f"- Accuracy: **{metrics['accuracy']:.2%}**.",
        f"- Macro F1: **{metrics['macro_f1']:.4f}**.",
        f"- Batch evaluation time: {metrics['elapsed_seconds']:.1f} seconds; {1000 * metrics['pipeline_seconds_per_image']:.1f} ms per image including test loading and transforms.",
        f"- Warm single-image preprocessing and inference: median {1000 * verification['warm_single_image_median_seconds']:.1f} ms on this CPU over five runs. This excludes uploading, decoding and initial model loading.", "",
        f"[Full metrics]({latest['test_evaluation']}/metrics.json) · [Every test prediction]({latest['test_evaluation']}/predictions.csv) · [Confusion matrix CSV]({latest['test_evaluation']}/confusion_matrix.csv)", "",
        f"![Row-normalized test confusion matrix]({latest['test_evaluation']}/confusion_matrix.png)", "",
        "## Per-class results", "",
        "| Class | Precision | Recall | F1 | Test images |", "|---|---:|---:|---:|---:|",
    ]
    for name in class_names():
        item = metrics["class_report"][name]
        report.append(f"| {name} | {item['precision']:.4f} | {item['recall']:.4f} | {item['f1-score']:.4f} | {int(item['support'])} |")
    resume_log = ROOT / "reports/full-training-resume.log"
    if resume_log.is_file() and latest["resnet18"] in resume_log.read_text(encoding="utf-8"):
        report += ["", "The full-data run resumed from the completed classifier checkpoint after an interruption. The unfinished fine-tuning epoch was restarted. Optimizer and random state were not restored for an exact batch-level continuation."]
    report += ["", "## Uncertainty and scope", ""]
    filtering = card["uncertainty_filter"]
    if filtering["status"] == "validation_estimate":
        report.append(f"The score threshold is {filtering['threshold']:.6f}. On validation data it accepts {filtering['accepted_images']:,}/{filtering['validation_images']:,} images ({filtering['coverage']:.2%} coverage), with {filtering['accepted_accuracy']:.2%} accuracy among accepted examples. This estimate is not probability calibration or a detector for unsupported inputs.")
    else:
        report.append("No uncertainty threshold met the validation acceptance target. Scores remain rankings among supported classes.")
    report += [
        "", "**Independent field-photo performance has not been measured.** PlantVillage results do not establish performance under field backgrounds, lighting or unsupported diseases. A verified independent-photo set can be evaluated separately with the command in the README.", "",
        "## Final verification", "",
        f"All {result.testsRun} regression checks passed, with no skipped tests. They cover split isolation, class coverage, upload size/type/pixel limits, RGB conversion, packaged label and preprocessing contracts, consistent saved-model reload predictions, resumable stage boundaries, and the app's classification and unreadable-upload paths. The app prediction was checked against the shared prediction module. Dependency consistency also passed.", "",
        "App tests inject JPEG bytes into the uploader using Streamlit AppTest and click the real classification button. They exercise the application script; they do not simulate a browser's native file-selection dialog.", "",
        "[Verification record](reports/project_verification.json) · [Model card](artifacts/model_bundle/model_card.json) · [Data audit](reports/data_audit.json)", "",
    ]
    if browser_verified:
        report += ["## Browser preview", "", "The final model was also tested through the local browser upload and classification flow using a public PlantVillage Apple scab photograph. The page displayed Apple scab and three ranked alternatives. The displayed score is a model score for this example.", "", "![Verified app prediction](reports/demo.jpg)", "", "[Browser verification record](reports/browser_verification.json)", ""]
    (ROOT / "EVALUATION.md").write_text("\n".join(report), encoding="utf-8")
    readme = ROOT / "README.md"
    content = readme.read_text(encoding="utf-8")
    if not browser_verified:
        content = content.split("\n## Browser preview\n", 1)[0].rstrip() + "\n"
    lines = []
    for line in content.splitlines():
        if line.startswith("Current progress:"):
            line = f"Current progress: **Complete. The final {card['architecture']} model scored {metrics['accuracy']:.2%} accuracy and {metrics['macro_f1']:.4f} macro F1 on all {metrics['images']:,} held-out images. All {result.testsRun} regression checks passed. See [the final evaluation](EVALUATION.md). Field-photo accuracy remains unmeasured.**"
        elif line.startswith("6. **Pilot complete:"):
            line = "6. **Complete:** Train and document the full-data CNN baseline."
        elif line.startswith("8. **Pilot complete:"):
            line = "8. **Complete:** Train ResNet18 on the full training split and select its best validation checkpoint."
        elif line.startswith("9. **Pilot complete:"):
            line = "9. **Complete:** Evaluate all 10,709 held-out images with per-class metrics and a confusion matrix. Independent-photo evaluation remains unmeasured."
        elif line.startswith("11. **Pilot complete:"):
            line = "11. **Complete:** Verify model reload, upload errors, app predictions, dependencies, and documentation."
        lines.append(line)
    content = "\n".join(lines) + "\n"
    if "## Final verification" not in content:
        content += "\n## Final verification\n\nStart the packaged app with `.\\start_app.ps1`, or use the Python command above. Run the final checks and regenerate the evaluation report with:\n\n```powershell\n.\\.venv\\Scripts\\python.exe scripts/verify_project.py\n```\n\nThe complete results are in [EVALUATION.md](EVALUATION.md); the verification record is in `reports/project_verification.json`. The earlier pilot results are retained as development history.\n"
    readme.write_text(content, encoding="utf-8")
    scope_path = ROOT / "PROJECT_SCOPE.md"
    scope = scope_path.read_text(encoding="utf-8")
    scope = scope.replace("Step 1 completed on 12 October 2026. This document defines the first version; implementation starts in Step 2.", "Scope established on 12 October 2026. The implemented project and full held-out results are documented in [EVALUATION.md](EVALUATION.md).")
    scope = scope.replace("Step 1 is complete when this scope, the supported-class contract, and the build sequence are saved. Model training and application behavior are future completion checks.", "The scope, class contract, and build sequence are implemented. Full-data training, held-out evaluation, packaged-model reload, upload handling, and app prediction checks have passed. Independent field-photo accuracy remains unmeasured.")
    scope_path.write_text(scope, encoding="utf-8")
    print(f"Final verification complete: {result.testsRun} tests passed; full-test accuracy={metrics['accuracy']:.2%}; report=EVALUATION.md", flush=True)


if __name__ == "__main__":
    main()
