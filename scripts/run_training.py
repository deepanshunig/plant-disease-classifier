"""Run training, packaging, evaluation, and inference checks in one interpreter."""

from __future__ import annotations

import argparse
import gc
import json
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime/matplotlib"))
PROGRESS_FILE = None


def run(module: str, arguments: list[str]):
    print(f"\nRunning {module}: {' '.join(arguments)}", flush=True)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__")
    gc.collect()


def main():
    global PROGRESS_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--name", default="pilot")
    parser.add_argument("--max-test-per-class", type=int)
    parser.add_argument("--resume", action="store_true", help="Resume the recorded pipeline from its last phase")
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    baseline = f"artifacts/runs/{args.name}-{run_id}-small-cnn"
    resnet = f"artifacts/runs/{args.name}-{run_id}-resnet18"
    evaluation = f"reports/{args.name}-{run_id}-test"
    from src.common import save_json
    progress_path = ROOT / f"reports/{args.name}-progress.json"
    progress = {"status": "running", "config": args.config, "baseline": baseline, "resnet18": resnet, "test_evaluation": evaluation}
    phases = ["training_baseline", "training_resnet18", "packaging_selected_model", "evaluating_held_out_test", "prediction_reload_check_passed"]
    start_phase = 0
    if args.resume:
        progress = json.loads(progress_path.read_text())
        if progress["config"] != args.config:
            raise ValueError("Resume config differs from the saved pipeline")
        if progress["status"] == "complete":
            raise ValueError("This pipeline is already complete")
        baseline, resnet, evaluation = (progress[key] for key in ("baseline", "resnet18", "test_evaluation"))
        start_phase = phases.index(progress["phase"])
        progress["status"] = "running"
        progress.pop("error", None)
    PROGRESS_FILE = progress_path
    def record_phase(phase):
        progress["phase"] = phase
        save_json(progress_path, progress)
    for phase_index, architecture, run_dir in [(0, "small_cnn", baseline), (1, "resnet18", resnet)]:
        if start_phase <= phase_index:
            record_phase(phases[phase_index])
            training_args = ["--config", args.config, "--architecture", architecture, "--run-dir", run_dir]
            if args.resume and (ROOT / run_dir / "best.pt").exists():
                training_args.append("--resume")
            run("src.train", training_args)
    if start_phase <= 2:
        record_phase("packaging_selected_model")
        run("src.package_model", ["--runs", baseline, resnet])
    evaluation_args = ["--checkpoint", "artifacts/model_bundle/model.pt", "--output-dir", evaluation]
    if args.max_test_per_class:
        evaluation_args += ["--max-per-class", str(args.max_test_per_class)]
    if start_phase <= 3:
        record_phase("evaluating_held_out_test")
        run("src.evaluate", evaluation_args)

    import pandas as pd
    from src.predict import Predictor, decode_image

    frame = pd.read_csv(ROOT / "data/manifests/images.csv")
    sample = ROOT / frame[frame.split == "test"].iloc[0].image_path
    image = decode_image(sample.read_bytes())
    predictor = Predictor(ROOT / "artifacts/model_bundle/model.pt")
    first = predictor.predict(image)
    second = Predictor(ROOT / "artifacts/model_bundle/model.pt").predict(image)
    if first != second or len(first["top_three"]) != 3:
        raise ValueError("Saved-model prediction consistency check failed")
    save_json(ROOT / "reports/prediction_check.json", {"status": "passed", "sample_image": str(sample), "result": first, "model_reload_predictions_match": True})
    save_json(ROOT / "reports/latest_run.json", {"baseline": baseline, "resnet18": resnet, "test_evaluation": evaluation})
    progress["status"] = "complete"
    record_phase("prediction_reload_check_passed")
    if predictor.checkpoint["training_profile"] == "full_data":
        readme_path = ROOT / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        lines = ["Current progress: **The first full-data run is complete, including held-out evaluation and saved-model inference verification. See reports/latest_run.json for results.**" if line.startswith("Current progress:") else line for line in lines]
        readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        run("scripts.verify_project", ["--progress", str(progress_path)])
    print(f"Complete: saved-model reload check passed. Evaluation report: {evaluation}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        if PROGRESS_FILE is not None:
            state = json.loads(PROGRESS_FILE.read_text()) if PROGRESS_FILE.exists() else {}
            state["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            state["error"] = str(error)
            PROGRESS_FILE.write_text(json.dumps(state, indent=2) + "\n")
        raise
