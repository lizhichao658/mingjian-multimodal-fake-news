"""Prepare a local model from the complete, separately obtained Weibo17 archive.

No research data or weights are downloaded or redistributed. The work directory
must be outside this checkout and empty. All generated data, logs and weights
remain there. ``--help`` uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SPLITS = {"train": 5415, "val": 843, "test": 1465}
HEALTH_TEXT = "这是一条自行编写的流程验收文本，请核查原始来源和发布时间。"


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="official full weibo.zip")
    parser.add_argument("--work-dir", required=True, type=Path, help="empty directory outside checkout")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--num-workers", type=_nonnegative, default=0)
    parser.add_argument("--threads", type=_positive, default=4, help="CPU thread budget per process")
    return parser.parse_args(argv)


def validate_paths(archive: Path, work_dir: Path) -> tuple[Path, Path]:
    archive, work_dir = archive.resolve(), work_dir.resolve()
    root = REPO_ROOT.resolve()
    if work_dir == root or root in work_dir.parents:
        raise ValueError("work directory must be outside the checkout")
    if not archive.is_file():
        raise ValueError("archive is not a file")
    if work_dir.exists() and (not work_dir.is_dir() or any(work_dir.iterdir())):
        raise ValueError("work directory must be new or empty; existing outputs are never replaced")
    return archive, work_dir


def runtime_info(requested: str) -> tuple[str, dict[str, Any]]:
    # Delay the optional dependency import until after CLI parsing and path checks.
    import torch

    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA requested but unavailable")
    device = "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"
    versions = {"python": platform.python_version(), "torch": str(torch.__version__)}
    for package in ("numpy", "pillow", "torchvision"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return device, {"versions": versions, "torch_cuda_build": torch.version.cuda}


def _write_report(work_dir: Path, report: dict[str, Any]) -> None:
    (work_dir / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _run_stage(
    name: str,
    command: list[str],
    *,
    work_dir: Path,
    env: dict[str, str],
    report: dict[str, Any],
) -> None:
    report["stage"] = name
    print(f"[{name}] running; detailed output stays in local logs", flush=True)
    stage: dict[str, Any] = {"name": name, "status": "failed", "returncode": None}
    started = time.perf_counter()
    try:
        with (work_dir / "logs" / f"{name}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        stage["returncode"] = completed.returncode
        if completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command)
        stage["status"] = "passed"
    finally:
        stage["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        report["stages"].append(stage)


def verify_artifacts(model_dir: Path) -> dict[str, Any]:
    for name in ("model.pt", "tokenizer.json", "metrics.json"):
        path = model_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"required nonempty artifact missing: {name}")
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    if not isinstance(tokenizer, dict) or not tokenizer:
        raise ValueError("invalid tokenizer")
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    splits = metrics["data"]["splits"]
    if {name: splits[name]["samples"] for name in EXPECTED_SPLITS} != EXPECTED_SPLITS:
        raise ValueError("training did not use all official splits")
    summary: dict[str, Any] = {"split_samples": EXPECTED_SPLITS.copy(), "evaluation": {}}
    for split in ("val", "test"):
        values = metrics["evaluation"][split]["threshold_0_5"]
        selected = {}
        for name in ("accuracy", "precision", "recall", "f1", "auc", "ece"):
            number = float(values[name])
            if not math.isfinite(number) or not 0 <= number <= 1:
                raise ValueError("invalid evaluation metric")
            selected[name] = number
        if values["samples"] != EXPECTED_SPLITS[split] or values["threshold"] != 0.5:
            raise ValueError("invalid evaluation scope")
        summary["evaluation"][split] = selected
    training = metrics["training"]
    if training["epochs_requested"] != 8 or not 1 <= training["best_epoch"] <= training["epochs_ran"]:
        raise ValueError("invalid training completion")
    summary["training"] = {
        key: training[key] for key in ("epochs_requested", "epochs_ran", "best_epoch")
    }
    return summary


def verify_cpu_health(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    probability = float(result["decision"]["probability_fake"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("invalid CPU inference probability")
    if result["modalities"]["has_image"] is not False:
        raise ValueError("text-only health check reported an image")
    if result["evidence_card"]["cross_modal"]["reason_code"] != "missing_image":
        raise ValueError("missing-image fallback was not verified")
    return {"device": "cpu", "text_only_inference": "passed", "missing_image_fallback": "passed"}


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        archive, work_dir = validate_paths(args.archive, args.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "logs").mkdir()
    except (OSError, ValueError) as exc:
        print(f"error: preflight failed ({type(exc).__name__}); no existing outputs changed", file=sys.stderr)
        return 1

    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "stage": "dependencies",
        "requested_device": args.device,
        "cpu_threads": args.threads,
        "num_workers": args.num_workers,
        "stages": [],
        "scope": "Local preparation; excludes dataset download and dependency installation.",
        "notice": "Keep raw/processed research data, weights and logs local; do not redistribute.",
        "outputs": {"processed": "processed", "model": "model", "logs": "logs"},
    }
    model_dir, processed = work_dir / "model", work_dir / "processed"
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    })
    try:
        device, environment = runtime_info(args.device)
        report.update({"actual_device": device, "environment": environment})
        _write_report(work_dir, report)
        _run_stage("build", [
            sys.executable, str(REPO_ROOT / "scripts/build_weibo17_local.py"),
            "--archive", str(archive), "--out", str(processed),
        ], work_dir=work_dir, env=env, report=report)
        _run_stage("train", [
            sys.executable, str(REPO_ROOT / "scripts/train_lite_baseline.py"),
            "--data-dir", str(processed), "--out", str(model_dir),
            "--epochs", "8", "--batch-size", "64", "--lr", "2e-3",
            "--device", device, "--num-workers", str(args.num_workers),
        ], work_dir=work_dir, env=env, report=report)
        report["stage"] = "artifact_validation"
        report["new_model_metrics"] = verify_artifacts(model_dir)
        health_path = work_dir / "cpu_health.json"
        _run_stage("cpu_explain", [
            sys.executable, str(REPO_ROOT / "scripts/explain_lite.py"),
            "--checkpoint", str(model_dir / "model.pt"), "--device", "cpu",
            "--text", HEALTH_TEXT, "--out", str(health_path),
        ], work_dir=work_dir, env=env, report=report)
        report["stage"] = "health_validation"
        report["cpu_health"] = verify_cpu_health(health_path)
        report.update({"status": "passed", "stage": "complete"})
    except (
        OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError,
        subprocess.SubprocessError, KeyboardInterrupt,
    ) as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        print(f"error: preparation failed during {report['stage']}; see local logs/report", file=sys.stderr)
        return 1
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        _write_report(work_dir, report)

    print("Model prepared and CPU text-only inference verified. New metrics are local to this run.")
    print("Keep data, weights and logs local; do not upload them with the source submission.")
    command = [
        sys.executable, str(REPO_ROOT / "scripts/run_web_demo.py"),
        "--checkpoint", str(model_dir / "model.pt"), "--device", "cpu", "--port", "8002",
    ]
    print("PowerShell start command:\n& " + " ".join(_ps_quote(value) for value in command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
