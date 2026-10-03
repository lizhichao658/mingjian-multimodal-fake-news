"""Train and evaluate the MingJian lite multimodal baseline on Weibo17.

Usage (from the repository root)::

    python scripts/train_lite_baseline.py ^
        --data-dir <weibo17_processed> ^
        --out artifacts/lite_v1

The processed data directory is produced by ``scripts/build_weibo17_local.py``.
Raw text, tweet ids and images stay local: they are read from ``--data-dir`` and
never written into the repository. Checkpoints, metrics and predictions go to
``--out`` (``artifacts/`` is git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mingjian.data.dataset import read_jsonl
from mingjian.data.tokenization import build_char_tokenizer
from mingjian.data.torch_dataset import Weibo17TorchDataset
from mingjian.evaluation.metrics import (
    binary_classification_metrics,
    expected_calibration_error,
)
from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig
from mingjian.training.lite_runner import (
    class_weight_for,
    collect_predictions,
    fit_lite_model,
)


def limit_split(samples: list, limit: int, seed: int) -> list:
    """Keep a class-balanced deterministic subset for smoke runs.

    The processed splits are ordered by source file, so naively truncating the
    list would yield a single class. ``--limit`` therefore samples evenly from
    both labels.
    """

    if limit <= 0 or limit >= len(samples):
        return samples
    by_label: dict[int, list] = {0: [], 1: []}
    for sample in samples:
        by_label[sample.label].append(sample)
    half = limit // 2
    chosen = by_label[0][:half] + by_label[1][: limit - half]
    random.Random(seed).shuffle(chosen)
    return chosen


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="processed Weibo17 directory")
    parser.add_argument("--out", type=Path, default=Path("artifacts/lite_v1"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-vocab", type=int, default=6000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=196)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision on GPU")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="keep only the first N samples of every split (smoke runs only)",
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - git may be absent
        return "unknown"
    return completed.stdout.strip() or "unknown"


def make_loader(
    dataset: Weibo17TorchDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    extra: dict[str, object] = {}
    if num_workers > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **extra,
    )


def split_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    metrics = dict(binary_classification_metrics(labels, probabilities, threshold=threshold))
    metrics["ece"] = float(expected_calibration_error(labels, probabilities))
    metrics["samples"] = int(labels.size)
    metrics["mean_probability_fake"] = float(probabilities.mean())
    return metrics


def write_predictions(
    path: Path,
    sample_ids: list[str],
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "probability_fake", "prediction"])
        for sample_id, label, probability in zip(sample_ids, labels, probabilities, strict=True):
            writer.writerow(
                [sample_id, int(label), f"{probability:.6f}", int(probability >= threshold)]
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    data_dir = args.data_dir.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    splits = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing split file: {path}")
        samples = limit_split(read_jsonl(path), args.limit, args.seed)
        splits[name] = samples
        fake = sum(1 for sample in samples if sample.label == 1)
        print(f"[data] {name}: {len(samples)} samples ({fake} fake / {len(samples) - fake} real)")

    tokenizer = build_char_tokenizer(
        (sample.text for sample in splits["train"]),
        max_length=args.max_length,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
    )
    tokenizer.save(out_dir / "tokenizer.json")
    print(f"[tokenizer] vocab_size={tokenizer.vocab_size} max_length={tokenizer.max_length}")

    datasets = {
        name: Weibo17TorchDataset(
            samples,
            root=data_dir,
            tokenizer=tokenizer,
            image_size=args.image_size,
            augment=name == "train",
        )
        for name, samples in splits.items()
    }
    loaders = {
        name: make_loader(
            dataset,
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }

    config = LiteModelConfig(vocab_size=tokenizer.vocab_size, dropout=args.dropout)
    model = LiteFusionClassifier(config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    pos_weight = class_weight_for([sample.label for sample in splits["train"]])
    print(f"[model] parameters={parameters} pos_weight={pos_weight:.4f} device={device}")

    def log_epoch(record: dict[str, float]) -> None:
        print(
            f"[epoch {int(record['epoch'])}] train_loss={record['train_loss']:.4f} "
            f"val_loss={record['val_loss']:.4f} val_f1={record['val_f1']:.4f} "
            f"val_auc={record['val_auc']:.4f} val_ece={record['val_ece']:.4f} "
            f"tuned_threshold={record['tuned_threshold']:.2f} tuned_f1={record['tuned_f1']:.4f}"
        )

    report = fit_lite_model(
        model,
        loaders["train"],
        loaders["val"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        patience=args.patience,
        amp=not args.no_amp,
        seed=args.seed,
        log=log_epoch,
    )
    if report.best_epoch == 0:
        raise SystemExit("training produced no finite validation AUC; nothing to save")
    model.load_state_dict(report.best_state)
    model.eval()
    print(
        f"[select] best_epoch={report.best_epoch} best_val_auc={report.best_val_auc:.4f} "
        f"threshold={report.best_threshold:.2f} stopped_early={report.stopped_early}"
    )

    evaluation: dict[str, dict[str, object]] = {}
    for name in ("val", "test"):
        _, labels, probabilities = collect_predictions(model, loaders[name], device)
        evaluation[name] = {
            "threshold_0_5": split_metrics(labels, probabilities, 0.5),
            "threshold_tuned_on_val": split_metrics(labels, probabilities, report.best_threshold),
        }
        write_predictions(
            out_dir / f"predictions_{name}.csv",
            [sample.sample_id for sample in splits[name]],
            labels,
            probabilities,
            report.best_threshold,
        )
        tuned = evaluation[name]["threshold_tuned_on_val"]
        print(
            f"[eval:{name}] accuracy={tuned['accuracy']:.4f} precision={tuned['precision']:.4f} "
            f"recall={tuned['recall']:.4f} f1={tuned['f1']:.4f} auc={tuned['auc']:.4f} "
            f"ece={tuned['ece']:.4f} threshold={report.best_threshold:.2f}"
        )

    commit = git_commit()
    torch.save(
        {
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "threshold": report.best_threshold,
            "tokenizer_file": "tokenizer.json",
            "git_commit": commit,
            "label_contract": {"0": "real/non-rumor", "1": "fake/rumor"},
        },
        out_dir / "model.pt",
    )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    payload = {
        "project": "MingJian",
        "task": "multimodal fake news detection (news text + single image)",
        "label_contract": {"0": "real/non-rumor", "1": "fake/rumor"},
        "git_commit": commit,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": gpu_name,
        },
        "data": {
            "source": "Weibo17 / EANN Weibo (public research release, not redistributed)",
            "dir": str(data_dir),
            "splits": {
                name: {
                    "samples": len(samples),
                    "fake": sum(1 for sample in samples if sample.label == 1),
                    "real": sum(1 for sample in samples if sample.label == 0),
                }
                for name, samples in splits.items()
            },
        },
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "patience": args.patience,
            "seed": args.seed,
            "image_size": args.image_size,
            "max_length": args.max_length,
            "max_vocab": args.max_vocab,
            "min_freq": args.min_freq,
            "pos_weight": pos_weight,
            "amp": not args.no_amp,
        },
        "model": config.to_dict(),
        "training": {
            "epochs_requested": args.epochs,
            "epochs_ran": report.epochs_ran,
            "best_epoch": report.best_epoch,
            "best_val_auc": report.best_val_auc,
            "best_threshold": report.best_threshold,
            "stopped_early": report.stopped_early,
            "history": report.history,
        },
        "evaluation": evaluation,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())