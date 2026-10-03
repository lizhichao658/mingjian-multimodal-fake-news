"""Training and evaluation loop for the lite fusion baseline.

Kept separate from ``mingjian.training.engine`` because the MVP pipeline needs
three extra behaviours the generic smoke-test loop does not provide: class
imbalance weighting, threshold tuning on the validation split, and epoch-level
model selection with early stopping.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from mingjian.evaluation.metrics import (
    binary_classification_metrics,
    expected_calibration_error,
)
from mingjian.models.lite import LiteFusionClassifier

DEFAULT_THRESHOLD_GRID: tuple[float, ...] = tuple(float(value) for value in np.linspace(0.05, 0.95, 19))


def class_weight_for(labels: Sequence[int] | np.ndarray) -> float:
    """Return ``pos_weight`` for BCE so the two classes contribute evenly."""

    array = np.asarray(labels, dtype=np.int64).reshape(-1)
    positives = int((array == 1).sum())
    negatives = int((array == 0).sum())
    if positives == 0 or negatives == 0:
        return 1.0
    return float(negatives / positives)


def select_threshold(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    grid: Iterable[float] = DEFAULT_THRESHOLD_GRID,
) -> tuple[float, dict[str, float | int | dict[str, int]]]:
    """Pick the threshold that maximises F1 (ties keep the lowest threshold)."""

    best_threshold = 0.5
    best_metrics = binary_classification_metrics(labels, probabilities, threshold=best_threshold)
    best_score = float(best_metrics["f1"])
    for threshold in grid:
        metrics = binary_classification_metrics(labels, probabilities, threshold=float(threshold))
        score = float(metrics["f1"])
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "images": batch["images"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device=device, dtype=torch.float32),
    }


def train_one_epoch_lite(
    model: LiteFusionClassifier,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip: float | None = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    for batch in batches:
        moved = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
            logits = model(
                input_ids=moved["input_ids"],
                images=moved["images"],
                attention_mask=moved["attention_mask"],
            ).logits
            loss = criterion(logits, moved["labels"])

        if amp_enabled:
            assert scaler is not None
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = int(moved["labels"].shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def collect_predictions(
    model: LiteFusionClassifier,
    batches: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    criterion: nn.Module | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return ``(mean_loss, labels, P(fake))`` for one data split."""

    model.eval()
    losses: list[float] = []
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    total_samples = 0

    for batch in batches:
        moved = _move_batch(batch, device)
        logits = model(
            input_ids=moved["input_ids"],
            images=moved["images"],
            attention_mask=moved["attention_mask"],
        ).logits
        if criterion is not None:
            loss = criterion(logits, moved["labels"])
            losses.append(float(loss.detach().cpu()) * int(moved["labels"].shape[0]))
        total_samples += int(moved["labels"].shape[0])
        labels_all.append(moved["labels"].detach().cpu().numpy().astype(np.int64))
        probabilities_all.append(torch.sigmoid(logits).detach().cpu().numpy())

    if not labels_all:
        return float("nan"), np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    labels = np.concatenate(labels_all)
    probabilities = np.concatenate(probabilities_all).astype(np.float64)
    mean_loss = float(sum(losses) / max(total_samples, 1)) if losses else float("nan")
    return mean_loss, labels, probabilities


@dataclass
class LiteTrainReport:
    """Everything a report or a checkpoint needs from one training run."""

    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0
    best_val_auc: float = float("-inf")
    best_threshold: float = 0.5
    best_val_metrics: dict[str, Any] = field(default_factory=dict)
    best_state: dict[str, torch.Tensor] = field(default_factory=dict)
    stopped_early: bool = False
    epochs_ran: int = 0


def fit_lite_model(
    model: LiteFusionClassifier,
    train_batches: Iterable[dict[str, Any]],
    val_batches: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float = 0.01,
    pos_weight: float | None = None,
    patience: int = 3,
    amp: bool = True,
    grad_clip: float = 1.0,
    seed: int = 42,
    log: Callable[[dict[str, float]], None] | None = None,
) -> LiteTrainReport:
    """Train ``model`` in place and keep the best validation-AUC weights."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if patience < 0:
        raise ValueError("patience must not be negative")

    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    weight = None if pos_weight is None else torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    report = LiteTrainReport()
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch_lite(
            model,
            train_batches,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            grad_clip=grad_clip,
        )
        val_loss, labels, probabilities = collect_predictions(
            model, val_batches, device, criterion=criterion
        )
        metrics = binary_classification_metrics(labels, probabilities, threshold=0.5)
        threshold, tuned_metrics = select_threshold(labels, probabilities)
        record = {
            "epoch": float(epoch),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(metrics["accuracy"]),
            "val_precision": float(metrics["precision"]),
            "val_recall": float(metrics["recall"]),
            "val_f1": float(metrics["f1"]),
            "val_auc": float(metrics["auc"]),
            "val_ece": float(expected_calibration_error(labels, probabilities)),
            "tuned_threshold": float(threshold),
            "tuned_f1": float(tuned_metrics["f1"]),
        }
        report.history.append(record)
        report.epochs_ran = epoch
        if log is not None:
            log(record)

        score = float(metrics["auc"])
        if np.isfinite(score) and score > report.best_val_auc:
            report.best_val_auc = score
            report.best_epoch = epoch
            report.best_threshold = float(threshold)
            report.best_val_metrics = dict(tuned_metrics)
            report.best_val_metrics["threshold_0_5"] = metrics
            report.best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience and epoch < epochs:
                report.stopped_early = True
                break

        scheduler.step()

    return report