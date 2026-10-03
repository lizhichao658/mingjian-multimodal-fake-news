"""Generic training loop for smoke tests and future real pipelines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            output = model(input_ids=input_ids, images=images)
            loss = criterion(output.logits, labels)

        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_binary(
    model: nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []

    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device=device, dtype=torch.float32)
        output = model(input_ids=input_ids, images=images)
        labels_all.append(labels.detach().cpu().numpy())
        probabilities_all.append(output.probability.detach().cpu().numpy())

    if not labels_all:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
    return np.concatenate(labels_all), np.concatenate(probabilities_all)