from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader

from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig
from mingjian.training.lite_runner import (
    class_weight_for,
    collect_predictions,
    fit_lite_model,
    select_threshold,
)


class _SyntheticSplit:
    """Fake posts carry a red-dominant image, real posts a blue-dominant one."""

    def __init__(self, size: int, *, seed: int = 0, constant_label: int | None = None) -> None:
        generator = torch.Generator().manual_seed(seed)
        labels = torch.randint(0, 2, (size,), generator=generator)
        if constant_label is not None:
            labels = torch.full((size,), constant_label, dtype=torch.long)
        self.labels = labels
        self.input_ids = torch.randint(1, 20, (size, 8), generator=generator)
        self.images = torch.randn(size, 3, 16, 16, generator=generator) * 0.1
        fake = labels.float().view(-1, 1, 1, 1)
        self.images = self.images + fake * torch.tensor([1.5, 0.0, -1.5]).view(1, 3, 1, 1)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": torch.ones(8),
            "images": self.images[index],
            "labels": self.labels[index].float(),
            "index": index,
            "sample_id": f"s{index}",
        }


def _model() -> LiteFusionClassifier:
    config = LiteModelConfig(
        vocab_size=20,
        text_embed_dim=8,
        text_num_filters=4,
        text_kernel_sizes=(2, 3),
        image_widths=(8, 16),
        image_feature_dim=16,
        hidden_dim=16,
        dropout=0.0,
    )
    return LiteFusionClassifier(config)


def test_class_weight_for_balances_imbalanced_splits() -> None:
    assert class_weight_for([0, 1]) == pytest.approx(1.0)
    assert class_weight_for([0, 0, 0, 1]) == pytest.approx(3.0)
    assert class_weight_for([1, 1, 1, 1]) == pytest.approx(1.0)
    assert class_weight_for(np.array([0, 0])) == pytest.approx(1.0)


def test_select_threshold_maximises_f1() -> None:
    labels = [0, 0, 1, 1]
    probabilities = [0.10, 0.40, 0.35, 0.80]
    threshold, metrics = select_threshold(labels, probabilities)
    # Every threshold in (0.10, 0.40] yields the same perfect-recall split;
    # the tie-break keeps the lowest one the grid actually visits.
    assert 0.10 < threshold <= 0.40
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(2 / 3)


def test_collect_predictions_covers_every_sample() -> None:
    split = _SyntheticSplit(20)
    loader = DataLoader(split, batch_size=8)
    loss, labels, probabilities = collect_predictions(_model(), loader, torch.device("cpu"))
    assert math.isnan(loss)  # no criterion -> NaN loss placeholder
    assert labels.shape == (20,)
    assert probabilities.shape == (20,)
    assert labels.dtype == np.int64
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_fit_lite_model_learns_the_synthetic_signal() -> None:
    train_split = _SyntheticSplit(64, seed=1)
    val_split = _SyntheticSplit(32, seed=2)
    model = _model()
    report = fit_lite_model(
        model,
        DataLoader(train_split, batch_size=16),
        DataLoader(val_split, batch_size=16),
        device=torch.device("cpu"),
        epochs=4,
        learning_rate=2e-2,
        patience=4,
        amp=False,
        seed=7,
    )
    assert len(report.history) == 4
    assert report.best_epoch >= 1
    assert report.best_val_auc > 0.8
    assert set(report.best_state) == set(model.state_dict())
    assert report.best_val_metrics["f1"] > 0.7
    assert 0.0 < report.best_threshold < 1.0
    assert report.stopped_early is False
    assert report.history[-1]["val_ece"] >= 0.0

    model.load_state_dict(report.best_state)
    _, labels, probabilities = collect_predictions(
        model, DataLoader(val_split, batch_size=16), torch.device("cpu")
    )
    predictions = (probabilities >= report.best_threshold).astype(int)
    assert (predictions == labels).mean() > 0.75


def test_fit_validates_arguments() -> None:
    split = _SyntheticSplit(8)
    loader = DataLoader(split, batch_size=4)
    common = {
        "train_batches": loader,
        "val_batches": loader,
        "device": torch.device("cpu"),
    }
    with pytest.raises(ValueError, match="epochs"):
        fit_lite_model(_model(), epochs=0, learning_rate=1e-3, **common)
    with pytest.raises(ValueError, match="learning_rate"):
        fit_lite_model(_model(), epochs=1, learning_rate=0.0, **common)
    with pytest.raises(ValueError, match="patience"):
        fit_lite_model(_model(), epochs=1, learning_rate=1e-3, patience=-1, **common)


def test_early_stopping_stops_when_auc_never_improves() -> None:
    single_class = _SyntheticSplit(16, constant_label=0)
    report = fit_lite_model(
        _model(),
        DataLoader(single_class, batch_size=8),
        DataLoader(single_class, batch_size=8),
        device=torch.device("cpu"),
        epochs=4,
        learning_rate=1e-3,
        patience=1,
        amp=False,
    )
    assert report.stopped_early is True
    assert report.epochs_ran == 1
    assert report.best_epoch == 0