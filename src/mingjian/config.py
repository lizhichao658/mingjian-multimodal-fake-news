"""Configuration objects for MingJian experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(frozen=True)
class DataConfig:
    train_jsonl: str = "data/processed/train.jsonl"
    val_jsonl: str = "data/processed/val.jsonl"
    test_jsonl: str = "data/processed/test.jsonl"
    text_max_length: int = 196
    image_size: int = 224

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> DataConfig:
        values = values or {}
        return cls(
            train_jsonl=str(values.get("train_jsonl", cls.train_jsonl)),
            val_jsonl=str(values.get("val_jsonl", cls.val_jsonl)),
            test_jsonl=str(values.get("test_jsonl", cls.test_jsonl)),
            text_max_length=int(values.get("text_max_length", cls.text_max_length)),
            image_size=int(values.get("image_size", cls.image_size)),
        )


@dataclass(frozen=True)
class ModelConfig:
    text_encoder: str = "hfl/chinese-roberta-wwm-ext"
    image_encoder: str = "google/vit-base-patch16-224"
    hidden_dim: int = 512
    dropout: float = 0.2
    freeze_encoders: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> ModelConfig:
        values = values or {}
        return cls(
            text_encoder=str(values.get("text_encoder", cls.text_encoder)),
            image_encoder=str(values.get("image_encoder", cls.image_encoder)),
            hidden_dim=int(values.get("hidden_dim", cls.hidden_dim)),
            dropout=float(values.get("dropout", cls.dropout)),
            freeze_encoders=bool(values.get("freeze_encoders", cls.freeze_encoders)),
        )


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 8
    learning_rate: float = 2.0e-6
    weight_decay: float = 0.01
    epochs: int = 5
    num_workers: int = 0
    mixed_precision: bool = True

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> TrainingConfig:
        values = values or {}
        return cls(
            seed=int(values.get("seed", cls.seed)),
            batch_size=int(values.get("batch_size", cls.batch_size)),
            learning_rate=float(values.get("learning_rate", cls.learning_rate)),
            weight_decay=float(values.get("weight_decay", cls.weight_decay)),
            epochs=int(values.get("epochs", cls.epochs)),
            num_workers=int(values.get("num_workers", cls.num_workers)),
            mixed_precision=bool(values.get("mixed_precision", cls.mixed_precision)),
        )


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> AppConfig:
        values = values or {}
        return cls(
            data=DataConfig.from_mapping(values.get("data")),
            model=ModelConfig.from_mapping(values.get("model")),
            training=TrainingConfig.from_mapping(values.get("training")),
        )


def load_config(path: str | Path) -> AppConfig:
    """Load a TOML configuration file."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)
    return AppConfig.from_mapping(raw)