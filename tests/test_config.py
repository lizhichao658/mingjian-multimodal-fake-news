from __future__ import annotations

from pathlib import Path

from mingjian.config import load_config


def test_baseline_config_loads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.toml"
    config = load_config(config_path)
    assert config.model.text_encoder == "hfl/chinese-roberta-wwm-ext"
    assert config.training.seed == 42
    assert config.data.text_max_length == 196