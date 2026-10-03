from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mingjian.data.schema import NewsSample

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_lite_baseline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_lite_baseline", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _samples(fake: int, real: int) -> list[NewsSample]:
    rows = [NewsSample(f"f{i}", "text", "a.png", 1) for i in range(fake)]
    rows += [NewsSample(f"r{i}", "text", "a.png", 0) for i in range(real)]
    return rows


def test_limit_split_is_class_balanced_and_deterministic() -> None:
    module = _load_script()
    samples = _samples(fake=50, real=50)
    limited = module.limit_split(samples, 20, 42)
    assert len(limited) == 20
    assert sum(1 for row in limited if row.label == 1) == 10
    assert sum(1 for row in limited if row.label == 0) == 10
    assert [row.sample_id for row in limited] == [
        row.sample_id for row in module.limit_split(samples, 20, 42)
    ]


def test_limit_split_keeps_everything_when_limit_is_large() -> None:
    module = _load_script()
    samples = _samples(fake=3, real=4)
    assert module.limit_split(samples, 0, 42) == samples
    assert module.limit_split(samples, 99, 42) == samples


def test_resolve_device_honours_explicit_requests() -> None:
    module = _load_script()
    assert module.resolve_device("cpu").type == "cpu"
    resolved = module.resolve_device("auto")
    assert resolved.type in {"cpu", "cuda"}


def test_parse_args_requires_data_dir() -> None:
    module = _load_script()
    with pytest.raises(SystemExit):
        module.parse_args([])
    args = module.parse_args(["--data-dir", "work/weibo17_processed", "--epochs", "3"])
    assert args.epochs == 3
    assert args.device == "auto"
    assert args.limit == 0