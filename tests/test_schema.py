from __future__ import annotations

import json

import pytest

from mingjian.data.dataset import JsonlNewsDataset, read_jsonl, write_jsonl
from mingjian.data.schema import NewsSample


def test_news_sample_roundtrip() -> None:
    sample = NewsSample(
        sample_id="demo-1",
        text="示例新闻文本",
        image_path="data/demo/images/demo-1.png",
        label=0,
        split="test",
        source="unit-test",
        metadata={"note": "fixture"},
    )
    restored = NewsSample.from_dict(sample.to_dict())
    assert restored == sample


def test_invalid_label_is_rejected() -> None:
    with pytest.raises(ValueError):
        NewsSample(
            sample_id="bad-1",
            text="text",
            image_path="image.png",
            label=2,
        )


def test_jsonl_roundtrip(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    samples = [
        NewsSample("a", "text a", "a.png", 0, "train"),
        NewsSample("b", "text b", "b.png", 1, "test"),
    ]
    assert write_jsonl(samples, path) == 2
    loaded = read_jsonl(path)
    assert [item.sample_id for item in loaded] == ["a", "b"]
    dataset = JsonlNewsDataset(path, split="test")
    assert len(dataset) == 1
    assert dataset[0].label == 1


def test_invalid_jsonl_row_reports_line_number(tmp_path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="broken.jsonl:1"):
        read_jsonl(path)