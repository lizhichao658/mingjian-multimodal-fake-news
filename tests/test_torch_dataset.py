from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mingjian.data.schema import NewsSample
from mingjian.data.tokenization import PAD_TOKEN, UNK_TOKEN, CharTokenizer
from mingjian.data.torch_dataset import Weibo17TorchDataset, load_image_tensor


def _tokenizer() -> CharTokenizer:
    return CharTokenizer(vocab=(PAD_TOKEN, UNK_TOKEN, "a", "b"), max_length=4)


def _write_image(path, color=(255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=color).save(path)


def test_load_image_tensor_shape_and_range(tmp_path) -> None:
    path = tmp_path / "a.png"
    _write_image(path, (0, 255, 0))
    tensor = load_image_tensor(path, 16)
    assert tensor.shape == (3, 16, 16)
    assert tensor.dtype == torch.float32
    # A saturated green channel maps to (1.0 - 0.5) / 0.5 = 1.0 after normalization.
    green = tensor[1].mean().item()
    assert abs(green - 1.0) < 1e-5
    assert abs(tensor[0].mean().item() + 1.0) < 1e-5


def test_load_image_tensor_reports_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image_tensor(tmp_path / "missing.png", 8)
    with pytest.raises(ValueError):
        load_image_tensor(tmp_path / "missing.png", 0)


def test_dataset_item_contract(tmp_path) -> None:
    _write_image(tmp_path / "images" / "a.png")
    samples = [NewsSample("s1", "ab", "images/a.png", 1, "train")]
    dataset = Weibo17TorchDataset(samples, root=tmp_path, tokenizer=_tokenizer(), image_size=12)

    item = dataset[0]
    assert len(dataset) == 1
    assert item["input_ids"].tolist() == [2, 3, 0, 0]
    assert item["attention_mask"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert item["images"].shape == (3, 12, 12)
    assert item["labels"].item() == 1.0
    assert item["sample_id"] == "s1"
    assert item["index"] == 0
    assert dataset.image_path(0) == tmp_path / "images" / "a.png"


def test_dataset_normalizes_whitespace_and_truncates(tmp_path) -> None:
    _write_image(tmp_path / "a.png")
    samples = [NewsSample("s1", "a  \n b", "a.png", 0)]
    dataset = Weibo17TorchDataset(samples, root=tmp_path, tokenizer=_tokenizer(), image_size=8)
    item = dataset[0]
    assert item["input_ids"].tolist() == [2, 1, 3, 0]
    assert item["attention_mask"].tolist() == [1.0, 1.0, 1.0, 0.0]

    long_samples = [NewsSample("s2", "ababab", "a.png", 0)]
    long_dataset = Weibo17TorchDataset(
        long_samples, root=tmp_path, tokenizer=_tokenizer(), image_size=8
    )
    assert long_dataset[0]["input_ids"].tolist() == [2, 3, 2, 3]


def test_dataset_raises_for_missing_image(tmp_path) -> None:
    samples = [NewsSample("s1", "ab", "nope.png", 0)]
    dataset = Weibo17TorchDataset(samples, root=tmp_path, tokenizer=_tokenizer(), image_size=8)
    with pytest.raises(FileNotFoundError, match="nope.png"):
        _ = dataset[0]


def test_augment_keeps_shape(tmp_path) -> None:
    _write_image(tmp_path / "a.png")
    samples = [NewsSample("s1", "ab", "a.png", 0)]
    dataset = Weibo17TorchDataset(
        samples, root=tmp_path, tokenizer=_tokenizer(), image_size=8, augment=True
    )
    torch.manual_seed(0)
    item = dataset[0]
    assert tuple(item["images"].shape) == (3, 8, 8)
    assert np.isfinite(item["images"].numpy()).all()