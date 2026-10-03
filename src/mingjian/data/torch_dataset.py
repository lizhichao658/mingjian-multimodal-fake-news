"""Torch datasets for the MingJian text-image MVP.

The dataset decodes one JPEG/PNG per sample with Pillow and applies a fixed
resize/normalize transform, so no torchvision transform API is required and the
pipeline stays reproducible across torch/torchvision versions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from mingjian.data.schema import NewsSample
from mingjian.data.tokenization import CharTokenizer

IMAGE_MEAN = 0.5
IMAGE_STD = 0.5


def load_image_tensor(
    path: str | Path,
    image_size: int,
    *,
    augment: bool = False,
) -> torch.Tensor:
    """Load one image as a normalized ``float32`` CHW tensor."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")
    try:
        with Image.open(source) as handle:
            image = handle.convert("RGB")
            if augment and torch.rand(1).item() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            resized = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError) as exc:  # pragma: no cover - corrupt input
        raise RuntimeError(f"failed to read image: {source}") from exc
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGE_MEAN) / IMAGE_STD


class Weibo17TorchDataset(Dataset):
    """Map validated JSONL samples to model-ready tensors."""

    def __init__(
        self,
        samples: Sequence[NewsSample],
        *,
        root: str | Path,
        tokenizer: CharTokenizer,
        image_size: int = 128,
        max_length: int | None = None,
        augment: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.root = Path(root)
        self.tokenizer = tokenizer
        self.image_size = int(image_size)
        self.max_length = tokenizer.max_length if max_length is None else int(max_length)
        self.augment = bool(augment)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

    def __len__(self) -> int:
        return len(self.samples)

    def image_path(self, index: int) -> Path:
        image_path = Path(self.samples[index].image_path)
        return image_path if image_path.is_absolute() else self.root / image_path

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        ids, mask = self.tokenizer.encode(sample.text, max_length=self.max_length)
        image = load_image_tensor(self.image_path(index), self.image_size, augment=self.augment)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.float32),
            "images": image,
            "labels": torch.tensor(float(sample.label), dtype=torch.float32),
            "index": index,
            "sample_id": sample.sample_id,
        }