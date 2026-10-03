"""Data contracts, adapters and dataset utilities."""

from .dataset import JsonlNewsDataset, read_jsonl, write_jsonl
from .schema import NewsSample
from .tokenization import CharTokenizer, build_char_tokenizer, normalize_text
from .torch_dataset import Weibo17TorchDataset, load_image_tensor
from .weibo17 import (
    Weibo17ArchiveError,
    assert_official_invariants,
    build_weibo17_dataset,
)

__all__ = [
    "CharTokenizer",
    "JsonlNewsDataset",
    "NewsSample",
    "Weibo17ArchiveError",
    "Weibo17TorchDataset",
    "assert_official_invariants",
    "build_char_tokenizer",
    "build_weibo17_dataset",
    "load_image_tensor",
    "normalize_text",
    "read_jsonl",
    "write_jsonl",
]