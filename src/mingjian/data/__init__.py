"""Data contracts and dataset utilities."""

from .dataset import JsonlNewsDataset, read_jsonl, write_jsonl
from .schema import NewsSample
from .weibo17 import (
    Weibo17ArchiveError,
    assert_official_invariants,
    build_weibo17_dataset,
)

__all__ = [
    "JsonlNewsDataset",
    "NewsSample",
    "Weibo17ArchiveError",
    "assert_official_invariants",
    "build_weibo17_dataset",
    "read_jsonl",
    "write_jsonl",
]
