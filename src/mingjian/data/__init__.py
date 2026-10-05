"""Data contracts, adapters and dataset utilities."""

from .dataset import JsonlNewsDataset, read_jsonl, write_jsonl
from .schema import NewsSample
from .tokenization import CharTokenizer, build_char_tokenizer, normalize_text
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
    "assert_official_invariants",
    "build_char_tokenizer",
    "build_weibo17_dataset",
    "normalize_text",
    "read_jsonl",
    "write_jsonl",
]

# Keep torch optional so the lightweight reproduction path can run without it.
try:
    from .torch_dataset import Weibo17TorchDataset, load_image_tensor
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__ += ["Weibo17TorchDataset", "load_image_tensor"]
