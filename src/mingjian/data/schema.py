"""Data schema for one multimodal news sample."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

VALID_LABELS = frozenset({0, 1})
VALID_SPLITS = frozenset({"train", "val", "test"})


@dataclass(frozen=True)
class NewsSample:
    """A single text-image news sample.

    label: 0 = fake, 1 = real.
    """

    sample_id: str
    text: str
    image_path: str
    label: int
    split: str = "train"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        if not isinstance(self.image_path, str) or not self.image_path.strip():
            raise ValueError("image_path must be a non-empty string")
        if self.label not in VALID_LABELS:
            raise ValueError("label must be 0 (fake) or 1 (real)")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dict")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "NewsSample":
        if not isinstance(raw, Mapping):
            raise TypeError("sample must be a mapping")
        metadata = raw.get("metadata", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        return cls(
            sample_id=str(raw.get("sample_id", "")),
            text=str(raw.get("text", "")),
            image_path=str(raw.get("image_path", "")),
            label=int(raw.get("label", -1)),
            split=str(raw.get("split", "train")),
            source=str(raw.get("source", "")),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "image_path": self.image_path,
            "label": self.label,
            "split": self.split,
            "source": self.source,
            "metadata": self.metadata,
        }