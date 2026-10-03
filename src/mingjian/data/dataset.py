"""JSONL dataset utilities.

The first version keeps file I/O and filtering dependency-free. Model-specific
tokenization and image transforms are added by the training pipeline later.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from .schema import NewsSample


def read_jsonl(path: str | Path) -> list[NewsSample]:
    """Read a JSONL file and validate every row."""

    source = Path(path)
    samples: list[NewsSample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                samples.append(NewsSample.from_dict(raw))
            except Exception as exc:  # pragma: no cover - message is the contract
                raise ValueError(f"{source}:{line_number}: invalid sample: {exc}") from exc
    return samples


def write_jsonl(samples: Iterable[NewsSample], path: str | Path) -> int:
    """Write samples to JSONL and return the number of rows written."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def ensure_unique_ids(samples: Iterable[NewsSample]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            duplicates.add(sample.sample_id)
        seen.add(sample.sample_id)
    if duplicates:
        raise ValueError(f"duplicate sample ids: {sorted(duplicates)}")


class JsonlNewsDataset:
    """A lightweight sequence over validated JSONL samples."""

    def __init__(self, path: str | Path, split: str | None = None) -> None:
        self.path = Path(path)
        samples = read_jsonl(self.path)
        ensure_unique_ids(samples)
        if split is not None:
            samples = [sample for sample in samples if sample.split == split]
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> NewsSample:
        return self.samples[index]

    def __iter__(self) -> Iterator[NewsSample]:
        return iter(self.samples)