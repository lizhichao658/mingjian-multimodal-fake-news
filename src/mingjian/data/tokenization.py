"""Character tokenizer for the MingJian text-image MVP.

Clean-room implementation written for this project. Chinese social-media posts
are short, so a character vocabulary keeps the first end-to-end baseline
reproducible without downloading pretrained weights or external tokenizers.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD_TOKEN, UNK_TOKEN)
TOKENIZER_KIND = "char"
TOKENIZER_VERSION = 1


def normalize_text(text: str) -> str:
    """Collapse whitespace runs so equivalent posts share one encoding."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.split())


@dataclass(frozen=True)
class CharTokenizer:
    """Deterministic character tokenizer with fixed pad/unk slots.

    Index 0 is padding, index 1 is the unknown character, and every other
    index maps to one character observed often enough in the training split.
    """

    vocab: tuple[str, ...]
    max_length: int = 196

    def __post_init__(self) -> None:
        if not isinstance(self.vocab, tuple):
            raise TypeError("vocab must be a tuple of characters")
        if len(self.vocab) < len(SPECIAL_TOKENS) or self.vocab[:2] != SPECIAL_TOKENS:
            raise ValueError("vocab must start with <pad> and <unk>")
        if len(set(self.vocab)) != len(self.vocab):
            raise ValueError("vocab must not contain duplicate tokens")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        object.__setattr__(self, "_token_to_id", {token: i for i, token in enumerate(self.vocab)})

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, self.unk_id)

    def encode(self, text: str, max_length: int | None = None) -> tuple[list[int], list[int]]:
        """Return (input_ids, attention_mask) padded/truncated to ``max_length``."""

        limit = self.max_length if max_length is None else max_length
        if limit <= 0:
            raise ValueError("max_length must be positive")
        token_to_id = self._token_to_id
        ids = [token_to_id.get(char, self.unk_id) for char in normalize_text(text)][:limit]
        mask = [1] * len(ids)
        padding = limit - len(ids)
        if padding > 0:
            ids = ids + [self.pad_id] * padding
            mask = mask + [0] * padding
        return ids, mask

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": TOKENIZER_KIND,
            "version": TOKENIZER_VERSION,
            "max_length": self.max_length,
            "vocab": list(self.vocab),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        target.write_text(payload + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("tokenizer file must contain a JSON object")
        if raw.get("kind") != TOKENIZER_KIND:
            raise ValueError(f"unsupported tokenizer kind: {raw.get('kind')!r}")
        if raw.get("version") != TOKENIZER_VERSION:
            raise ValueError(f"unsupported tokenizer version: {raw.get('version')!r}")
        vocab = raw.get("vocab")
        if not isinstance(vocab, list) or not all(isinstance(item, str) for item in vocab):
            raise ValueError("tokenizer vocab must be a list of strings")
        return cls(vocab=tuple(vocab), max_length=int(raw.get("max_length", 196)))


def build_char_tokenizer(
    texts: Iterable[str],
    *,
    max_length: int = 196,
    max_vocab: int = 6000,
    min_freq: int = 2,
) -> CharTokenizer:
    """Build a character vocabulary from training text only."""

    if max_vocab < len(SPECIAL_TOKENS):
        raise ValueError("max_vocab must leave room for <pad> and <unk>")
    if min_freq < 1:
        raise ValueError("min_freq must be at least 1")

    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(normalize_text(text))

    candidates = [(char, count) for char, count in counter.items() if count >= min_freq]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    keep = max_vocab - len(SPECIAL_TOKENS)
    chars = [char for char, _ in candidates[:keep]]
    return CharTokenizer(vocab=SPECIAL_TOKENS + tuple(chars), max_length=max_length)