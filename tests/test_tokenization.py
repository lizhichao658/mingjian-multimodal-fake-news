from __future__ import annotations

import json

import pytest

from mingjian.data.tokenization import (
    PAD_TOKEN,
    UNK_TOKEN,
    CharTokenizer,
    build_char_tokenizer,
    normalize_text,
)


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("  hello\n\nworld\t!  ") == "hello world !"
    with pytest.raises(TypeError):
        normalize_text(None)  # type: ignore[arg-type]


def test_vocab_reserves_pad_and_unk_slots() -> None:
    tokenizer = build_char_tokenizer(["啊啊", "啊"], min_freq=1, max_vocab=10, max_length=8)
    assert tokenizer.vocab[0] == PAD_TOKEN
    assert tokenizer.vocab[1] == UNK_TOKEN
    assert tokenizer.pad_id == 0
    assert tokenizer.unk_id == 1
    ids, mask = tokenizer.encode("啊？")
    assert ids[0] == tokenizer.token_to_id("啊")
    assert ids[1] == tokenizer.unk_id
    assert mask == [1, 1, 0, 0, 0, 0, 0, 0]


def test_encode_pads_and_truncates_to_limit() -> None:
    tokenizer = CharTokenizer(vocab=(PAD_TOKEN, UNK_TOKEN, "a"), max_length=4)
    ids, mask = tokenizer.encode("aa", max_length=4)
    assert ids == [2, 2, 0, 0]
    assert mask == [1, 1, 0, 0]
    ids, mask = tokenizer.encode("aaaa", max_length=3)
    assert ids == [2, 2, 2]
    assert mask == [1, 1, 1]


def test_builder_orders_by_frequency_then_codepoint_and_filters_rare() -> None:
    tokenizer = build_char_tokenizer(["aab", "bc"], min_freq=2, max_vocab=10)
    assert tokenizer.vocab == (PAD_TOKEN, UNK_TOKEN, "a", "b")
    assert "c" not in tokenizer.vocab

    rare = build_char_tokenizer(["xyz"], min_freq=2, max_vocab=10)
    assert rare.vocab == (PAD_TOKEN, UNK_TOKEN)


def test_builder_validates_arguments() -> None:
    with pytest.raises(ValueError, match="max_vocab"):
        build_char_tokenizer(["a"], max_vocab=1)
    with pytest.raises(ValueError, match="min_freq"):
        build_char_tokenizer(["a"], min_freq=0)
    with pytest.raises(ValueError, match="max_length"):
        CharTokenizer(vocab=(PAD_TOKEN, UNK_TOKEN), max_length=0)
    with pytest.raises(ValueError, match="duplicate"):
        CharTokenizer(vocab=(PAD_TOKEN, UNK_TOKEN, "a", "a"))


def test_tokenizer_round_trip(tmp_path) -> None:
    tokenizer = build_char_tokenizer(["你好世界"], min_freq=1, max_length=8)
    path = tokenizer.save(tmp_path / "tokenizer.json")
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["kind"] == "char"
    assert payload["version"] == 1

    restored = CharTokenizer.load(path)
    assert restored == tokenizer
    assert restored.encode("你好世界") == tokenizer.encode("你好世界")


def test_tokenizer_load_rejects_other_formats(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "bpe", "version": 1, "vocab": ["<pad>", "<unk>"]}), "utf-8")
    with pytest.raises(ValueError, match="kind"):
        CharTokenizer.load(path)