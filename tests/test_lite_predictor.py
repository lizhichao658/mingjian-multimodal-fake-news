from __future__ import annotations

import json

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from mingjian.data.tokenization import CharTokenizer
from mingjian.inference import LitePredictor
from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig


def _write_checkpoint(tmp_path):
    config = LiteModelConfig(
        vocab_size=12,
        text_embed_dim=8,
        text_num_filters=4,
        text_kernel_sizes=(2, 3),
        image_widths=(8, 16),
        image_feature_dim=16,
        hidden_dim=12,
        dropout=0.0,
    )
    model = LiteFusionClassifier(config)
    tokenizer = CharTokenizer(("<pad>", "<unk>", "a", "b", "c", "d"), max_length=8)
    tokenizer.save(tmp_path / "tokenizer.json")
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "threshold": 0.4,
            "tokenizer_file": "tokenizer.json",
            "git_commit": "test",
        },
        checkpoint,
    )
    return checkpoint


def test_lite_predictor_loads_checkpoint_and_explains_missing_image(tmp_path) -> None:
    predictor = LitePredictor.from_checkpoint(
        _write_checkpoint(tmp_path),
        device="cpu",
        image_size=16,
    )
    result = predictor.analyze("abc", sample_id="missing-image")

    assert result["model"]["image_provided"] is False
    assert result["model"]["checkpoint_threshold"] == 0.4
    assert result["model"]["decision_threshold"] == 0.5
    assert result["decision"]["threshold"] == 0.5
    assert result["model"]["metadata"]["git_commit"] == "test"
    assert json.dumps(result, ensure_ascii=False)

    override = predictor.analyze("abc", decision_threshold=0.4)
    assert override["model"]["checkpoint_threshold"] == 0.4
    assert override["model"]["decision_threshold"] == 0.4
    assert override["decision"]["threshold"] == 0.4


def test_lite_predictor_loads_local_image(tmp_path) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 24), color=(64, 128, 192)).save(image_path)

    predictor = LitePredictor.from_checkpoint(checkpoint, device="cpu", image_size=16)
    result = predictor.analyze("abcd", image_path=image_path)

    assert result["model"]["image_provided"] is True
    assert result["modalities"]["has_image"] is True
    assert result["image_evidence"]["grid_shape"] == [4, 4]

