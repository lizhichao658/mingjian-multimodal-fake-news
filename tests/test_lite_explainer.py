from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from mingjian.data.tokenization import CharTokenizer
from mingjian.explain import LiteExplainer
from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig


def _config() -> LiteModelConfig:
    return LiteModelConfig(
        vocab_size=12,
        text_embed_dim=8,
        text_num_filters=4,
        text_kernel_sizes=(2, 3),
        image_widths=(8, 16),
        image_feature_dim=16,
        hidden_dim=12,
        dropout=0.0,
    )


def _tokenizer() -> CharTokenizer:
    return CharTokenizer(("<pad>", "<unk>", "a", "b", "c", "d"), max_length=8)


def test_lite_explainer_returns_json_ready_text_and_image_evidence() -> None:
    model = LiteFusionClassifier(_config()).eval()
    explainer = LiteExplainer(model, _tokenizer(), device="cpu", image_size=16, threshold=0.5)
    image = torch.randn(3, 16, 16)
    result = explainer.explain("abcd", image=image, sample_id="case-1")

    assert result["sample_id"] == "case-1"
    assert 0.0 <= result["decision"]["probability_fake"] <= 1.0
    assert result["modalities"]["has_image"] is True
    assert result["image_evidence"]["grid_shape"] == [4, 4]
    assert len(result["image_evidence"]["cam"]) == 4
    assert result["text_evidence"]
    assert result["risk"]["level"] in {"high", "medium", "low", "uncertain"}
    assert json.loads(json.dumps(result, ensure_ascii=False))["sample_id"] == "case-1"


def test_lite_explainer_supports_missing_image_degradation() -> None:
    model = LiteFusionClassifier(_config()).eval()
    explainer = LiteExplainer(model, _tokenizer(), device="cpu", image_size=16)
    result = explainer.explain("abc", image=None)

    assert result["modalities"]["has_image"] is False
    assert result["modalities"]["image_contribution"] == 0.0
    assert any("缺图降级" in item for item in result["limitations"])


def test_lite_explainer_validates_inputs() -> None:
    model = LiteFusionClassifier(_config()).eval()
    with pytest.raises(ValueError, match="threshold"):
        LiteExplainer(model, _tokenizer(), threshold=1.0)
    with pytest.raises(ValueError, match="shape"):
        LiteExplainer(model, _tokenizer(), image_size=16).explain(
            "abc", image=torch.randn(4, 16, 16)
        )


