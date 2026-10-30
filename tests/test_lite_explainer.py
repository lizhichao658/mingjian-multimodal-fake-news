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
    assert len(result["image_evidence"]["signed_cam"]) == 4
    assert result["image_evidence"]["top_regions"]
    assert result["text_evidence"]
    assert result["evidence_card"]["state"] in {"support", "conflict", "insufficient"}
    assert result["evidence_card"]["text_units"]
    assert "cross_modal" in result["evidence_card"]
    assert result["risk"]["level"] in {"high", "medium", "low", "uncertain"}
    assert result["decision"]["binary_label"] == result["decision"]["label"]
    assert result["decision"]["binary_label_name_zh"] == result["decision"]["label_name_zh"]
    assert set(result["decision"]["label_semantics"]) == {"label", "display_label"}
    expected_display = (
        "uncertain" if result["risk"]["is_uncertain"] else result["decision"]["label"]
    )
    assert result["decision"]["display_label"] == expected_display
    assert result["decision"]["display_label_name_zh"] == (
        "待复核" if expected_display == "uncertain" else result["decision"]["label_name_zh"]
    )
    assert json.loads(json.dumps(result, ensure_ascii=False))["sample_id"] == "case-1"


def test_lite_risk_uses_pending_review_band_at_decision_boundary() -> None:
    model = LiteFusionClassifier(_config()).eval()
    explainer = LiteExplainer(model, _tokenizer(), device="cpu", image_size=16)

    for probability in (0.40, 0.50, 0.60):
        risk = explainer._risk(probability)
        assert risk["level"] == "uncertain"
        assert risk["is_uncertain"] is True

    assert explainer._risk(0.39)["level"] == "low"
    assert explainer._risk(0.61)["level"] == "medium"
    assert explainer._risk(0.74)["level"] == "medium"
    assert explainer._risk(0.75)["level"] == "high"


@pytest.mark.parametrize(
    "probability,threshold,dtype,published_probability,expected_risk,expected_binary",
    [
        (0.399999, 0.5, torch.float32, 0.399999, "low", "real"),
        (0.3999999, 0.4, torch.float32, 0.4, "uncertain", "real"),
        (0.39999999, 0.4, torch.float64, 0.4, "uncertain", "real"),
        (0.4, 0.5, torch.float32, 0.4, "uncertain", "real"),
        (0.557922, 0.5, torch.float32, 0.557922, "uncertain", "fake"),
        (0.6, 0.5, torch.float32, 0.6, "uncertain", "fake"),
        (0.600001, 0.5, torch.float32, 0.600001, "medium", "fake"),
        (0.74999997, 0.5, torch.float32, 0.75, "high", "fake"),
        (0.75, 0.5, torch.float32, 0.75, "high", "fake"),
    ],
)
def test_explain_keeps_published_score_risk_and_verdict_consistent(
    probability: float,
    threshold: float,
    dtype: torch.dtype,
    published_probability: float,
    expected_risk: str,
    expected_binary: str,
) -> None:
    model = LiteFusionClassifier(_config()).to(dtype=dtype).eval()
    # A constant final logit exercises the complete tensor -> explanation -> JSON path.
    # Float32 sigmoid(0.6's logit) exceeds 0.6 before serialization; float64 also
    # covers the exact 0.39999999 case without losing its below-threshold raw label.
    with torch.no_grad():
        model.classifier[-1].weight.zero_()
        model.classifier[-1].bias.copy_(torch.logit(torch.tensor([probability], dtype=dtype)))
    explainer = LiteExplainer(
        model, _tokenizer(), device="cpu", image_size=16, threshold=threshold
    )
    result = explainer.explain("abcd", image=torch.zeros(3, 16, 16, dtype=dtype))
    serialized = json.loads(json.dumps(result, ensure_ascii=False))
    decision = serialized["decision"]

    assert decision["probability_fake"] == published_probability
    assert serialized["risk"]["level"] == expected_risk
    assert serialized["risk"]["is_uncertain"] is (expected_risk == "uncertain")
    assert decision["label"] == decision["binary_label"] == expected_binary
    expected_display = "uncertain" if expected_risk == "uncertain" else expected_binary
    assert decision["display_label"] == expected_display
    assert decision["display_label_name_zh"] == {
        "uncertain": "待复核",
        "fake": "疑似虚假",
        "real": "倾向真实",
    }[expected_display]
    assert decision["binary_label_name_zh"] == decision["label_name_zh"]


def test_lite_explainer_supports_missing_image_degradation() -> None:
    model = LiteFusionClassifier(_config()).eval()
    explainer = LiteExplainer(model, _tokenizer(), device="cpu", image_size=16)
    result = explainer.explain("abc", image=None)

    assert result["modalities"]["has_image"] is False
    assert result["modalities"]["image_contribution"] == 0.0
    assert result["evidence_card"]["state"] == "insufficient"
    assert result["evidence_card"]["image_regions"] == []
    assert any("缺图降级" in item for item in result["limitations"])


def test_missing_image_has_zero_contribution_when_text_feature_energy_is_zero() -> None:
    model = LiteFusionClassifier(_config()).eval()
    # Simulate a degenerate checkpoint through the full explanation/JSON path,
    # including nonzero image projection bias before the missing-image mask.
    with torch.no_grad():
        model.text_projection.weight.zero_()
        model.text_projection.bias.zero_()
        model.image_projection.bias.fill_(1.0)
    explainer = LiteExplainer(model, _tokenizer(), device="cpu", image_size=16)
    result = json.loads(json.dumps(explainer.explain("abc", image=None)))

    assert result["modalities"]["has_image"] is False
    assert result["modalities"]["image_contribution"] == 0.0
    assert result["modalities"]["text_contribution"] == 1.0
    assert result["evidence_card"]["state"] == "insufficient"
    assert result["evidence_card"]["cross_modal"]["reason_code"] == "missing_image"
    assert result["evidence_card"]["image_regions"] == []


def test_lite_explainer_validates_inputs() -> None:
    model = LiteFusionClassifier(_config()).eval()
    with pytest.raises(ValueError, match="threshold"):
        LiteExplainer(model, _tokenizer(), threshold=1.0)
    with pytest.raises(ValueError, match="min_strength"):
        LiteExplainer(model, _tokenizer(), min_strength=0.5)
    with pytest.raises(ValueError, match="shape"):
        LiteExplainer(model, _tokenizer(), image_size=16).explain(
            "abc", image=torch.randn(4, 16, 16)
        )


def _feature_with_grad(values: list[float], grads: list[float]) -> torch.Tensor:
    feature = torch.tensor(values, dtype=torch.float32, requires_grad=True)
    (feature * torch.tensor(grads, dtype=torch.float32)).sum().backward()
    return feature


def test_cross_modal_evidence_three_states_are_reachable() -> None:
    text = _feature_with_grad([1.0, 0.0], [1.0, 0.0])
    image = _feature_with_grad([1.0, 0.0], [1.0, 0.0])
    support = LiteExplainer._cross_modal_evidence(text, image, has_image=True)
    assert support["state"] == "support"
    assert support["state_zh"] == "证据同向"
    assert support["reason_code"] == "same_direction"
    assert support["text_direction"] == "fake"
    assert support["image_direction"] == "fake"
    assert support["text_strength"] == pytest.approx(0.5)
    assert support["text_coherence"] == pytest.approx(1.0)
    assert support["image_coherence"] == pytest.approx(1.0)

    text = _feature_with_grad([1.0, 0.0], [1.0, 0.0])
    image = _feature_with_grad([-1.0, 0.0], [1.0, 0.0])
    conflict = LiteExplainer._cross_modal_evidence(text, image, has_image=True)
    assert conflict["state"] == "conflict"
    assert conflict["reason_code"] == "opposite_direction"
    assert conflict["text_direction"] == "fake"
    assert conflict["image_direction"] == "real"

    text = _feature_with_grad([1.0, 0.0], [1.0, 0.0])
    image = _feature_with_grad([0.05, 0.0], [1.0, 0.0])
    weak_image = LiteExplainer._cross_modal_evidence(text, image, has_image=True)
    assert weak_image["state"] == "insufficient"
    assert weak_image["reason_code"] == "weak_modality"
    assert weak_image["image_strength"] < weak_image["min_strength"]

    text = _feature_with_grad([0.0, 0.0], [1.0, 1.0])
    image = _feature_with_grad([1.0, 0.0], [1.0, 0.0])
    neutral_text = LiteExplainer._cross_modal_evidence(text, image, has_image=True)
    assert neutral_text["state"] == "insufficient"
    assert neutral_text["reason_code"] == "neutral_direction"
    assert neutral_text["text_direction"] == "neutral"

    text = torch.tensor([1.0, 0.9], requires_grad=True)
    (text * torch.tensor([1.0, -0.99])).sum().backward()
    image = _feature_with_grad([0.5, 0.0], [1.0, 0.0])
    unstable = LiteExplainer._cross_modal_evidence(text, image, has_image=True)
    assert unstable["state"] == "insufficient"
    assert unstable["reason_code"] == "unstable_direction"
    assert unstable["text_coherence"] < unstable["min_coherence"]


def test_cross_modal_evidence_missing_image_is_insufficient() -> None:
    text = _feature_with_grad([1.0, 0.0], [1.0, 0.0])
    image = torch.zeros(2)
    result = LiteExplainer._cross_modal_evidence(text, image, has_image=False)
    assert result["state"] == "insufficient"
    assert result["reason_code"] == "missing_image"
    assert result["image_direction"] == "neutral"
    assert result["image_strength"] == 0.0
