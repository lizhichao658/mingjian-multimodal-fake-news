from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mingjian.models.baseline import FeatureFusionBaseline, TinyTextImageBaseline


def test_feature_fusion_baseline_shapes() -> None:
    model = FeatureFusionBaseline(text_dim=16, image_dim=8, hidden_dim=12)
    output = model(torch.randn(4, 16), torch.randn(4, 8))
    assert output.logits.shape == (4,)
    assert output.probability.shape == (4,)


def test_tiny_baseline_backward() -> None:
    model = TinyTextImageBaseline(vocab_size=64, text_embed_dim=16, hidden_dim=8)
    input_ids = torch.randint(1, 64, (2, 7))
    images = torch.randn(2, 3, 16, 16)
    labels = torch.tensor([0.0, 1.0])
    output = model(input_ids=input_ids, images=images)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(output.logits, labels)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())