from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig


def _config(**overrides) -> LiteModelConfig:
    values = {
        "vocab_size": 50,
        "text_embed_dim": 16,
        "text_num_filters": 8,
        "text_kernel_sizes": (2, 3),
        "image_widths": (8, 16),
        "image_feature_dim": 16,
        "hidden_dim": 12,
        "dropout": 0.0,
        "padding_idx": 0,
    }
    values.update(overrides)
    return LiteModelConfig(**values)


def _batch(batch_size: int = 3, seq_len: int = 9, image_size: int = 16):
    input_ids = torch.randint(1, 50, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    images = torch.randn(batch_size, 3, image_size, image_size)
    return input_ids, attention_mask, images


def test_config_validates_and_round_trips() -> None:
    config = _config()
    restored = LiteModelConfig.from_dict(config.to_dict())
    assert restored == config
    assert config.text_feature_dim == 16

    with pytest.raises(ValueError, match="vocab_size"):
        _config(vocab_size=0)
    with pytest.raises(ValueError, match="text_kernel_sizes"):
        _config(text_kernel_sizes=())
    with pytest.raises(ValueError, match="image_feature_dim"):
        _config(image_feature_dim=32)
    with pytest.raises(ValueError, match="dropout"):
        _config(dropout=1.0)
    with pytest.raises(ValueError, match="padding_idx"):
        _config(padding_idx=99)


def test_forward_shapes_and_probability_range() -> None:
    model = LiteFusionClassifier(_config()).eval()
    input_ids, attention_mask, images = _batch()
    output = model(input_ids=input_ids, images=images, attention_mask=attention_mask)
    assert output.logits.shape == (3,)
    assert output.probability.shape == (3,)
    assert torch.all(output.probability >= 0) and torch.all(output.probability <= 1)
    assert output.gate is None


def test_auxiliary_tensors_support_explainability() -> None:
    model = LiteFusionClassifier(_config()).eval()
    input_ids, attention_mask, images = _batch()
    output = model(
        input_ids=input_ids, images=images, attention_mask=attention_mask, return_aux=True
    )
    assert output.gate is not None and output.gate.shape == (3, 12)
    assert output.text_features is not None and output.text_features.shape == (3, 12)
    assert output.image_features is not None and output.image_features.shape == (3, 12)
    assert output.text_token_features is not None
    assert output.text_token_features.shape == (3, 9, 16)
    assert output.image_feature_maps is not None
    assert output.image_feature_maps.shape == (3, 16, 4, 4)


def test_masked_positions_cannot_change_logits() -> None:
    model = LiteFusionClassifier(_config()).eval()
    input_ids, attention_mask, images = _batch(batch_size=2, seq_len=8)
    attention_mask[:, 5:] = 0.0
    noisy = input_ids.clone()
    noisy[:, 5:] = torch.randint(1, 50, (2, 3))

    clean = model(input_ids=input_ids, images=images, attention_mask=attention_mask).logits
    dirty = model(input_ids=noisy, images=images, attention_mask=attention_mask).logits
    assert torch.allclose(clean, dirty, atol=1e-5)


def test_fully_padded_row_stays_finite() -> None:
    model = LiteFusionClassifier(_config()).eval()
    input_ids, attention_mask, images = _batch(batch_size=2)
    attention_mask[1] = 0.0
    output = model(input_ids=input_ids, images=images, attention_mask=attention_mask)
    assert torch.isfinite(output.logits).all()


def test_short_sequence_is_rejected() -> None:
    model = LiteFusionClassifier(_config(text_kernel_sizes=(2, 5))).eval()
    with pytest.raises(ValueError, match="shorter than the largest kernel"):
        model(input_ids=torch.ones(1, 3, dtype=torch.long), images=torch.randn(1, 3, 16, 16))


def test_missing_image_flag_keeps_output_finite() -> None:
    model = LiteFusionClassifier(_config()).eval()
    input_ids, attention_mask, images = _batch(batch_size=2)
    output = model(
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        has_image=torch.zeros(2),
    )
    assert torch.isfinite(output.logits).all()
    assert output.probability.shape == (2,)

    with pytest.raises(ValueError, match="one flag per sample"):
        model(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            has_image=torch.zeros(3),
        )


def test_backward_reaches_both_encoders() -> None:
    model = LiteFusionClassifier(_config())
    input_ids, attention_mask, images = _batch()
    output = model(input_ids=input_ids, images=images, attention_mask=attention_mask)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.logits, torch.tensor([0.0, 1.0, 1.0])
    )
    loss.backward()
    assert model.text_encoder.embedding.weight.grad is not None
    text_conv_weight = model.text_encoder.convolutions[0].weight
    assert text_conv_weight.grad is not None and text_conv_weight.grad.abs().sum() > 0
    image_conv_weight = model.image_encoder.stages[0].weight
    assert image_conv_weight.grad is not None and image_conv_weight.grad.abs().sum() > 0


def test_default_configuration_stays_lite() -> None:
    model = LiteFusionClassifier(LiteModelConfig(vocab_size=6000))
    total = sum(parameter.numel() for parameter in model.parameters())
    assert total < 5_000_000