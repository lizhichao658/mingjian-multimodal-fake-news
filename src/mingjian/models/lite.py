"""Self-contained lite fusion model for the MingJian MVP.

Design goals, in order:

1. Run end-to-end on one consumer GPU (8 GB) without downloading weights.
2. Expose enough structure for the later explainability work:
   ``TextCnnEncoder`` keeps token-level features for gradient saliency, and
   ``ImageCnnEncoder`` keeps the last convolutional feature map for Grad-CAM.
3. Stay small enough that a full Weibo17 run is a few minutes, so the
   competition reproduction path is realistic.

The architecture is a clean-room baseline: a multi-kernel character TextCNN, a
small strided CNN image encoder, and a gated fusion head that produces
P(fake). No code is taken from EANN, MIAN, IMOL or any other public project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LiteModelConfig:
    """Hyper-parameters of the lite fusion model."""

    vocab_size: int
    text_embed_dim: int = 128
    text_num_filters: int = 128
    text_kernel_sizes: tuple[int, ...] = (2, 3, 4, 5)
    image_feature_dim: int = 256
    image_widths: tuple[int, ...] = (32, 64, 128, 256)
    hidden_dim: int = 256
    dropout: float = 0.3
    padding_idx: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        for name in ("text_embed_dim", "text_num_filters", "image_feature_dim", "hidden_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.text_kernel_sizes:
            raise ValueError("text_kernel_sizes must not be empty")
        if any(size <= 0 for size in self.text_kernel_sizes):
            raise ValueError("text_kernel_sizes must be positive")
        if not self.image_widths:
            raise ValueError("image_widths must not be empty")
        if any(width <= 0 for width in self.image_widths):
            raise ValueError("image_widths must be positive")
        if self.image_widths[-1] != self.image_feature_dim:
            raise ValueError("image_feature_dim must match the last entry of image_widths")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 <= self.padding_idx < self.vocab_size:
            raise ValueError("padding_idx must index the embedding table")

    @property
    def text_feature_dim(self) -> int:
        return self.text_num_filters * len(self.text_kernel_sizes)

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["text_kernel_sizes"] = list(self.text_kernel_sizes)
        payload["image_widths"] = list(self.image_widths)
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LiteModelConfig:
        values = dict(raw)
        values["text_kernel_sizes"] = tuple(values.get("text_kernel_sizes", (2, 3, 4, 5)))
        values["image_widths"] = tuple(values.get("image_widths", (32, 64, 128, 256)))
        return cls(**values)


@dataclass(frozen=True)
class LiteOutput:
    """Model output plus optional interpretability tensors."""

    logits: torch.Tensor
    probability: torch.Tensor
    gate: torch.Tensor | None = None
    text_features: torch.Tensor | None = None
    image_features: torch.Tensor | None = None
    text_token_features: torch.Tensor | None = None
    image_feature_maps: torch.Tensor | None = None


class TextCnnEncoder(nn.Module):
    """Multi-kernel character CNN with padding-aware max pooling."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 128,
        kernel_sizes: tuple[int, ...] = (2, 3, 4, 5),
        dropout: float = 0.1,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embed_dim, num_filters, kernel_size=size) for size in self.kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(self.kernel_sizes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        max_kernel = max(self.kernel_sizes)
        if input_ids.shape[1] < max_kernel:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} is shorter than the largest kernel {max_kernel}"
            )
        if attention_mask is None:
            attention_mask = (input_ids != self.padding_idx).to(dtype=torch.float32)
        else:
            attention_mask = attention_mask.to(dtype=torch.float32)
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")

        embeddings = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = embeddings.transpose(1, 2)
        lengths = attention_mask.sum(dim=-1)
        positions = torch.arange(hidden.shape[-1], device=hidden.device).unsqueeze(0)

        pooled: list[torch.Tensor] = []
        for kernel_size, convolution in zip(self.kernel_sizes, self.convolutions, strict=True):
            activations = torch.relu(convolution(hidden))
            window_valid = positions[:, : activations.shape[-1]] + kernel_size <= lengths.unsqueeze(1)
            activations = activations.masked_fill(~window_valid.unsqueeze(1), float("-inf"))
            # All conv outputs are ReLU'd, so leftover -inf (fully padded rows) clamps to 0.
            pooled.append(activations.max(dim=-1).values.clamp(min=0.0))

        return {
            "pooled_features": self.dropout(torch.cat(pooled, dim=-1)),
            "token_features": embeddings,
        }


class ImageCnnEncoder(nn.Module):
    """Small strided CNN that keeps its last feature map for Grad-CAM."""

    def __init__(
        self,
        in_channels: int = 3,
        widths: tuple[int, ...] = (32, 64, 128, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        stages: list[nn.Module] = []
        current = in_channels
        for width in widths:
            stages.extend(
                [
                    nn.Conv2d(current, width, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                ]
            )
            current = width
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = current

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        feature_maps = self.stages(images)
        pooled = self.pool(feature_maps).flatten(1)
        return {"pooled_features": self.dropout(pooled), "feature_maps": feature_maps}


class LiteFusionClassifier(nn.Module):
    """Character TextCNN + image CNN + gate fusion head for P(fake)."""

    def __init__(self, config: LiteModelConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = TextCnnEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.text_embed_dim,
            num_filters=config.text_num_filters,
            kernel_sizes=config.text_kernel_sizes,
            dropout=config.dropout,
            padding_idx=config.padding_idx,
        )
        self.image_encoder = ImageCnnEncoder(
            widths=config.image_widths,
            dropout=config.dropout,
        )
        self.text_projection = nn.Linear(self.text_encoder.output_dim, config.hidden_dim)
        self.image_projection = nn.Linear(self.image_encoder.output_dim, config.hidden_dim)
        self.gate_layer = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.fusion_norm = nn.LayerNorm(config.hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        has_image: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> LiteOutput:
        if images is None:
            raise ValueError("images are required by the first-version MVP model")

        text_out = self.text_encoder(input_ids, attention_mask=attention_mask)
        image_out = self.image_encoder(images)

        text_features = self.text_projection(text_out["pooled_features"])
        image_features = self.image_projection(image_out["pooled_features"])
        if has_image is not None:
            availability = has_image.to(dtype=image_features.dtype).view(-1, 1)
            if availability.shape[0] != image_features.shape[0]:
                raise ValueError("has_image must have one flag per sample")
            image_features = image_features * availability

        gate = torch.sigmoid(self.gate_layer(torch.cat([text_features, image_features], dim=-1)))
        fused = self.fusion_norm(gate * text_features + (1.0 - gate) * image_features)
        logits = self.classifier(fused).squeeze(-1)

        if not return_aux:
            return LiteOutput(logits=logits, probability=torch.sigmoid(logits))
        return LiteOutput(
            logits=logits,
            probability=torch.sigmoid(logits),
            gate=gate,
            text_features=text_features,
            image_features=image_features,
            text_token_features=text_out["token_features"],
            image_feature_maps=image_out["feature_maps"],
        )