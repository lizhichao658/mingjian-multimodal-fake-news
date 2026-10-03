"""Baseline models for the first text-image MVP.

``FeatureFusionBaseline`` is the real baseline interface: it consumes pooled
text and image features. ``TinyTextImageBaseline`` is a small, dependency-light
model used to verify forward/backward behavior before HuggingFace encoders are
connected.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    probability: torch.Tensor


class FeatureFusionBaseline(nn.Module):
    """Concatenate pooled text/image features and classify fake(0)/real(1)."""

    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if text_dim <= 0 or image_dim <= 0 or hidden_dim <= 0:
            raise ValueError("feature dimensions and hidden_dim must be positive")
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
    ) -> ModelOutput:
        if text_features.ndim != 2 or image_features.ndim != 2:
            raise ValueError("features must have shape [batch, dimension]")
        if text_features.shape[0] != image_features.shape[0]:
            raise ValueError("text and image batch sizes must match")
        fused = torch.cat(
            [self.text_projection(text_features), self.image_projection(image_features)],
            dim=-1,
        )
        logits = self.classifier(fused).squeeze(-1)
        return ModelOutput(logits=logits, probability=torch.sigmoid(logits))


class TinyTextImageBaseline(nn.Module):
    """Small synthetic model for tests before real encoders are installed."""

    def __init__(
        self,
        vocab_size: int = 128,
        text_embed_dim: int = 32,
        image_channels: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_embedding = nn.Embedding(vocab_size, text_embed_dim, padding_idx=0)
        self.text_projection = nn.Linear(text_embed_dim, hidden_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(8 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, channels, height, width]")
        if attention_mask is None:
            attention_mask = (input_ids != 0).to(dtype=torch.float32)
        else:
            attention_mask = attention_mask.to(dtype=torch.float32)

        embeddings = self.text_embedding(input_ids)
        mask = attention_mask.unsqueeze(-1)
        pooled_text = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        image_features = self.image_encoder(images)
        fused = torch.cat([self.text_projection(pooled_text), image_features], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return ModelOutput(logits=logits, probability=torch.sigmoid(logits))