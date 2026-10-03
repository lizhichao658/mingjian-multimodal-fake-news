"""Optional HuggingFace encoder wrappers.

These wrappers are intentionally lazy: importing this module does not require
Transformers, and tests can skip them when the dependency is unavailable.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _require_transformers() -> Any:
    try:
        import transformers
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "transformers is required for HuggingFace encoders; "
            "install the project dependencies first"
        ) from exc
    return transformers


class HuggingFaceTextEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True) -> None:
        super().__init__()
        transformers = _require_transformers()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return {"token_features": outputs.last_hidden_state, "pooled_features": pooled}


class HuggingFaceImageEncoder(nn.Module):
    def __init__(self, model_name: str, freeze: bool = True) -> None:
        super().__init__()
        transformers = _require_transformers()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.model(pixel_values=pixel_values)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state.mean(dim=1)
        return {"patch_features": outputs.last_hidden_state, "pooled_features": pooled}