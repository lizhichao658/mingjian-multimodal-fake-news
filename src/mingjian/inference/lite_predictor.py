"""Inference helpers for the self-contained MingJian lite checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mingjian.data.tokenization import CharTokenizer
from mingjian.data.torch_dataset import load_image_tensor
from mingjian.explain import LiteExplainer
from mingjian.models.lite import LiteFusionClassifier, LiteModelConfig


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve ``auto`` to CUDA when available and validate explicit devices."""

    if isinstance(requested, torch.device):
        device = requested
    elif requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_threshold(value: float, *, name: str) -> float:
    threshold = float(value)
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return threshold


@dataclass
class LitePredictor:
    """Load one lite checkpoint and expose a compact explanation API."""

    model: LiteFusionClassifier
    tokenizer: CharTokenizer
    threshold: float
    decision_threshold: float
    device: torch.device
    image_size: int
    metadata: dict[str, Any]

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        tokenizer_path: str | Path | None = None,
        device: str | torch.device = "auto",
        image_size: int = 128,
        decision_threshold: float = 0.5,
    ) -> LitePredictor:
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_file}")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        decision_threshold = _validate_threshold(
            decision_threshold,
            name="decision_threshold",
        )

        resolved_device = resolve_device(device)
        try:
            payload = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch fallback
            payload = torch.load(checkpoint_file, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError("checkpoint must contain a mapping")
        config_raw = payload.get("config")
        state_dict = payload.get("state_dict")
        if not isinstance(config_raw, dict) or not isinstance(state_dict, dict):
            raise TypeError("checkpoint must contain config and state_dict mappings")

        tokenizer_file = (
            Path(tokenizer_path)
            if tokenizer_path is not None
            else checkpoint_file.parent / str(payload.get("tokenizer_file", "tokenizer.json"))
        )
        tokenizer = CharTokenizer.load(tokenizer_file)
        model = LiteFusionClassifier(LiteModelConfig.from_dict(config_raw))
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()
        checkpoint_threshold = _validate_threshold(
            payload.get("threshold", 0.5),
            name="checkpoint threshold",
        )
        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"config", "state_dict", "tokenizer_file"}
        }
        return cls(
            model=model,
            tokenizer=tokenizer,
            threshold=checkpoint_threshold,
            decision_threshold=decision_threshold,
            device=resolved_device,
            image_size=int(image_size),
            metadata=metadata,
        )

    def analyze(
        self,
        text: str,
        *,
        image_path: str | Path | None = None,
        sample_id: str = "",
        top_k: int = 12,
        decision_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Analyze one text/image pair and return JSON-ready evidence."""

        active_threshold = (
            self.decision_threshold
            if decision_threshold is None
            else _validate_threshold(decision_threshold, name="decision_threshold")
        )
        image = None
        if image_path is not None and str(image_path).strip():
            image = load_image_tensor(image_path, self.image_size)
        explainer = LiteExplainer(
            self.model,
            self.tokenizer,
            device=self.device,
            image_size=self.image_size,
            threshold=active_threshold,
            top_k=top_k,
        )
        result = explainer.explain(text, image=image, sample_id=sample_id or "adhoc")
        result["model"] = {
            "checkpoint_threshold": self.threshold,
            "decision_threshold": active_threshold,
            "device": str(self.device),
            "image_size": self.image_size,
            "image_provided": image is not None,
            "metadata": self.metadata,
        }
        return result