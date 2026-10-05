"""Gradient-based evidence for the self-contained MingJian lite model.

This module deliberately keeps the first explainability pass cheap and honest:

* text evidence: gradient x embedding attribution over character tokens;
* image evidence: Grad-CAM over the last image CNN feature map;
* cross-modal evidence: magnitude of each modality's contribution after gating.

These are explanation aids, not causal proofs or pixel-level tamper detection.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from mingjian.data.tokenization import CharTokenizer, normalize_text
from mingjian.models.lite import LiteFusionClassifier


def _rounded_vector(values: torch.Tensor, *, digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values.detach().cpu().tolist()]


class LiteExplainer:
    """Produce structured, JSON-serialisable evidence for one input pair."""

    def __init__(
        self,
        model: LiteFusionClassifier,
        tokenizer: CharTokenizer,
        *,
        device: str | torch.device = "cpu",
        image_size: int = 128,
        threshold: float = 0.5,
        top_k: int = 12,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.model.to(self.device)

    def _prepare_image(self, image: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, bool]:
        if image is None:
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            has_image = torch.zeros(1, device=self.device)
            return dummy, has_image, False
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor or None")
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("image must have shape [3, H, W] or [1, 3, H, W]")
        if image.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(f"image spatial size must be {self.image_size}x{self.image_size}")
        return image.to(self.device), torch.ones(1, device=self.device), True

    def _text_evidence(
        self,
        text: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> list[dict[str, Any]]:
        if embeddings.grad is None:
            raise RuntimeError("text embeddings did not receive gradients")
        gradient = embeddings.grad[0]
        signed_scores = (gradient * embeddings[0]).sum(dim=-1)
        magnitudes = signed_scores.abs()
        max_magnitude = float(magnitudes.max().detach().cpu())
        if max_magnitude <= 1e-12:
            return []

        chars = list(normalize_text(text))[: self.tokenizer.max_length]
        valid_positions = [
            position
            for position, is_valid in enumerate(attention_mask[0].detach().cpu().tolist())
            if is_valid and position < len(chars)
        ]
        ranked = sorted(valid_positions, key=lambda position: float(magnitudes[position].detach().cpu()), reverse=True)
        evidence: list[dict[str, Any]] = []
        for position in ranked[: self.top_k]:
            signed = float(signed_scores[position].detach().cpu()) / max_magnitude
            evidence.append(
                {
                    "position": position,
                    "token": chars[position],
                    "importance": round(float(magnitudes[position].detach().cpu()) / max_magnitude, 6),
                    "signed_score": round(signed, 6),
                    "direction": "fake" if signed >= 0 else "real",
                }
            )
        return evidence

    @staticmethod
    def _grad_cam(feature_maps: torch.Tensor) -> list[list[float]]:
        if feature_maps.grad is None:
            return []
        activations = feature_maps.detach()[0]
        gradients = feature_maps.grad.detach()[0]
        weights = gradients.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
        max_value = float(cam.max().detach().cpu())
        if max_value > 1e-12:
            cam = cam / max_value
        return [[round(float(value), 6) for value in row] for row in cam.cpu().tolist()]

    def _modality_evidence(
        self,
        gate: torch.Tensor,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        has_image: bool,
    ) -> dict[str, Any]:
        text_energy = float((gate * text_features).abs().mean().detach().cpu())
        image_energy = float(((1.0 - gate) * image_features).abs().mean().detach().cpu())
        if not has_image:
            image_energy = 0.0
        total = text_energy + image_energy
        text_contribution = text_energy / total if total > 1e-12 else 0.5
        image_contribution = image_energy / total if total > 1e-12 else 0.5
        return {
            "text_gate_mean": round(float(gate.mean().detach().cpu()), 6),
            "image_gate_mean": round(float((1.0 - gate).mean().detach().cpu()), 6),
            "text_contribution": round(text_contribution, 6),
            "image_contribution": round(image_contribution, 6),
            "note": "本项是门控融合贡献的幅度估计，不代表因果关系。",
        }

    def _risk(self, probability_fake: float) -> dict[str, Any]:
        confidence = max(probability_fake, 1.0 - probability_fake)
        entropy = 0.0
        for probability in (probability_fake, 1.0 - probability_fake):
            if probability > 0:
                entropy -= probability * math.log(probability)
        entropy = entropy / math.log(2.0)

        if 0.4 <= probability_fake <= 0.6 or confidence < 0.6:
            level = "uncertain"
            reason = "概率接近决策边界，建议人工复核。"
        elif probability_fake >= 0.75:
            level = "high"
            reason = "假新闻概率较高，但结论仍需人工核验证据。"
        elif probability_fake >= 0.55:
            level = "medium"
            reason = "存在假新闻风险，建议结合证据复核。"
        else:
            level = "low"
            reason = "当前模型给出的假新闻风险较低。"
        return {
            "level": level,
            "reason": reason,
            "confidence": round(confidence, 6),
            "normalized_entropy": round(entropy, 6),
            "is_uncertain": level == "uncertain",
        }

    def explain(
        self,
        text: str,
        image: torch.Tensor | None = None,
        *,
        sample_id: str = "",
    ) -> dict[str, Any]:
        """Return one structured explanation without requiring an API key."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        prepared_image, has_image, has_image_bool = self._prepare_image(image)
        ids, mask = self.tokenizer.encode(text, max_length=self.tokenizer.max_length)
        input_ids = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        attention_mask = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        captured: dict[str, torch.Tensor] = {}
        handles = []
        was_training = self.model.training
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        def capture_embeddings(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            output.retain_grad()
            captured["embeddings"] = output

        def capture_feature_maps(_module: Any, _inputs: Any, output: torch.Tensor) -> None:
            output.retain_grad()
            captured["feature_maps"] = output

        handles.append(self.model.text_encoder.embedding.register_forward_hook(capture_embeddings))
        handles.append(self.model.image_encoder.stages.register_forward_hook(capture_feature_maps))
        try:
            with torch.enable_grad():
                output = self.model(
                    input_ids=input_ids,
                    images=prepared_image,
                    attention_mask=attention_mask,
                    has_image=has_image,
                    return_aux=True,
                )
                if output.gate is None or output.text_features is None or output.image_features is None:
                    raise RuntimeError("model did not return auxiliary tensors")
                output.logits[0].backward()
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self.model.train()

        probability_fake = float(output.probability[0].detach().cpu())
        probability_real = 1.0 - probability_fake
        predicted_label = "fake" if probability_fake >= self.threshold else "real"
        text_evidence = self._text_evidence(
            text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            embeddings=captured["embeddings"],
        )
        modality = self._modality_evidence(
            output.gate,
            output.text_features,
            output.image_features,
            has_image_bool,
        )
        cam = self._grad_cam(captured["feature_maps"])
        limitations = [
            "本结果是辅助研判，不构成事实认定、法律意见或行政处置依据。",
            "文本证据来自梯度归因，图像证据来自粗粒度 Grad-CAM，均不等于因果证明。",
            "当前模型仅在 Weibo17 上完成首轮训练，跨域泛化与概率校准仍需继续验证。",
        ]
        if not has_image_bool:
            limitations.append("未提供图片，本次调用已执行缺图降级，图像贡献被置零。")

        return {
            "sample_id": sample_id,
            "decision": {
                "label": predicted_label,
                "label_name_zh": "疑似虚假" if predicted_label == "fake" else "倾向真实",
                "probability_fake": round(probability_fake, 6),
                "probability_real": round(probability_real, 6),
                "threshold": self.threshold,
            },
            "risk": self._risk(probability_fake),
            "modalities": {
                "has_image": has_image_bool,
                **modality,
            },
            "text_evidence": text_evidence,
            "image_evidence": {
                "method": "grad-cam",
                "grid_shape": [len(cam), len(cam[0])] if cam and cam[0] else [0, 0],
                "cam": cam,
            },
            "limitations": limitations,
        }


