"""Gradient-based evidence for the self-contained MingJian lite model.

This module deliberately keeps the first explainability pass cheap and honest:

* text evidence: gradient x embedding attribution over character tokens;
* image evidence: Grad-CAM over the last image CNN feature map;
* cross-modal evidence: model-gradient direction comparison, not semantic consistency.

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


def _direction(value: float, *, epsilon: float = 1e-12) -> str:
    if value > epsilon:
        return "fake"
    if value < -epsilon:
        return "real"
    return "neutral"


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
        min_strength: float = 0.1,
        min_coherence: float = 0.1,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 <= min_strength < 0.5:
            raise ValueError("min_strength must be in [0, 0.5)")
        if not 0.0 <= min_coherence <= 1.0:
            raise ValueError("min_coherence must be in [0, 1]")
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.threshold = float(threshold)
        self.top_k = int(top_k)
        self.min_strength = float(min_strength)
        self.min_coherence = float(min_coherence)
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

    def _signed_token_scores(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        if embeddings.grad is None:
            raise RuntimeError("text embeddings did not receive gradients")
        gradient = embeddings.grad[0]
        signed_scores = (gradient * embeddings[0]).sum(dim=-1)
        magnitudes = signed_scores.abs()
        valid = attention_mask[0].bool()
        max_magnitude = (
            float(magnitudes[valid].max().detach().cpu()) if bool(valid.any()) else 0.0
        )
        return signed_scores, magnitudes, max_magnitude

    def _text_evidence(
        self,
        text: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> list[dict[str, Any]]:
        signed_scores, magnitudes, max_magnitude = self._signed_token_scores(
            embeddings,
            attention_mask,
        )
        if max_magnitude <= 1e-12:
            return []

        chars = list(normalize_text(text))[: self.tokenizer.max_length]
        valid_positions = [
            position
            for position, is_valid in enumerate(attention_mask[0].detach().cpu().tolist())
            if is_valid and position < len(chars)
        ]
        ranked = sorted(
            valid_positions,
            key=lambda position: float(magnitudes[position].detach().cpu()),
            reverse=True,
        )
        evidence: list[dict[str, Any]] = []
        for position in ranked[: self.top_k]:
            signed = float(signed_scores[position].detach().cpu()) / max_magnitude
            evidence.append(
                {
                    "position": position,
                    "token": chars[position],
                    "importance": round(
                        float(magnitudes[position].detach().cpu()) / max_magnitude,
                        6,
                    ),
                    "signed_score": round(signed, 6),
                    "direction": _direction(signed),
                }
            )
        return evidence

    def _text_units(
        self,
        text: str,
        attention_mask: torch.Tensor,
        signed_scores: torch.Tensor,
        magnitudes: torch.Tensor,
        max_magnitude: float,
    ) -> list[dict[str, Any]]:
        """Group character evidence into human-readable clauses."""
        if max_magnitude <= 1e-12:
            return []

        chars = list(normalize_text(text))[: self.tokenizer.max_length]
        if not chars:
            return []

        separators = set("。！？!?；;")
        spans: list[tuple[int, int]] = []
        start = 0
        for index, char in enumerate(chars):
            if char in separators:
                if start <= index:
                    spans.append((start, index + 1))
                start = index + 1
        if start < len(chars):
            spans.append((start, len(chars)))
        if not spans:
            spans = [(0, len(chars))]

        valid_positions = {
            position
            for position, is_valid in enumerate(attention_mask[0].detach().cpu().tolist())
            if is_valid and position < len(chars)
        }
        units: list[dict[str, Any]] = []
        for start_char, end_char in spans:
            positions = [
                position
                for position in range(start_char, min(end_char, len(chars)))
                if position in valid_positions
            ]
            if not positions:
                continue
            unit_text = "".join(chars[start_char:end_char]).strip()
            if not unit_text:
                continue
            unit_importance = max(
                float(magnitudes[position].detach().cpu()) for position in positions
            ) / max_magnitude
            unit_signed = sum(
                float(signed_scores[position].detach().cpu()) for position in positions
            ) / len(positions) / max_magnitude
            top_tokens = [
                {
                    "position": position,
                    "token": chars[position],
                    "importance": round(
                        float(magnitudes[position].detach().cpu()) / max_magnitude,
                        6,
                    ),
                }
                for position in sorted(
                    positions,
                    key=lambda item: float(magnitudes[item].detach().cpu()),
                    reverse=True,
                )[:3]
            ]
            units.append(
                {
                    "start_char": start_char,
                    "end_char": end_char,
                    "text": unit_text[:80] + ("…" if len(unit_text) > 80 else ""),
                    "importance": round(unit_importance, 6),
                    "signed_score": round(unit_signed, 6),
                    "direction": _direction(unit_signed),
                    "top_tokens": top_tokens,
                }
            )

        units.sort(key=lambda item: item["importance"], reverse=True)
        return units[: self.top_k]

    @staticmethod
    def _grad_cam_maps(feature_maps: torch.Tensor) -> dict[str, list[list[float]]]:
        """Return positive Grad-CAM and a signed direction map."""
        if feature_maps.grad is None:
            return {"positive": [], "signed": []}
        activations = feature_maps.detach()[0]
        gradients = feature_maps.grad.detach()[0]
        weights = gradients.mean(dim=(1, 2))
        signed = (weights[:, None, None] * activations).sum(dim=0)
        positive = torch.relu(signed)

        max_abs = float(signed.abs().max().detach().cpu())
        if max_abs > 1e-12:
            signed = signed / max_abs
        max_positive = float(positive.max().detach().cpu())
        if max_positive > 1e-12:
            positive = positive / max_positive

        return {
            "positive": [
                [round(float(value), 6) for value in row] for row in positive.cpu().tolist()
            ],
            "signed": [
                [round(float(value), 6) for value in row] for row in signed.cpu().tolist()
            ],
        }

    @staticmethod
    def _image_regions(
        signed_cam: list[list[float]],
        *,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        if not signed_cam or not signed_cam[0]:
            return []
        height = len(signed_cam)
        width = len(signed_cam[0])
        regions: list[dict[str, Any]] = []
        for row in range(height):
            for col in range(width):
                score = float(signed_cam[row][col])
                regions.append(
                    {
                        "row": row,
                        "col": col,
                        "score": round(score, 6),
                        "direction": _direction(score),
                        "bbox": [
                            round(col / width, 4),
                            round(row / height, 4),
                            round((col + 1) / width, 4),
                            round((row + 1) / height, 4),
                        ],
                    }
                )
        regions.sort(key=lambda item: abs(item["score"]), reverse=True)
        return regions[:limit]

    @staticmethod
    def _cross_modal_evidence(
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        *,
        has_image: bool,
        min_strength: float = 0.1,
        min_coherence: float = 0.1,
    ) -> dict[str, Any]:
        """Compare model-evidence directions, not factual consistency."""
        text_signed = 0.0
        image_signed = 0.0
        text_coherence = 0.0
        image_coherence = 0.0
        if text_features.grad is not None:
            text_products = text_features.grad * text_features
            text_signed = float(text_products.sum().detach().cpu())
            text_mass = float(text_products.abs().sum().detach().cpu())
            text_coherence = abs(text_signed) / text_mass if text_mass > 1e-12 else 0.0
        if has_image and image_features.grad is not None:
            image_products = image_features.grad * image_features
            image_signed = float(image_products.sum().detach().cpu())
            image_mass = float(image_products.abs().sum().detach().cpu())
            image_coherence = abs(image_signed) / image_mass if image_mass > 1e-12 else 0.0

        text_abs = abs(text_signed)
        image_abs = abs(image_signed)
        total = text_abs + image_abs
        text_strength = text_abs / total if total > 1e-12 else 0.0
        image_strength = image_abs / total if total > 1e-12 else 0.0

        text_direction = _direction(text_signed)
        image_direction = _direction(image_signed)
        if not has_image:
            state = "insufficient"
            state_zh = "证据不足"
            reason_code = "missing_image"
            summary = "未提供图片，无法比较图文证据方向。"
            action = "建议补充图片或相关信息后复核。"
        elif "neutral" in {text_direction, image_direction}:
            state = "insufficient"
            state_zh = "证据不足"
            reason_code = "neutral_direction"
            summary = "至少一个模态未给出非中性的梯度证据方向，暂不形成图文方向判断。"
            action = "建议补充信息或转人工复核。"
        elif text_strength < min_strength or image_strength < min_strength:
            state = "insufficient"
            state_zh = "证据不足"
            reason_code = "weak_modality"
            summary = "至少一个模态的相对证据强度不足，暂不形成图文方向判断。"
            action = "建议补充信息或转人工复核。"
        elif text_coherence < min_coherence or image_coherence < min_coherence:
            state = "insufficient"
            state_zh = "证据不足"
            reason_code = "unstable_direction"
            summary = "至少一个模态的梯度方向被内部正负贡献大量抵消，方向稳定性不足。"
            action = "建议补充信息或转人工复核。"
        elif text_direction == image_direction:
            state = "support"
            state_zh = "证据同向"
            reason_code = "same_direction"
            summary = (
                "文本证据与图像证据在模型中的梯度方向相同；"
                "这只表示模型内部证据方向同向，不代表语义一致或事实成立。"
            )
            action = "结合风险等级和原始来源进行人工复核。"
        else:
            state = "conflict"
            state_zh = "方向冲突"
            reason_code = "opposite_direction"
            summary = (
                "文本证据与图像证据在模型中的梯度方向相反；"
                "这提示两模态给出了不同方向的信号，不等于语义矛盾。"
            )
            action = "优先人工复核，并检查图文是否来自同一事件。"

        return {
            "state": state,
            "state_zh": state_zh,
            "reason_code": reason_code,
            "summary": summary,
            "recommended_action": action,
            "text_signed": round(text_signed, 6),
            "image_signed": round(image_signed, 6),
            "text_strength": round(text_strength, 6),
            "image_strength": round(image_strength, 6),
            "text_coherence": round(text_coherence, 6),
            "image_coherence": round(image_coherence, 6),
            "text_direction": text_direction,
            "image_direction": image_direction,
            "min_strength": min_strength,
            "min_coherence": min_coherence,
            "note": "本项比较的是模型梯度证据方向，不代表语义真伪、事实认定或因果关系。",
        }

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
        if not has_image:
            # An absent modality must never receive fallback contribution, even
            # when the available text features have zero measurable energy.
            text_contribution, image_contribution = 1.0, 0.0
        elif total > 1e-12:
            text_contribution = text_energy / total
            image_contribution = image_energy / total
        else:
            text_contribution = image_contribution = 0.5
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
            reason = "虚假概率位于 40%–60% 待复核区间，建议人工复核。"
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
                if (
                    output.gate is None
                    or output.text_features is None
                    or output.image_features is None
                ):
                    raise RuntimeError("model did not return auxiliary tensors")
                output.text_features.retain_grad()
                output.image_features.retain_grad()
                output.logits[0].backward()
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                self.model.train()

        probability_fake = float(output.probability[0].detach().cpu())
        probability_real = 1.0 - probability_fake
        predicted_label = "fake" if probability_fake >= self.threshold else "real"
        # Classify the published score so JSON and UI agree at rounded band boundaries.
        # Preserve the raw thresholded label above for metrics and existing API consumers.
        published_probability_fake = round(probability_fake, 6)
        risk = self._risk(published_probability_fake)
        display_label = "uncertain" if risk["is_uncertain"] else predicted_label
        binary_label_name_zh = "疑似虚假" if predicted_label == "fake" else "倾向真实"
        display_label_name_zh = {
            "uncertain": "待复核",
            "fake": "疑似虚假",
            "real": "倾向真实",
        }[display_label]
        signed_scores, magnitudes, max_magnitude = self._signed_token_scores(
            captured["embeddings"],
            attention_mask,
        )
        text_evidence = self._text_evidence(
            text,
            input_ids=input_ids,
            attention_mask=attention_mask,
            embeddings=captured["embeddings"],
        )
        text_units = self._text_units(
            text,
            attention_mask,
            signed_scores,
            magnitudes,
            max_magnitude,
        )
        modality = self._modality_evidence(
            output.gate,
            output.text_features,
            output.image_features,
            has_image_bool,
        )
        cam_maps = self._grad_cam_maps(captured["feature_maps"])
        cam = cam_maps["positive"]
        signed_cam = cam_maps["signed"]
        image_regions = self._image_regions(signed_cam) if has_image_bool else []
        cross_modal = self._cross_modal_evidence(
            output.text_features,
            output.image_features,
            has_image=has_image_bool,
            min_strength=self.min_strength,
            min_coherence=self.min_coherence,
        )
        limitations = [
            "本结果是辅助研判，不构成事实认定、法律意见或行政处置依据。",
            "文本证据来自梯度归因，图像证据来自粗粒度 Grad-CAM，均不等于因果证明。",
            "证据卡中的同向/冲突是模型证据方向，不是语义事实或因果结论。",
            "当前模型仅在 Weibo17 上完成首轮训练，跨域泛化与概率校准仍需继续验证。",
        ]
        if not has_image_bool:
            limitations.append("未提供图片，本次调用已执行缺图降级，图像贡献被置零。")

        return {
            "sample_id": sample_id,
            "decision": {
                # label/binary_label are the raw thresholded result for metrics and compatibility.
                "label": predicted_label,
                "label_name_zh": binary_label_name_zh,
                "binary_label": predicted_label,
                "binary_label_name_zh": binary_label_name_zh,
                "label_semantics": {
                    "label": "原始概率按决策阈值得到的二分类结果，用于评测和兼容；不是页面主结论。",
                    "display_label": (
                        "用户主结论，依据已发布的六位小数虚假概率；"
                        "40%–60%（含端点）统一为待复核。"
                    ),
                },
                # display_label is the user-facing three-state result; use risk.level for review UI.
                "display_label": display_label,
                "display_label_name_zh": display_label_name_zh,
                "probability_fake": published_probability_fake,
                "probability_real": round(probability_real, 6),
                "threshold": self.threshold,
            },
            "risk": risk,
            "modalities": {
                "has_image": has_image_bool,
                **modality,
            },
            "text_evidence": text_evidence,
            "image_evidence": {
                "method": "grad-cam",
                "grid_shape": [len(cam), len(cam[0])] if cam and cam[0] else [0, 0],
                "cam": cam,
                "signed_cam": signed_cam,
                "top_regions": image_regions,
            },
            "evidence_card": {
                "state": cross_modal["state"],
                "state_zh": cross_modal["state_zh"],
                "summary": cross_modal["summary"],
                "recommended_action": cross_modal["recommended_action"],
                "text_units": text_units,
                "image_regions": image_regions,
                "cross_modal": cross_modal,
            },
            "limitations": limitations,
        }


