r"""Audit evidence-card states on a local JSONL split.

This script does not train or modify the model. It runs the same checkpoint and
explainer used by the demo, then reports the observed distribution of
'support / conflict / insufficient' states. Missing images are reported
separately and are expected to yield 'insufficient'.

Example::

    python scripts/audit_evidence_cards.py ^
        --checkpoint artifacts/lite_v1/model.pt ^
        --data <local-weibo17-processed>/test.jsonl ^
        --limit 200 ^
        --out artifacts/evidence_card_audit.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mingjian.data.dataset import read_jsonl
from mingjian.inference import LitePredictor


def _resolve_image(raw: str, data_dir: Path) -> Path | None:
    if not raw:
        return None
    candidate = Path(raw)
    options = [candidate] if candidate.is_absolute() else [Path.cwd() / candidate, data_dir / candidate]
    for option in options:
        if option.is_file():
            return option
    return None


def _redact_error(exc: BaseException, *, data_dir: Path) -> str:
    """Keep audit errors useful without leaking local absolute paths."""
    message = f"{type(exc).__name__}: {exc}"
    for prefix in (str(REPO_ROOT), str(data_dir), str(Path.cwd())):
        if prefix:
            message = message.replace(prefix, "<local>")
    message = re.sub(r"[A-Za-z]:\\[^\s\"']+", "<local-path>", message)
    return message


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True, help="JSONL split with text/image_path/label")
    parser.add_argument("--limit", type=int, default=200, help="maximum samples; <=0 means all")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-strength", type=float, default=0.1)
    parser.add_argument("--min-coherence", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    predictor = LitePredictor.from_checkpoint(
        args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        image_size=args.image_size,
    )
    samples = read_jsonl(args.data)
    if args.limit > 0:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("no samples found")

    states: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    by_label: dict[int, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, str]] = []
    text_strengths: list[float] = []
    image_strengths: list[float] = []
    text_coherences: list[float] = []
    image_coherences: list[float] = []
    missing_images = 0
    data_dir = args.data.resolve().parent

    for sample_index, sample in enumerate(samples):
        image_path = _resolve_image(str(sample.image_path or ""), data_dir)
        if image_path is None:
            missing_images += 1
        try:
            result = predictor.analyze(
                sample.text,
                image_path=image_path,
                sample_id=sample.sample_id,
                top_k=args.top_k,
                min_strength=args.min_strength,
                min_coherence=args.min_coherence,
            )
        except Exception as exc:  # noqa: BLE001 - audit must record every failed sample
            errors.append(
                {
                    "sample_index": sample_index,
                    "error": _redact_error(exc, data_dir=data_dir),
                }
            )
            continue

        card = result["evidence_card"]
        cross = card["cross_modal"]
        state = str(card["state"])
        states[state] += 1
        reason_codes[str(cross.get("reason_code", "unknown"))] += 1
        by_label[int(sample.label)][state] += 1
        directions[str(cross["text_direction"]) + "->" + str(cross["image_direction"])] += 1
        text_strengths.append(float(cross["text_strength"]))
        image_strengths.append(float(cross["image_strength"]))
        text_coherences.append(float(cross["text_coherence"]))
        image_coherences.append(float(cross["image_coherence"]))
        if len(examples[state]) < 3:
            examples[state].append(
                {
                    "sample_index": sample_index,
                    "label": int(sample.label),
                    "probability_fake": result["decision"]["probability_fake"],
                    "state_zh": card["state_zh"],
                    "reason_code": cross.get("reason_code"),
                    "summary": card["summary"],
                    "text_direction": cross["text_direction"],
                    "image_direction": cross["image_direction"],
                    "text_strength": cross["text_strength"],
                    "image_strength": cross["image_strength"],
                    "image_provided": image_path is not None,
                }
            )

    audited = sum(states.values())
    summary = {
        "checkpoint_file": args.checkpoint.name,
        "data_file": args.data.name,
        "input_paths_redacted": True,
        "requested_limit": args.limit,
        "loaded_samples": len(samples),
        "audited_samples": audited,
        "error_samples": len(errors),
        "missing_images": missing_images,
        "thresholds": {
            "min_strength": args.min_strength,
            "min_coherence": args.min_coherence,
        },
        "states": dict(states),
        "reason_codes": dict(reason_codes),
        "states_by_label": {str(label): dict(counts) for label, counts in sorted(by_label.items())},
        "direction_pairs": dict(directions),
        "strength": {
            "text_mean": round(statistics.fmean(text_strengths), 6) if text_strengths else None,
            "image_mean": round(statistics.fmean(image_strengths), 6) if image_strengths else None,
        },
        "coherence": {
            "text_mean": round(statistics.fmean(text_coherences), 6) if text_coherences else None,
            "image_mean": round(statistics.fmean(image_coherences), 6) if image_coherences else None,
        },
        "examples": dict(examples),
        "errors": errors,
        "note": "States are model-evidence directions, not semantic consistency or factual truth.",
    }
    output = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(output)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
