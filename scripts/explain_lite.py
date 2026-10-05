r"""Run one explainable inference with the MingJian lite checkpoint.

Example::

    python scripts/explain_lite.py ^
        --checkpoint artifacts/lite_v1/model.pt ^
        --text "某条新闻文本" ^
        --image D:\local\news.jpg ^
        --out artifacts\one_explanation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mingjian.inference import LitePredictor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--image", type=Path, default=None, help="optional local image")
    parser.add_argument("--sample-id", default="adhoc")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
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
    result = predictor.analyze(
        args.text,
        image_path=args.image,
        sample_id=args.sample_id,
        top_k=args.top_k,
    )
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

