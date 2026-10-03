"""Create tiny synthetic image/text fixtures for pipeline smoke tests.

These samples are NOT training data and must not be presented as real news.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "demo"
IMAGE_DIR = OUTPUT_DIR / "images"


def _make_image(path: Path, index: int, size: int = 64) -> None:
    image = Image.new("RGB", (size, size), color=(240, 240, 240))
    draw = ImageDraw.Draw(image)
    if index % 4 == 0:
        draw.rectangle((8, 8, size - 8, size - 8), outline=(30, 80, 160), width=4)
    elif index % 4 == 1:
        for offset in range(0, size, 8):
            draw.line((offset, 0, offset, size), fill=(200, 120, 40), width=2)
    elif index % 4 == 2:
        draw.ellipse((12, 12, size - 12, size - 12), outline=(30, 140, 80), width=4)
    else:
        draw.line((0, size, size, 0), fill=(160, 40, 120), width=4)
    image.save(path)


def main() -> int:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    samples = []
    for index in range(1, 5):
        image_path = IMAGE_DIR / f"demo-{index:04d}.png"
        _make_image(image_path, index)
        samples.append(
            {
                "sample_id": f"demo-{index:04d}",
                "text": f"这是第 {index} 条合成联调样例，仅用于验证数据管道，不代表真实新闻。",
                "image_path": image_path.relative_to(ROOT).as_posix(),
                "label": index % 2,
                "split": "test",
                "source": "synthetic-smoke-fixture",
                "metadata": {
                    "synthetic": True,
                    "purpose": "pipeline smoke test only",
                    "not_training_data": True,
                },
            }
        )
    output_path = OUTPUT_DIR / "sample_cases.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} synthetic samples to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())