"""Dependency-aware smoke test for the MingJian repository."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from mingjian.data.dataset import read_jsonl, write_jsonl  # noqa: E402
from mingjian.data.schema import NewsSample  # noqa: E402
from mingjian.evaluation.metrics import binary_classification_metrics  # noqa: E402


def main() -> int:
    print("[1/4] data schema and JSONL roundtrip")
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "samples.jsonl"
        samples = [
            NewsSample("smoke-1", "示例文本", "image.png", 0, "test"),
            NewsSample("smoke-2", "示例文本", "image.png", 1, "test"),
        ]
        assert write_jsonl(samples, path) == 2
        assert read_jsonl(path) == samples
    print("      OK")

    print("[2/4] metrics")
    metrics = binary_classification_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    print("      OK")

    print("[3/4] configuration")
    from mingjian.config import load_config

    config = load_config(ROOT / "configs" / "baseline.toml")
    assert config.training.seed == 42
    print("      OK")

    print("[4/4] model forward/backward")
    if importlib.util.find_spec("torch") is None:
        print("      SKIP: PyTorch is not installed")
        return 0

    import torch  # noqa: E402

    from mingjian.models.baseline import TinyTextImageBaseline  # noqa: E402

    model = TinyTextImageBaseline(vocab_size=64, text_embed_dim=16, hidden_dim=8)
    output = model(torch.randint(1, 64, (2, 7)), torch.randn(2, 3, 16, 16))
    assert output.logits.shape == (2,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output.logits, torch.tensor([0.0, 1.0])
    )
    loss.backward()
    print("      OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())