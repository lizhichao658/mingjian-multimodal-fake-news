from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_training_cli_seed_controls_model_initialization(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/train_lite_baseline.py"
    spec = importlib.util.spec_from_file_location("training_seed_cli", path)
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    for split in ("train", "val", "test"):
        (tmp_path / f"{split}.jsonl").touch()
    samples = [SimpleNamespace(label=i % 2, text="合成训练流程样例") for i in range(4)]
    monkeypatch.setattr(training, "read_jsonl", lambda path: samples)

    class Tokenizer:
        vocab_size = 20
        max_length = 196

        def save(self, path):
            path.write_text("{}")

    monkeypatch.setattr(training, "build_char_tokenizer", lambda *args, **kwargs: Tokenizer())
    monkeypatch.setattr(training, "Weibo17TorchDataset", lambda samples, **kwargs: samples)
    monkeypatch.setattr(training, "make_loader", lambda dataset, **kwargs: dataset)
    initial_states = []

    class BeforeFit(Exception):
        pass

    def capture_model(model, *args, **kwargs):
        initial_states.append(torch.cat([p.detach().cpu().flatten() for p in model.parameters()]))
        raise BeforeFit

    monkeypatch.setattr(training, "fit_lite_model", capture_model)
    for run, seed in enumerate((42, 42, 43)):
        torch.manual_seed(800 + run)
        torch.rand(100 + run)
        with pytest.raises(BeforeFit):
            training.main([
                "--data-dir", str(tmp_path), "--out", str(tmp_path / f"run_{run}"),
                "--device", "cpu", "--num-workers", "0", "--seed", str(seed),
            ])
    assert torch.equal(initial_states[0], initial_states[1])
    assert not torch.equal(initial_states[0], initial_states[2])
