"""Safety and completion checks for the local-data training orchestrator."""

from __future__ import annotations

import builtins
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def preparation(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_local_model.py"
    spec = importlib.util.spec_from_file_location("local_model_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "runtime_info", lambda device: ("cpu", {"versions": {"test": "mock"}}))
    return module


def _metrics() -> dict:
    values = {name: 0.5 for name in ("accuracy", "precision", "recall", "f1", "auc", "ece")}
    return {
        "data": {
            "dir": "PRIVATE_LOCAL_PATH_DO_NOT_EXPORT",
            "splits": {name: {"samples": count} for name, count in (
                ("train", 5415), ("val", 843), ("test", 1465),
            )},
        },
        "training": {"epochs_requested": 8, "epochs_ran": 8, "best_epoch": 2},
        "evaluation": {
            split: {"threshold_0_5": {**values, "samples": count, "threshold": 0.5}}
            for split, count in (("val", 843), ("test", 1465))
        },
    }


def _health() -> dict:
    return {
        "decision": {"probability_fake": 0.52},
        "modalities": {"has_image": False},
        "evidence_card": {"cross_modal": {"reason_code": "missing_image"}},
    }


def _fake_processes(monkeypatch, preparation, *, fail=None, missing=None, health=None, metrics=None):
    calls = []

    def run(command, **kwargs):
        script = Path(command[1]).stem
        calls.append((script, command, kwargs))
        if script == fail:
            return subprocess.CompletedProcess(command, 7)
        if script == "train_lite_baseline":
            model = Path(command[command.index("--out") + 1])
            model.mkdir()
            outputs = {
                "model.pt": b"mock-checkpoint",
                "tokenizer.json": b'{"vocab": {"<pad>": 0}}',
                "metrics.json": json.dumps(metrics or _metrics()).encode(),
            }
            for name, content in outputs.items():
                if name != missing:
                    (model / name).write_bytes(content)
        elif script == "explain_lite":
            destination = Path(command[command.index("--out") + 1])
            destination.write_text(json.dumps(_health() if health is None else health))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(preparation.subprocess, "run", run)
    return calls


def _args(tmp_path):
    archive = tmp_path / "weibo.zip"
    archive.write_bytes(b"mock: the real adapter verifies the official archive")
    work = tmp_path / "new_local_run"
    return ["--archive", str(archive), "--work-dir", str(work)], work


def test_help_does_not_import_torch(preparation, monkeypatch, capsys):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("--help must work without torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as stopped:
        preparation.main(["--help"])
    assert stopped.value.code == 0
    assert "--archive" in capsys.readouterr().out


@pytest.mark.parametrize("relative", (".", "data/processed", "artifacts/new_model"))
def test_rejects_outputs_inside_checkout(preparation, tmp_path, relative):
    archive = tmp_path / "weibo.zip"
    archive.touch()
    with pytest.raises(ValueError, match="outside"):
        preparation.validate_paths(archive, preparation.REPO_ROOT / relative)


def test_existing_output_is_never_overwritten(preparation, tmp_path, monkeypatch):
    args, work = _args(tmp_path)
    work.mkdir()
    marker = work / "model.pt"
    marker.write_bytes(b"keep-this-model")
    calls = _fake_processes(monkeypatch, preparation)
    assert preparation.main(args) == 1
    assert marker.read_bytes() == b"keep-this-model"
    assert not (work / "preparation_report.json").exists()
    assert calls == []


def test_success_requires_train_and_cpu_inference(preparation, tmp_path, monkeypatch):
    args, work = _args(tmp_path)
    calls = _fake_processes(monkeypatch, preparation)
    assert preparation.main(args + ["--threads", "3"]) == 0
    assert [call[0] for call in calls] == [
        "build_weibo17_local", "train_lite_baseline", "explain_lite",
    ]
    assert all(command[0] == preparation.sys.executable for _, command, _ in calls)
    assert "--skip-integrity-check" not in calls[0][1]
    assert "--limit" not in calls[1][1]
    for flag, value in (("--epochs", "8"), ("--batch-size", "64"), ("--lr", "2e-3")):
        command = calls[1][1]
        assert command[command.index(flag) + 1] == value
    assert calls[2][1][calls[2][1].index("--device") + 1] == "cpu"
    assert all(call[2]["env"]["OMP_NUM_THREADS"] == "3" for call in calls)
    raw = (work / "preparation_report.json").read_text(encoding="utf-8")
    report = json.loads(raw)
    assert report["status"] == "passed"
    assert report["cpu_health"]["missing_image_fallback"] == "passed"
    assert report["new_model_metrics"]["split_samples"]["test"] == 1465
    assert str(work) not in raw
    assert "PRIVATE_LOCAL_PATH_DO_NOT_EXPORT" not in raw


@pytest.mark.parametrize("failing_script, expected_calls, stage", (
    ("build_weibo17_local", 1, "build"),
    ("train_lite_baseline", 2, "train"),
    ("explain_lite", 3, "cpu_explain"),
))
def test_subprocess_failure_stops_later_stages(
    preparation, tmp_path, monkeypatch, failing_script, expected_calls, stage
):
    args, work = _args(tmp_path)
    calls = _fake_processes(monkeypatch, preparation, fail=failing_script)
    assert preparation.main(args) == 1
    assert len(calls) == expected_calls
    report = json.loads((work / "preparation_report.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == stage
    assert report["stages"][-1]["returncode"] == 7


@pytest.mark.parametrize("missing", ("model.pt", "tokenizer.json", "metrics.json"))
def test_train_exit_zero_without_artifacts_is_failure(preparation, tmp_path, monkeypatch, missing):
    args, work = _args(tmp_path)
    calls = _fake_processes(monkeypatch, preparation, missing=missing)
    assert preparation.main(args) == 1
    assert len(calls) == 2
    report = json.loads((work / "preparation_report.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "artifact_validation"
    assert "cpu_health" not in report


def test_subset_training_cannot_be_reported_as_full_preparation(preparation, tmp_path, monkeypatch):
    args, work = _args(tmp_path)
    metrics = _metrics()
    metrics["data"]["splits"]["train"]["samples"] = 32
    calls = _fake_processes(monkeypatch, preparation, metrics=metrics)
    assert preparation.main(args) == 1
    assert len(calls) == 2
    assert json.loads((work / "preparation_report.json").read_text())["status"] == "failed"


def test_cpu_process_exit_zero_without_health_evidence_is_failure(preparation, tmp_path, monkeypatch):
    args, work = _args(tmp_path)
    _fake_processes(monkeypatch, preparation, health={})
    assert preparation.main(args) == 1
    report = json.loads((work / "preparation_report.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "health_validation"
    assert "cpu_health" not in report


def test_dependency_failure_has_failed_report_without_launching_stages(
    preparation, tmp_path, monkeypatch
):
    args, work = _args(tmp_path)

    def missing_dependency(device):
        raise ModuleNotFoundError("torch")

    monkeypatch.setattr(preparation, "runtime_info", missing_dependency)
    calls = _fake_processes(monkeypatch, preparation)
    assert preparation.main(args) == 1
    report = json.loads((work / "preparation_report.json").read_text())
    assert report["status"] == "failed"
    assert report["stage"] == "dependencies"
    assert calls == []
