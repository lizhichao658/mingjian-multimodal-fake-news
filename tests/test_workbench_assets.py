from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO / "scripts/sync_workbench.py"
UI = REPO / "src/mingjian/web/workbench.html"
NODE = os.environ.get("MINGJIAN_NODE") or shutil.which("node")


def test_standalone_export_is_exact_and_checks_without_rewriting(tmp_path) -> None:
    target = tmp_path / "standalone" / "review.html"
    result = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--output", str(target)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert target.read_bytes() == UI.read_bytes()
    target.write_text("outdated exported UI", encoding="utf-8")
    checked = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--output", str(target), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert checked.returncode == 1
    assert "MISMATCH" in checked.stderr
    assert target.read_text(encoding="utf-8") == "outdated exported UI"


def test_export_check_reports_a_missing_file_without_creating_it(tmp_path) -> None:
    target = tmp_path / "not-created.html"
    checked = subprocess.run(
        [sys.executable, str(EXPORT_SCRIPT), "--output", str(target), "--check"],
        capture_output=True, text=True, check=False,
    )
    assert checked.returncode == 1
    assert not target.exists()


@pytest.mark.skipif(NODE is None, reason="Node is optional for UI regression")
def test_workspace_browser_logic_regression() -> None:
    checked = subprocess.run(
        [NODE, str(REPO / "scripts/check_workbench.cjs")],
        capture_output=True, text=True, check=False, timeout=30,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "assertions passed" in checked.stdout
