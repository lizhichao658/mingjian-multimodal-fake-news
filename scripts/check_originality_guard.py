"""Fail when forbidden project names or model artifacts enter source directories.

Documentation may legitimately mention a reference project as background, but
runtime source code must remain clean-room and independent.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [ROOT / "src", ROOT / "scripts", ROOT / "tests"]
FORBIDDEN_TERMS = ["ComP", "comp_source"]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".safetensors", ".ckpt"}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for directory in SCAN_DIRS:
        if directory.exists():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    return files


def main() -> int:
    problems: list[str] = []
    for path in iter_files():
        if path.name == "check_originality_guard.py":
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"model artifact inside source tree: {relative}")
            continue
        if path.suffix.lower() not in {".py", ".ps1", ".md", ".toml", ".yaml", ".yml", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for term in FORBIDDEN_TERMS:
            if term in text:
                problems.append(f"forbidden term {term!r} found in {relative}")

    if problems:
        print("originality guard failed:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("originality guard passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())