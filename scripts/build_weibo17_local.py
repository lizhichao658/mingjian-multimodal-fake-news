"""Build the local Weibo17 JSONL view from the official archive.

Usage (from the repository root)::

    python scripts/build_weibo17_local.py --out work/weibo17_processed

Raw text, tweet ids, image URLs and images stay in the local output directory
and must never be committed to Git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mingjian.data.weibo17 import (
    ARCHIVE_FILE_NAME,
    Weibo17ArchiveError,
    assert_official_invariants,
    build_weibo17_dataset,
)


def _default_archive() -> Path:
    return Path.home() / "Downloads" / ARCHIVE_FILE_NAME


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=_default_archive(),
        help=f"path to the local {ARCHIVE_FILE_NAME} (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="local output directory for JSONL, images and manifest.json",
    )
    parser.add_argument(
        "--skip-integrity-check",
        action="store_true",
        help="skip the archive size and SHA-256 check (development only)",
    )
    parser.add_argument(
        "--extract-all-images",
        action="store_true",
        help="extract every archive image instead of only the selected images",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_weibo17_dataset(
            args.archive,
            args.out,
            verify_integrity=not args.skip_integrity_check,
            extract_all_images=args.extract_all_images,
        )
        assert_official_invariants(manifest)
    except Weibo17ArchiveError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    summary = {
        "output_dir": str(args.out),
        "counts": manifest["counts"],
        "image_matching": {
            key: manifest["image_matching"][key]
            for key in (
                "matched_records",
                "unmatched_records",
                "multi_match_records",
                "unique_selected_images",
                "extracted_images",
            )
        },
        "duplicates": manifest["duplicates"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())