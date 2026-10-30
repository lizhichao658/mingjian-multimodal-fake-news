"""Export the single workspace UI or verify an exported/live copy."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src/mingjian/web/workbench.html"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Standalone HTML export to write or check")
    parser.add_argument("--check", action="store_true", help="Check the export without changing it")
    parser.add_argument("--live-url", help="Also compare the running HTTP UI with the source")
    args = parser.parse_args(argv)
    if args.output is None and not args.live_url:
        parser.error("provide --output or --live-url")
    if args.check and args.output is None:
        parser.error("--check requires --output")
    content = SOURCE.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if args.output is not None:
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != content:
                print("MISMATCH: standalone export differs from workspace source", file=sys.stderr)
                return 1
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(content)
    if args.live_url:
        try:
            with urllib.request.urlopen(args.live_url, timeout=10) as response:
                live = response.read(len(content) + 1)
        except (OSError, urllib.error.URLError) as exc:
            print(f"UNAVAILABLE: live UI cannot be read ({type(exc).__name__})", file=sys.stderr)
            return 1
        if live != content:
            print("MISMATCH: live UI differs from source; restart the local service", file=sys.stderr)
            return 1
    print(f"WORKBENCH_SYNC_OK sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
