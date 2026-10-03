"""Command-line entry points for MingJian."""

from __future__ import annotations

import importlib.util
import sys


def main() -> int:
    """Run a dependency-aware smoke check."""

    required = ["numpy", "pandas", "PIL"]
    optional = ["torch", "transformers"]
    print("MingJian environment check")
    for name in required:
        print(f"  required  {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
    for name in optional:
        print(f"  optional  {name}: {'OK' if importlib.util.find_spec(name) else 'MISSING'}")
    if any(importlib.util.find_spec(name) is None for name in ["numpy"]):
        print("Please install project dependencies: pip install -e '.[dev]'")
        return 1
    print("Base dependencies are sufficient for schema and metric tests.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())