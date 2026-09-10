#!/usr/bin/env python3
"""Run the Problem 3.2 nationwide constraint-group workflow."""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from problem_3_2.q32_constraint_groups.pipeline import main


if __name__ == "__main__":
    try:
        raise SystemExit(main(Path(__file__).resolve().parents[2]))
    except (ValueError, FileNotFoundError, ImportError, KeyError, RuntimeError) as exc:
        print(f"Problem 3.2 ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
