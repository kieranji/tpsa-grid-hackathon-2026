#!/usr/bin/env python3
"""Run the Problem 3.1 Q5 wind-farm shift-factor workflow from repository root.

Examples:
    python -u src/problem_3_1/question_5_shift_factors.py --check
    python -u src/problem_3_1/question_5_shift_factors.py --smoke
    python -u src/problem_3_1/question_5_shift_factors.py --run
"""
from pathlib import Path
import sys

from q5_shift_factors.pipeline import main


if __name__ == "__main__":
    try:
        raise SystemExit(main(Path(__file__).resolve().parents[2]))
    except (ValueError, FileNotFoundError, ImportError, KeyError, RuntimeError) as exc:
        print(f"Q5 ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
