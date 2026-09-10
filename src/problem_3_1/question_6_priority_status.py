#!/usr/bin/env python3
"""Run Problem 3.1 Q6 priority-status counterfactuals from repository root.

Examples:
    python -u src/problem_3_1/question_6_priority_status.py --check
    python -u src/problem_3_1/question_6_priority_status.py --smoke
    python -u src/problem_3_1/question_6_priority_status.py --run
"""
from pathlib import Path
import sys

from q6_priority_dispatch.pipeline import main


if __name__ == "__main__":
    try:
        raise SystemExit(main(Path(__file__).resolve().parents[2]))
    except (ValueError, FileNotFoundError, ImportError, KeyError, RuntimeError) as exc:
        print(f"Q6 ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
