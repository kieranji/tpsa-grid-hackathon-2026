#!/usr/bin/env python3
"""Run from repository root. Start with --check, then --smoke.

python -u src/problem_3_1/question_4_techno_economics.py --check
python -u src/problem_3_1/question_4_techno_economics.py --smoke --allow-illustrative-economics
"""
from pathlib import Path
import sys
from q4_techno_economic.pipeline import main

if __name__ == "__main__":
    try:
        raise SystemExit(main(Path(__file__).resolve().parents[2]))
    except (ValueError, FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"Q4 ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
