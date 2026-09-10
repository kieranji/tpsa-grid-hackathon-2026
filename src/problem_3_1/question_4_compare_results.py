#!/usr/bin/env python3
"""Build an auditable presentation bundle from completed Q4 runs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from q4_techno_economic.comparison import write_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare fixed-line, DLR and distributed Q4 runs")
    parser.add_argument("--fixed-run", type=Path, required=True)
    parser.add_argument("--dlr-run", type=Path, required=True)
    parser.add_argument("--distributed-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = write_bundle(
        args.out,
        args.fixed_run,
        args.dlr_run,
        args.distributed_run,
    )
    print(f"Q4 comparison written to {output}")
    print("Read PRESENTATION_CALCULATIONS.md before quoting any result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
