"""Diagnose a merchant candidate inside the supplied 15-node technical model.

These system-cost dispatch solves do not validate merchant operation or a grid
connection. The output keeps that limitation explicit and checks the same BESS
with and without the target line constraint at beginning and end of life.
"""
from pathlib import Path
import argparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/problem_3_1"))
import question_2_battery_siting_sizing as q2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-mw", type=float, default=85.)
    parser.add_argument("--energy-mwh", type=float, default=310.)
    parser.add_argument("--rte", type=float, default=.85)
    parser.add_argument("--site", default="Croaghonagh")
    parser.add_argument("--target-line", default="5041-17010-2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/network_validation/WP2033/supplied_15_node_candidate_diagnostics.csv")
    args = parser.parse_args()
    context = q2.build_study_context("WP2033", "north-west", args.target_line, "line")
    rows = []
    for soh in (1., .8):
        model = q2.BatteryModel("merchant_assumptions_diagnostic", .1, .9, args.rte, soh)
        row, _, _ = q2.run_battery_trial(context, [args.site], args.power_mw,
            args.energy_mwh, model, .99, "benchmark",
            q2.TrialCache(Path("unused"), False), force_solve=True, return_network=True)
        row["network_interpretation"] = "supplied_15_node_static_boundary_model_only"
        row["input_network_sha256"] = context.input_identity
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result[["state_of_health_pct", "target_recovery_pct", "target_binding_hours",
                  "same_bess_residual_dispatch_down_mwh", "same_bess_target_residual_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
