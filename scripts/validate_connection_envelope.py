"""Full-island N-0 fixed-injection tests with other generators redispatched.

Holding +/-85 MW throughout a synthetic week deliberately ignores battery energy.
It tests whether the dispatch model can accommodate that power at the named bus;
it is not a chronological BESS schedule, N-1 study, or a connection agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/problem_3_1"))
from network_validation import network_input_fingerprint


def dense(network, table, attribute):
    static = getattr(network, table)
    frame = pd.DataFrame(np.tile(static[attribute].to_numpy(), (len(network.snapshots), 1)),
                         index=network.snapshots, columns=static.index, dtype=float)
    dynamic = getattr(network, table + "_t")[attribute]
    for column in dynamic.columns.intersection(frame.columns):
        frame[column] = dynamic[column].reindex(frame.index)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"Incomplete {table}.{attribute}")
    return frame


def metrics(n, label, bus, injection_mw):
    weights = n.snapshot_weightings.generators
    weather = n.generators.index[n.generators.carrier.isin(["wind", "solar"])]
    available = dense(n, "generators", "p_max_pu")[weather].mul(n.generators.loc[weather, "p_nom"])
    down = (available - n.generators_t.p[weather]).clip(lower=0).sum(axis=1)
    shed = n.generators.index[n.generators.carrier.eq("load shedding")]
    unserved = n.generators_t.p[shed].sum(axis=1)
    maximum_loading = pd.Series(0., index=n.snapshots)
    maximum_violation = pd.Series(0., index=n.snapshots)
    branch_rows = []
    for table in ("lines", "transformers"):
        static = getattr(n, table)
        limits = dense(n, table, "s_max_pu").mul(static.s_nom)
        flows = getattr(n, table + "_t").p0.abs()
        loading = flows / limits
        violation = (flows - limits).clip(lower=0)
        maximum_loading = pd.concat([maximum_loading, loading.max(axis=1)], axis=1).max(axis=1)
        maximum_violation = pd.concat([maximum_violation, violation.max(axis=1)], axis=1).max(axis=1)
        for branch in static.index:
            branch_rows.append({"case": label, "component": table, "branch": branch,
                                "max_loading_pu": float(loading[branch].max()),
                                "binding_hours": int((loading[branch] >= .999).sum()),
                                "max_thermal_violation_mw": float(violation[branch].max())})
    actual = (n.generators_t.p["ENVELOPE_FIXED_INJECTION"] if injection_mw else pd.Series(0., index=n.snapshots))
    injection_error = float((actual - injection_mw).abs().max())
    summary = {"case": label, "bus": bus, "fixed_injection_mw": injection_mw,
               "snapshots": len(n.snapshots), "solve_status": "ok/optimal",
               "solved_state_sha256": network_input_fingerprint(n),
               "objective_model_cost_units": float(n.objective),
               "renewable_dispatch_down_mwh": float(down.mul(weights).sum()),
               "unserved_mwh": float(unserved.mul(weights).sum()),
               "max_passive_loading_pu": float(maximum_loading.max()),
               "max_thermal_violation_mw": float(maximum_violation.max()),
               "max_fixed_injection_error_mw": injection_error,
               "thermal_and_fixed_injection_constraints_pass": bool(maximum_violation.max() <= 1e-4 and injection_error <= 1e-4)}
    hourly = pd.DataFrame({"case": label, "injection_mw": actual,
                          "unserved_mw": unserved, "renewable_dispatch_down_mw": down,
                          "max_passive_loading_pu": maximum_loading,
                          "max_thermal_violation_mw": maximum_violation})
    return summary, hourly, pd.DataFrame(branch_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, default=ROOT / "data/participant-kit/networks/WP2033_all-island.nc")
    parser.add_argument("--bus", default="51911")
    parser.add_argument("--power-mw", type=float, default=85.)
    parser.add_argument("--output", type=Path, default=ROOT / "results/network_validation/WP2033/connection_envelope")
    args = parser.parse_args()
    if not np.isfinite(args.power_mw) or args.power_mw <= 0:
        parser.error("power must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for label, injection in (("baseline", 0.), ("fixed_export", args.power_mw), ("fixed_import", -args.power_mw)):
        n = pypsa.Network(args.network)
        if args.bus not in n.buses.index:
            raise KeyError(f"Exact source bus {args.bus} absent")
        identity = network_input_fingerprint(n)
        if injection:
            n.add("Carrier", "connection_envelope")
            n.add("Generator", "ENVELOPE_FIXED_INJECTION", bus=args.bus,
                  carrier="connection_envelope", p_nom=abs(injection),
                  p_min_pu=float(np.sign(injection)), p_max_pu=float(np.sign(injection)),
                  marginal_cost=0.)
        print(f"Solving {label}: {injection:+.1f} MW at bus {args.bus}", flush=True)
        status, condition = n.optimize(solver_name="highs", solver_options={"output_flag": False}, progress=False)
        if status != "ok" or condition != "optimal":
            raise RuntimeError(f"{label}: {status}/{condition}")
        summary, hourly, branches = metrics(n, label, args.bus, injection)
        summary["source_input_sha256"] = identity
        summaries.append(summary)
        hourly.to_csv(args.output / f"{label}_hourly.csv", index_label="snapshot")
        branches.to_csv(args.output / f"{label}_branches.csv", index=False)
        pd.DataFrame(summaries).to_csv(args.output / "case_summary.csv", index=False)
        del n
    result = pd.DataFrame(summaries)
    for field in ("objective_model_cost_units", "renewable_dispatch_down_mwh", "unserved_mwh"):
        result["delta_" + field] = result[field] - result.loc[0, field]
    result["n0_envelope_without_additional_shedding_pass"] = (
        result.thermal_and_fixed_injection_constraints_pass & (result.delta_unserved_mwh <= 1e-3))
    result.to_csv(args.output / "case_summary.csv", index=False)
    statement = {"source_file_sha256": hashlib.sha256(args.network.read_bytes()).hexdigest(),
                 "bus": args.bus, "bus_mapping": "51911=CROAGHONAGH in the official source network" if args.bus == "51911" else "explicit source bus ID",
                 "claim": "N-0 fixed-power accommodation with full-island redispatch in the supplied synthetic168h model",
                 "battery_energy_soc_and_degradation_modelled": False,
                 "n_minus_1_security_validated": False,
                 "chronological_merchant_schedule_validated": False,
                 "formal_grid_connection_right_verified": False,
                 "unused_headroom_or_curtailment_free_access_validated": False,
                 "other_generators_may_be_redispatched_and_renewables_curtailed": True,
                 "cost_unit_interpretation": "official placeholder dispatch objective; not project income",
                 "all_cases_pass": bool(result.n0_envelope_without_additional_shedding_pass.all())}
    (args.output / "interpretation.json").write_text(json.dumps(statement, indent=2), encoding="utf-8")
    print(result.to_string(index=False), flush=True)
    return 0 if statement["all_cases_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
