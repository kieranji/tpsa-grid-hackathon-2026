"""Revalue unchanged dispatch evidence without repeating the MILP solves.

Only reporting/finance code may change. Battery configuration, price/CPI inputs
and dispatch implementation must match the producing manifest exactly.
"""
from pathlib import Path
import hashlib
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bess_market.dispatch import Battery
from bess_market.pipeline import digest, installed_cost, lifecycle, toll_floor


def main():
    out = ROOT / "results/bess_market_public_benchmarks"
    manifest_path = out / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    hashes = {key.replace("\\", "/"): value for key, value in manifest["source_sha256"].items()}
    for path in ("configs/bess_market_public_benchmarks.json", "data/market/ireland_cpi_monthly.csv",
                 "src/bess_market/dispatch.py"):
        if digest(ROOT / path) != hashes[path]:
            raise ValueError(f"Cannot reuse dispatch: changed {path}")
    for dataset, data in manifest["datasets"].items():
        if digest(ROOT / f"data/market/semopx_dam_{dataset}.csv") != data["price_file_sha256"]:
            raise ValueError(f"Cannot reuse changed prices: {dataset}")
    config = manifest["config"]
    battery = Battery(**config["battery"])
    original = pd.read_csv(out / "project_scenarios.csv")
    backtests = pd.read_csv(out / "backtests.csv")
    projects, cashflows = [], []
    for row in original.to_dict("records"):
        labels = {key: row[key] for key in ("dataset", "policy", "case", "cost_scenario", "contract_years")}
        subset = backtests[(backtests.dataset == row["dataset"]) & (backtests.policy == row["policy"])]
        selected = subset[subset["case"] == row["case"]].iloc[0].to_dict()
        energy_only = subset[subset["case"] == "energy_only"].iloc[0].to_dict()
        case = next(c for c in config["market_cases"] if c["name"] == row["case"])
        factor = manifest["datasets"][row["dataset"]]["current_money_deflation_factor"]
        cost = installed_cost(config, battery, row["cost_scenario"], factor)
        result = lifecycle(config, battery, selected, energy_only, case, cost, int(row["contract_years"]), factor)
        # Current change adds a toll hurdle and validation only; verify the
        # existing central financial conclusion before replacing any output.
        if abs(result["npv_eur"] - row["npv_eur"]) > 1e-5:
            raise AssertionError("Revaluation unexpectedly changed NPV")
        projects.append({**row, "alternative_whole_asset_toll_floor_eur_year": toll_floor(config, cost)})
        cashflows.extend([{**labels, **year} for year in result["annual_ledger"]])
    pd.DataFrame(projects).to_csv(out / "project_scenarios.csv", index=False)
    pd.DataFrame(cashflows).to_csv(out / "annual_cashflows.csv", index=False)
    manifest["finance_revaluation"] = {
        "existing_npv_values_reproduced": True,
        "dispatch_configuration_and_price_hashes_verified": True,
        "source_sha256": {p: digest(ROOT / p) for p in
            ("src/bess_market/pipeline.py", "src/bess_market/finance.py", "scripts/revalue_bess_cashflows.py")},
        "note": "Added alternative whole-asset toll hurdle; no rescheduling or change to existing NPV values."}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Reproduced {len(projects)} project NPVs; added toll hurdles.")


if __name__ == "__main__":
    main()
