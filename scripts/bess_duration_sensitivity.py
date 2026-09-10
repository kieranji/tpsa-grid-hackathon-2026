"""Bounded 85 MW BESS energy-duration sensitivity on the recent public price window.

Compares only 170, 310 and 510 DC MWh. Results are conditional screening cases,
not a size optimization, forward forecast, FASS qualification or secured revenue.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_market.dispatch import Battery, backtest
from bess_market.finance import reference_storage_derating
from bess_market.pipeline import digest, installed_cost, lifecycle, load_real_prices


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run(root: Path, output: Path):
    started = time.time()
    config_path = root / "configs/bess_market_public_benchmarks.json"
    cpi_path = root / "data/market/ireland_cpi_monthly.csv"
    prices_path = root / "data/market/semopx_dam_recent.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base_battery = Battery(**config["battery"])
    if base_battery.power_mw != 85 or base_battery.round_trip_efficiency != .85:
        raise ValueError("This bounded study requires 85 MW / 85% RTE")
    if base_battery.max_efc_per_day != 1 or config["lifetime_years"] != 15:
        raise ValueError("Inclusive-maintenance study requires one EFC/day and 15 years")
    if not config["cost_basis"]["fixed_opex_includes_augmentation"]:
        raise ValueError("Maintained-capacity assumption requires inclusive augmentation OPEX")
    selected_names = ("energy_only", "stack_t4_services4")
    cases = [next(case for case in config["market_cases"] if case["name"] == name) for name in selected_names]
    prices, money_factor = load_real_prices(prices_path, cpi_path)
    observed_hours = float(prices.duration_hours.sum())
    if observed_hours != 8712:
        raise ValueError(f"Expected 8712 recent observed hours; found {observed_hours}")
    output.mkdir(parents=True, exist_ok=True)
    code_paths = [root / "src/bess_market" / name for name in ("pipeline.py", "dispatch.py", "finance.py")]
    # Match the main pipeline exactly for optional 310-MWh cache reuse.
    pipeline_hashes = {str(path.relative_to(root)): digest(path) for path in code_paths + [config_path, cpi_path]}
    all_source_hashes = {**pipeline_hashes, str(Path(__file__).resolve().relative_to(root)): digest(__file__),
                         str(prices_path.relative_to(root)): digest(prices_path)}
    dataset = {
        "price_file_sha256": digest(prices_path),
        "start_utc": str(prices.timestamp_utc.iloc[0]),
        "end_utc_exclusive": str(prices.timestamp_utc.iloc[-1] + pd.Timedelta(hours=prices.duration_hours.iloc[-1])),
        "observed_hours": observed_hours,
        "cpi_proxy_hours": float(prices.loc[prices.cpi_last_observed_proxy, "duration_hours"].sum()),
        "actual_price_gaps_filled": False,
        "current_money_deflation_factor": money_factor,
    }
    manifest = {
        "study": "three_point_energy_duration_sensitivity_at_fixed_85_MW",
        "source_sha256": all_source_hashes, "config": config, "dataset": dataset,
        "energy_dc_mwh_candidates": [170, 310, 510], "cases": cases,
        "policies": ["causal", "daily_oracle"], "conditional_contract_term_years": 15,
        "maintained_capacity_by_inclusive_augmentation_assumption": True,
        "system_benefits_included_in_project_cashflow": False,
        "no_build_npv_eur": 0, "all_sizes_or_locations_optimized": False,
        "repeated_historical_pattern_is_not_forward_forecast": True,
        "capacity_and_services_secured_eur": 0,
        "source_dispatches": [],
    }
    financial_rows, backtest_rows, annual_rows = [], [], []
    # Larger/smaller cases execute while the main pipeline completes 310 MWh.
    for energy_dc_mwh in (170, 510, 310):
        battery = replace(base_battery, energy_dc_mwh=float(energy_dc_mwh))
        duration = battery.deliverable_ac_mwh / battery.power_mw
        factor = reference_storage_derating(duration, battery.power_mw)
        for policy in ("causal", "daily_oracle"):
            summaries = {}
            for case in cases:
                award = (battery.power_mw * battery.capability_fraction * factor *
                         config["capacity_award_fraction_if_conditional"] if case["capacity_reference"] else 0.0)
                key = (case["reserve_eur2024_per_mw_h"], award)
                identity_payload = {"source": all_source_hashes, "dataset": dataset, "battery": asdict(battery),
                                    "policy": policy, "key": key}
                identity = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()[:20]
                cache = output / f"dispatch_{energy_dc_mwh}_{policy}_{case['name']}_{identity}"
                summary = None
                provenance = "fresh_run"
                reused_source = None
                if cache.with_suffix(".json").exists() and cache.with_suffix(".csv.gz").exists():
                    summary = json.loads(cache.with_suffix(".json").read_text())
                    provenance = "matched_duration_study_cache"
                elif asdict(battery) == asdict(base_battery):
                    main_payload = json.dumps({"source": pipeline_hashes, "dataset": dataset,
                                               "policy": policy, "key": key}, sort_keys=True)
                    main_identity = hashlib.sha256(main_payload.encode()).hexdigest()[:20]
                    main_cache = root / "results/bess_market_public_benchmarks" / f"dispatch_recent_{policy}_{main_identity}"
                    if main_cache.with_suffix(".json").exists() and main_cache.with_suffix(".csv.gz").exists():
                        try:
                            summary = json.loads(main_cache.with_suffix(".json").read_text())
                        except json.JSONDecodeError:
                            summary = None  # A still-being-written result is never consumed.
                        if summary is not None:
                            provenance = "exact_main_pipeline_cache"
                            reused_source = str(main_cache.relative_to(root))
                if summary is None:
                    before = time.monotonic()
                    dispatch, summary = backtest(
                        prices, battery, policy, key[0], key[1],
                        config["capacity_strike_eur2024_mwh_sensitivity"] if award else None,
                    )
                    dispatch.to_csv(cache.with_suffix(".csv.gz"), index=False, compression="gzip")
                    write_json(cache.with_suffix(".json"), summary)
                    print(f"{energy_dc_mwh} MWh {policy} {case['name']}: "
                          f"cash EUR {summary['net_trading_and_services_eur']:,.0f}, "
                          f"{time.monotonic() - before:.1f}s", flush=True)
                else:
                    print(f"{energy_dc_mwh} MWh {policy} {case['name']}: {provenance}", flush=True)
                source = reused_source or str(cache.relative_to(root))
                manifest["source_dispatches"].append({
                    "energy_dc_mwh": energy_dc_mwh, "policy": policy, "case": case["name"],
                    "provenance": provenance, "dispatch_path_without_suffix": source,
                    "summary_sha256": digest(root / (source + ".json")),
                    "dispatch_sha256": digest(root / (source + ".csv.gz")),
                })
                summaries[case["name"]] = summary
                labels = {
                    "dataset": "recent", "energy_dc_mwh": energy_dc_mwh, "power_mw": 85,
                    "nameplate_dc_hours": energy_dc_mwh / 85,
                    "deliverable_ac_mwh_before_capability": battery.deliverable_ac_mwh,
                    "deliverable_ac_hours_before_capability": duration,
                    "reference_storage_derating_factor": factor,
                    "conditional_capacity_awarded_mw": award,
                    "policy": policy, "case": case["name"], "source_dispatch": source,
                    "source_identity": identity,
                }
                backtest_rows.append({**labels, **summary})
                for cost_name in config["cost_scenarios"]:
                    cost = installed_cost(config, battery, cost_name, money_factor)
                    term = 15 if case["name"] != "energy_only" else 0
                    result = lifecycle(config, battery, summary, summaries["energy_only"],
                                       case, cost, term, money_factor)
                    year_one = result["annual_ledger"][0]
                    financial_rows.append({
                        **labels, "cost_scenario": cost_name, "conditional_contract_years": term,
                        **cost, "real_discount_rate": config["real_discount_rate"],
                        "operating_years": config["lifetime_years"], "money_basis": config["money_basis"],
                        "annual_energy_sales_eur": year_one["energy_sales_eur"],
                        "annual_energy_purchases_eur": year_one["energy_purchase_eur"],
                        "annual_services_gross_eur": year_one["services_revenue_eur"],
                        "annual_capacity_gross_eur": year_one["capacity_revenue_eur"],
                        "annual_gross_receipts_eur": (
                            year_one["energy_sales_eur"] + year_one["services_revenue_eur"]
                            + year_one["capacity_revenue_eur"]),
                        "annual_energy_and_service_trading_cash_eur": (
                            summary["net_trading_and_services_eur"] * summary["annual_equivalent_multiplier"]),
                        "year_1_net_operating_cashflow_eur": year_one["net_operating_cashflow_eur"],
                        "year_1_net_cashflow_eur": year_one["net_cashflow_eur"],
                        "npv_eur": result["npv_eur"], "no_build_npv_eur": 0.0,
                        "npv_vs_no_build_eur": result["npv_eur"],
                        "preferred_to_no_build_by_npv": result["npv_eur"] > 0,
                        "irr": result["irr"], "irr_status": result["irr_status"],
                        "simple_payback_years": result["simple_payback_years"],
                        "discounted_payback_years": result["discounted_payback_years"],
                        "additional_annual_net_contract_eur": result["additional_break_even_net_contract"]["additional_annual_net_contract_eur"],
                        "grid_feasibility_verified": False,
                        "capacity_and_services_secured_eur": 0.0,
                        "service_price_is_hypothetical": True,
                        "maintained_capacity_by_inclusive_augmentation_assumption": True,
                        "future_pattern_repeat_is_not_forecast": True,
                    })
                    annual_rows.extend([{**labels, "cost_scenario": cost_name, **row}
                                        for row in result["annual_ledger"]])
                # Atomic progress allows read-only review of completed cases.
                write_json(output / "progress.json", {
                    "completed_dispatch_cases": len(backtest_rows), "required_dispatch_cases": 12,
                    "elapsed_seconds": time.time() - started, "last_case": labels,
                })
    # A changed source invalidates the run rather than disguising mixed versions.
    for path, expected in all_source_hashes.items():
        if digest(root / path) != expected:
            raise RuntimeError(f"Source changed while running: {path}; rerun under one version")
    columns = ["energy_dc_mwh", "policy", "case", "cost_scenario"]
    pd.DataFrame(financial_rows).sort_values(columns).to_csv(output / "project_scenarios.csv", index=False)
    pd.DataFrame(backtest_rows).sort_values(columns[:-1]).to_csv(output / "backtests.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(output / "annual_cashflows.csv", index=False)
    manifest["completed_dispatch_cases"] = len(backtest_rows)
    manifest["financial_scenarios"] = len(financial_rows)
    manifest["elapsed_seconds"] = time.time() - started
    write_json(output / "run_manifest.json", manifest)
    print(f"Complete: {len(financial_rows)} financial cases in {time.time() - started:.1f}s; {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bess_duration_sensitivity")
    arguments = parser.parse_args()
    run(ROOT, ROOT / arguments.output)

