"""Run public-price backtests and explicitly conditional investment sensitivities."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import pandas as pd

from .dispatch import Battery, backtest, validate_prices
from .finance import (AnnualOperatingInputs, CAPACITY_REFERENCE_PRICES,
                      reference_storage_derating, project_cashflows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_real_prices(price_path: Path, cpi_path: Path):
    df = validate_prices(pd.read_csv(price_path))
    cpi = pd.read_csv(cpi_path)
    mapping = dict(zip(cpi.month.astype(str), cpi.cpi_dec2023_100.astype(float)))
    base = 100.7  # CSO annual 2024, Dec 2023 = 100.
    latest = max(mapping)
    month = pd.to_datetime(df.market_date).dt.strftime("%Y-%m")
    used = month.map(mapping)
    # Only months after the last publication can use the explicit last-observed proxy.
    if used[month <= latest].isna().any():
        raise ValueError("Missing published historical CPI; no interpolation")
    df["cpi_last_observed_proxy"] = used.isna()
    used = used.fillna(mapping[latest])
    df["price_nominal_eur_mwh"] = df.price_eur_mwh
    df["price_eur_mwh"] = df.price_eur_mwh * base / used
    df["cpi_index_used"] = used
    return df, base/mapping[latest]


def installed_cost(config: dict, battery: Battery, name: str, current_money_factor: float):
    basis, scenario = config["cost_basis"], config["cost_scenarios"][name]
    # Duration scaling is an inference from the 2024 power/energy split, not an
    # NREL quote for this exact 2.69-hour DC-nameplate project.
    starting_usd = (battery.deliverable_ac_mwh*1000*basis["nrel_2024_energy_usd_per_usable_ac_kwh"]
                    +battery.power_mw*1000*basis["nrel_2024_power_usd_per_kw"])
    plant = (starting_usd*scenario["nrel_2033_usd_per_ac_kwh_4h"]
             /basis["nrel_2024_four_hour_usd_per_usable_ac_kwh"]
             /basis["usd_per_eur_2024_average"]*scenario["ireland_installation_multiplier"])
    return {"installed_plant_eur": plant,
            "connection_allowance_eur": scenario["connection_allowance_eur2024"],
            "upfront_capex_eur": plant+scenario["connection_allowance_eur2024"],
            "annual_fixed_opex_including_augmentation_eur": plant*basis["fixed_opex_fraction_of_installed_plant"],
            "annual_gtuos_proxy_eur": scenario["gtuos_eur2026_per_kw_year"]*1000*battery.power_mw*current_money_factor,
            "annual_house_load_incremental_cost_eur": basis["house_load_and_incremental_fees_eur_year"]}


def toll_floor(config, cost):
    """Alternative whole-asset toll: counterparty receives ALL market rights.

    Owner pays the listed fixed/grid/house-load costs. Energy purchases, trading
    income and service/capacity income belong to the toll counterparty and are
    excluded. The result is a zero-NPV fee hurdle, not an offered contract.
    """
    years = config["lifetime_years"]
    result = project_cashflows(cost["upfront_capex_eur"], [AnnualOperatingInputs(
        year=y, fixed_opex_eur=cost["annual_fixed_opex_including_augmentation_eur"],
        fixed_opex_includes_replacement_and_augmentation=True,
        generator_tuos_eur=cost["annual_gtuos_proxy_eur"],
        house_load_grid_cost_eur=cost["annual_house_load_incremental_cost_eur"],
        decommissioning_cost_eur=(cost["upfront_capex_eur"]*config["cost_basis"]["end_of_life_decommission_fraction"]
                                  if y == years else 0)) for y in range(1, years+1)], config["real_discount_rate"])
    return result["additional_break_even_net_contract"]["additional_annual_net_contract_eur"]


def lifecycle(config, battery, selected, energy_only, case, cost, contract_years, money_factor):
    years = config["lifetime_years"]
    factor = reference_storage_derating(battery.deliverable_ac_mwh/battery.power_mw, battery.power_mw)
    award = battery.power_mw*battery.capability_fraction*factor*config["capacity_award_fraction_if_conditional"]
    cap_price = (CAPACITY_REFERENCE_PRICES[case["capacity_reference"]]["eur_per_derated_mw_year"]*money_factor
                 if case["capacity_reference"] else 0)
    annual = []
    for year in range(1, years+1):
        active = year <= contract_years
        s = selected if active else energy_only
        scale = s["annual_equivalent_multiplier"]
        annual.append(AnnualOperatingInputs(
            year=year,
            energy_sales_eur=s["energy_sales_eur"]*scale,
            energy_purchase_eur=s["energy_purchases_eur"]*scale,
            services_revenue_eur=s["service_gross_eur"]*scale,
            service_nonperformance_cost_eur=s["service_deductions_eur"]*scale,
            capacity_revenue_eur=award*cap_price if active else 0,
            capacity_difference_charges_eur=s["capacity_difference_charge_eur"]*scale,
            fees_eur=s["execution_fees_eur"]*scale,
            other_grid_cost_eur=s["import_adder_eur"]*scale,
            fixed_opex_eur=cost["annual_fixed_opex_including_augmentation_eur"],
            fixed_opex_includes_replacement_and_augmentation=True,
            discharge_throughput_mwh=s["discharge_mwh"]*scale,
            generator_tuos_eur=cost["annual_gtuos_proxy_eur"],
            house_load_grid_cost_eur=cost["annual_house_load_incremental_cost_eur"],
            decommissioning_cost_eur=(cost["upfront_capex_eur"]*config["cost_basis"]["end_of_life_decommission_fraction"]
                                      if year == years else 0)))
    result = project_cashflows(cost["upfront_capex_eur"], annual, config["real_discount_rate"])
    result.update({"conditional_contract_years": contract_years,
                   "capacity_reference_derating": factor,
                   "capacity_reference_mw_if_awarded": award if cap_price else 0,
                   "money_basis": config["money_basis"],
                   "maintained_capacity_by_inclusive_augmentation_assumption": True,
                   "future_historical_energy_pattern_repeat_is_not_forecast": True,
                   "secured_capacity_and_services_eur": 0,
                   "site_grid_feasibility_verified": False})
    return result


def run(root: Path, output: Path, datasets=("2023", "recent")):
    config_path = root/"configs/bess_market_public_benchmarks.json"
    config = json.loads(config_path.read_text())
    battery = Battery(**config["battery"])
    if not config["cost_basis"]["fixed_opex_includes_augmentation"]:
        raise ValueError("Maintained-capacity screening requires inclusive augmentation FOM; use an explicit degradation model otherwise")
    if config["lifetime_years"] > 15 or battery.max_efc_per_day > 1.0:
        raise ValueError("NREL maintained-capacity premise supports this 15-year, at-most-one-EFC/day screen only")
    output.mkdir(parents=True, exist_ok=True)
    cpi_path = root/"data/market/ireland_cpi_monthly.csv"
    code_paths = [Path(__file__), Path(__file__).with_name("dispatch.py"), Path(__file__).with_name("finance.py")]
    source_hashes = {str(p.relative_to(root)): digest(p) for p in code_paths+[config_path,cpi_path]}
    run_manifest = {"config": config, "source_sha256": source_hashes, "datasets": {},
                    "battery": asdict(battery), "deliverable_ac_mwh": battery.deliverable_ac_mwh,
                    "deliverable_hours": battery.deliverable_ac_mwh/battery.power_mw}
    backtest_rows, project_rows, cashflow_rows = [], [], []
    for dataset in datasets:
        prices_path = root/f"data/market/semopx_dam_{dataset}.csv"
        prices, money_factor = load_real_prices(prices_path, cpi_path)
        run_manifest["datasets"][dataset] = {"price_file_sha256": digest(prices_path),
            "start_utc": str(prices.timestamp_utc.iloc[0]),
            "end_utc_exclusive": str(prices.timestamp_utc.iloc[-1]+pd.Timedelta(hours=prices.duration_hours.iloc[-1])),
            "observed_hours": float(prices.duration_hours.sum()),
            "cpi_proxy_hours": float(prices.loc[prices.cpi_last_observed_proxy, "duration_hours"].sum()),
            "actual_price_gaps_filled": False, "current_money_deflation_factor": money_factor}
        for policy in ("causal", "daily_oracle"):
            summaries, physical_cache = {}, {}
            cases = config["market_cases"] if dataset in ("2023", "recent") else config["market_cases"][:1]
            for case in cases:
                awarded = (battery.power_mw*battery.capability_fraction*reference_storage_derating(battery.deliverable_ac_mwh/battery.power_mw)
                           *config["capacity_award_fraction_if_conditional"] if case["capacity_reference"] else 0.)
                key = (case["reserve_eur2024_per_mw_h"], awarded)
                if key not in physical_cache:
                    payload = json.dumps({"source": source_hashes, "dataset": run_manifest["datasets"][dataset],
                                          "policy": policy, "key": key}, sort_keys=True)
                    identity = hashlib.sha256(payload.encode()).hexdigest()[:20]
                    cache = output/f"dispatch_{dataset}_{policy}_{identity}"
                    if cache.with_suffix(".json").exists() and cache.with_suffix(".csv.gz").exists():
                        summary = json.loads(cache.with_suffix(".json").read_text())
                    else:
                        start = time.monotonic()
                        dispatch, summary = backtest(prices, battery, policy, key[0], key[1],
                            config["capacity_strike_eur2024_mwh_sensitivity"] if awarded else None)
                        dispatch.to_csv(cache.with_suffix(".csv.gz"), index=False, compression="gzip")
                        cache.with_suffix(".json").write_text(json.dumps(summary, indent=2))
                        print(f"{dataset} {policy} reserve={key[0]} capacity={awarded:.3f}: "
                              f"EUR {summary['net_trading_and_services_eur']:,.0f}, {time.monotonic()-start:.1f}s", flush=True)
                    physical_cache[key] = (summary, cache.name)
                summary, dispatch_file = physical_cache[key]
                summaries[case["name"]] = summary
                backtest_rows.append({"dataset": dataset, "case": case["name"], "dispatch_file": dispatch_file, **summary})
                for cost_name in config["cost_scenarios"]:
                    cost = installed_cost(config, battery, cost_name, money_factor)
                    terms = config["contract_term_cases_years"] if case["name"] != "energy_only" else [0]
                    for term in terms:
                        result = lifecycle(config, battery, summary, summaries["energy_only"], case, cost, term, money_factor)
                        labels = {"dataset": dataset, "policy": policy, "case": case["name"], "cost_scenario": cost_name,
                                  "contract_years": term}
                        project_rows.append({**labels, **cost,
                            "alternative_whole_asset_toll_floor_eur_year": toll_floor(config, cost),
                            **{k: v for k, v in result.items() if not isinstance(v, (dict, list))},
                            "additional_annual_net_contract_eur": result["additional_break_even_net_contract"]["additional_annual_net_contract_eur"],
                            "year_1_net_operating_cashflow_eur": result["annual_ledger"][0]["net_operating_cashflow_eur"]})
                        cashflow_rows.extend([{**labels, **row} for row in result["annual_ledger"]])
    pd.DataFrame(backtest_rows).to_csv(output/"backtests.csv", index=False)
    pd.DataFrame(project_rows).to_csv(output/"project_scenarios.csv", index=False)
    pd.DataFrame(cashflow_rows).to_csv(output/"annual_cashflows.csv", index=False)
    (output/"run_manifest.json").write_text(json.dumps(run_manifest, indent=2))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/bess_market_public_benchmarks")
    parser.add_argument("--datasets", nargs="+", default=["2023", "recent"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    run(root, root/args.output, tuple(args.datasets))


if __name__ == "__main__":
    main()
