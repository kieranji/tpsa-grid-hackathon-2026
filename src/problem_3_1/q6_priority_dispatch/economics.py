"""Auditable bridge from Q6 allocation effects to existing Q4 economics."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


Q4_COLUMNS = [
    "case_id",
    "technology",
    "site_count",
    "sites",
    "allocations",
    "battery_mw",
    "battery_mwh",
    "lifetime_net_renewable_gain_mwh",
    "year_1_net_cashflow_eur",
    "year_1_project_revenue_eur",
    "year_1_operating_cost_eur",
    "lifetime_project_revenue_eur",
    "lifetime_energy_purchase_eur",
    "lifetime_fixed_and_variable_om_eur",
    "lifetime_maintenance_capex_eur",
    "initial_capex_eur",
    "lifetime_system_dispatch_cost_saving_eur",
    "pv_system_dispatch_cost_saving_eur",
    "system_saving_included_in_project_cashflow",
    "npv_eur",
    "simple_payback_years",
    "discounted_payback_years",
    "additional_break_even_payment_eur_kw_year",
    "assumptions_status",
]


def annuity_present_value_factor(rate: float, years: int) -> float:
    if years < 1 or not 0 <= rate < 1:
        raise ValueError("Use a positive horizon and fractional non-negative discount rate")
    if abs(rate) <= 1e-14:
        return float(years)
    return float((1.0 - (1.0 + rate) ** -years) / rate)


def q4_case_rows(
    root: Path,
    size_run: Path,
    distributed_run: Path,
    size_pairs: Iterable[tuple[float, float]],
) -> tuple[pd.DataFrame, list[Path]]:
    """Read canonical Q4 rankings without altering their project cash flows."""
    sources: list[Path] = []
    rows: list[pd.DataFrame] = []
    size_path = root / size_run / "01_project_ranking.csv"
    distributed_path = root / distributed_run / "01_project_ranking.csv"
    for family, path in (("size", size_path), ("distributed", distributed_path)):
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        missing = sorted(set(Q4_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} lacks Q4 economic columns: {missing}")
        if family == "size":
            keep = pd.Series(False, index=frame.index)
            for mw, mwh in size_pairs:
                keep |= np.isclose(frame["battery_mw"], mw) & np.isclose(
                    frame["battery_mwh"], mwh
                )
            selected = frame.loc[keep, Q4_COLUMNS].copy()
        else:
            selected = frame.loc[
                np.isclose(frame["battery_mw"], 45.0)
                & np.isclose(frame["battery_mwh"], 220.0)
                & frame["site_count"].isin([1, 2, 3, 4]),
                Q4_COLUMNS,
            ].copy()
        selected.insert(0, "comparison_family", family)
        selected.insert(1, "source_ranking_file", str(path))
        rows.append(selected)
        sources.append(path)
    combined = pd.concat(rows, ignore_index=True)
    no_build = {column: np.nan for column in combined.columns}
    no_build.update(
        {
            "comparison_family": "size_and_distributed_reference",
            "source_ranking_file": "constructed zero-investment comparator",
            "case_id": "NO_BUILD",
            "technology": "no_build",
            "site_count": 0,
            "sites": "-",
            "allocations": "",
            "battery_mw": 0.0,
            "battery_mwh": 0.0,
            "lifetime_net_renewable_gain_mwh": 0.0,
            "year_1_net_cashflow_eur": 0.0,
            "year_1_project_revenue_eur": 0.0,
            "year_1_operating_cost_eur": 0.0,
            "lifetime_project_revenue_eur": 0.0,
            "lifetime_energy_purchase_eur": 0.0,
            "lifetime_fixed_and_variable_om_eur": 0.0,
            "lifetime_maintenance_capex_eur": 0.0,
            "initial_capex_eur": 0.0,
            "lifetime_system_dispatch_cost_saving_eur": 0.0,
            "pv_system_dispatch_cost_saving_eur": 0.0,
            "system_saving_included_in_project_cashflow": False,
            "npv_eur": 0.0,
            "simple_payback_years": 0.0,
            "discounted_payback_years": 0.0,
            "additional_break_even_payment_eur_kw_year": 0.0,
            "assumptions_status": "zero_investment_reference",
        }
    )
    combined = pd.concat([pd.DataFrame([no_build]), combined], ignore_index=True)
    combined["q6_wind_value_included_in_bess_project_revenue"] = False
    combined["q6_system_saving_included_in_bess_project_revenue"] = False
    combined["priority_aware_bess_redispatch_performed"] = False
    combined["bridge_interpretation"] = (
        "Q4 project cash flow retained; Q6 wind-owner allocation is a separate ledger"
    )
    return combined, sources


def wind_value_distribution(
    impacts: pd.DataFrame,
    *,
    block_hours: float,
    annual_hours: float,
    energy_value_eur_mwh: float,
    discount_rate: float,
    years: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Value Q6 farm energy redistribution without calling it BESS revenue."""
    if block_hours <= 0 or annual_hours <= 0 or energy_value_eur_mwh < 0:
        raise ValueError("Invalid Q6 economic bridge assumptions")
    factor = annual_hours / block_hours
    pv_factor = annuity_present_value_factor(discount_rate, years)
    result = impacts.copy()
    result["dispatched_energy_change_mwh"] = -result["additional_dispatch_down_mwh"]
    result["annualised_dispatched_energy_change_mwh"] = (
        result["dispatched_energy_change_mwh"] * factor
    )
    result["energy_value_eur_mwh"] = float(energy_value_eur_mwh)
    result["annual_energy_value_transfer_eur"] = (
        result["annualised_dispatched_energy_change_mwh"] * energy_value_eur_mwh
    )
    result["annuity_present_value_factor"] = pv_factor
    result["pv_energy_value_transfer_eur"] = (
        result["annual_energy_value_transfer_eur"] * pv_factor
    )
    result["data_class"] = "COMMERCIAL_ASSUMPTION"
    result["included_in_bess_npv"] = False
    result["included_in_system_dispatch_cost"] = False
    result["interpretation_limit"] = (
        "Energy-only owner allocation proxy; no contract, imbalance or curtailment compensation data supplied"
    )
    audit = {
        "annualisation_factor": float(factor),
        "energy_value_eur_mwh": float(energy_value_eur_mwh),
        "discount_rate": float(discount_rate),
        "years": int(years),
        "annuity_present_value_factor": float(pv_factor),
        "aggregate_annual_energy_value_transfer_eur": float(
            result["annual_energy_value_transfer_eur"].sum()
        ),
        "aggregate_pv_energy_value_transfer_eur": float(
            result["pv_energy_value_transfer_eur"].sum()
        ),
    }
    return result, audit


def collect_q4_sensitivity(
    root: Path,
    run_directories: Iterable[Path],
    selected_case_ids: set[str],
) -> pd.DataFrame:
    """Collect existing fixed-path Q4 sensitivities for linked cases."""
    rows: list[pd.DataFrame] = []
    for run in run_directories:
        cases = root / run / "cases"
        if not cases.exists():
            continue
        for case_id in sorted(selected_case_ids):
            path = cases / case_id / "financial_sensitivity.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frame.insert(0, "case_id", case_id)
            frame.insert(1, "source_file", str(path))
            frame["physical_dispatch_rerun"] = False
            frame["q6_status_effect_included"] = False
            rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
