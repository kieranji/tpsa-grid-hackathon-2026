"""Auditable cross-run comparison for Q4 capacity, distributed BESS and Q3 DLR."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_SUMMARY_COLUMNS = {
    "case_id", "technology", "site_count", "sites", "allocations",
    "wind_mw", "battery_mw", "battery_mwh",
    "lifetime_net_renewable_gain_mwh", "lifetime_project_revenue_eur",
    "lifetime_energy_purchase_eur", "lifetime_fixed_and_variable_om_eur",
    "lifetime_maintenance_capex_eur", "initial_capex_eur",
    "lifetime_decommission_eur", "lifetime_residual_eur",
    "lifetime_net_project_cashflow_eur", "lifecycle_cost_pv_eur",
    "lifetime_system_dispatch_cost_saving_eur",
    "pv_system_dispatch_cost_saving_eur",
    "system_saving_included_in_project_cashflow", "npv_eur",
    "simple_payback_years", "discounted_payback_years",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_run(directory: Path) -> tuple[pd.DataFrame, dict, dict]:
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    ranking_path = directory / "checkpoint_ranking.csv"
    baseline_path = directory / "00_baseline_metrics.json"
    for path in (manifest_path, ranking_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(ranking_path)
    missing = REQUIRED_SUMMARY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{directory}: summary is missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{directory}: no completed Q4 cases")
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{directory}: duplicate completed case IDs")
    return frame, manifest, baseline


def _same_basis(fixed_manifest: dict, dlr_manifest: dict, distributed_manifest: dict) -> None:
    manifests = (fixed_manifest, dlr_manifest, distributed_manifest)
    fixed_network = fixed_manifest["network"]
    dlr_network = dlr_manifest["network"]
    distributed_network = distributed_manifest["network"]
    if fixed_network["network_hash"] != distributed_network["network_hash"]:
        raise ValueError("Fixed-line and distributed runs use different network hashes")
    # Applying DLR intentionally changes the solved network hash. Verify the immutable
    # official source identity/topology/time basis instead of requiring that post-DLR
    # hash to equal the fixed-line network.
    identity_keys = (
        "origin", "network_meta", "buses", "lines", "transformers", "links", "snapshots"
    )
    for key in identity_keys:
        if json.dumps(fixed_network[key], sort_keys=True) != json.dumps(
            dlr_network[key], sort_keys=True
        ):
            raise ValueError(f"DLR run differs in base-network field {key!r}")
    if not dlr_network.get("dlr_source_sha256"):
        raise ValueError("DLR run has no source-file fingerprint")
    keys = ("annualisation", "prices", "costs", "battery", "real_discount_rate", "currency_basis")
    for key in keys:
        values = [json.dumps(m["config"][key], sort_keys=True) for m in manifests]
        if len(set(values)) != 1:
            raise ValueError(f"Runs are not comparable: config field {key!r} differs")
    if fixed_manifest["network"].get("dlr_label", "none") != "none":
        raise ValueError("Fixed-line run unexpectedly contains DLR")
    if dlr_manifest["network"].get("dlr_label", "none") == "none":
        raise ValueError("DLR run has no DLR audit label")


def _with_total_cost(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["lifetime_total_project_cost_undiscounted_eur"] = (
        out["initial_capex_eur"]
        + out["lifetime_energy_purchase_eur"]
        + out["lifetime_fixed_and_variable_om_eur"]
        + out["lifetime_maintenance_capex_eur"]
        + out["lifetime_decommission_eur"]
        - out["lifetime_residual_eur"]
    )
    return out


def capacity_dlr_comparison(
    fixed: pd.DataFrame,
    dlr: pd.DataFrame,
    fixed_manifest: dict,
    fixed_baseline: dict,
    dlr_baseline: dict,
) -> tuple[pd.DataFrame, dict]:
    keys = ["technology", "site_count", "sites", "wind_mw", "battery_mw", "battery_mwh"]
    metrics = [
        "lifetime_net_renewable_gain_mwh", "lifetime_project_revenue_eur",
        "lifetime_total_project_cost_undiscounted_eur",
        "lifetime_system_dispatch_cost_saving_eur", "npv_eur",
        "simple_payback_years", "discounted_payback_years",
    ]
    fixed = _with_total_cost(fixed)
    dlr = _with_total_cost(dlr)
    left = fixed[keys + metrics].rename(columns={c: f"fixed_{c}" for c in metrics})
    right = dlr[keys + metrics].rename(columns={c: f"dlr_{c}" for c in metrics})
    out = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not out["_merge"].eq("both").all():
        raise ValueError("Fixed and DLR runs do not contain the same capacity designs")
    out = out.drop(columns="_merge")
    snapshots = int(fixed_manifest["network"]["snapshots"])
    hours_per_year = float(fixed_manifest["config"]["annualisation"]["hours_per_year"])
    factor = hours_per_year / snapshots
    years = int(fixed_manifest["config"]["years"])
    block_gain = (
        float(fixed_baseline["renewable_dispatch_down_mwh"])
        - float(dlr_baseline["renewable_dispatch_down_mwh"])
    )
    dlr_lifetime_gain = block_gain * factor * years
    out["dlr_standalone_lifetime_gain_mwh"] = dlr_lifetime_gain
    out["dlr_plus_bess_total_gain_vs_fixed_mwh"] = (
        dlr_lifetime_gain + out["dlr_lifetime_net_renewable_gain_mwh"]
    )
    out["dlr_change_in_bess_incremental_gain_mwh"] = (
        out["dlr_lifetime_net_renewable_gain_mwh"]
        - out["fixed_lifetime_net_renewable_gain_mwh"]
    )
    out["dlr_change_in_conditional_bess_npv_eur"] = (
        out["dlr_npv_eur"] - out["fixed_npv_eur"]
    )
    out["dlr_cost_included_in_conditional_bess_npv"] = False
    out = out.sort_values(["technology", "battery_mw", "battery_mwh"]).reset_index(drop=True)
    audit = {
        "block_hours": snapshots,
        "hours_per_year": hours_per_year,
        "annualisation_factor": factor,
        "project_years": years,
        "fixed_baseline_dispatch_down_mwh_per_block": float(
            fixed_baseline["renewable_dispatch_down_mwh"]
        ),
        "dlr_baseline_dispatch_down_mwh_per_block": float(
            dlr_baseline["renewable_dispatch_down_mwh"]
        ),
        "dlr_standalone_gain_mwh_per_block": block_gain,
        "dlr_standalone_lifetime_gain_mwh": dlr_lifetime_gain,
    }
    return out, audit


def distributed_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    out = _with_total_cost(frame)
    single = out.loc[out["site_count"].eq(1), [
        "battery_mw", "battery_mwh", "lifetime_net_renewable_gain_mwh",
        "lifetime_total_project_cost_undiscounted_eur", "npv_eur",
    ]].rename(columns={
        "lifetime_net_renewable_gain_mwh": "single_site_lifetime_gain_mwh",
        "lifetime_total_project_cost_undiscounted_eur":
            "single_site_lifetime_total_project_cost_undiscounted_eur",
        "npv_eur": "single_site_npv_eur",
    })
    if single.duplicated(["battery_mw", "battery_mwh"]).any():
        raise ValueError("Strict distributed run has more than one single-site reference per size")
    out = out.merge(single, on=["battery_mw", "battery_mwh"], how="left", validate="many_to_one")
    if out["single_site_npv_eur"].isna().any():
        raise ValueError("Every distributed size requires a single-site reference")
    out["technical_gain_delta_vs_single_mwh"] = (
        out["lifetime_net_renewable_gain_mwh"] - out["single_site_lifetime_gain_mwh"]
    )
    out["undiscounted_cost_delta_vs_single_eur"] = (
        out["lifetime_total_project_cost_undiscounted_eur"]
        - out["single_site_lifetime_total_project_cost_undiscounted_eur"]
    )
    out["npv_delta_vs_single_eur"] = out["npv_eur"] - out["single_site_npv_eur"]
    return out.sort_values(["battery_mwh", "site_count"]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str], headings: list[str]) -> list[str]:
    rows = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame[columns].itertuples(index=False, name=None):
        rows.append("| " + " | ".join(str(value) for value in record) + " |")
    return rows


def write_bundle(
    out: Path,
    fixed_run: Path,
    dlr_run: Path,
    distributed_run: Path,
) -> Path:
    fixed, fixed_manifest, fixed_baseline = load_run(fixed_run)
    dlr, dlr_manifest, dlr_baseline = load_run(dlr_run)
    distributed, distributed_manifest, _ = load_run(distributed_run)
    _same_basis(fixed_manifest, dlr_manifest, distributed_manifest)
    capacity, audit = capacity_dlr_comparison(
        fixed, dlr, fixed_manifest, fixed_baseline, dlr_baseline
    )
    distribution = distributed_comparison(distributed)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new comparison output directory")
    capacity.to_csv(out / "01_capacity_fixed_vs_dlr.csv", index=False)
    distribution.to_csv(out / "02_distributed_strict_comparison.csv", index=False)
    full_breakdown = pd.concat([
        _with_total_cost(fixed).assign(scenario="fixed_lines"),
        _with_total_cost(dlr).assign(scenario="q3_dlr"),
        _with_total_cost(distributed).assign(scenario="strict_distributed_45mw"),
    ], ignore_index=True)
    full_breakdown.to_csv(out / "03_full_project_breakdown.csv", index=False)

    cap_view = capacity[[
        "battery_mw", "battery_mwh", "fixed_lifetime_net_renewable_gain_mwh",
        "dlr_lifetime_net_renewable_gain_mwh", "dlr_plus_bess_total_gain_vs_fixed_mwh",
        "fixed_npv_eur", "dlr_npv_eur",
    ]].copy()
    for column in cap_view.columns:
        decimals = 2 if column in ("battery_mw", "battery_mwh") else 0
        cap_view[column] = cap_view[column].map(
            lambda value, d=decimals: (
                (f"{value:,.{d}f}".rstrip("0").rstrip(".") if d else f"{value:,.0f}")
                if pd.notna(value) else "-"
            )
        )
    dist_view = distribution[[
        "site_count", "sites", "lifetime_net_renewable_gain_mwh",
        "technical_gain_delta_vs_single_mwh", "initial_capex_eur",
        "npv_eur", "npv_delta_vs_single_eur",
    ]].copy()
    for column in dist_view.columns:
        if column != "sites":
            dist_view[column] = dist_view[column].map(
                lambda value: f"{0.0 if abs(value) < 0.5 else value:,.0f}" if pd.notna(value) else "-"
            )

    factor = audit["annualisation_factor"]
    block_gain = audit["dlr_standalone_gain_mwh_per_block"]
    lifetime_gain = audit["dlr_standalone_lifetime_gain_mwh"]
    best_fixed = fixed.loc[fixed["npv_eur"].idxmax()]
    technical_fixed = fixed.loc[fixed["lifetime_net_renewable_gain_mwh"].idxmax()]
    lines = [
        "# Q4 presentation calculations",
        "",
        "Status: the network and Q3 DLR series are project data; prices, costs, repeated-week",
        "annualisation and all investment conclusions remain illustrative scenario assumptions.",
        "",
        "Q2 source correction: the repository final Q2 recommendation is 85 MW / 310 MWh.",
        "The 90 MW / 440 MWh point is retained only as an independently requested envelope case.",
        "",
        "## Step-by-step calculation",
        "",
        f"1. Annualisation factor = 8,760 / {audit['block_hours']} = {factor:.6f}.",
        "2. Annual technical gain = factor × [(renewable dispatch with design − scenario-matched",
        "   no-build counterfactual renewable dispatch) − project battery losses].",
        "3. Lifetime technical gain is the undiscounted sum of the annual technical gain",
        "   after the battery degradation and maintenance trajectory is re-solved each year.",
        "4. Project revenue = common-meter external energy sales + configured grid-service",
        "   revenue. External charging imports are paid; internal hybrid transfers are not",
        "   counted as a sale.",
        "5. Net project cash flow = sales + grid services + residual value − imports − fixed",
        "   and variable O&M − replacement/augmentation/overhaul − decommissioning − year-0 CAPEX.",
        "6. NPV = sum from year 0 to year N of net cash flow divided by (1 + real discount",
        "   rate)^year. System dispatch-cost savings are reported separately and never enter",
        "   developer revenue, project cash flow or NPV.",
        "7. Simple and discounted payback are blank when cumulative cash flow never crosses zero.",
        "",
        "## Capacity: fixed lines versus Q3 DLR",
        "",
        *_markdown_table(
            cap_view,
            list(cap_view.columns),
            ["MW", "MWh", "Fixed gain MWh", "BESS gain with DLR MWh",
             "DLR+BESS gain vs fixed MWh", "Fixed NPV EUR", "Conditional DLR NPV EUR"],
        ),
        "",
        f"DLR standalone gain = ({audit['fixed_baseline_dispatch_down_mwh_per_block']:.3f}",
        f"− {audit['dlr_baseline_dispatch_down_mwh_per_block']:.3f}) × {factor:.6f}",
        f"× {audit['project_years']} = {lifetime_gain:,.0f} MWh over the modeled lifetime.",
        "The conditional DLR NPV columns contain battery-project cash flows only. No DLR",
        "CAPEX, OPEX or DLR-owner revenue is available in the source and none is invented.",
        "",
        f"Technical maximum among the fixed-line battery candidates: {technical_fixed['battery_mw']:g}",
        f"MW / {technical_fixed['battery_mwh']:g} MWh. Best battery-project NPV among built",
        f"candidates: EUR {best_fixed['npv_eur']:,.0f}; because this is below zero, the",
        "economic choice under the stated scenario is NO_BUILD with NPV EUR 0.",
        "",
        "## Strict 45 MW / 220 MWh distributed tie-break",
        "",
        *_markdown_table(
            dist_view,
            list(dist_view.columns),
            ["Sites", "Placement", "Lifetime gain MWh", "Gain delta vs single MWh",
             "Initial CAPEX EUR", "NPV EUR", "NPV delta vs single EUR"],
        ),
        "",
        "The strict run uses a 0.0001% MILP relative gap. Equal technical gains within",
        "floating-point precision mean the extra sites do not improve this modeled network",
        "case; they only add site, connection and O&M costs.",
        "",
        "## Decision boundary",
        "",
        "These results do not establish a merchant business case: the model has no real SEM",
        "price series, DS3/FASS bid stack, taxes, debt, grants, endogenous price response,",
        "AC voltage or N-1 security. Replacing only prices does not make the synthetic wind",
        "and load week a historical joint time series.",
        "",
    ]
    (out / "PRESENTATION_CALCULATIONS.md").write_text("\n".join(lines), encoding="utf-8")
    comparison_manifest = {
        "fixed_run": str(Path(fixed_run).resolve()),
        "dlr_run": str(Path(dlr_run).resolve()),
        "distributed_run": str(Path(distributed_run).resolve()),
        "input_sha256": {
            "fixed_manifest": _sha256(Path(fixed_run) / "manifest.json"),
            "dlr_manifest": _sha256(Path(dlr_run) / "manifest.json"),
            "distributed_manifest": _sha256(Path(distributed_run) / "manifest.json"),
        },
        "network_hash": fixed_manifest["network"]["network_hash"],
        "dlr_label": dlr_manifest["network"]["dlr_label"],
        "audit": audit,
        "dlr_cost_included": False,
        "system_savings_in_project_cashflow": False,
    }
    (out / "manifest.json").write_text(
        json.dumps(comparison_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out
