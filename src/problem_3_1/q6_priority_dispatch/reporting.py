"""English-only Q6 result package and presentation calculations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_clean(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def build_summary(
    manifest: dict[str, Any],
    scenarios: pd.DataFrame,
    impacts: pd.DataFrame,
    line_attribution: pd.DataFrame,
    q4_bridge: pd.DataFrame,
) -> str:
    config = manifest["config"]
    static = scenarios.set_index("scenario")
    neutral = static.loc["network_neutral_static"]
    priority = static.loc["network_priority_static"]
    copper_neutral = static.loc["copperplate_neutral"]
    copper_priority = static.loc["copperplate_priority"]
    total_delta = float(
        priority["wind_dispatch_down_mwh"] - neutral["wind_dispatch_down_mwh"]
    )
    network_only_delta = float(
        (priority["wind_dispatch_down_mwh"] - copper_priority["wind_dispatch_down_mwh"])
        - (neutral["wind_dispatch_down_mwh"] - copper_neutral["wind_dispatch_down_mwh"])
    )
    protected = impacts.loc[impacts["additional_dispatch_down_mwh"] < -1e-6]
    burden = impacts.loc[impacts["additional_dispatch_down_mwh"] > 1e-6]
    protected_energy = float(-protected["additional_dispatch_down_mwh"].sum())
    burden_energy = float(burden["additional_dispatch_down_mwh"].sum())
    lines = [
        "# Q6 Priority and Non-Priority Wind Dispatch",
        "",
        "**The official project network contains no verified legal priority-status field. "
        "All status labels in this run are explicit scenario choices, not Irish operational facts.**",
        "",
        f"- Network: {config['scenario']} {config['scope']}",
        f"- Run horizon: {int(priority['snapshots'])} snapshots / {float(priority['hours']):.1f} hours",
        f"- Priority farms in the illustrative assignment: {int(priority['priority_generator_count'])}",
        f"- Non-priority farms: {int(priority['non_priority_generator_count'])}",
        f"- Solver rule: exact lexicographic load-shedding, priority-energy, total-wind-energy, then physical-cost stages",
        "",
        "## Direct answer",
        "",
        f"The illustrative priority rule changes total wind dispatch-down by "
        f"{total_delta:+,.6f} MWh over the model block relative to neutral maximum-wind dispatch.",
        "",
        f"After subtracting the corresponding copperplate allocation effect, the "
        f"network-specific difference-in-differences is {network_only_delta:+,.6f} MWh.",
        "",
        f"Priority status protects {protected_energy:,.6f} MWh across farms with reduced "
        f"dispatch-down and transfers {burden_energy:,.6f} MWh of additional burden to farms "
        f"with increased dispatch-down. Protection and burden need not be equal when the "
        f"priority rule changes total technically feasible wind energy.",
        "",
        "## Why allocation and inefficiency are different",
        "",
        "A status rule can move dispatch-down from one owner to another while leaving total "
        "wind energy unchanged. That is a distributional effect. It is a technical "
        "inefficiency only when the same network could accept more total wind under the "
        "neutral maximum-wind solution.",
        "",
    ]
    if not impacts.empty:
        top_burden = impacts.sort_values("additional_dispatch_down_mwh", ascending=False).iloc[0]
        top_protected = impacts.sort_values("additional_dispatch_down_mwh").iloc[0]
        lines.extend(
            [
                "## Largest farm-level changes",
                "",
                f"- Largest additional burden: {top_burden['generator']} at "
                f"{top_burden['bus']}, {float(top_burden['additional_dispatch_down_mwh']):+,.6f} MWh.",
                f"- Largest protected amount: {top_protected['generator']} at "
                f"{top_protected['bus']}, {float(top_protected['additional_dispatch_down_mwh']):+,.6f} MWh "
                "(negative means less dispatch-down).",
                "",
            ]
        )
    if not line_attribution.empty:
        top_line = line_attribution.sort_values(
            "status_inefficiency_reduction_when_relaxed_mwh",
            ascending=False,
        ).iloc[0]
        lines.extend(
            [
                "## Where the inefficiency is located",
                "",
                f"The largest tested non-additive line-relaxation attribution is "
                f"{top_line['branch']}: relaxing only this branch changes the status "
                f"inefficiency by "
                f"{float(top_line['status_inefficiency_reduction_when_relaxed_mwh']):+,.6f} MWh.",
                "",
                "Line attributions are marginal counterfactuals. They are not additive when "
                "several constraints bind together.",
                "",
            ]
        )
    if "network_neutral_dlr" in static.index and "network_priority_dlr" in static.index:
        dlr_neutral = static.loc["network_neutral_dlr"]
        dlr_priority = static.loc["network_priority_dlr"]
        dlr_delta = float(
            dlr_priority["wind_dispatch_down_mwh"]
            - dlr_neutral["wind_dispatch_down_mwh"]
        )
        lines.extend(
            [
                "## DLR interaction",
                "",
                f"Under the linked Q3 rating series, the priority-minus-neutral total "
                f"dispatch-down difference is {dlr_delta:+,.6f} MWh.",
                "",
                "DLR changes hourly thermal headroom, not the PTDF. It can therefore change "
                "how often a priority allocation conflicts with a network limit without "
                "changing the underlying linear sensitivity.",
                "",
            ]
        )
    if not q4_bridge.empty:
        economic = q4_bridge.sort_values("npv_eur", ascending=False).iloc[0]
        lines.extend(
            [
                "## Q4 economic bridge",
                "",
                f"The greatest conditional project NPV in the linked Q4 comparison is "
                f"{economic['case_id']} at EUR {float(economic['npv_eur']):,.0f}.",
                "",
                "Q6 wind-revenue transfers are reported separately. They are not booked as "
                "stand-alone BESS revenue, system savings, or a change in battery NPV. The "
                "official input supplies neither contractual compensation rules nor a status "
                "right for stored energy.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "1. Status assignments are scenario choices because the supplied network has no verified priority field.",
            "2. The priority rule is a transparent lexicographic counterfactual, not a reproduction of SEM settlement or EirGrid control-room rules.",
            "3. The 168-hour profiles are synthetic. Repeat-block annualisation is an illustrative extrapolation.",
            "4. The network model is lossless DC power flow and excludes voltage, reactive power, inertia, SNSP and transient security.",
            "5. Multiple simultaneous constraints make single-line causal attribution non-additive.",
            "6. Physical generator cost uses original marginal costs only; status objectives are never reported as cost.",
            "",
            "See 12_conclusion_provenance.csv and manifest.json for exact source and formula lineage.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_calculations(
    manifest: dict[str, Any],
    scenarios: pd.DataFrame,
    impacts: pd.DataFrame,
    wind_value: pd.DataFrame,
) -> str:
    static = scenarios.set_index("scenario")
    neutral = static.loc["network_neutral_static"]
    priority = static.loc["network_priority_static"]
    delta = float(
        priority["wind_dispatch_down_mwh"] - neutral["wind_dispatch_down_mwh"]
    )
    annual_factor = float(manifest["economic_bridge"]["annualisation_factor"])
    price = float(manifest["economic_bridge"]["energy_value_eur_mwh"])
    pv_factor = float(manifest["economic_bridge"]["annuity_present_value_factor"])
    annual_value = -delta * annual_factor * price
    pv_value = annual_value * pv_factor
    lines = [
        "# Q6 Presentation Calculations",
        "",
        "## Step 1 - Define dispatch-down",
        "",
        "For wind farm g and hour t:",
        "",
        "$$c_{g,t}=a_{g,t}-w_{g,t}$$",
        "",
        "where a is available wind and w is dispatched wind.",
        "",
        "## Step 2 - Apply exact priority stages",
        "",
        "Neutral dispatch first minimises unserved energy, then maximises total wind, "
        "then minimises original physical operating cost. Priority dispatch inserts a "
        "stage that maximises priority-wind energy before maximising total wind.",
        "",
        "The optimum of every earlier stage is fixed within the configured tolerance, so "
        "later stages cannot trade it away.",
        "",
        "## Step 3 - Measure technical inefficiency",
        "",
        f"- Neutral dispatch-down: {float(neutral['wind_dispatch_down_mwh']):,.9f} MWh",
        f"- Priority dispatch-down: {float(priority['wind_dispatch_down_mwh']):,.9f} MWh",
        f"- Priority minus neutral: {float(priority['wind_dispatch_down_mwh']):,.9f} "
        f"- {float(neutral['wind_dispatch_down_mwh']):,.9f} = {delta:+,.9f} MWh",
        "",
        "A positive difference is lost feasible wind energy. A zero difference with non-zero "
        "farm changes is redistribution rather than technical inefficiency.",
        "",
        "## Step 4 - Keep project and system money separate",
        "",
        f"The illustrative annualisation factor is 8,760 / {float(priority['hours']):.1f} "
        f"= {annual_factor:.9f}. With an illustrative energy value of EUR {price:.2f}/MWh:",
        "",
        f"$$Annual\ wind\ value\ change = -({delta:+.9f}) "
        f"\times {annual_factor:.9f} \times {price:.2f} "
        f"= EUR\ {annual_value:,.2f}/year$$",
        "",
        f"Using the Q4 real discount rate and horizon, the annuity present-value factor is "
        f"{pv_factor:.9f}, giving an aggregate wind-value transfer proxy of "
        f"EUR {pv_value:,.2f}.",
        "",
        "This value belongs only in the wind-owner allocation scenario. It is not BESS "
        "revenue and is not the model's system dispatch-cost saving.",
        "",
    ]
    if not impacts.empty:
        example = impacts.reindex(
            impacts["additional_dispatch_down_mwh"].abs().sort_values(ascending=False).index
        ).iloc[0]
        lines.extend(
            [
                "## Step 5 - Farm-level example",
                "",
                f"{example['generator']} changes by "
                f"{float(example['additional_dispatch_down_mwh']):+,.9f} MWh of dispatch-down. "
                "Positive means extra burden; negative means protected energy.",
                "",
            ]
        )
    if not wind_value.empty:
        example = wind_value.reindex(
            wind_value["pv_energy_value_transfer_eur"].abs().sort_values(ascending=False).index
        ).iloc[0]
        lines.extend(
            [
                f"Its annualised energy-value transfer at the stated assumption is "
                f"EUR {float(example['annual_energy_value_transfer_eur']):+,.2f}/year, "
                f"with present value EUR {float(example['pv_energy_value_transfer_eur']):+,.2f}.",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    out: Path,
    *,
    manifest: dict[str, Any],
    assignment: pd.DataFrame,
    scenarios: pd.DataFrame,
    generator_hourly: pd.DataFrame,
    system_hourly: pd.DataFrame,
    branch_hourly: pd.DataFrame,
    impacts: pd.DataFrame,
    hotspots: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    permutations: pd.DataFrame,
    dlr_comparison: pd.DataFrame,
    line_attribution: pd.DataFrame,
    q4_bridge: pd.DataFrame,
    wind_value: pd.DataFrame,
    sensitivity: pd.DataFrame,
    provenance: pd.DataFrame,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "00_input_audit.json", manifest["input_audit"])
    assignment.to_csv(out / "01_priority_status_assignment.csv", index=False)
    scenarios.to_csv(out / "02_scenario_summary.csv", index=False)
    generator_hourly.to_csv(out / "03_generator_hourly_dispatch.csv", index=False)
    system_hourly.to_csv(out / "04_hourly_system_metrics.csv", index=False)
    branch_hourly.to_csv(out / "05_line_hour_metrics.csv", index=False)
    impacts.to_csv(out / "06_generator_status_impacts.csv", index=False)
    hotspots.to_csv(out / "07_line_hour_inefficiency_hotspots.csv", index=False)
    leave_one_out.to_csv(out / "08_leave_one_status_out.csv", index=False)
    permutations.to_csv(out / "09_status_permutation_results.csv", index=False)
    dlr_comparison.to_csv(out / "10_static_vs_dlr_priority_comparison.csv", index=False)
    line_attribution.to_csv(out / "11_line_relaxation_attribution.csv", index=False)
    q4_bridge.to_csv(out / "12_q4_economic_bridge.csv", index=False)
    wind_value.to_csv(out / "13_wind_revenue_distribution.csv", index=False)
    sensitivity.to_csv(out / "14_q4_financial_sensitivity.csv", index=False)
    provenance.to_csv(out / "15_conclusion_provenance.csv", index=False)
    write_json(out / "manifest.json", manifest)
    (out / "SUMMARY.md").write_text(
        build_summary(manifest, scenarios, impacts, line_attribution, q4_bridge),
        encoding="utf-8",
    )
    (out / "PRESENTATION_CALCULATIONS.md").write_text(
        build_calculations(manifest, scenarios, impacts, wind_value),
        encoding="utf-8",
    )
