"""Machine-readable comparisons and compact plots; no unsupported investment claims."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .finance import pareto_flags


def build_comparisons(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = summary.copy()
    summary["pareto_within_technology"] = False
    rows = []
    for tech, group in summary.groupby("technology"):
        vals = group["lifetime_net_renewable_gain_mwh"].to_numpy()
        returns = group["npv_eur"].to_numpy()
        # Include no-build when assessing dominance and the commercial decision.
        keep = pareto_flags(np.r_[vals, 0.0], np.r_[returns, 0.0])[:-1]
        summary.loc[group.index, "pareto_within_technology"] = keep
        technical_max = float(group["lifetime_net_renewable_gain_mwh"].max())
        technical_tolerance = max(1e-6, abs(technical_max) * 1e-9)
        technical_candidates = group.loc[
            group["lifetime_net_renewable_gain_mwh"] >= technical_max - technical_tolerance
        ].copy()
        site_values = (
            technical_candidates["site_count"]
            if "site_count" in technical_candidates
            else pd.Series(1, index=technical_candidates.index, dtype=float)
        )
        capex_values = (
            technical_candidates["initial_capex_eur"]
            if "initial_capex_eur" in technical_candidates
            else pd.Series(np.inf, index=technical_candidates.index, dtype=float)
        )
        technical_candidates["_tie_site_count"] = pd.to_numeric(
            site_values, errors="coerce"
        ).fillna(1)
        technical_candidates["_tie_capex"] = pd.to_numeric(
            capex_values, errors="coerce"
        ).fillna(np.inf)
        technical = technical_candidates.sort_values(
            ["_tie_site_count", "_tie_capex", "case_id"], kind="stable"
        ).iloc[0]
        best = group.loc[group.npv_eur.idxmax()]
        for label, selected in (("technical_max_net_RE_gain", technical), ("economic_max_conditional_NPV", best)):
            no_build = (label.startswith("economic") and selected.npv_eur <= 0) or (
                label.startswith("technical") and selected.lifetime_net_renewable_gain_mwh <= 0)
            rows.append({"technology": tech, "selection": label,
                "case_id": "NO_BUILD" if no_build else selected.case_id,
                "bus": "-" if no_build else selected.bus,
                "site_count": 0 if no_build else selected.get("site_count", 1),
                "sites": "-" if no_build else selected.get("sites", selected.bus),
                "allocations": "" if no_build else selected.get("allocations", ""),
                "wind_mw": 0 if no_build else selected.wind_mw,
                "battery_mw": 0 if no_build else selected.battery_mw,
                "battery_mwh": 0 if no_build else selected.battery_mwh,
                "npv_eur": 0 if no_build else selected.npv_eur,
                "lifetime_net_renewable_gain_mwh": 0 if no_build else selected.lifetime_net_renewable_gain_mwh,
                "interpretation": "best among evaluated designs under stated operator dispatch and assumptions"})
    return summary, pd.DataFrame(rows)


def write_report(out: Path, summary: pd.DataFrame, config: dict, metadata: dict,
                 failures: list[dict], planned: int, smoke: bool) -> None:
    if summary.empty:
        (out / "SUMMARY.md").write_text("# Q4\n\nNo successful cases. Inspect failures.json and the terminal output.\n", encoding="utf-8")
        return
    summary, comparisons = build_comparisons(summary)
    summary.sort_values(["technology", "npv_eur"], ascending=[True, False]).to_csv(out / "01_project_ranking.csv", index=False)
    comparisons.to_csv(out / "02_technical_vs_economic.csv", index=False)
    summary.loc[summary.pareto_within_technology].to_csv(out / "03_pareto_frontier.csv", index=False)
    breakdown_columns = ["case_id", "technology", "site_count", "sites", "allocations",
        "wind_mw", "battery_mw", "battery_mwh", "lifetime_net_renewable_gain_mwh",
        "year_1_project_revenue_eur", "year_1_operating_cost_eur",
        "lifetime_project_revenue_eur", "lifetime_energy_purchase_eur",
        "lifetime_fixed_and_variable_om_eur", "lifetime_maintenance_capex_eur",
        "initial_capex_eur", "lifetime_decommission_eur", "lifetime_residual_eur",
        "lifetime_net_project_cashflow_eur", "lifecycle_cost_pv_eur",
        "lifetime_system_dispatch_cost_saving_eur", "pv_system_dispatch_cost_saving_eur",
        "system_saving_included_in_project_cashflow", "npv_eur",
        "simple_payback_years", "discounted_payback_years",
        "additional_break_even_payment_eur_kw_year"]
    breakdown = summary.reindex(columns=breakdown_columns).copy()
    no_build = {column: 0.0 for column in breakdown_columns}
    no_build.update(case_id="NO_BUILD", technology="no_build", site_count=0, sites="-", allocations="",
                    system_saving_included_in_project_cashflow=False)
    breakdown = pd.concat([pd.DataFrame([no_build]), breakdown], ignore_index=True)
    breakdown.to_csv(out / "04_technical_revenue_cost_npv_payback.csv", index=False)
    for tech in summary.technology.unique():
        frame = summary.loc[summary.technology.eq(tech)]
        frame.to_csv(out / f"ranking_{tech}.csv", index=False)
    lines = ["# Q4 Techno-Economic Siting Analysis", "",
        "**This document is scenario modelling, not investment advice or a connection commitment.**", "",
        f"- Backend: {metadata['backend']}",
        f"- Cases completed: {len(summary)} / {planned}; failed: {len(failures)}",
        f"- Financial horizon: {config['years']} years; real discount rate: {config['real_discount_rate']:.1%}",
        f"- Currency: {config['currency_basis']}",
        f"- Assumptions status: {config['assumptions_status']}",
        f"- Annualisation: {config['annualisation']}", "",
        "SMOKE TEST ONLY — the shortened time series and project life cannot support investment conclusions." if smoke else "",
        "## Comparison basis", "",
        "Technical optimum: the evaluated candidate with the greatest lifetime proxy for additional renewable energy used, net of project battery losses.",
        "This is not source-by-source MWh tracing and is not a certification of total system energy including existing storage or controllable-link losses.",
        "Economic optimum: the evaluated candidate with the greatest lifecycle NPV under the same system-operator dispatch rule; choose no build when every NPV is non-positive.",
        "The NPV is not a global optimum for autonomous developer bidding or arbitrage. Both optima are limited to the discrete candidates evaluated.",
        "System dispatch-cost savings are reported separately and are never counted as developer revenue, project cash flow, or NPV.",
        "04_technical_revenue_cost_npv_payback.csv separates technical improvement, first-year and lifetime project revenue, electricity purchases, O&M, maintenance capital, decommissioning, residual value, system savings, NPV, and payback.", "",
        "## Results", ""]
    for row in comparisons.to_dict("records"):
        lines.append(f"**{row['technology']} / {row['selection']}**: {row['case_id']} @ {row['sites']}, NPV €{row['npv_eur']:,.0f}.")
        lines.append("")
    lines += ["## Limitations that must be retained", "",
        "1. The official network's DC flow representation, boundaries, and parallel circuits are preserved. Q4 does not replace 15/16-node or all-island validation.",
        "2. Repeating a synthetic representative week to a year is a scenario extrapolation. Replacing the price series with actual prices does not turn synthetic wind and load into an actual joint time series.",
        "3. Capacity-constrained dispatch is rerun for each battery year. Capacity is held at its beginning-of-year value within a year; degradation and maintenance occur at year end.",
        "4. The model excludes tax, debt, subsidies, actual system-service bidding, endogenous price feedback, AC voltage, and N-1 validation. Network topology and background generation remain unchanged over the project life.",
        "5. Added wind uses a common weather profile by default to isolate grid-location effects; this is not a site-specific wind-resource assessment. Co-located wind farms share available power proportionally and no dispatch priority is modeled.",
        "6. LCOS is reported only for stand-alone batteries. Hybrid projects use net settlement at a common meter and do not count internal charging as a sale.",
        "7. O&M excludes cell augmentation or replacement, which is listed separately. The dispatch wear penalty is not deducted again as a cash cost.",
        "8. Financial sensitivity revalues a fixed physical operating path. Changes to degradation, efficiency, or life require a new configured run.",
        "9. The first-year constraint split is a counterfactual that relaxes Line and Transformer thermal limits; it is not SNSP curtailment.", "",
        "Annual cash flows, first-year hourly power and state of charge, and full-network branch metrics are in cases/<case_id>/.",
        "Inputs, versions, and data fingerprints are in manifest.json; failed cases do not enter the ranking."]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    plot_outputs(out, summary, is_demo="demo" in metadata["backend"])


def plot_outputs(out: Path, summary: pd.DataFrame, is_demo: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    title_prefix = "SYNTHETIC DEMO - NOT IRELAND\n" if is_demo else ""
    directory = out / "figures"
    directory.mkdir(exist_ok=True)
    for tech, group in summary.groupby("technology"):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.scatter(group.lifetime_net_renewable_gain_mwh / 1e3, group.npv_eur / 1e6)
        ax.scatter([0], [0], marker="x", label="No build")
        frontier = group.loc[group.pareto_within_technology].sort_values("lifetime_net_renewable_gain_mwh")
        if len(frontier) > 1:
            ax.plot(frontier.lifetime_net_renewable_gain_mwh / 1e3, frontier.npv_eur / 1e6, linestyle="--", label="Evaluated Pareto frontier")
        for _, row in group.nlargest(3, "npv_eur").iterrows():
            label = f"{row.get('sites', row.bus)} | W{row.wind_mw:g} B{row.battery_mw:g}/{row.battery_mwh:g}"
            ax.annotate(label, (row.lifetime_net_renewable_gain_mwh / 1e3, row.npv_eur / 1e6),
                        xytext=(-4, 5) if row.lifetime_net_renewable_gain_mwh > group.lifetime_net_renewable_gain_mwh.median() else (4, 5),
                        ha="right" if row.lifetime_net_renewable_gain_mwh > group.lifetime_net_renewable_gain_mwh.median() else "left",
                        textcoords="offset points", fontsize=7)
        ax.axhline(0, linewidth=0.7)
        ax.set(xlabel="Lifetime net renewable gain proxy (GWh)", ylabel="Conditional project NPV (EUR million)",
               title=title_prefix + f"{tech}: technical benefit vs economic return - scenario model")
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(directory / f"pareto_{tech}.png", dpi=150); plt.close(fig)
        best = group.loc[group.npv_eur.idxmax()]
        path = out / "cases" / best.case_id / "annual_cashflows.csv"
        cf = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(cf.year, cf.cumulative_cashflow_eur / 1e6, label="Undiscounted")
        ax.plot(cf.year, cf.cumulative_discounted_cashflow_eur / 1e6, label="Discounted")
        ax.axhline(0, linewidth=0.7)
        ax.set(xlabel="Project year", ylabel="Cumulative cash flow (EUR million)",
               title=title_prefix + f"Best evaluated {tech} candidate - compare with no build")
        ax.legend(); ax.grid(alpha=0.25); fig.tight_layout()
        fig.savefig(directory / f"cashflow_{tech}.png", dpi=150); plt.close(fig)

        sensitivity_path = path.parent / "financial_sensitivity.csv"
        if sensitivity_path.exists():
            sens = pd.read_csv(sensitivity_path)
            ranges = sens.groupby("parameter").npv_eur.agg(["min", "max"])
            ranges = ranges.loc[(ranges["max"] - ranges["min"]).sort_values().index]
            fig, ax = plt.subplots(figsize=(10, 5.5))
            ax.barh(ranges.index, (ranges["max"] - ranges["min"]) / 1e6,
                    left=ranges["min"] / 1e6)
            ax.axvline(best.npv_eur / 1e6, linestyle="--", label="Base NPV")
            ax.set(xlabel="Conditional NPV range (EUR million)",
                   title=title_prefix + f"{tech}: fixed-dispatch financial sensitivity")
            ax.legend(); ax.grid(axis="x", alpha=0.25); fig.tight_layout()
            fig.savefig(directory / f"sensitivity_{tech}.png", dpi=150); plt.close(fig)
