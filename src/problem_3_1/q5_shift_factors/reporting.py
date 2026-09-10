"""English-only machine-readable outputs, plots, and presentation summary for Q5."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    """Write JSON atomically so interrupted runs do not leave partial metadata."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _markdown_ranking(frame: pd.DataFrame) -> list[str]:
    lines = [
        "| Rank | Wind farm | Substation | Shift factor | Relief per MW curtailed | Group |",
        "|---:|---|---|---:|---:|:---:|",
    ]
    for _, row in frame.sort_values(["monitored_line", "rank_abs_shift_factor", "wind_farm"]).iterrows():
        lines.append(
            "| {rank} | {farm} | {bus} | {factor:+.6f} | {relief:+.6f} | {group} |".format(
                rank=int(row["rank_abs_shift_factor"]),
                farm=row["wind_farm"],
                bus=row["nearest_substation"],
                factor=float(row["primary_shift_factor"]),
                relief=float(row["relief_mw_per_mw_curtailment"]),
                group="yes" if bool(row["directional_constraint_group_member"]) else "no",
            )
        )
    return lines


def build_summary(
    manifest: dict,
    ranking: pd.DataFrame,
    validation: pd.DataFrame,
    invariance: pd.DataFrame,
    q4_crosswalk: pd.DataFrame,
    context: pd.DataFrame,
    smoke: bool,
) -> str:
    """Build a concise presentation-ready explanation with explicit sign conventions."""
    network = manifest["network"]
    config = manifest["config"]
    references = manifest["references"]
    primary = references["primary_label"]
    max_error = float(validation["absolute_error"].max()) if len(validation) else float("nan")
    all_passed = bool(validation["passed"].all()) if len(validation) else False
    lines = [
        "# Q5 Wind-Farm Shift Factors",
        "",
        "**This is a DC-network study result, not an operational dispatch instruction or a connection offer.**",
        "",
        f"- Network: {network['scenario']} {network['scope']} ({network['buses']} buses, "
        f"{network['passive_branches']} passive branches, {network['connected_components']} passive AC components, "
        f"{network['snapshots']} hourly snapshots)",
        f"- Wind farms assessed: {network['wind_farms']}",
        f"- Primary balancing reference: {primary}",
        f"- Single-bus comparison reference: {references['single_slack_bus']}",
        f"- Constraint-group threshold: {config['constraint_group_threshold']:.1%}",
        f"- Run mode: {'SMOKE TEST' if smoke else 'FULL ANALYSIS'}",
        "",
    ]
    if smoke:
        lines.extend(
            [
                "**Smoke-test outputs use a shortened snapshot set and a limited farm set. "
                "They must not be used as final study results.**",
                "",
            ]
        )
    lines.extend(
        [
            "## Definition and sign convention",
            "",
            "A shift factor is the change in monitored branch flow, in MW, caused by a "
            "+1 MW injection at the wind farm's model-assigned substation and an equal "
            "withdrawal at the declared balancing reference.",
            "",
            "Load-weighted and uniform balancing are normalized separately inside the source bus's passive AC component. A farm outside the monitored line's component therefore has zero effect on that line. A single-slack transfer across disconnected components is undefined and is left blank in the CSV.",
            "",
            "Positive branch flow follows bus0 to bus1. A negative shift factor therefore "
            "does not mean low impact; it means the incremental flow is in the opposite "
            "direction. The directional relief column combines the factor with the observed "
            "binding-flow direction. A positive relief value means that reducing the farm by "
            "1 MW would reduce absolute loading on the active constraint.",
            "",
        ]
    )
    for _, row in context.iterrows():
        lines.extend(
            [
                f"## Monitored circuit {row['monitored_line']}",
                "",
                f"- Orientation: {row['bus0']} to {row['bus1']} is positive.",
                f"- Observed binding direction: {row['dominant_binding_flow_direction']}.",
                f"- Binding hours: {int(row['binding_hours'])}; "
                f"positive-direction hours: {int(row['positive_binding_hours'])}; "
                f"negative-direction hours: {int(row['negative_binding_hours'])}.",
                f"- Maximum absolute flow: {float(row['maximum_absolute_flow_mw']):.3f} MW "
                f"against {float(row['rating_at_peak_mva']):.3f} MVA at the peak.",
                "",
            ]
        )
        selected = ranking.loc[ranking["monitored_line"].eq(row["monitored_line"])]
        lines.extend(_markdown_ranking(selected))
        lines.append("")
    lines.extend(
        [
            "## Validation",
            "",
            f"- Independent PyPSA LPF finite-difference checks passed: {all_passed}.",
            f"- Maximum absolute analytical-versus-LPF error: {max_error:.3e}.",
            f"- Maximum PTDF change after a rating-only DLR test: "
            f"{float(invariance['maximum_absolute_shift_factor_change'].max()):.3e}.",
            "- DLR changes thermal headroom. It does not change PTDF unless topology or "
            "reactance also changes.",
            "",
            "## Link to the Q4 battery comparison",
            "",
        ]
    )
    sites = q4_crosswalk.loc[q4_crosswalk["record_type"].eq("site")]
    portfolios = q4_crosswalk.loc[q4_crosswalk["record_type"].eq("portfolio")]
    if len(sites):
        grouped = sites.groupby("monitored_line")["primary_shift_factor"].agg(["min", "max"])
        if bool(((grouped["max"] - grouped["min"]).abs() <= 1e-10).all()):
            factor = float(sites.iloc[0]["primary_shift_factor"])
            relief = float(sites.iloc[0]["relief_mw_per_mw_bess_charge"])
            lines.append(
                f"All evaluated Q4 battery sites have the same primary shift factor "
                f"({factor:+.6f}) on the monitored circuit. One MW of charging provides "
                f"{relief:+.6f} MW of directional relief under the observed constraint direction."
            )
            lines.append("")
            lines.append(
                "This explains why splitting the same total MW/MWh across those sites did not "
                "improve the strict Q4 technical result: their aggregate PTDF is unchanged, "
                "while extra sites add fixed connection and site costs."
            )
        else:
            lines.append(
                "The evaluated Q4 sites have different factors, so a portfolio's weighted "
                "factor must be considered together with its dispatch and connection limits."
            )
    if len(portfolios):
        lines.extend(
            [
                "",
                "The site and allocation arithmetic is recorded in "
                "07_q4_site_and_portfolio_crosswalk.csv.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "1. The official model already assigns each aggregated wind generator to a bus. "
            "No independent wind-farm coordinates are provided, so 'nearest substation' means "
            "that official model connection bus; it is not a new geospatial nearest-neighbour study.",
            "2. Shift factors are linear DC sensitivities. They exclude voltage, reactive power, "
            "losses, dynamic stability, fault levels, and N-1 contingencies.",
            "3. Values depend on the balancing reference. The load-weighted reference is primary; "
            "uniform and single-bus values are retained so that the convention remains auditable. Balancing is component-local; cross-component single-slack entries are undefined.",
            "4. A high shift factor shows network leverage, not project profitability or a right "
            "to dispatch. Q4 economics remain separate.",
            "5. The synthetic 168-hour profiles provide operational context only. The PTDF itself "
            "is set by topology and reactance, not by the hourly profile.",
            "",
            "See manifest.json for versions, hashes, references, and input provenance.",
        ]
    )
    return "\n".join(lines) + "\n"



def build_presentation_calculations(
    manifest: dict,
    ranking: pd.DataFrame,
    q4_crosswalk: pd.DataFrame,
    context: pd.DataFrame,
) -> str:
    """Explain the calculation in a slide-ready sequence with worked examples."""
    primary_label = manifest["references"]["primary_label"]
    line_context = context.iloc[0]
    line = str(line_context["monitored_line"])
    line_rows = ranking.loc[ranking["monitored_line"].eq(line)].copy()
    preferred = line_rows.loc[line_rows["wind_farm"].eq("Croaghonagh wind")]
    example = preferred.iloc[0] if len(preferred) else line_rows.nlargest(
        1, "relief_mw_per_mw_curtailment"
    ).iloc[0]
    q1_rows = manifest["input_audit"]["q1_evidence"]["rows"]
    q1 = next(row for row in q1_rows if str(row["line"]) == line)
    q3_stats = manifest["input_audit"]["q3_dlr_evidence"].get(
        "monitored_line_statistics", {}
    ).get(line, {})
    q4_audit = manifest["input_audit"]["q4_portfolios"]
    sizes = q4_audit.get("battery_sizes_mw_mwh", [])
    battery_mw = float(sizes[0][0]) if sizes else 45.0
    sites = q4_crosswalk.loc[
        q4_crosswalk["record_type"].eq("site")
        & q4_crosswalk["monitored_line"].eq(line)
    ]
    portfolios = q4_crosswalk.loc[
        q4_crosswalk["record_type"].eq("portfolio")
        & q4_crosswalk["monitored_line"].eq(line)
    ]
    example_factor = float(example["primary_shift_factor"])
    binding_sign = int(line_context["dominant_binding_flow_sign"])
    example_relief = float(example["relief_mw_per_mw_curtailment"])
    full_relief = float(example["relief_mw_if_fully_curtailed"])
    site_factor = float(sites.iloc[0]["primary_shift_factor"]) if len(sites) else float("nan")
    charge_relief = (
        float(sites.iloc[0]["relief_mw_per_mw_bess_charge"]) if len(sites) else float("nan")
    )
    lines = [
        "# Q5 Presentation Calculations",
        "",
        "## Step 1 — Fix the monitored circuit and direction",
        "",
        f"The Q1/Q3 monitored circuit is {line}, oriented "
        f"{line_context['bus0']} to {line_context['bus1']}. Its fixed rating is "
        f"{float(line_context['nominal_rating_mva']):.1f} MVA.",
        "",
        f"The solved 168-hour case reaches the rating for {int(line_context['binding_hours'])} "
        f"hours. All binding hours have sign {binding_sign:+d}, so the active direction is "
        f"{line_context['dominant_binding_flow_direction']}.",
        "",
        "Q1 cross-check:",
        "",
        f"- Q1 binding hours: {int(q1['binding_hours'])}",
        f"- Q1 energy recovered by a 25% rating uplift: "
        f"{float(q1['saved_dispatch_down_plus_25_mwh']):.3f} MWh",
        "",
        "## Step 2 — Build the DC PTDF",
        "",
        "Let K be the bus-by-branch incidence matrix and B the diagonal branch "
        "susceptance matrix, with branch susceptance equal to 1/x_pu_eff.",
        "",
        "$$L = K B K^T$$",
        "",
        "$$PTDF = B K^T L^+$$",
        "",
        "L+ is the Moore-Penrose pseudoinverse. Positive flow follows each branch's "
        "bus0-to-bus1 orientation.",
        "",
        "## Step 3 — Apply the balancing reference",
        "",
        "For a farm g connected at bus b(g), the primary component-local load-weighted "
        "shift factor is:",
        "",
        "$$SF_{line,g} = PTDF_{line,b(g)} - "
        "\\sum_{j \\in component(g)} w_j PTDF_{line,j}$$",
        "",
        "where the mean-load weights w sum to one inside the farm's passive AC component. "
        "A farm in another disconnected component has zero effect on this line.",
        "",
        "## Step 4 — Convert the signed factor into directional relief",
        "",
        "For a 1 MW curtailment, the injection change is -1 MW. The first-order reduction "
        "in absolute loading is:",
        "",
        "$$Relief_{curtail} = s_{binding} \\times SF_{line,g}$$",
        "",
        f"Worked example — {example['wind_farm']}:",
        "",
        f"- Primary shift factor: {example_factor:+.9f}",
        f"- Binding-flow sign: {binding_sign:+d}",
        f"- Relief per MW curtailed: ({binding_sign:+d}) × "
        f"({example_factor:+.9f}) = {example_relief:+.9f} MW/MW",
        f"- Installed capacity: {float(example['installed_capacity_mw']):.3f} MW",
        f"- Maximum linear relief if reduced from full output to zero: "
        f"{float(example['installed_capacity_mw']):.3f} × {max(example_relief, 0.0):.9f} "
        f"= {full_relief:.3f} MW",
        "",
        "Positive relief means curtailment unloads the active direction. Negative relief "
        "means curtailing that farm would move the circuit the wrong way.",
        "",
        "## Step 5 — Explain the Q4 distributed-battery result",
        "",
        f"Every evaluated Q4 site has primary factor approximately {site_factor:+.9f}. "
        "Charging is a negative injection, so its directional relief is:",
        "",
        f"$$Relief_{{charge}} = {binding_sign:+d} \\times ({site_factor:+.9f}) "
        f"= {charge_relief:+.9f} MW/MW$$",
        "",
        f"For the {battery_mw:.1f} MW Q4 comparison, simultaneous charging at full power "
        f"therefore provides about {battery_mw:.1f} × {charge_relief:.9f} "
        f"= {battery_mw * charge_relief:.3f} MW of first-order relief.",
        "",
    ]
    if len(portfolios):
        minimum = float(portfolios["primary_shift_factor"].min())
        maximum = float(portfolios["primary_shift_factor"].max())
        lines.extend(
            [
                f"The 1–{int(portfolios['site_count'].max())}-site portfolio factors range only "
                f"from {minimum:+.12f} to {maximum:+.12f}; the differences are numerical roundoff.",
                "",
                "Therefore, splitting the same total MW/MWh among these buses does not change "
                "aggregate PTDF leverage. The strict Q4 dispatch result stays the same while "
                "additional sites add fixed costs.",
                "",
            ]
        )
    lines.extend(
        [
            "## Step 6 — Connect the result to DLR",
            "",
            "A rating-only DLR multiplier changes available headroom but not x_pu_eff, K, B, "
            "or the PTDF. The computed maximum shift-factor change is "
            f"{float(manifest['numerical_checks']['maximum_rating_only_dlr_ptdf_change']):.3e}.",
            "",
        ]
    )
    if q3_stats:
        mean_multiplier = float(q3_stats["mean"])
        maximum_multiplier = float(q3_stats["maximum"])
        rating = float(line_context["nominal_rating_mva"])
        lines.extend(
            [
                f"The linked Q3 series has mean multiplier {mean_multiplier:.6f} and maximum "
                f"{maximum_multiplier:.6f}. Applied to {rating:.1f} MVA, these correspond to "
                f"{rating * mean_multiplier:.3f} MVA mean dynamic capacity and "
                f"{rating * maximum_multiplier:.3f} MVA maximum dynamic capacity.",
                "",
            ]
        )
    lines.extend(
        [
            "## Step 7 — Numerical validation",
            "",
            f"- {manifest['network']['wind_farms']} farm transfers were checked against "
            "independent PyPSA LPF perturbations.",
            f"- Maximum analytical-versus-LPF error: "
            f"{float(manifest['numerical_checks']['maximum_lpf_finite_difference_error']):.3e}.",
            f"- Maximum component-local reference residual: "
            f"{float(manifest['numerical_checks']['maximum_primary_reference_weighted_mean']):.3e}.",
            "",
            "These calculations quantify network sensitivity only. They do not assign market "
            "revenue, dispatch priority, or project NPV.",
        ]
    )
    return "\n".join(lines) + "\n"

def plot_outputs(
    out: Path,
    ranking: pd.DataFrame,
    reference_columns: list[str],
    smoke: bool,
) -> None:
    """Write compact figures without changing numerical results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = out / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    prefix = "SMOKE TEST - " if smoke else ""
    for line, group in ranking.groupby("monitored_line"):
        ordered = group.sort_values(
            ["relief_mw_per_mw_curtailment", "wind_farm"], ascending=[True, True]
        )
        colors = np.where(
            ordered["relief_mw_per_mw_curtailment"] > 0,
            "#217a52",
            "#b14a42",
        )
        fig, ax = plt.subplots(figsize=(10.5, 6.6))
        ax.barh(ordered["wind_farm"], ordered["primary_shift_factor"], color=colors)
        ax.axvline(0.0, color="#333333", linewidth=0.8)
        ax.set(
            xlabel="Shift factor (MW on circuit per MW injected)",
            ylabel="",
            title=f"{prefix}{line}: load-weighted wind-farm shift factors",
        )
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"01_target_shift_factors_{line}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10.5, 6.6))
        x = np.arange(len(group))
        width = 0.8 / max(len(reference_columns), 1)
        for number, column in enumerate(reference_columns):
            ax.bar(
                x + (number - (len(reference_columns) - 1) / 2) * width,
                group[column],
                width=width,
                label=column.removeprefix("shift_factor_").replace("_", " "),
            )
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(group["wind_farm"], rotation=65, ha="right")
        ax.set(
            ylabel="Shift factor",
            title=f"{prefix}{line}: balancing-reference comparison",
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"02_reference_comparison_{line}.png", dpi=160)
        plt.close(fig)


def write_outputs(
    out: Path,
    manifest: dict,
    farm_table: pd.DataFrame,
    factor_table: pd.DataFrame,
    ranking: pd.DataFrame,
    all_branch_matrix: pd.DataFrame,
    validation: pd.DataFrame,
    invariance: pd.DataFrame,
    q4_crosswalk: pd.DataFrame,
    context: pd.DataFrame,
    smoke: bool,
    make_plots: bool = True,
) -> None:
    """Write the complete auditable Q5 result package."""
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "00_input_audit.json", manifest["input_audit"])
    farm_table.reset_index().to_csv(
        out / "01_wind_farm_nearest_substation.csv", index=False
    )
    factor_table.to_csv(out / "02_shift_factor_matrix.csv", index=False)
    ranking.to_csv(out / "03_target_line_ranking.csv", index=False)
    all_branch_matrix.reset_index().to_csv(
        out / "04_all_branch_load_weighted_shift_factors.csv", index=False
    )
    validation.to_csv(out / "05_finite_difference_validation.csv", index=False)
    invariance.to_csv(out / "06_dlr_rating_invariance.csv", index=False)
    q4_crosswalk.to_csv(out / "07_q4_site_and_portfolio_crosswalk.csv", index=False)
    context.to_csv(out / "08_monitored_line_context.csv", index=False)
    write_json(out / "manifest.json", manifest)
    (out / "SUMMARY.md").write_text(
        build_summary(
            manifest,
            ranking,
            validation,
            invariance,
            q4_crosswalk,
            context,
            smoke,
        ),
        encoding="utf-8",
    )
    (out / "PRESENTATION_CALCULATIONS.md").write_text(
        build_presentation_calculations(manifest, ranking, q4_crosswalk, context),
        encoding="utf-8",
    )
    if make_plots:
        reference_columns = [
            column for column in factor_table.columns if column.startswith("shift_factor_")
        ]
        plot_outputs(out, ranking, reference_columns, smoke)
