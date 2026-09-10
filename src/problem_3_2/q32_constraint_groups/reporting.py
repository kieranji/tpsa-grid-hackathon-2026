"""Output writer for nationwide Problem 3.2 constraint groups."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_plot(output: Path, modes: pd.DataFrame, zones: pd.DataFrame, buses: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    top = modes.head(20).sort_values("event_weighted_severity_hours")
    fig, ax = plt.subplots(figsize=(10, 7))
    if len(top):
        ax.barh(top["mode_id"], top["event_weighted_severity_hours"], color="#006B5E")
    ax.set_xlabel("Event-weighted congestion severity (hours)")
    ax.set_title("Top nationwide directional constraint modes")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "01_top_directional_modes.png", dpi=180)
    plt.close(fig)

    bus_table = buses.reset_index().rename(columns={buses.index.name or "index": "bus"})
    merged = zones.merge(bus_table, on="bus", how="left", validate="one_to_one")
    merged = merged.loc[np.isfinite(merged["x"]) & np.isfinite(merged["y"])]
    fig, ax = plt.subplots(figsize=(9, 10))
    if len(merged):
        codes = pd.Categorical(merged["response_zone_id"]).codes
        ax.scatter(merged["x"], merged["y"], c=codes, cmap="turbo", s=8, alpha=0.8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("All-island electrical response zones")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "02_response_zone_map.png", dpi=180)
    plt.close(fig)


def write_outputs(
    output: Path,
    *,
    manifest: dict[str, Any],
    input_audit: dict[str, Any],
    modes: pd.DataFrame,
    memberships: pd.DataFrame,
    merged_modes: pd.DataFrame,
    mode_pairs: pd.DataFrame,
    zones: pd.DataFrame,
    zone_summary: pd.DataFrame,
    nonlocal_pairs: pd.DataFrame,
    reference_audit: pd.DataFrame,
    lpf_validation: pd.DataFrame,
    north_west_consistency: pd.DataFrame,
    dlr_comparison: pd.DataFrame,
    q6_crosswalk: pd.DataFrame,
    event_weights: pd.DataFrame,
    relief_load: pd.DataFrame,
    relief_uniform: pd.DataFrame,
    scope_gaps: pd.DataFrame,
    renewable_crosswalk: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    buses: pd.DataFrame,
    make_plots: bool,
) -> None:
    """Write deterministic tables, audit prose, and presentation calculations."""
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new --out directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / "00_input_audit.json").write_text(
        json.dumps(input_audit, indent=2, default=_json_default), encoding="utf-8"
    )
    tables = [
        ("01_directional_constraint_modes.csv", modes),
        ("02_overlapping_constraint_memberships.csv", memberships),
        ("03_merged_mode_map.csv", merged_modes),
        ("04_mode_similarity_audit.csv", mode_pairs),
        ("05_bus_response_zones.csv", zones),
        ("06_response_zone_summary.csv", zone_summary),
        ("07_nonlocal_similar_bus_pairs.csv", nonlocal_pairs),
        ("08_reference_sensitivity.csv", reference_audit),
        ("09_lpf_sample_validation.csv", lpf_validation),
        ("10_north_west_q5_consistency.csv", north_west_consistency),
        ("11_static_vs_dlr.csv", dlr_comparison),
        ("12_q6_priority_crosswalk.csv", q6_crosswalk),
        ("16_model_scope_and_gaps.csv", scope_gaps),
        ("17_renewable_resource_constraint_memberships.csv", renewable_crosswalk),
    ]
    for name, frame in tables:
        frame.to_csv(output / name, index=False)
    event_weights.to_csv(output / "13_mode_event_weights.csv.gz", compression="gzip")
    relief_load.to_csv(output / "14_directional_relief_load_weighted.csv.gz", compression="gzip")
    relief_uniform.to_csv(output / "15_directional_relief_uniform.csv.gz", compression="gzip")
    similarity_matrix.to_csv(output / "18_bus_electrical_similarity.csv.gz", compression="gzip")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    if make_plots:
        _write_plot(output, modes, zones, buses)

    mode_count = len(modes)
    merged_count = merged_modes["merged_constraint_group_id"].nunique() if len(merged_modes) else 0
    zone_count = zones["response_zone_id"].nunique() if len(zones) else 0
    member_count = len(memberships)
    renewable_count = int(renewable_crosswalk["renewable_resource"].nunique())
    renewable_grouped = int(
        renewable_crosswalk.loc[renewable_crosswalk["mode_id"].notna(), "renewable_resource"].nunique()
    )
    validation_error = float(lpf_validation["absolute_error"].max()) if len(lpf_validation) else float("nan")
    reference_centered = (
        float(reference_audit["maximum_absolute_centered_response_change"].max())
        if len(reference_audit) else float("nan")
    )
    summary = f"""# Problem 3.2 — Nationwide Constraint Group Generation

## Direct answer

The official all-island network was solved for **{manifest['network']['snapshots']} hours**. It produced **{mode_count} directional near-congestion modes**, retained as overlapping constraint groups with **{member_count} bus memberships**, and condensed into **{merged_count} non-exclusive merged mode groups** only where both membership and relief-vector tests passed.

A separate, mutually exclusive electrical-response zoning layer contains **{zone_count} zones**. Constraint groups and response zones are not interchangeable: a bus may belong to many directional constraint groups, but exactly one response zone inside its passive AC component.

The explicit RES crosswalk covers **{renewable_count} wind, solar, hydro, and biomass generators**; **{renewable_grouped}** have at least one active directional group in this simulated horizon. File 17 retains each official generator-to-bus mapping. File 18 is the compressed full bus-by-bus electrical response similarity matrix; cross-component and featureless comparisons are intentionally blank.

## Key safeguards

- Positive flow always follows branch bus0-to-bus1 orientation; each opposite direction is a different mode.
- Relief is MW of directional branch relief per MW of curtailment or BESS charging.
- Load-weighted and uniform balancing results are both published. Their raw relief and memberships may differ.
- Response zones use centered pairwise response differences. The maximum reference-change residual after centering is **{reference_centered:.3e}**.
- Sampled independent LPF validation has maximum absolute error **{validation_error:.3e}**.
- DLR changes event weights and headroom, not the PTDF. Q3 DLR coverage is limited to the supplied target line.
- Q6 priority status changes allocation and is reported separately from physical constraint similarity.

## Interpretation

The national model identifies electrically related resources that need not be geographically adjacent. File 07_nonlocal_similar_bus_pairs.csv gives explicit examples. These are model-derived planning signals, not a substitute for protection, voltage, stability, outage, or market studies.

## Evidence map

Every headline metric is in manifest.json; row-level evidence is in files 01–18. File 00_input_audit.json records official inputs and hashes. PRESENTATION_CALCULATIONS.md gives worked formulas and exact table locators.
"""
    (output / "SUMMARY.md").write_text(summary, encoding="utf-8")

    top_mode = modes.iloc[0].to_dict() if len(modes) else {}
    example_member = memberships.iloc[0].to_dict() if len(memberships) else {}
    calculations = f"""# Problem 3.2 Presentation Calculations

## 1. Directional congestion event

For each branch and direction:

loading = abs(flow_MW) / effective_rating_MVA

event_weight = snapshot_hours times clip((loading - {manifest['config']['near_congestion_threshold_pu']}) / (1 - {manifest['config']['near_congestion_threshold_pu']}), 0, 1)

Example top mode: {top_mode.get('mode_id', 'none')}. Its maximum loading is {top_mode.get('maximum_loading_pu', float('nan'))} pu and its event-weighted severity is {top_mode.get('event_weighted_severity_hours', float('nan'))} hours. Source: 01_directional_constraint_modes.csv, key mode_id={top_mode.get('mode_id', 'none')}.

## 2. Directional relief and overlapping membership

relief(mode, bus) = flow_sign(mode) times PTDF(branch, bus)

normalised_relief = max(relief, 0) / max_bus(max(relief, 0))

A bus is a member when normalised relief is at least {manifest['config']['constraint_group_threshold']}. Example: mode {example_member.get('mode_id', 'none')}, bus {example_member.get('bus', 'none')}, raw relief {example_member.get('relief_mw_per_mw_curtailment_or_charge', float('nan'))}, normalised relief {example_member.get('normalized_positive_relief', float('nan'))}. Source: 02_overlapping_constraint_memberships.csv with the exact mode and bus keys.

## 3. Mode merging

Two modes merge only when they are in the same passive AC component and both membership Jaccard is at least {manifest['config']['mode_merge_jaccard_threshold']} and positive-relief cosine is at least {manifest['config']['mode_merge_cosine_threshold']}. Pair-level audit: 04_mode_similarity_audit.csv. Final mapping: 03_merged_mode_map.csv.

## 4. Electrical response zones

For each component, the bus feature for each active mode is the centered directional PTDF multiplied by the square root of that mode's normalized congestion-event weight. Centering removes the reference-dependent row constant. Average-linkage clustering chooses the number of zones by maximum silhouette, with a minimum useful score of 0.05. Sources: 05_bus_response_zones.csv and 06_response_zone_summary.csv.

## 5. Reference test

The raw load-weighted and uniform factors can change because the balancing injection changes. For every mode, subtracting the within-component mean leaves pairwise bus-response differences. The resulting maximum discrepancy is {reference_centered:.3e}. Source: 08_reference_sensitivity.csv.

## 6. Independent electrical validation

A 1 MW source injection is balanced over load in the same passive AC component, and two PyPSA LPFs measure the branch-flow delta. This is compared against the analytical PTDF in 09_lpf_sample_validation.csv. Maximum absolute error: {validation_error:.3e}.

## 7. Q5, DLR, and Q6 links

- File 10_north_west_q5_consistency.csv recomputes the Q5 target-line factors on the North-West network and separately shows the all-island factor where the identifier exists.
- File 11_static_vs_dlr.csv changes the Q3 target rating while holding the solved-flow trace fixed; this isolates headroom/event classification, not redispatch.
- File 12_q6_priority_crosswalk.csv imports Q6 policy allocation metrics. Priority status is not an electrical-similarity input and is not developer BESS revenue.

## 8. Explicit RES correlation map

File 17_renewable_resource_constraint_memberships.csv maps every configured renewable generator through its official model connection bus to zero or more overlapping directional groups and exactly one response zone. File 18_bus_electrical_similarity.csv.gz gives the full centered, event-weighted cosine matrix for reproducible national correlation analysis.
"""
    (output / "PRESENTATION_CALCULATIONS.md").write_text(calculations, encoding="utf-8")
    verification = f"""mode_count={mode_count}
merged_group_count={merged_count}
response_zone_count={zone_count}
lpf_max_abs_error={validation_error:.12g}
reference_centered_max_abs_change={reference_centered:.12g}
all_lpf_checks_passed={manifest['numerical_checks']['all_lpf_checks_passed']}
all_buses_have_one_zone={manifest['numerical_checks']['all_buses_have_exactly_one_response_zone']}
"""
    (output / "VERIFICATION.log").write_text(verification, encoding="utf-8")
