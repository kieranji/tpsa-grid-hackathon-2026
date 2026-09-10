"""TPSA Hackathon 2026 — Problem 3.1, Question 1.

Primary study: the supplied WP2033 North-West regional model.
Question: which lines are constrained most often, and how does thermal rating
change the amount of renewable dispatch-down?

The all-island solve in this script is a robustness check only.  It does not
replace the North-West model requested by Problem 3.1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KIT_DIR = PROJECT_ROOT / "data" / "participant-kit"

if not (KIT_DIR / "gridkit.py").exists():
    raise FileNotFoundError(
        "Cannot find data/participant-kit/gridkit.py. "
        "Place this file under src/problem_3_1/."
    )

sys.path.insert(0, str(KIT_DIR))
import gridkit  # noqa: E402


DEFAULT_SCENARIO = "WP2033"
REGIONAL_SCOPE = "north-west"
FULL_SCOPE = "all-island"
BINDING_THRESHOLD = 0.999
NEAR_LIMIT_THRESHOLD = 0.90
LINE_REINFORCEMENT_MULTIPLIER = 1.25
RATING_RELAXATION_MULTIPLIER = 1000.0
NETWORK_RATING_MULTIPLIERS = (0.75, 1.00, 1.10, 1.25, 1.50, 2.00)
TOP_N = 10
NUMERICAL_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument(
        "--skip-all-island-check",
        action="store_true",
        help="Skip the full-grid robustness comparison.",
    )
    return parser.parse_args()


def load_network(scenario: str, scope: str):
    return gridkit.load(scenario, scope)


def solve_checked(network, label: str) -> None:
    result = gridkit.solve(network)

    if isinstance(result, tuple) and len(result) >= 2:
        status = str(result[0]).lower()
        condition = str(result[1]).lower()
        if status != "ok" or "optimal" not in condition:
            raise RuntimeError(
                f"{label} failed: status={result[0]!r}, "
                f"condition={result[1]!r}"
            )

    if not len(network.generators_t.p.columns):
        raise RuntimeError(f"{label} produced no generator dispatch.")
    if len(network.lines) and not len(network.lines_t.p0.columns):
        raise RuntimeError(f"{label} produced no line flows.")


def dispatch_down_mwh(network) -> float:
    table = gridkit.dispatch_down(network)
    return 0.0 if table.empty else float(table["dispatch_down_mwh"].sum())


def unserved_energy_mwh(network) -> float:
    series = gridkit.unserved(network)
    return 0.0 if series.empty else float(series.sum())


def zero_small(value: float) -> float:
    return 0.0 if abs(value) < NUMERICAL_TOLERANCE else float(value)


def line_label(line_name: str, bus0: str, bus1: str) -> str:
    return f"{line_name}: {bus0} ↔ {bus1}"


def network_components(network) -> list[set[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)
    graph.add_edges_from(zip(network.lines["bus0"], network.lines["bus1"]))
    if len(network.transformers):
        graph.add_edges_from(
            zip(network.transformers["bus0"], network.transformers["bus1"])
        )
    return [set(component) for component in nx.connected_components(graph)]


def boundary_diagnostics(network) -> pd.DataFrame:
    if "carrier" not in network.generators.columns:
        return pd.DataFrame()

    boundary = network.generators[
        network.generators["carrier"].astype(str).eq("boundary")
    ].copy()
    if boundary.empty:
        return pd.DataFrame()

    boundary["fixed_by_static_bounds"] = np.isclose(
        boundary["p_min_pu"].astype(float),
        boundary["p_max_pu"].astype(float),
    )

    rows: list[dict[str, object]] = []
    for name, row in boundary.iterrows():
        dispatch = network.generators_t.p[name]
        rows.append(
            {
                "generator": name,
                "bus": row["bus"],
                "p_nom_mw": float(row["p_nom"]),
                "p_min_pu": float(row["p_min_pu"]),
                "p_max_pu": float(row["p_max_pu"]),
                "fixed_by_static_bounds": bool(row["fixed_by_static_bounds"]),
                "dispatch_min_mw": float(dispatch.min()),
                "dispatch_mean_mw": float(dispatch.mean()),
                "dispatch_max_mw": float(dispatch.max()),
                "dispatch_range_mw": float(dispatch.max() - dispatch.min()),
                "dispatch_std_mw": float(dispatch.std()),
            }
        )
    return pd.DataFrame(rows)


def regional_line_table(network, loading: pd.DataFrame) -> pd.DataFrame:
    hours = len(network.snapshots)
    lines = network.lines[["bus0", "bus1", "s_nom"]].copy()
    lines = lines.rename(columns={"s_nom": "rating_mva"})
    lines.index.name = "line"

    lines["label"] = [
        line_label(name, row.bus0, row.bus1)
        for name, row in lines.iterrows()
    ]
    lines["binding_hours"] = (
        loading >= BINDING_THRESHOLD
    ).sum().reindex(lines.index).fillna(0).astype(int)
    lines["binding_share_pct"] = 100.0 * lines["binding_hours"] / hours
    lines["hours_at_or_above_90_pct"] = (
        loading >= NEAR_LIMIT_THRESHOLD
    ).sum().reindex(lines.index).fillna(0).astype(int)
    lines["mean_loading_pct"] = 100.0 * loading.mean().reindex(lines.index)
    lines["p95_loading_pct"] = 100.0 * loading.quantile(0.95).reindex(lines.index)
    lines["max_loading_pct"] = 100.0 * loading.max().reindex(lines.index)
    lines["peak_loading_snapshot"] = [
        loading[name].idxmax() for name in lines.index
    ]

    return lines.sort_values(
        ["binding_hours", "p95_loading_pct", "max_loading_pct"],
        ascending=False,
    )


def run_all_branch_rating_case(
    scenario: str,
    multiplier: float,
) -> dict[str, float | int]:
    network = load_network(scenario, REGIONAL_SCOPE)
    network.lines.loc[:, "s_nom"] *= float(multiplier)
    if len(network.transformers):
        network.transformers.loc[:, "s_nom"] *= float(multiplier)

    solve_checked(network, f"regional all-rating case ×{multiplier:.2f}")
    return {
        "rating_multiplier": float(multiplier),
        "dispatch_down_mwh": dispatch_down_mwh(network),
        "unserved_mwh": unserved_energy_mwh(network),
        "binding_line_count": int(
            len(gridkit.binding(network, threshold=BINDING_THRESHOLD))
        ),
    }


def line_reinforcement_trials(
    scenario: str,
    baseline_dispatch_down: float,
    ordered_lines: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for position, line_name in enumerate(ordered_lines, start=1):
        print(
            f"[{position:02d}/{len(ordered_lines):02d}] "
            f"Testing {line_name} at +25% rating"
        )
        experiment = load_network(scenario, REGIONAL_SCOPE)
        original_rating = float(experiment.lines.at[line_name, "s_nom"])
        new_rating = original_rating * LINE_REINFORCEMENT_MULTIPLIER
        gridkit.set_rating(experiment, line_name, new_rating)
        solve_checked(experiment, f"+25% rating trial for {line_name}")

        new_dispatch_down = dispatch_down_mwh(experiment)
        saved_dispatch_down = zero_small(
            baseline_dispatch_down - new_dispatch_down
        )
        added_rating = new_rating - original_rating
        changed_loading = gridkit.line_loading(experiment)[line_name]

        rows.append(
            {
                "line": line_name,
                "new_rating_mva": new_rating,
                "added_rating_mva": added_rating,
                "dispatch_down_after_plus_25_mwh": new_dispatch_down,
                "saved_dispatch_down_plus_25_mwh": saved_dispatch_down,
                "saved_mwh_per_added_mva": (
                    saved_dispatch_down / added_rating
                    if added_rating > NUMERICAL_TOLERANCE
                    else np.nan
                ),
                "binding_hours_after_plus_25": int(
                    (changed_loading >= BINDING_THRESHOLD).sum()
                ),
                "max_loading_after_plus_25_pct": float(
                    100.0 * changed_loading.max()
                ),
                "unserved_after_plus_25_mwh": unserved_energy_mwh(experiment),
            }
        )

    return pd.DataFrame(rows).set_index("line")


def network_rating_sensitivity(
    scenario: str,
    baseline_dispatch_down: float,
    baseline_unserved: float,
    baseline_binding_lines: int,
    relaxed_dispatch_down: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []

    for multiplier in NETWORK_RATING_MULTIPLIERS:
        if abs(multiplier - 1.0) < NUMERICAL_TOLERANCE:
            case = {
                "dispatch_down_mwh": baseline_dispatch_down,
                "unserved_mwh": baseline_unserved,
                "binding_line_count": baseline_binding_lines,
            }
        else:
            print(
                f"[network] Testing all regional branch ratings at "
                f"{multiplier:.0%}"
            )
            case = run_all_branch_rating_case(scenario, multiplier)

        total_down = float(case["dispatch_down_mwh"])
        # This split is valid inside the supplied regional model.  It should
        # not be interpreted as an operational EirGrid curtailment figure.
        rating_sensitive_down = max(
            zero_small(total_down - relaxed_dispatch_down), 0.0
        )

        rows.append(
            {
                "rating_multiplier": float(multiplier),
                "total_dispatch_down_mwh": total_down,
                "regional_rating_sensitive_dispatch_down_mwh": rating_sensitive_down,
                "saved_vs_baseline_mwh": zero_small(
                    baseline_dispatch_down - total_down
                ),
                "unserved_mwh": float(case["unserved_mwh"]),
                "binding_line_count": int(case["binding_line_count"]),
            }
        )

    return pd.DataFrame(rows).sort_values("rating_multiplier")


def all_island_robustness(
    scenario: str,
    regional_lines: pd.DataFrame,
) -> pd.DataFrame:
    print("\nRunning all-island robustness check...")
    full = load_network(scenario, FULL_SCOPE)
    solve_checked(full, "all-island robustness solve")
    full_loading = gridkit.line_loading(full)

    common = [name for name in regional_lines.index if name in full_loading.columns]
    rows: list[dict[str, object]] = []

    for name in regional_lines.index:
        regional = regional_lines.loc[name]
        if name not in common:
            rows.append(
                {
                    "line": name,
                    "bus0": regional["bus0"],
                    "bus1": regional["bus1"],
                    "regional_binding_hours": int(regional["binding_hours"]),
                    "regional_p95_loading_pct": float(regional["p95_loading_pct"]),
                    "regional_max_loading_pct": float(regional["max_loading_pct"]),
                    "present_in_all_island": False,
                    "all_island_binding_hours": np.nan,
                    "all_island_p95_loading_pct": np.nan,
                    "all_island_max_loading_pct": np.nan,
                }
            )
            continue

        series = full_loading[name]
        rows.append(
            {
                "line": name,
                "bus0": regional["bus0"],
                "bus1": regional["bus1"],
                "regional_binding_hours": int(regional["binding_hours"]),
                "regional_p95_loading_pct": float(regional["p95_loading_pct"]),
                "regional_max_loading_pct": float(regional["max_loading_pct"]),
                "present_in_all_island": True,
                "all_island_binding_hours": int(
                    (series >= BINDING_THRESHOLD).sum()
                ),
                "all_island_p95_loading_pct": float(100.0 * series.quantile(0.95)),
                "all_island_max_loading_pct": float(100.0 * series.max()),
            }
        )

    return pd.DataFrame(rows).set_index("line")


def plot_results(
    scenario: str,
    lines: pd.DataFrame,
    sensitivity: pd.DataFrame,
    figure_dir: Path,
    robustness: pd.DataFrame | None,
) -> None:
    frequency = lines.head(TOP_N).sort_values("binding_hours")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(frequency["label"], frequency["binding_hours"])
    ax.set_xlabel(f"Hours at or above {BINDING_THRESHOLD:.1%} of rating")
    ax.set_title(f"{scenario} North-West: most frequently binding lines")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "01_binding_hours.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    impact = lines.sort_values(
        "saved_dispatch_down_plus_25_mwh", ascending=False
    ).head(TOP_N)
    impact = impact.sort_values("saved_dispatch_down_plus_25_mwh")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(impact["label"], impact["saved_dispatch_down_plus_25_mwh"])
    ax.set_xlabel("Dispatch-down recovered by +25% line rating (MWh)")
    ax.set_title(f"{scenario} North-West: one-line reinforcement sensitivity")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(
        figure_dir / "02_recovered_energy_plus_25_pct.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(
        sensitivity["rating_multiplier"],
        sensitivity["total_dispatch_down_mwh"],
        marker="o",
        label="total dispatch-down",
    )
    ax.plot(
        sensitivity["rating_multiplier"],
        sensitivity["regional_rating_sensitive_dispatch_down_mwh"],
        marker="o",
        label="rating-sensitive component",
    )
    ax.set_xlabel("Multiplier applied to all regional branch ratings")
    ax.set_ylabel("Energy over 168-hour model (MWh)")
    ax.set_title(f"{scenario} North-West: thermal-rating sensitivity")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        figure_dir / "03_network_rating_sensitivity.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    if robustness is not None and not robustness.empty:
        comparable = robustness[robustness["present_in_all_island"]].copy()
        comparable = comparable.sort_values(
            "regional_max_loading_pct", ascending=False
        ).head(TOP_N)
        comparable = comparable.iloc[::-1]

        y = np.arange(len(comparable))
        fig, ax = plt.subplots(figsize=(11, 6.5))
        ax.barh(
            y - 0.18,
            comparable["regional_max_loading_pct"],
            height=0.35,
            label="North-West regional model",
        )
        ax.barh(
            y + 0.18,
            comparable["all_island_max_loading_pct"],
            height=0.35,
            label="same circuit in all-island model",
        )
        ax.set_yticks(y, comparable.index)
        ax.set_xlabel("Maximum loading over 168 hours (%)")
        ax.set_title("Scope robustness check for North-West circuits")
        ax.grid(axis="x", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figure_dir / "04_scope_robustness_max_loading.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()
    scenario = args.scenario
    output_dir = PROJECT_ROOT / "results" / "problem_3_1" / "question_1_final"
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    gridkit.quiet()
    print("\nTPSA Hackathon 2026 — Problem 3.1, Question 1")
    print(f"Primary model: {scenario} / {REGIONAL_SCOPE}\n")

    baseline = load_network(scenario, REGIONAL_SCOPE)
    solve_checked(baseline, "North-West baseline")

    loading = gridkit.line_loading(baseline)
    if float(loading.max().max()) > 1.0 + 1e-5:
        raise RuntimeError("Solved baseline contains a line above its rating.")

    baseline_dispatch_down = dispatch_down_mwh(baseline)
    baseline_unserved = unserved_energy_mwh(baseline)
    lines = regional_line_table(baseline, loading)
    loading.to_csv(table_dir / "01_hourly_line_loading_ratio.csv", index_label="snapshot")

    components = network_components(baseline)
    component_rows = []
    for number, component in enumerate(components, start=1):
        for bus in sorted(component):
            component_rows.append(
                {
                    "component": number,
                    "component_size": len(component),
                    "bus": bus,
                    "isolated_bus": len(component) == 1,
                }
            )
    pd.DataFrame(component_rows).to_csv(
        table_dir / "00_network_components.csv", index=False
    )

    boundaries = boundary_diagnostics(baseline)
    if not boundaries.empty:
        boundaries.to_csv(table_dir / "00_boundary_diagnostics.csv", index=False)

    print("Estimating dispatch-down remaining after regional ratings are relaxed...")
    relaxed = run_all_branch_rating_case(scenario, RATING_RELAXATION_MULTIPLIER)
    relaxed_dispatch_down = float(relaxed["dispatch_down_mwh"])
    regional_rating_sensitive_down = max(
        zero_small(baseline_dispatch_down - relaxed_dispatch_down), 0.0
    )

    trials = line_reinforcement_trials(
        scenario,
        baseline_dispatch_down,
        list(lines.index),
    )
    lines = lines.join(trials)

    if regional_rating_sensitive_down > NUMERICAL_TOLERANCE:
        lines["regional_rating_sensitive_recovered_pct"] = (
            100.0
            * lines["saved_dispatch_down_plus_25_mwh"]
            / regional_rating_sensitive_down
        )
    else:
        lines["regional_rating_sensitive_recovered_pct"] = np.nan

    lines.to_csv(
        table_dir / "02_line_constraint_and_rating_results.csv",
        index_label="line",
    )
    lines.sort_values(
        "saved_dispatch_down_plus_25_mwh", ascending=False
    ).to_csv(
        table_dir / "03_lines_ranked_by_recovered_energy.csv",
        index_label="line",
    )

    sensitivity = network_rating_sensitivity(
        scenario=scenario,
        baseline_dispatch_down=baseline_dispatch_down,
        baseline_unserved=baseline_unserved,
        baseline_binding_lines=int((lines["binding_hours"] > 0).sum()),
        relaxed_dispatch_down=relaxed_dispatch_down,
    )
    sensitivity.to_csv(table_dir / "04_network_rating_sensitivity.csv", index=False)

    robustness: pd.DataFrame | None = None
    if not args.skip_all_island_check:
        robustness = all_island_robustness(scenario, lines)
        robustness.to_csv(
            table_dir / "05_all_island_scope_robustness.csv",
            index_label="line",
        )

    plot_results(scenario, lines, sensitivity, figure_dir, robustness)

    dominant = lines.sort_values(
        [
            "binding_hours",
            "saved_dispatch_down_plus_25_mwh",
            "p95_loading_pct",
        ],
        ascending=False,
    ).iloc[0]
    dominant_name = dominant.name

    print("\n=== PRIMARY NORTH-WEST RESULT ===")
    print(f"Snapshots:                         {len(baseline.snapshots)}")
    print(f"Buses / lines:                     {len(baseline.buses)} / {len(baseline.lines)}")
    print(f"Connected components:              {len(components)}")
    isolated = [sorted(c)[0] for c in components if len(c) == 1]
    print(f"Isolated buses:                    {', '.join(isolated) if isolated else 'none'}")
    if not boundaries.empty:
        fixed_count = int(boundaries["fixed_by_static_bounds"].sum())
        print(f"Boundary generators:               {len(boundaries)}")
        print(f"Fixed / non-fixed boundaries:      {fixed_count} / {len(boundaries) - fixed_count}")
        most_variable = boundaries.sort_values("dispatch_range_mw", ascending=False).iloc[0]
        print(
            "Largest boundary dispatch range:    "
            f"{most_variable['generator']} = {most_variable['dispatch_range_mw']:.2f} MW"
        )

    print(f"Baseline dispatch-down:             {baseline_dispatch_down:,.2f} MWh")
    print(f"Ratings-relaxed dispatch-down:      {relaxed_dispatch_down:,.2f} MWh")
    print(
        "Regional rating-sensitive component:  "
        f"{regional_rating_sensitive_down:,.2f} MWh"
    )
    print(f"Unserved energy:                    {baseline_unserved:,.6f} MWh")

    print("\nDominant line in supplied regional model:")
    print(f"  {dominant_name}: {dominant['bus0']} ↔ {dominant['bus1']}")
    print(f"  rating:                    {dominant['rating_mva']:.2f} MVA")
    print(f"  binding hours:             {int(dominant['binding_hours'])}")
    print(f"  p95 / max loading:         {dominant['p95_loading_pct']:.2f}% / {dominant['max_loading_pct']:.2f}%")
    print(
        "  +25% rating saves:         "
        f"{dominant['saved_dispatch_down_plus_25_mwh']:.2f} MWh"
    )

    print("\n=== IMPORTANT MODEL INTERPRETATION ===")
    print(
        "This is the answer inside the supplied 15-node North-West regional "
        "model. The regional model has pinned boundary injections, synthetic "
        "hourly profiles, and known aggregation limitations, so the result "
        "must be reported as model-specific rather than as an operational "
        "EirGrid finding."
    )

    if robustness is not None and dominant_name in robustness.index:
        row = robustness.loc[dominant_name]
        if bool(row["present_in_all_island"]):
            print("\nAll-island robustness check for the same circuit:")
            print(
                f"  regional:   binding={int(row['regional_binding_hours'])} h, "
                f"max={row['regional_max_loading_pct']:.2f}%"
            )
            print(
                f"  all-island: binding={int(row['all_island_binding_hours'])} h, "
                f"max={row['all_island_max_loading_pct']:.2f}%"
            )
            print(
                "  This comparison is a sensitivity check on network scope; "
                "it does not replace the North-West analysis required by Problem 3.1."
            )

    print(f"\nOutputs saved to:\n{output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
