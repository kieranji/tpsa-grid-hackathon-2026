from __future__ import annotations #标准开头
from pathlib import Path #统一Apple和Windows的/和\

import sys #让系统可以找到其他文件夹里面的内容
import matplotlib #绘图功能
matplotlib.use("Agg") #不是弹窗而是直接保存图片
import matplotlib.pyplot as plt #绘图接口
import numpy as np #数学计算工具
import pandas as pd #表格分析工具

# -----------------------------------------
# A模块：查找项目并且倒入官方工具包
# -----------------------------------------

SCRIPT_PATH=Path(__file__).resolve() #获取项目文件路径
PROJECT_ROOT=SCRIPT_PATH.parents[2] #记录文件路径顶层文件夹（嵌套两层）
KIT_DIR=PROJECT_ROOT/"data"/"participant-kit" #工具包路径
GRIDKIT_PATH=KIT_DIR/"gridkit.py" #定位到工具文件
if not GRIDKIT_PATH.exists(): #如果工具文件不存在
    raise FileNotFoundError(
        "Cannot find data/participant-kit/gridkit.py.\n"
        "Save this file as "
        "src/problem_3_1/question_1_line_constraints.py."
    )
sys.path.insert(0,str(KIT_DIR)) #把participant-kit放在最前面，避免不小心导入其他文件
import gridkit #导入工具函数

# ----------------------------------------
# B模块：分析设定
# ----------------------------------------

# Use the 2033 winter-peak case recommended by the organisers as a starting case.
SCENARIO = "WP2033"

# Use the North-West subnetwork corresponding to Problem 3.1.
SCOPE = "north-west"

# Match gridkit.binding(): a line is binding at 99.9% or more of rating.
BINDING_THRESHOLD = 0.999

# Test a 25% reinforcement for every line, one line at a time.
LINE_REINFORCEMENT_MULTIPLIER = 1.25

# Use a huge multiplier to estimate dispatch-down with no binding branch ratings.
COPPER_PLATE_MULTIPLIER = 1000.0

# Test how the whole network responds to different thermal-rating levels.
NETWORK_RATING_MULTIPLIERS = (0.75, 1.00, 1.10, 1.25, 1.50, 2.00)

# Print and plot no more than ten lines.
TOP_N = 10

# Treat extremely small floating-point differences as zero.
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# C. OUTPUT FOLDERS
# ---------------------------------------------------------------------------

# Keep all outputs for this question together.
OUTPUT_DIR = PROJECT_ROOT / "results" / "problem_3_1" / "question_1"

# Store data tables separately from charts.
TABLE_DIR = OUTPUT_DIR / "tables"

# Store charts separately from data tables.
FIGURE_DIR = OUTPUT_DIR / "figures"

# Create the table folder and any missing parent folders.
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# Create the figure folder and any missing parent folders.
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# D. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def load_network():
    """Return a fresh, untouched copy of the selected official network."""

    # Loading a fresh copy prevents one experiment from changing another.
    return gridkit.load(SCENARIO, SCOPE)


def solve_checked(network, label: str) -> None:
    """Solve a network and fail clearly if no valid result is produced."""

    # Run the organiser's wrapper around PyPSA and the HiGHS solver.
    result = gridkit.solve(network)

    # PyPSA normally returns a tuple such as ("ok", "optimal").
    if isinstance(result, tuple) and len(result) >= 2:
        # Convert the broad solver status to lowercase text.
        status = str(result[0]).lower()

        # Convert the termination condition to lowercase text.
        condition = str(result[1]).lower()

        # Reject unsuccessful or non-optimal solves.
        if status != "ok" or "optimal" not in condition:
            raise RuntimeError(
                f"{label} failed: status={result[0]!r}, "
                f"condition={result[1]!r}"
            )

    # The official examples also check that dispatch results were created.
    if not len(network.generators_t.p.columns):
        raise RuntimeError(f"{label} produced no generator-dispatch results.")

    # This question also requires solved line flows.
    if len(network.lines) and not len(network.lines_t.p0.columns):
        raise RuntimeError(f"{label} produced no line-flow results.")


def dispatch_down_mwh(network) -> float:
    """Return total wind/solar dispatch-down across the supplied week."""

    # Ask the official helper for per-generator dispatch-down.
    table = gridkit.dispatch_down(network)

    # Return zero if there are no weather-driven generators.
    if table.empty:
        return 0.0

    # Sum all generators' withheld energy.
    return float(table["dispatch_down_mwh"].sum())


def unserved_energy_mwh(network) -> float:
    """Return total unmet electricity demand across the supplied week."""

    # Ask the official helper for unmet demand by bus.
    series = gridkit.unserved(network)

    # Return zero when all demand was served.
    if series.empty:
        return 0.0

    # Sum unmet demand across all buses.
    return float(series.sum())


def clean_number(value: float) -> float:
    """Replace tiny solver noise with an exact zero."""

    # Numerical solvers can produce differences such as -0.00000001.
    if abs(value) < TOLERANCE:
        return 0.0

    # Keep any meaningful value unchanged.
    return float(value)


def make_line_label(line_name: str, bus0: str, bus1: str) -> str:
    """Create a readable label containing a line ID and its endpoints."""

    # Include both the official line name and the two connected substations.
    return f"{line_name}: {bus0} ↔ {bus1}"


def run_all_branch_rating_case(multiplier: float) -> dict[str, float]:
    """Multiply every line/transformer rating, solve, and return key results."""

    # Start from a clean copy of the official network.
    network = load_network()

    # Scale every AC line rating by the requested multiplier.
    network.lines.loc[:, "s_nom"] = (
        network.lines["s_nom"] * float(multiplier)
    )

    # Scale transformer ratings too if this network contains transformers.
    if len(network.transformers):
        network.transformers.loc[:, "s_nom"] = (
            network.transformers["s_nom"] * float(multiplier)
        )

    # Solve the changed network for all 168 hours.
    solve_checked(network, f"all-branch rating case × {multiplier:.2f}")

    # Count lines that bind for at least one hour.
    binding_line_count = len(
        gridkit.binding(network, threshold=BINDING_THRESHOLD)
    )

    # Return the system-level results required later.
    return {
        "rating_multiplier": float(multiplier),
        "dispatch_down_mwh": dispatch_down_mwh(network),
        "unserved_mwh": unserved_energy_mwh(network),
        "binding_line_count": int(binding_line_count),
    }


# ---------------------------------------------------------------------------
# E. MAIN ANALYSIS
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the complete first-question analysis."""

    # Suppress long PyPSA/HiGHS logs without changing any result.
    gridkit.quiet()

    # Print a clear start message.
    print("\nTPSA Hackathon 2026 — Problem 3.1, first question")

    # Print the selected scenario and geographical scope.
    print(f"Running {SCENARIO} / {SCOPE}\n")

    # -----------------------------------------------------------------------
    # 1. SOLVE THE ORIGINAL 168-HOUR NETWORK
    # -----------------------------------------------------------------------

    # Load the untouched official model.
    baseline = load_network()

    # Show its size before solving.
    print("Network summary:")
    print(gridkit.summary(baseline).to_string())

    # Solve the least-cost dispatch and transmission constraints.
    solve_checked(baseline, "baseline")

    # Count the supplied hourly snapshots.
    number_of_hours = len(baseline.snapshots)

    # Calculate |flow| / rating for every line in every hour.
    loading = gridkit.line_loading(baseline)

    # Calculate total renewable dispatch-down in the original model.
    baseline_down = dispatch_down_mwh(baseline)

    # Calculate unmet demand in the original model.
    baseline_unserved = unserved_energy_mwh(baseline)

    # Begin the line result table with endpoints and original rating.
    lines = baseline.lines[["bus0", "bus1", "s_nom"]].copy()

    # Rename s_nom to make its meaning and unit explicit.
    lines = lines.rename(columns={"s_nom": "rating_mva"})

    # Build a readable label for each line.
    lines["label"] = [
        make_line_label(line_name, row.bus0, row.bus1)
        for line_name, row in lines.iterrows()
    ]

    # Count hours at or above 99.9% of rating.
    lines["binding_hours"] = (
        loading >= BINDING_THRESHOLD
    ).sum().reindex(lines.index).fillna(0).astype(int)

    # Express binding time as a percentage of all supplied hours.
    lines["binding_share_pct"] = (
        100.0 * lines["binding_hours"] / number_of_hours
    )

    # Count hours at or above 90% of rating.
    lines["hours_at_or_above_90_pct"] = (
        loading >= 0.90
    ).sum().reindex(lines.index).fillna(0).astype(int)

    # Calculate mean loading across the whole week.
    lines["mean_loading_pct"] = (
        100.0 * loading.mean().reindex(lines.index)
    )

    # Calculate 95th-percentile loading as a measure of sustained high loading.
    lines["p95_loading_pct"] = (
        100.0 * loading.quantile(0.95).reindex(lines.index)
    )

    # Calculate the maximum loading reached by each line.
    lines["max_loading_pct"] = (
        100.0 * loading.max().reindex(lines.index)
    )

    # Record the hour at which each line reached its maximum loading.
    lines["peak_loading_snapshot"] = [
        loading[line_name].idxmax()
        for line_name in lines.index
    ]

    # Name the index so exported CSV files have a clear first column.
    lines.index.name = "line"

    # Sort by binding frequency, then by sustained and maximum loading.
    lines = lines.sort_values(
        ["binding_hours", "p95_loading_pct", "max_loading_pct"],
        ascending=False,
    )

    # Save the complete 168-hour loading matrix for reproducibility.
    loading.to_csv(
        TABLE_DIR / "01_hourly_line_loading_ratio.csv",
        index_label="snapshot",
    )

    # -----------------------------------------------------------------------
    # 2. ESTIMATE SURPLUS-BASED VS CONSTRAINT-BASED DISPATCH-DOWN
    # -----------------------------------------------------------------------

    # Explain that the next solve removes practical branch limits.
    print("\nEstimating the surplus floor with all ratings lifted...")

    # Solve a counterfactual with every line/transformer rating multiplied by 1000.
    copper_plate = run_all_branch_rating_case(COPPER_PLATE_MULTIPLIER)

    # What remains withheld is treated as surplus-based dispatch-down.
    surplus_floor = float(copper_plate["dispatch_down_mwh"])

    # The recovered amount is the network-constraint component.
    constraint_based_down = max(
        clean_number(baseline_down - surplus_floor),
        0.0,
    )

    # -----------------------------------------------------------------------
    # 3. INCREASE EACH LINE RATING BY 25%, ONE LINE AT A TIME
    # -----------------------------------------------------------------------

    # Prepare one intervention-result row per line.
    intervention_rows: list[dict[str, float | int | str]] = []

    # Remember the number of lines for readable progress messages.
    total_lines = len(lines)

    # Loop over every official line ID.
    for position, line_name in enumerate(lines.index, start=1):
        # Print progress because each line requires a new optimisation.
        print(
            f"[{position:02d}/{total_lines:02d}] "
            f"Testing {line_name} at +25% rating"
        )

        # Load a clean network for this one-line experiment.
        experiment = load_network()

        # Read the original rating of the selected line.
        original_rating = float(experiment.lines.at[line_name, "s_nom"])

        # Calculate its new 25%-higher rating.
        new_rating = original_rating * LINE_REINFORCEMENT_MULTIPLIER

        # Change only this line, leaving all other inputs unchanged.
        gridkit.set_rating(experiment, line_name, new_rating)

        # Re-solve all 168 hours after the intervention.
        solve_checked(experiment, f"+25% experiment for {line_name}")

        # Calculate total dispatch-down after the intervention.
        new_down = dispatch_down_mwh(experiment)

        # Calculate renewable energy recovered relative to baseline.
        saved_down = clean_number(baseline_down - new_down)

        # Calculate the amount of thermal capacity added.
        added_rating = new_rating - original_rating

        # Calculate MWh recovered per extra MVA for cross-line comparison.
        saved_per_added_mva = (
            saved_down / added_rating
            if added_rating > TOLERANCE
            else np.nan
        )

        # Recalculate the changed line's loading after redispatch.
        changed_loading = gridkit.line_loading(experiment)[line_name]

        # Store the experiment's results.
        intervention_rows.append(
            {
                "line": line_name,
                "new_rating_mva": new_rating,
                "added_rating_mva": added_rating,
                "dispatch_down_after_plus_25_mwh": new_down,
                "saved_dispatch_down_plus_25_mwh": saved_down,
                "saved_mwh_per_added_mva": saved_per_added_mva,
                "binding_hours_after_plus_25": int(
                    (changed_loading >= BINDING_THRESHOLD).sum()
                ),
                "max_loading_after_plus_25_pct": float(
                    100.0 * changed_loading.max()
                ),
                "unserved_after_plus_25_mwh": unserved_energy_mwh(
                    experiment
                ),
            }
        )

    # Convert intervention rows into a table indexed by line ID.
    interventions = pd.DataFrame(intervention_rows).set_index("line")

    # Join baseline line statistics with the +25% intervention results.
    lines = lines.join(interventions)

    # Calculate the share of the baseline constraint component recovered.
    if constraint_based_down > TOLERANCE:
        lines["constraint_component_recovered_pct"] = (
            100.0
            * lines["saved_dispatch_down_plus_25_mwh"]
            / constraint_based_down
        )
    else:
        lines["constraint_component_recovered_pct"] = np.nan

    # Save the main combined line-results table.
    lines.to_csv(
        TABLE_DIR / "02_line_constraint_and_rating_results.csv",
        index_label="line",
    )

    # Save another copy ranked by the effect of reinforcement.
    lines.sort_values(
        "saved_dispatch_down_plus_25_mwh",
        ascending=False,
    ).to_csv(
        TABLE_DIR / "03_lines_ranked_by_recovered_energy.csv",
        index_label="line",
    )

    # -----------------------------------------------------------------------
    # 4. SCALE ALL NETWORK RATINGS TOGETHER
    # -----------------------------------------------------------------------

    # Prepare rows for the network-wide thermal-rating curve.
    network_rows: list[dict[str, float | int]] = []

    # Test each requested multiplier.
    for multiplier in NETWORK_RATING_MULTIPLIERS:
        # Reuse the already-solved baseline when the multiplier is exactly 1.
        if abs(multiplier - 1.0) < TOLERANCE:
            case = {
                "rating_multiplier": 1.0,
                "dispatch_down_mwh": baseline_down,
                "unserved_mwh": baseline_unserved,
                "binding_line_count": int(
                    (lines["binding_hours"] > 0).sum()
                ),
            }
        else:
            # Print the current system-wide experiment.
            print(
                f"[network] Testing all branch ratings at "
                f"{multiplier:.0%} of baseline"
            )

            # Solve the system-wide rating case.
            case = run_all_branch_rating_case(multiplier)

        # Calculate dispatch-down above the common surplus floor.
        constrained_part = max(
            clean_number(
                float(case["dispatch_down_mwh"]) - surplus_floor
            ),
            0.0,
        )

        # Store the network-wide result.
        network_rows.append(
            {
                "rating_multiplier": float(multiplier),
                "total_dispatch_down_mwh": float(
                    case["dispatch_down_mwh"]
                ),
                "constraint_based_dispatch_down_mwh": constrained_part,
                "saved_vs_baseline_mwh": clean_number(
                    baseline_down - float(case["dispatch_down_mwh"])
                ),
                "unserved_mwh": float(case["unserved_mwh"]),
                "binding_line_count": int(case["binding_line_count"]),
            }
        )

    # Convert and sort the network-wide results.
    network_sensitivity = pd.DataFrame(network_rows).sort_values(
        "rating_multiplier"
    )

    # Save the network-wide rating-response table.
    network_sensitivity.to_csv(
        TABLE_DIR / "04_network_rating_sensitivity.csv",
        index=False,
    )

    # -----------------------------------------------------------------------
    # 5. CALCULATE DESCRIPTIVE CORRELATIONS
    # -----------------------------------------------------------------------

    # Calculate ordinary linear correlation across lines.
    pearson = lines["rating_mva"].corr(
        lines["binding_hours"],
        method="pearson",
    )

    # Calculate a rank correlation without adding a SciPy dependency.
    spearman = (
        lines["rating_mva"].rank(method="average").corr(
            lines["binding_hours"].rank(method="average"),
            method="pearson",
        )
    )

    # -----------------------------------------------------------------------
    # 6. DRAW THREE OUTPUT CHARTS
    # -----------------------------------------------------------------------

    # Select the most frequently binding lines.
    frequency_chart = lines.head(TOP_N).sort_values(
        "binding_hours",
        ascending=True,
    )

    # Create the first independent figure.
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Draw binding hours as horizontal bars.
    ax.barh(frequency_chart["label"], frequency_chart["binding_hours"])

    # Label the numerical axis.
    ax.set_xlabel(
        f"Hours at or above {BINDING_THRESHOLD:.1%} of rating"
    )

    # Give the chart a precise title.
    ax.set_title(
        f"{SCENARIO} {SCOPE}: most frequently constrained lines"
    )

    # Draw a light grid behind the bars.
    ax.grid(axis="x", alpha=0.3)

    # Prevent long endpoint labels from being cut off.
    fig.tight_layout()

    # Save the first chart.
    fig.savefig(
        FIGURE_DIR / "01_binding_hours.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Release the first figure.
    plt.close(fig)

    # Select lines with the greatest recovered energy.
    impact_chart = lines.sort_values(
        "saved_dispatch_down_plus_25_mwh",
        ascending=False,
    ).head(TOP_N)

    # Reverse them for a readable horizontal chart.
    impact_chart = impact_chart.sort_values(
        "saved_dispatch_down_plus_25_mwh",
        ascending=True,
    )

    # Create the second independent figure.
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Draw recovered dispatch-down as horizontal bars.
    ax.barh(
        impact_chart["label"],
        impact_chart["saved_dispatch_down_plus_25_mwh"],
    )

    # Label the numerical axis.
    ax.set_xlabel(
        "Renewable dispatch-down recovered by +25% rating (MWh)"
    )

    # Give the chart a precise title.
    ax.set_title(
        f"{SCENARIO} {SCOPE}: effect of reinforcing one line"
    )

    # Draw a light grid behind the bars.
    ax.grid(axis="x", alpha=0.3)

    # Prevent long endpoint labels from being cut off.
    fig.tight_layout()

    # Save the second chart.
    fig.savefig(
        FIGURE_DIR / "02_recovered_energy_plus_25_pct.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Release the second figure.
    plt.close(fig)

    # Create the third independent figure.
    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    # Plot total dispatch-down against the network-wide rating multiplier.
    ax.plot(
        network_sensitivity["rating_multiplier"],
        network_sensitivity["total_dispatch_down_mwh"],
        marker="o",
        label="total dispatch-down",
    )

    # Plot only the constraint-based part above the surplus floor.
    ax.plot(
        network_sensitivity["rating_multiplier"],
        network_sensitivity["constraint_based_dispatch_down_mwh"],
        marker="o",
        label="constraint-based component",
    )

    # Label the horizontal axis.
    ax.set_xlabel("Multiplier applied to every branch thermal rating")

    # Label the vertical axis.
    ax.set_ylabel("Energy over the 168-hour model (MWh)")

    # Give the chart a precise title.
    ax.set_title(
        f"{SCENARIO} {SCOPE}: thermal rating versus dispatch-down"
    )

    # Draw a light grid.
    ax.grid(alpha=0.3)

    # Identify the two curves.
    ax.legend()

    # Prevent labels from being cut off.
    fig.tight_layout()

    # Save the third chart.
    fig.savefig(
        FIGURE_DIR / "03_network_rating_sensitivity.png",
        dpi=200,
        bbox_inches="tight",
    )

    # Release the third figure.
    plt.close(fig)

    # -----------------------------------------------------------------------
    # 7. PRINT THE IMPORTANT RESULTS
    # -----------------------------------------------------------------------

    # Print the system-level baseline and counterfactual totals.
    print("\n=== SYSTEM RESULTS ===")
    print(f"Hourly snapshots:                  {number_of_hours}")
    print(f"Baseline dispatch-down:            {baseline_down:,.2f} MWh")
    print(f"Surplus floor, ratings lifted:     {surplus_floor:,.2f} MWh")
    print(f"Constraint-based component:        {constraint_based_down:,.2f} MWh")
    print(f"Baseline unserved energy:          {baseline_unserved:,.6f} MWh")

    # Choose concise columns for the frequency ranking.
    frequency_columns = [
        "bus0",
        "bus1",
        "rating_mva",
        "binding_hours",
        "binding_share_pct",
        "p95_loading_pct",
        "max_loading_pct",
    ]

    # Print the frequency ranking.
    print("\n=== MOST FREQUENTLY CONSTRAINED LINES ===")
    print(
        lines[frequency_columns]
        .head(TOP_N)
        .round(2)
        .to_string()
    )

    # Choose concise columns for the intervention ranking.
    impact_columns = [
        "bus0",
        "bus1",
        "rating_mva",
        "binding_hours",
        "saved_dispatch_down_plus_25_mwh",
        "saved_mwh_per_added_mva",
        "binding_hours_after_plus_25",
    ]

    # Print the causal +25% reinforcement ranking.
    print("\n=== LARGEST EFFECT OF A +25% RATING INCREASE ===")
    print(
        lines.sort_values(
            "saved_dispatch_down_plus_25_mwh",
            ascending=False,
        )[impact_columns]
        .head(TOP_N)
        .round(2)
        .to_string()
    )

    # Print cross-sectional correlations as secondary descriptive evidence.
    print("\n=== DESCRIPTIVE RATING/FREQUENCY ASSOCIATION ===")
    print(f"Pearson correlation: {pearson:.3f}")
    print(f"Rank correlation:    {spearman:.3f}")

    # Warn that cross-line correlation is not a causal result.
    print(
        "Correlation is descriptive only; topology, reactance, "
        "generation location, and demand location also affect congestion."
    )

    # Show the exact output folder.
    print(f"\nAll outputs saved to:\n{OUTPUT_DIR}")

    # Return success to the terminal.
    return 0


# Execute main only when this file is run directly.
if __name__ == "__main__":
    # Pass main's return value back to the shell as the exit status.
    raise SystemExit(main())