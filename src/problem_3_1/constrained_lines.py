"""TPSA Hackathon 2026 — Problem 3.1, Question 1.

Ranks constrained lines in the WP2033 North-West model and measures how thermal-rating changes affect renewable dispatch-down."""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KIT_DIR = PROJECT_ROOT / 'data' / 'participant-kit'
GRIDKIT_PATH = KIT_DIR / 'gridkit.py'
if not GRIDKIT_PATH.exists():
    raise FileNotFoundError('Cannot find data/participant-kit/gridkit.py.\nSave this file as src/problem_3_1/constrained_lines.py.')
sys.path.insert(0, str(KIT_DIR))
import gridkit
SCENARIO = 'WP2033'
SCOPE = 'north-west'
# Treat a line as binding at 99.9% or more of its rating.
BINDING_THRESHOLD = 0.999
LINE_REINFORCEMENT_MULTIPLIER = 1.25
# Counterfactual only: remove branch-capacity bottlenecks without changing topology.
COPPER_PLATE_MULTIPLIER = 1000.0
NETWORK_RATING_MULTIPLIERS = (0.75, 1.0, 1.1, 1.25, 1.5, 2.0)
TOP_N = 10
TOLERANCE = 1e-06
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1'
TABLE_DIR = OUTPUT_DIR / 'tables'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def load_network():
    return gridkit.load(SCENARIO, SCOPE)

def solve_checked(network, label: str) -> None:
    result = gridkit.solve(network)
    if isinstance(result, tuple) and len(result) >= 2:
        status = str(result[0]).lower()
        condition = str(result[1]).lower()
        if status != 'ok' or 'optimal' not in condition:
            raise RuntimeError(f'{label} failed: status={result[0]!r}, condition={result[1]!r}')
    if not len(network.generators_t.p.columns):
        raise RuntimeError(f'{label} produced no generator-dispatch results.')
    if len(network.lines) and (not len(network.lines_t.p0.columns)):
        raise RuntimeError(f'{label} produced no line-flow results.')

def dispatch_down_mwh(network) -> float:
    table = gridkit.dispatch_down(network)
    if table.empty:
        return 0.0
    return float(table['dispatch_down_mwh'].sum())

def unserved_energy_mwh(network) -> float:
    series = gridkit.unserved(network)
    if series.empty:
        return 0.0
    return float(series.sum())

def clean_number(value: float) -> float:
    if abs(value) < TOLERANCE:
        return 0.0
    return float(value)

def make_line_label(line_name: str, bus0: str, bus1: str) -> str:
    return f'{line_name}: {bus0} ↔ {bus1}'

def run_all_branch_rating_case(multiplier: float) -> dict[str, float]:
    network = load_network()
    network.lines.loc[:, 's_nom'] = network.lines['s_nom'] * float(multiplier)
    if len(network.transformers):
        network.transformers.loc[:, 's_nom'] = network.transformers['s_nom'] * float(multiplier)
    solve_checked(network, f'all-branch rating case × {multiplier:.2f}')
    binding_line_count = len(gridkit.binding(network, threshold=BINDING_THRESHOLD))
    return {'rating_multiplier': float(multiplier), 'dispatch_down_mwh': dispatch_down_mwh(network), 'unserved_mwh': unserved_energy_mwh(network), 'binding_line_count': int(binding_line_count)}

def main() -> int:
    gridkit.quiet()
    print('\nTPSA Hackathon 2026 — Problem 3.1, first question')
    print(f'Running {SCENARIO} / {SCOPE}\n')
    baseline = load_network()
    print('Network summary:')
    print(gridkit.summary(baseline).to_string())
    solve_checked(baseline, 'baseline')
    number_of_hours = len(baseline.snapshots)
    loading = gridkit.line_loading(baseline)
    baseline_down = dispatch_down_mwh(baseline)
    baseline_unserved = unserved_energy_mwh(baseline)
    lines = baseline.lines[['bus0', 'bus1', 's_nom']].copy()
    lines = lines.rename(columns={'s_nom': 'rating_mva'})
    lines['label'] = [make_line_label(line_name, row.bus0, row.bus1) for line_name, row in lines.iterrows()]
    lines['binding_hours'] = (loading >= BINDING_THRESHOLD).sum().reindex(lines.index).fillna(0).astype(int)
    lines['binding_share_pct'] = 100.0 * lines['binding_hours'] / number_of_hours
    lines['hours_at_or_above_90_pct'] = (loading >= 0.9).sum().reindex(lines.index).fillna(0).astype(int)
    lines['mean_loading_pct'] = 100.0 * loading.mean().reindex(lines.index)
    lines['p95_loading_pct'] = 100.0 * loading.quantile(0.95).reindex(lines.index)
    lines['max_loading_pct'] = 100.0 * loading.max().reindex(lines.index)
    lines['peak_loading_snapshot'] = [loading[line_name].idxmax() for line_name in lines.index]
    lines.index.name = 'line'
    lines = lines.sort_values(['binding_hours', 'p95_loading_pct', 'max_loading_pct'], ascending=False)
    loading.to_csv(TABLE_DIR / '01_hourly_line_loading_ratio.csv', index_label='snapshot')
    print('\nEstimating the surplus floor with all ratings lifted...')
    copper_plate = run_all_branch_rating_case(COPPER_PLATE_MULTIPLIER)
    surplus_floor = float(copper_plate['dispatch_down_mwh'])
    constraint_based_down = max(clean_number(baseline_down - surplus_floor), 0.0)
    intervention_rows: list[dict[str, float | int | str]] = []
    total_lines = len(lines)
    for position, line_name in enumerate(lines.index, start=1):
        print(f'[{position:02d}/{total_lines:02d}] Testing {line_name} at +25% rating')
        experiment = load_network()
        original_rating = float(experiment.lines.at[line_name, 's_nom'])
        new_rating = original_rating * LINE_REINFORCEMENT_MULTIPLIER
        gridkit.set_rating(experiment, line_name, new_rating)
        solve_checked(experiment, f'+25% experiment for {line_name}')
        new_down = dispatch_down_mwh(experiment)
        saved_down = clean_number(baseline_down - new_down)
        added_rating = new_rating - original_rating
        saved_per_added_mva = saved_down / added_rating if added_rating > TOLERANCE else np.nan
        changed_loading = gridkit.line_loading(experiment)[line_name]
        intervention_rows.append({'line': line_name, 'new_rating_mva': new_rating, 'added_rating_mva': added_rating, 'dispatch_down_after_plus_25_mwh': new_down, 'saved_dispatch_down_plus_25_mwh': saved_down, 'saved_mwh_per_added_mva': saved_per_added_mva, 'binding_hours_after_plus_25': int((changed_loading >= BINDING_THRESHOLD).sum()), 'max_loading_after_plus_25_pct': float(100.0 * changed_loading.max()), 'unserved_after_plus_25_mwh': unserved_energy_mwh(experiment)})
    interventions = pd.DataFrame(intervention_rows).set_index('line')
    lines = lines.join(interventions)
    if constraint_based_down > TOLERANCE:
        lines['constraint_component_recovered_pct'] = 100.0 * lines['saved_dispatch_down_plus_25_mwh'] / constraint_based_down
    else:
        lines['constraint_component_recovered_pct'] = np.nan
    lines.to_csv(TABLE_DIR / '02_line_constraint_and_rating_results.csv', index_label='line')
    lines.sort_values('saved_dispatch_down_plus_25_mwh', ascending=False).to_csv(TABLE_DIR / '03_lines_ranked_by_recovered_energy.csv', index_label='line')
    network_rows: list[dict[str, float | int]] = []
    for multiplier in NETWORK_RATING_MULTIPLIERS:
        if abs(multiplier - 1.0) < TOLERANCE:
            case = {'rating_multiplier': 1.0, 'dispatch_down_mwh': baseline_down, 'unserved_mwh': baseline_unserved, 'binding_line_count': int((lines['binding_hours'] > 0).sum())}
        else:
            print(f'[network] Testing all branch ratings at {multiplier:.0%} of baseline')
            case = run_all_branch_rating_case(multiplier)
        constrained_part = max(clean_number(float(case['dispatch_down_mwh']) - surplus_floor), 0.0)
        network_rows.append({'rating_multiplier': float(multiplier), 'total_dispatch_down_mwh': float(case['dispatch_down_mwh']), 'constraint_based_dispatch_down_mwh': constrained_part, 'saved_vs_baseline_mwh': clean_number(baseline_down - float(case['dispatch_down_mwh'])), 'unserved_mwh': float(case['unserved_mwh']), 'binding_line_count': int(case['binding_line_count'])})
    network_sensitivity = pd.DataFrame(network_rows).sort_values('rating_multiplier')
    network_sensitivity.to_csv(TABLE_DIR / '04_network_rating_sensitivity.csv', index=False)
    pearson = lines['rating_mva'].corr(lines['binding_hours'], method='pearson')
    spearman = lines['rating_mva'].rank(method='average').corr(lines['binding_hours'].rank(method='average'), method='pearson')
    frequency_chart = lines.head(TOP_N).sort_values('binding_hours', ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(frequency_chart['label'], frequency_chart['binding_hours'])
    ax.set_xlabel(f'Hours at or above {BINDING_THRESHOLD:.1%} of rating')
    ax.set_title(f'{SCENARIO} {SCOPE}: most frequently constrained lines')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / '01_binding_hours.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    impact_chart = lines.sort_values('saved_dispatch_down_plus_25_mwh', ascending=False).head(TOP_N)
    impact_chart = impact_chart.sort_values('saved_dispatch_down_plus_25_mwh', ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(impact_chart['label'], impact_chart['saved_dispatch_down_plus_25_mwh'])
    ax.set_xlabel('Renewable dispatch-down recovered by +25% rating (MWh)')
    ax.set_title(f'{SCENARIO} {SCOPE}: effect of reinforcing one line')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / '02_recovered_energy_plus_25_pct.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.plot(network_sensitivity['rating_multiplier'], network_sensitivity['total_dispatch_down_mwh'], marker='o', label='total dispatch-down')
    ax.plot(network_sensitivity['rating_multiplier'], network_sensitivity['constraint_based_dispatch_down_mwh'], marker='o', label='constraint-based component')
    ax.set_xlabel('Multiplier applied to every branch thermal rating')
    ax.set_ylabel('Energy over the 168-hour model (MWh)')
    ax.set_title(f'{SCENARIO} {SCOPE}: thermal rating versus dispatch-down')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / '03_network_rating_sensitivity.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('\n=== SYSTEM RESULTS ===')
    print(f'Hourly snapshots:                  {number_of_hours}')
    print(f'Baseline dispatch-down:            {baseline_down:,.2f} MWh')
    print(f'Surplus floor, ratings lifted:     {surplus_floor:,.2f} MWh')
    print(f'Constraint-based component:        {constraint_based_down:,.2f} MWh')
    print(f'Baseline unserved energy:          {baseline_unserved:,.6f} MWh')
    frequency_columns = ['bus0', 'bus1', 'rating_mva', 'binding_hours', 'binding_share_pct', 'p95_loading_pct', 'max_loading_pct']
    print('\n=== MOST FREQUENTLY CONSTRAINED LINES ===')
    print(lines[frequency_columns].head(TOP_N).round(2).to_string())
    impact_columns = ['bus0', 'bus1', 'rating_mva', 'binding_hours', 'saved_dispatch_down_plus_25_mwh', 'saved_mwh_per_added_mva', 'binding_hours_after_plus_25']
    print('\n=== LARGEST EFFECT OF A +25% RATING INCREASE ===')
    print(lines.sort_values('saved_dispatch_down_plus_25_mwh', ascending=False)[impact_columns].head(TOP_N).round(2).to_string())
    print('\n=== DESCRIPTIVE RATING/FREQUENCY ASSOCIATION ===')
    print(f'Pearson correlation: {pearson:.3f}')
    print(f'Rank correlation:    {spearman:.3f}')
    print('Correlation is descriptive only; topology, reactance, generation location, and demand location also affect congestion.')
    print(f'\nAll outputs saved to:\n{OUTPUT_DIR}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
