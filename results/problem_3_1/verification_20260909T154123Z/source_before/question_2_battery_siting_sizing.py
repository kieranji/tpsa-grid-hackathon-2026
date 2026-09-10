#!/usr/bin/env python3
"""TPSA Hackathon 2026 — Problem 3.1, Question 2.

Finds effective BESS locations and searches MW/MWh sizes for theoretical, health-aware, and lifetime-aware designs on the 168-hour WP2033 North-West model.

The health and lifetime parameters are modelling assumptions and should be reported as such."""
from __future__ import annotations
import argparse
import hashlib
from itertools import combinations
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
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
    raise FileNotFoundError(f'Cannot find the official participant kit.\nExpected gridkit.py at: {GRIDKIT_PATH}\nRun this script from the repository structure described at the top of the file, and make sure data/participant-kit has been copied.')
sys.path.insert(0, str(KIT_DIR))
import gridkit
gridkit.quiet()
DEFAULT_SCENARIO = 'WP2033'
DEFAULT_SCOPE = 'north-west'
BINDING_THRESHOLD = 0.999
DEFAULT_RECOVERY_TARGET = 0.99
# Counterfactual only: relax the target line/corridor to estimate its constraint impact.
RELAX_RATING_MULTIPLIER = 1000.0
# Health-aware BESS assumptions; these are not manufacturer-specific operating limits.
HEALTH_SOC_MIN = 0.1
HEALTH_SOC_MAX = 0.9
HEALTH_ROUND_TRIP_EFFICIENCY = 0.9
DEFAULT_EOL_SOH = 0.8
DEFAULT_ENGINEERING_MARGIN = 0.1
DEFAULT_BATTERY_MARGINAL_COST = 0.5
ENERGY_TOLERANCE_MWH = 0.001
OUTPUT_ROOT = PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_2_final'
TABLE_DIR = OUTPUT_ROOT / 'tables'
FIGURE_DIR = OUTPUT_ROOT / 'figures'
CACHE_PATH = TABLE_DIR / '_trial_cache.csv'
CACHE_SCHEMA_VERSION = 'q2-final-v2.0'
FIRST_QUESTION_RESULT_CANDIDATES = [
    PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1_final' / 'tables' / '03_lines_ranked_by_recovered_energy.csv',
    PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1_final' / 'tables' / '02_line_constraint_and_rating_results.csv',
    PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1' / 'tables' / '03_lines_ranked_by_recovered_energy.csv',
    PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1' / 'tables' / '02_line_constraint_and_rating_results.csv',
    PROJECT_ROOT / 'results' / 'tables' / '03_lines_ranked_by_recovered_energy.csv',
    PROJECT_ROOT / 'results' / 'tables' / '02_line_constraint_and_rating_results.csv',
]

@dataclass(frozen=True)
class BatteryModel:
    name: str
    soc_min: float
    soc_max: float
    round_trip_efficiency: float
    state_of_health: float = 1.0
    marginal_cost_eur_per_mwh: float = DEFAULT_BATTERY_MARGINAL_COST
    standing_loss_per_hour: float = 0.0

    @property
    def soc_window_fraction(self) -> float:
        return self.soc_max - self.soc_min

    @property
    def one_way_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    def usable_energy_mwh(self, nameplate_energy_mwh: float) -> float:
        return float(nameplate_energy_mwh) * self.state_of_health * self.soc_window_fraction

    def validate(self) -> None:
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError(f'Invalid SoC window for {self.name}: {self.soc_min:.3f}–{self.soc_max:.3f}.')
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError(f'Round-trip efficiency must be in (0, 1], got {self.round_trip_efficiency}.')
        if not 0.0 < self.state_of_health <= 1.0:
            raise ValueError(f'State of health must be in (0, 1], got {self.state_of_health}.')
        if self.marginal_cost_eur_per_mwh < 0.0:
            raise ValueError('Battery marginal cost cannot be negative.')
        if not 0.0 <= self.standing_loss_per_hour < 1.0:
            raise ValueError('Standing loss must be between 0 and 1 per hour.')

@dataclass
class StudyContext:
    scenario: str
    scope: str
    target_mode: str
    target_lines: list[str]
    target_label: str
    target_endpoints: tuple[str, str]
    original_ratings_mva: dict[str, float]
    baseline_network: Any
    relaxed_network: Any
    baseline_dispatch_down_mwh: float
    relaxed_dispatch_down_mwh: float
    target_impact_mwh: float
    baseline_unserved_mwh: float
    baseline_target_binding_hours: int
    baseline_target_binding_line_hours: int
    baseline_target_max_loading_pct: float
    affected_generators: list[str]
    affected_target_gain_mwh: float
    baseline_affected_dispatch_mwh: float
    excess_profile_mw: pd.Series
    power_hint_mw: float
    usable_energy_hint_mwh: float

def ensure_output_directories() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def safe_filename(value: str) -> str:
    cleaned = ''.join((character if character.isalnum() else '_' for character in value))
    return '_'.join((part for part in cleaned.split('_') if part))

def stable_hash(text: str, length: int=16) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]

def round_up(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError('round_up step must be positive')
    return math.ceil(float(value) / step - 1e-12) * step

def unique_sorted_positive(values: Iterable[float], decimals: int=6) -> list[float]:
    cleaned: set[float] = set()
    for value in values:
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            cleaned.add(round(number, decimals))
    return sorted(cleaned)

def weighted_sum_over_time(network: Any, frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    weighting = pd.Series(1.0, index=frame.index, dtype=float)
    snapshot_weightings = getattr(network, 'snapshot_weightings', None)
    if isinstance(snapshot_weightings, pd.DataFrame):
        if 'generators' in snapshot_weightings.columns:
            weighting = snapshot_weightings['generators'].reindex(frame.index).fillna(1.0)
        elif 'objective' in snapshot_weightings.columns:
            weighting = snapshot_weightings['objective'].reindex(frame.index).fillna(1.0)
    elif isinstance(snapshot_weightings, pd.Series):
        weighting = snapshot_weightings.reindex(frame.index).fillna(1.0)
    return frame.mul(weighting, axis=0).sum(axis=0)

def storage_weighted_sum_over_time(network: Any, frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    weighting = pd.Series(1.0, index=frame.index, dtype=float)
    snapshot_weightings = getattr(network, 'snapshot_weightings', None)
    if isinstance(snapshot_weightings, pd.DataFrame):
        if 'stores' in snapshot_weightings.columns:
            weighting = snapshot_weightings['stores'].reindex(frame.index).fillna(1.0)
        elif 'objective' in snapshot_weightings.columns:
            weighting = snapshot_weightings['objective'].reindex(frame.index).fillna(1.0)
    elif isinstance(snapshot_weightings, pd.Series):
        weighting = snapshot_weightings.reindex(frame.index).fillna(1.0)
    return frame.mul(weighting, axis=0).sum(axis=0)

def solve_checked(network: Any, label: str) -> None:
    result = gridkit.solve(network)
    if isinstance(result, tuple) and len(result) >= 2:
        status = str(result[0]).lower()
        condition = str(result[1]).lower()
        if status != 'ok' or 'optimal' not in condition:
            raise RuntimeError(f'{label} solve was not optimal: status={result[0]!r}, condition={result[1]!r}.')
    generator_dispatch = getattr(network.generators_t, 'p', pd.DataFrame())
    if generator_dispatch.empty:
        raise RuntimeError(f'{label} solve returned no generator dispatch. Check the HiGHS installation and network feasibility.')

def total_dispatch_down_mwh(network: Any) -> float:
    result = gridkit.dispatch_down(network)
    if result.empty:
        return 0.0
    return float(result['dispatch_down_mwh'].sum())

def total_unserved_mwh(network: Any) -> float:
    result = gridkit.unserved(network)
    if result.empty:
        return 0.0
    return float(result.sum())

def weather_generator_names(network: Any) -> list[str]:
    time_series = getattr(network.generators_t, 'p_max_pu', pd.DataFrame())
    if time_series.empty:
        return []
    return [name for name in time_series.columns if name in network.generators.index]

def generator_dispatch_energy_mwh(network: Any, names: Sequence[str]) -> pd.Series:
    dispatch = getattr(network.generators_t, 'p', pd.DataFrame())
    selected = [name for name in names if name in dispatch.columns]
    if not selected:
        return pd.Series(dtype=float)
    return weighted_sum_over_time(network, dispatch[selected])

def hourly_dispatch_down_mw(network: Any) -> pd.Series:
    names = weather_generator_names(network)
    if not names:
        return pd.Series(0.0, index=network.snapshots, name='dispatch_down_mw')
    p_max_pu = network.generators_t.p_max_pu[names]
    p_nom = network.generators.loc[names, 'p_nom'].astype(float)
    offered = p_max_pu.mul(p_nom, axis=1).sum(axis=1)
    dispatched = network.generators_t.p[names].sum(axis=1)
    result = (offered - dispatched).clip(lower=0.0)
    result.name = 'dispatch_down_mw'
    return result

def target_loading_metrics(network: Any, target_lines: Sequence[str], threshold: float=BINDING_THRESHOLD) -> dict[str, float]:
    loading = gridkit.line_loading(network)
    existing = [line for line in target_lines if line in loading.columns]
    if not existing:
        raise KeyError('None of the target lines exist in the solved line-loading table.')
    target_loading = loading[existing]
    binding_matrix = target_loading >= threshold
    binding_hours_any = int(binding_matrix.any(axis=1).sum())
    binding_line_hours = int(binding_matrix.sum().sum())
    return {'target_binding_hours': float(binding_hours_any), 'target_binding_line_hours': float(binding_line_hours), 'target_max_loading_pct': float(target_loading.max().max() * 100.0), 'target_mean_max_loading_pct': float(target_loading.max(axis=1).mean() * 100.0)}

def max_consecutive_energy_mwh(power_series_mw: pd.Series, tolerance_mw: float=1e-06) -> float:
    current = 0.0
    maximum = 0.0
    for value in power_series_mw.fillna(0.0).astype(float):
        if value > tolerance_mw:
            current += value
            maximum = max(maximum, current)
        else:
            current = 0.0
    return maximum

def line_column_from_frame(frame: pd.DataFrame) -> str | None:
    preferred = ['line', 'line_name', 'name', 'index', 'Unnamed: 0']
    for column in preferred:
        if column in frame.columns:
            return column
    return None

def choose_target_line_from_first_question(network: Any) -> tuple[str | None, str]:
    for path in FIRST_QUESTION_RESULT_CANDIDATES:
        if not path.exists():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            print(f'[warning] Could not read first-question result {path}: {error}')
            continue
        line_column = line_column_from_frame(frame)
        if line_column is None:
            continue
        frame = frame.copy()
        frame[line_column] = frame[line_column].astype(str)
        frame = frame[frame[line_column].isin(network.lines.index.astype(str))]
        if frame.empty:
            continue
        if 'binding_hours' in frame.columns:
            binding = pd.to_numeric(frame['binding_hours'], errors='coerce').fillna(0.0)
            binding_rows = frame[binding > 0.0]
            if not binding_rows.empty:
                frame = binding_rows
        impact_columns = ['saved_dispatch_down_plus_25_mwh', 'saved_mwh_plus_25', 'saved_mwh_if_line_unlimited', 'constraint_component_recovered_pct']
        impact_column = next((column for column in impact_columns if column in frame.columns), None)
        sort_columns: list[str] = []
        ascending: list[bool] = []
        if impact_column is not None:
            frame[impact_column] = pd.to_numeric(frame[impact_column], errors='coerce').fillna(0.0)
            sort_columns.append(impact_column)
            ascending.append(False)
        if 'binding_hours' in frame.columns:
            frame['binding_hours'] = pd.to_numeric(frame['binding_hours'], errors='coerce').fillna(0.0)
            sort_columns.append('binding_hours')
            ascending.append(False)
        if sort_columns:
            frame = frame.sort_values(sort_columns, ascending=ascending)
        selected = str(frame.iloc[0][line_column])
        reason = f'auto-selected from first-question results: {path.relative_to(PROJECT_ROOT)}'
        return (selected, reason)
    return (None, 'no usable first-question result CSV was found')

def choose_target_line(baseline_network: Any, requested_line: str | None) -> tuple[str, str]:
    if requested_line is not None:
        if requested_line not in baseline_network.lines.index:
            available = ', '.join(map(str, baseline_network.lines.index[:10]))
            raise KeyError(f'Target line {requested_line!r} is not in the network. The first ten line IDs are: {available}')
        return (requested_line, 'explicitly supplied through --target-line')
    from_csv, reason = choose_target_line_from_first_question(baseline_network)
    if from_csv is not None:
        return (from_csv, reason)
    binding = gridkit.binding(baseline_network)
    if not binding.empty:
        selected = str(binding.index[0])
        return (selected, f'fallback: highest baseline binding-hours line ({reason})')
    maximum_loading = gridkit.line_loading(baseline_network).max().sort_values(ascending=False)
    if maximum_loading.empty:
        raise RuntimeError('The network contains no line loading results.')
    selected = str(maximum_loading.index[0])
    return (selected, f'fallback: highest maximum-loading line ({reason})')

def target_line_group(network: Any, selected_line: str, target_mode: str) -> list[str]:
    if selected_line not in network.lines.index:
        raise KeyError(f'Unknown target line: {selected_line}')
    if target_mode == 'line':
        return [selected_line]
    target = network.lines.loc[selected_line]
    endpoint_set = frozenset((str(target['bus0']), str(target['bus1'])))
    group: list[str] = []
    for line_name, row in network.lines.iterrows():
        row_endpoints = frozenset((str(row['bus0']), str(row['bus1'])))
        if row_endpoints == endpoint_set:
            group.append(str(line_name))
    return sorted(group)

def build_study_context(scenario: str, scope: str, requested_line: str | None, target_mode: str) -> StudyContext:
    print('\n[baseline] Loading and solving the original network ...')
    baseline = gridkit.load(scenario, scope)
    solve_checked(baseline, 'baseline')
    selected_line, selection_reason = choose_target_line(baseline, requested_line)
    target_lines = target_line_group(baseline, selected_line, target_mode)
    selected_row = baseline.lines.loc[selected_line]
    endpoints = (str(selected_row['bus0']), str(selected_row['bus1']))
    if len(target_lines) == 1:
        target_label = target_lines[0]
    else:
        target_label = f'corridor {endpoints[0]} <-> {endpoints[1]}'
    print(f'[target] {target_label}')
    print(f'[target] Selection reason: {selection_reason}')
    if len(target_lines) > 1:
        print(f"[target] Parallel line members: {', '.join(target_lines)}")
    original_ratings = {line: float(baseline.lines.at[line, 's_nom']) for line in target_lines}
    baseline_down = total_dispatch_down_mwh(baseline)
    baseline_unserved = total_unserved_mwh(baseline)
    baseline_loading = target_loading_metrics(baseline, target_lines)
    print(f"[baseline] Dispatch-down = {baseline_down:,.2f} MWh; target binding hours = {int(baseline_loading['target_binding_hours'])}; max target loading = {baseline_loading['target_max_loading_pct']:.3f}%")
    print('[counterfactual] Relaxing only the target line/corridor and re-solving ...')
    relaxed = gridkit.load(scenario, scope)
    for line, rating in original_ratings.items():
        gridkit.set_rating(relaxed, line, rating * RELAX_RATING_MULTIPLIER)
    solve_checked(relaxed, 'target-relaxed counterfactual')
    relaxed_down = total_dispatch_down_mwh(relaxed)
    target_impact = max(0.0, baseline_down - relaxed_down)
    weather_names = weather_generator_names(baseline)
    baseline_dispatch = generator_dispatch_energy_mwh(baseline, weather_names)
    relaxed_dispatch = generator_dispatch_energy_mwh(relaxed, weather_names)
    dispatch_gain = relaxed_dispatch.sub(baseline_dispatch, fill_value=0.0)
    positive_gain = dispatch_gain[dispatch_gain > ENERGY_TOLERANCE_MWH].sort_values(ascending=False)
    affected_generators = list(positive_gain.index.astype(str))
    affected_target_gain = float(positive_gain.sum())
    baseline_affected_dispatch = float(baseline_dispatch.reindex(affected_generators).fillna(0.0).sum())
    relaxed_flows = relaxed.lines_t.p0[target_lines].abs()
    excess_columns: dict[str, pd.Series] = {}
    for line in target_lines:
        original_limit = original_ratings[line] * BINDING_THRESHOLD
        excess_columns[line] = (relaxed_flows[line] - original_limit).clip(lower=0.0)
    excess_frame = pd.DataFrame(excess_columns, index=relaxed.snapshots)
    excess_profile = excess_frame.sum(axis=1)
    excess_profile.name = 'estimated_target_excess_mw'
    power_hint = float(excess_profile.max()) if not excess_profile.empty else 0.0
    energy_hint = max_consecutive_energy_mwh(excess_profile)
    total_target_rating = sum(original_ratings.values())
    if power_hint <= ENERGY_TOLERANCE_MWH:
        power_hint = max(25.0, total_target_rating * 0.25)
    if energy_hint <= ENERGY_TOLERANCE_MWH:
        energy_hint = max(target_impact, power_hint * 4.0)
    print(f'[counterfactual] Dispatch-down = {relaxed_down:,.2f} MWh; target-attributable reduction = {target_impact:,.2f} MWh')
    print(f'[counterfactual] {len(affected_generators)} weather-driven generators gain {affected_target_gain:,.2f} MWh in total.')
    print(f'[search hints] estimated power = {power_hint:,.1f} MW; continuous usable energy = {energy_hint:,.1f} MWh')
    return StudyContext(scenario=scenario, scope=scope, target_mode=target_mode, target_lines=target_lines, target_label=target_label, target_endpoints=endpoints, original_ratings_mva=original_ratings, baseline_network=baseline, relaxed_network=relaxed, baseline_dispatch_down_mwh=baseline_down, relaxed_dispatch_down_mwh=relaxed_down, target_impact_mwh=target_impact, baseline_unserved_mwh=baseline_unserved, baseline_target_binding_hours=int(baseline_loading['target_binding_hours']), baseline_target_binding_line_hours=int(baseline_loading['target_binding_line_hours']), baseline_target_max_loading_pct=float(baseline_loading['target_max_loading_pct']), affected_generators=affected_generators, affected_target_gain_mwh=affected_target_gain, baseline_affected_dispatch_mwh=baseline_affected_dispatch, excess_profile_mw=excess_profile, power_hint_mw=power_hint, usable_energy_hint_mwh=energy_hint)

class TrialCache:

    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.rows: dict[str, dict[str, Any]] = {}
        if self.enabled and self.path.exists():
            try:
                frame = pd.read_csv(self.path)
                if 'trial_key' in frame.columns:
                    for _, row in frame.iterrows():
                        self.rows[str(row['trial_key'])] = row.to_dict()
                    print(f'[cache] Loaded {len(self.rows)} completed trials.')
            except Exception as error:
                print(f'[warning] Existing cache could not be read: {error}')

    def get(self, trial_key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        row = self.rows.get(trial_key)
        return dict(row) if row is not None else None

    def put(self, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        trial_key = str(row['trial_key'])
        self.rows[trial_key] = dict(row)
        frame = pd.DataFrame(self.rows.values()).sort_values('trial_key')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(self.path, index=False)

def normalise_allocations(number_of_sites: int, allocations: Sequence[float] | None) -> list[float]:
    if number_of_sites <= 0:
        raise ValueError('At least one battery site is required.')
    if allocations is None:
        return [1.0 / number_of_sites] * number_of_sites
    if len(allocations) != number_of_sites:
        raise ValueError('The number of allocation weights must equal the number of sites.')
    values = [float(value) for value in allocations]
    if any((value <= 0.0 for value in values)):
        raise ValueError('Every allocation weight must be positive.')
    total = sum(values)
    return [value / total for value in values]

def add_battery_fleet(network: Any, sites: Sequence[str], allocations: Sequence[float], total_power_mw: float, total_nameplate_energy_mwh: float, model: BatteryModel) -> list[str]:
    model.validate()
    if total_power_mw <= 0.0:
        raise ValueError('Battery power must be positive.')
    if total_nameplate_energy_mwh <= 0.0:
        raise ValueError('Battery energy must be positive.')
    for site in sites:
        if site not in network.buses.index:
            raise KeyError(f'Battery site {site!r} is not a network bus.')
    if 'battery' not in network.carriers.index:
        network.add('Carrier', 'battery')
    names: list[str] = []
    for position, (site, share) in enumerate(zip(sites, allocations), start=1):
        site_power = float(total_power_mw) * float(share)
        site_nameplate_energy = float(total_nameplate_energy_mwh) * float(share)
        site_usable_energy = model.usable_energy_mwh(site_nameplate_energy)
        if site_usable_energy <= 0.0:
            raise ValueError('The selected SoC window and SoH leave no usable energy.')
        max_hours = site_usable_energy / site_power
        name = f'Q2 battery {position}'
        network.add('StorageUnit', name, bus=site, p_nom=site_power, max_hours=max_hours, efficiency_store=model.one_way_efficiency, efficiency_dispatch=model.one_way_efficiency, standing_loss=model.standing_loss_per_hour, # Prevent end-of-horizon energy dumping by returning to the initial state of charge.
            cyclic_state_of_charge=True, carrier='battery', marginal_cost=model.marginal_cost_eur_per_mwh)
        names.append(name)
    return names

def build_trial_key(context: StudyContext, sites: Sequence[str], allocations: Sequence[float], total_power_mw: float, total_energy_mwh: float, model: BatteryModel, recovery_target: float, success_mode: str) -> str:
    raw = '|'.join([CACHE_SCHEMA_VERSION, context.scenario, context.scope, context.target_mode, ','.join(context.target_lines), ','.join(map(str, sites)), ','.join((f'{value:.8f}' for value in allocations)), f'P={total_power_mw:.8f}', f'E={total_energy_mwh:.8f}', f'model={model.name}', f'soc={model.soc_min:.8f},{model.soc_max:.8f}', f'rte={model.round_trip_efficiency:.8f}', f'soh={model.state_of_health:.8f}', f'mc={model.marginal_cost_eur_per_mwh:.8f}', f'sl={model.standing_loss_per_hour:.8f}', f'recovery={recovery_target:.8f}', f'success={success_mode}'])
    return stable_hash(raw, length=24)

def evaluate_success(context: StudyContext, row: dict[str, Any], recovery_target: float, success_mode: str) -> dict[str, Any]:
    unserved_tolerance = max(0.1, context.baseline_unserved_mwh * 0.001)
    unserved_ok = float(row['unserved_mwh']) <= context.baseline_unserved_mwh + unserved_tolerance
    target_fraction = float(row['target_recovery_fraction'])
    benchmark_available = context.target_impact_mwh > ENERGY_TOLERANCE_MWH
    benchmark_success = benchmark_available and target_fraction + 1e-09 >= recovery_target and unserved_ok
    headroom_success = int(round(float(row['target_binding_hours']))) == 0 and unserved_ok
    if not benchmark_available:
        selected_success = headroom_success
        effective_mode = 'headroom-fallback'
    elif success_mode == 'benchmark':
        selected_success = benchmark_success
        effective_mode = 'benchmark'
    elif success_mode == 'headroom':
        selected_success = headroom_success
        effective_mode = 'headroom'
    elif success_mode == 'both':
        selected_success = benchmark_success and headroom_success
        effective_mode = 'both'
    else:
        raise ValueError(f'Unknown success mode: {success_mode}')
    return {'unserved_ok': bool(unserved_ok), 'benchmark_available': bool(benchmark_available), 'benchmark_success': bool(benchmark_success), 'headroom_success': bool(headroom_success), 'success': bool(selected_success), 'effective_success_mode': effective_mode}

def run_battery_trial(context: StudyContext, sites: Sequence[str], total_power_mw: float, total_nameplate_energy_mwh: float, model: BatteryModel, recovery_target: float, success_mode: str, cache: TrialCache, allocations: Sequence[float] | None=None, force_solve: bool=False, return_network: bool=False, progress_label: str | None=None) -> tuple[dict[str, Any], Any | None, list[str]]:
    site_list = [str(site) for site in sites]
    allocation_list = normalise_allocations(len(site_list), allocations)
    trial_key = build_trial_key(context=context, sites=site_list, allocations=allocation_list, total_power_mw=total_power_mw, total_energy_mwh=total_nameplate_energy_mwh, model=model, recovery_target=recovery_target, success_mode=success_mode)
    if not force_solve and (not return_network):
        cached = cache.get(trial_key)
        if cached is not None:
            if progress_label:
                print(f'{progress_label} [cached]')
            return (cached, None, [])
    if progress_label:
        print(progress_label)
    row: dict[str, Any] = {'trial_key': trial_key, 'scenario': context.scenario, 'scope': context.scope, 'target_label': context.target_label, 'target_lines': ';'.join(context.target_lines), 'site_count': len(site_list), 'sites': ';'.join(site_list), 'allocations': ';'.join((f'{value:.6f}' for value in allocation_list)), 'model': model.name, 'soc_min_pct': model.soc_min * 100.0, 'soc_max_pct': model.soc_max * 100.0, 'round_trip_efficiency_pct': model.round_trip_efficiency * 100.0, 'state_of_health_pct': model.state_of_health * 100.0, 'power_mw': float(total_power_mw), 'nameplate_energy_mwh': float(total_nameplate_energy_mwh), 'usable_energy_mwh': model.usable_energy_mwh(total_nameplate_energy_mwh), 'nameplate_duration_hours': float(total_nameplate_energy_mwh) / float(total_power_mw), 'usable_duration_hours': model.usable_energy_mwh(total_nameplate_energy_mwh) / float(total_power_mw), 'solve_ok': False, 'error': ''}
    try:
        network = gridkit.load(context.scenario, context.scope)
        battery_names = add_battery_fleet(network=network, sites=site_list, allocations=allocation_list, total_power_mw=total_power_mw, total_nameplate_energy_mwh=total_nameplate_energy_mwh, model=model)
        solve_checked(network, f'battery trial {trial_key}')
        trial_dispatch_down = total_dispatch_down_mwh(network)
        trial_unserved = total_unserved_mwh(network)
        dispatch_down_saved = context.baseline_dispatch_down_mwh - trial_dispatch_down
        if context.affected_generators:
            trial_affected_dispatch = generator_dispatch_energy_mwh(network, context.affected_generators).sum()
            affected_gain = float(trial_affected_dispatch - context.baseline_affected_dispatch_mwh)
        else:
            affected_gain = dispatch_down_saved
        if context.target_impact_mwh > ENERGY_TOLERANCE_MWH:
            total_recovery_fraction = dispatch_down_saved / context.target_impact_mwh
        else:
            total_recovery_fraction = float('nan')
        if context.affected_target_gain_mwh > ENERGY_TOLERANCE_MWH:
            affected_recovery_fraction = affected_gain / context.affected_target_gain_mwh
        else:
            affected_recovery_fraction = total_recovery_fraction
        finite_fractions = [value for value in (total_recovery_fraction, affected_recovery_fraction) if math.isfinite(value)]
        target_recovery_fraction = min(finite_fractions) if finite_fractions else 0.0
        loading = target_loading_metrics(network, context.target_lines)
        p_store_table = getattr(network.storage_units_t, 'p_store', pd.DataFrame())
        p_dispatch_table = getattr(network.storage_units_t, 'p_dispatch', pd.DataFrame())
        soc_table = getattr(network.storage_units_t, 'state_of_charge', pd.DataFrame())
        existing_batteries = [name for name in battery_names if name in p_store_table.columns]
        if existing_batteries:
            charge_mwh = float(storage_weighted_sum_over_time(network, p_store_table[existing_batteries]).sum())
            discharge_mwh = float(storage_weighted_sum_over_time(network, p_dispatch_table[existing_batteries]).sum())
        else:
            charge_mwh = 0.0
            discharge_mwh = 0.0
        throughput_mwh = charge_mwh + discharge_mwh
        efc_nameplate = throughput_mwh / (2.0 * float(total_nameplate_energy_mwh))
        usable_energy = model.usable_energy_mwh(total_nameplate_energy_mwh)
        efc_usable = throughput_mwh / (2.0 * usable_energy)
        if existing_batteries:
            aggregate_charge = p_store_table[existing_batteries].sum(axis=1)
            aggregate_discharge = p_dispatch_table[existing_batteries].sum(axis=1)
            simultaneous = np.minimum(aggregate_charge, aggregate_discharge)
            simultaneous_hours = int((simultaneous > 0.0001).sum())
            simultaneous_mwh = float(simultaneous.sum())
            absolute_power = aggregate_charge + aggregate_discharge
            average_absolute_power_mw = float(absolute_power.mean())
            average_power_utilisation_pct = 100.0 * average_absolute_power_mw / float(total_power_mw)
            active_hours = int((absolute_power > 0.0001).sum())
            idle_hours = int(len(absolute_power) - active_hours)
        else:
            simultaneous_hours = 0
            simultaneous_mwh = 0.0
            average_absolute_power_mw = 0.0
            average_power_utilisation_pct = 0.0
            active_hours = 0
            idle_hours = int(len(network.snapshots))
        if existing_batteries and (not soc_table.empty):
            model_soc_total = soc_table[existing_batteries].sum(axis=1)
            current_physical_capacity = float(total_nameplate_energy_mwh) * model.state_of_health
            physical_floor = current_physical_capacity * model.soc_min
            physical_soc_pct = (model_soc_total + physical_floor) / current_physical_capacity * 100.0
            soc_min_pct_observed = float(physical_soc_pct.min())
            soc_mean_pct_observed = float(physical_soc_pct.mean())
            soc_max_pct_observed = float(physical_soc_pct.max())
        else:
            soc_min_pct_observed = float('nan')
            soc_mean_pct_observed = float('nan')
            soc_max_pct_observed = float('nan')
        row.update({'solve_ok': True, 'objective': float(network.objective), 'total_dispatch_down_mwh': trial_dispatch_down, 'dispatch_down_saved_mwh': dispatch_down_saved, 'affected_generator_gain_mwh': affected_gain, 'total_recovery_fraction': total_recovery_fraction, 'affected_recovery_fraction': affected_recovery_fraction, 'target_recovery_fraction': target_recovery_fraction, 'target_recovery_pct': target_recovery_fraction * 100.0, 'unserved_mwh': trial_unserved, **loading, 'charge_mwh': charge_mwh, 'discharge_mwh': discharge_mwh, 'throughput_mwh': throughput_mwh, 'equivalent_full_cycles_nameplate': efc_nameplate, 'equivalent_full_cycles_usable': efc_usable, 'simultaneous_charge_discharge_hours': simultaneous_hours, 'simultaneous_charge_discharge_mwh': simultaneous_mwh, 'average_absolute_power_mw': average_absolute_power_mw, 'average_power_utilisation_pct': average_power_utilisation_pct, 'active_hours': active_hours, 'idle_hours': idle_hours, 'observed_physical_soc_min_pct': soc_min_pct_observed, 'observed_physical_soc_mean_pct': soc_mean_pct_observed, 'observed_physical_soc_max_pct': soc_max_pct_observed})
        row.update(evaluate_success(context=context, row=row, recovery_target=recovery_target, success_mode=success_mode))
        cache.put(row)
        return (row, network if return_network else None, battery_names)
    except Exception as error:
        row.update({'solve_ok': False, 'error': f'{type(error).__name__}: {error}', 'objective': float('nan'), 'total_dispatch_down_mwh': float('nan'), 'dispatch_down_saved_mwh': float('nan'), 'affected_generator_gain_mwh': float('nan'), 'total_recovery_fraction': float('nan'), 'affected_recovery_fraction': float('nan'), 'target_recovery_fraction': float('nan'), 'target_recovery_pct': float('nan'), 'unserved_mwh': float('nan'), 'target_binding_hours': float('nan'), 'target_binding_line_hours': float('nan'), 'target_max_loading_pct': float('nan'), 'target_mean_max_loading_pct': float('nan'), 'charge_mwh': float('nan'), 'discharge_mwh': float('nan'), 'throughput_mwh': float('nan'), 'equivalent_full_cycles_nameplate': float('nan'), 'equivalent_full_cycles_usable': float('nan'), 'simultaneous_charge_discharge_hours': float('nan'), 'simultaneous_charge_discharge_mwh': float('nan'), 'average_absolute_power_mw': float('nan'), 'average_power_utilisation_pct': float('nan'), 'active_hours': float('nan'), 'idle_hours': float('nan'), 'observed_physical_soc_min_pct': float('nan'), 'observed_physical_soc_mean_pct': float('nan'), 'observed_physical_soc_max_pct': float('nan'), 'unserved_ok': False, 'benchmark_available': context.target_impact_mwh > ENERGY_TOLERANCE_MWH, 'benchmark_success': False, 'headroom_success': False, 'success': False, 'effective_success_mode': success_mode})
        print(f"[warning] Trial failed: {row['error']}")
        cache.put(row)
        if return_network or force_solve:
            raise
        return (row, None, [])

def connected_component_buses(network: Any, seed_bus: str) -> set[str]:
    adjacency: dict[str, set[str]] = {str(bus): set() for bus in network.buses.index}
    for _, row in network.lines.iterrows():
        a, b = str(row['bus0']), str(row['bus1'])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    for _, row in network.transformers.iterrows():
        a, b = str(row['bus0']), str(row['bus1'])
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seed = str(seed_bus)
    if seed not in adjacency:
        raise KeyError(f'Unknown seed bus {seed!r}.')
    visited = {seed}
    stack = [seed]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                stack.append(neighbour)
    return visited


def candidate_buses(network: Any, mode: str, target_endpoints: Sequence[str]) -> list[str]:
    all_buses = set(network.buses.index.astype(str))
    renewable_mask = network.generators['carrier'].astype(str).isin(['wind', 'solar'])
    renewable_buses = set(network.generators.loc[renewable_mask, 'bus'].astype(str))
    load_buses = set(network.loads['bus'].astype(str)) if len(network.loads) else set()
    endpoint_buses = {str(bus) for bus in target_endpoints}
    if mode == 'all':
        selected = all_buses
    elif mode == 'renewable':
        selected = renewable_buses | endpoint_buses
    elif mode == 'eligible':
        selected = renewable_buses | load_buses | endpoint_buses
    else:
        raise ValueError(f'Unknown candidate mode: {mode}')

    component = connected_component_buses(network, str(target_endpoints[0]))
    if str(target_endpoints[1]) not in component:
        raise RuntimeError('The two target-line endpoints are not in the same connected component.')
    selected &= all_buses & component
    if not selected:
        raise RuntimeError('No candidate battery buses were found in the target-line component.')
    return sorted(selected)

def bus_metadata(network: Any, bus: str, target_endpoints: Sequence[str]) -> dict[str, Any]:
    generators = network.generators[network.generators['bus'].astype(str) == str(bus)]
    renewable = generators[generators['carrier'].astype(str).isin(['wind', 'solar'])]
    renewable_capacity = float(renewable['p_nom'].sum()) if len(renewable) else 0.0
    load_names = network.loads.index[network.loads['bus'].astype(str) == str(bus)]
    if len(load_names) and (not network.loads_t.p_set.empty):
        peak_load = float(network.loads_t.p_set[list(load_names)].sum(axis=1).max())
    else:
        peak_load = 0.0
    return {'renewable_capacity_mw': renewable_capacity, 'peak_load_mw': peak_load, 'is_target_endpoint': str(bus) in {str(value) for value in target_endpoints}}

def rank_screening_frame(frame: pd.DataFrame, success_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked['_solve_rank'] = boolean_series(ranked['solve_ok']).astype(int)
    if success_mode == 'headroom':
        ranked['_success_rank'] = boolean_series(ranked['headroom_success']).astype(int)
    elif success_mode == 'both':
        ranked['_success_rank'] = (
            boolean_series(ranked['benchmark_success'])
            & boolean_series(ranked['headroom_success'])
        ).astype(int)
    else:
        ranked['_success_rank'] = boolean_series(ranked['benchmark_success']).astype(int)

    ranked['_recovery'] = pd.to_numeric(ranked['target_recovery_fraction'], errors='coerce').fillna(-1e9)
    ranked['_binding'] = pd.to_numeric(ranked['target_binding_hours'], errors='coerce').fillna(1e9)
    ranked['_saved'] = pd.to_numeric(ranked['dispatch_down_saved_mwh'], errors='coerce').fillna(-1e9)
    ranked['_throughput'] = pd.to_numeric(ranked['throughput_mwh'], errors='coerce').fillna(1e9)

    if success_mode in {'headroom', 'both'}:
        sort_columns = ['_solve_rank', '_success_rank', '_binding', '_recovery', '_saved', '_throughput']
        ascending = [False, False, True, False, False, True]
    else:
        sort_columns = ['_solve_rank', '_success_rank', '_recovery', '_binding', '_saved', '_throughput']
        ascending = [False, False, False, True, False, True]

    ranked = ranked.sort_values(sort_columns, ascending=ascending, na_position='last').reset_index(drop=True)
    return ranked.drop(columns=[c for c in ranked.columns if c.startswith('_')])

def make_probe_size(
    context: StudyContext,
    health_model: BatteryModel,
    requested_power_mw: float | None = None,
    requested_hours: float = 4.0,
) -> tuple[float, float]:
    """Build one common moderate BESS probe for fair site screening."""

    health_model.validate()

    if requested_hours <= 0.0:
        raise ValueError("requested_hours must be positive.")

    if requested_power_mw is not None:
        if requested_power_mw <= 0.0:
            raise ValueError("requested_power_mw must be positive.")
        probe_power = float(requested_power_mw)
    else:
        # Use the target-line counterfactual as a sizing hint, but avoid
        # an oversized probe that makes every candidate site saturate.
        probe_power = round_up(
            max(25.0, context.power_hint_mw * 0.75),
            5.0,
        )

        probe_power = min(probe_power, 100.0)

    # --probe-hours is defined as NAMEPLATE duration.
    probe_nameplate_energy = probe_power * float(requested_hours)

    probe_nameplate_energy = round_up(
        probe_nameplate_energy,
        5.0,
    )

    return probe_power, probe_nameplate_energy

def screen_single_sites(
    context: StudyContext,
    buses: Sequence[str],
    probe_power_mw: float,
    probe_energy_mwh: float,
    model: BatteryModel,
    recovery_target: float,
    success_mode: str,
    cache: TrialCache,
) -> pd.DataFrame:
    print(f'\n[siting] Screening {len(buses)} single-bus sites with the same {probe_power_mw:.0f} MW / {probe_energy_mwh:.0f} MWh probe battery ...')
    rows: list[dict[str, Any]] = []
    for number, bus in enumerate(buses, start=1):
        row, _, _ = run_battery_trial(
            context=context,
            sites=[bus],
            total_power_mw=probe_power_mw,
            total_nameplate_energy_mwh=probe_energy_mwh,
            model=model,
            recovery_target=recovery_target,
            success_mode=success_mode,
            cache=cache,
            progress_label=f'[siting {number:02d}/{len(buses):02d}] {bus}',
        )
        row = dict(row)
        row.update(bus_metadata(context.baseline_network, bus, context.target_endpoints))
        row['site'] = bus
        rows.append(row)
    return rank_screening_frame(pd.DataFrame(rows), success_mode)


def screen_site_pairs(
    context: StudyContext,
    single_site_results: pd.DataFrame,
    top_n: int,
    probe_power_mw: float,
    probe_energy_mwh: float,
    model: BatteryModel,
    recovery_target: float,
    success_mode: str,
    cache: TrialCache,
) -> pd.DataFrame:
    top_sites = single_site_results.loc[
        boolean_series(single_site_results['solve_ok']), 'site'
    ].head(top_n).astype(str).tolist()
    pairs = list(combinations(top_sites, 2))
    if not pairs:
        return pd.DataFrame()
    print(f'\n[siting] Screening {len(pairs)} two-site combinations with the SAME total probe capacity ...')
    rows: list[dict[str, Any]] = []
    for number, pair in enumerate(pairs, start=1):
        row, _, _ = run_battery_trial(
            context=context,
            sites=list(pair),
            allocations=[0.5, 0.5],
            total_power_mw=probe_power_mw,
            total_nameplate_energy_mwh=probe_energy_mwh,
            model=model,
            recovery_target=recovery_target,
            success_mode=success_mode,
            cache=cache,
            progress_label=f'[pair {number:02d}/{len(pairs):02d}] {pair[0]} + {pair[1]} (50/50)',
        )
        row = dict(row)
        row['site_pair'] = ' + '.join(pair)
        rows.append(row)
    return rank_screening_frame(pd.DataFrame(rows), success_mode)


def choose_architecture(
    single_results: pd.DataFrame,
    pair_results: pd.DataFrame,
    recovery_tolerance: float = 0.02,
) -> tuple[list[str], list[float], str]:
    valid_single = single_results[boolean_series(single_results['solve_ok'])]
    if valid_single.empty:
        raise RuntimeError('Every single-site screening solve failed.')
    best_single = valid_single.iloc[0]
    single_sites = [str(best_single['site'])]
    if pair_results.empty:
        return single_sites, [1.0], 'best single-site fixed-probe result'

    valid_pair = pair_results[boolean_series(pair_results['solve_ok'])]
    if valid_pair.empty:
        return single_sites, [1.0], 'all two-site screening solves failed'
    best_pair = valid_pair.iloc[0]

    single_binding = float(best_single['target_binding_hours'])
    pair_binding = float(best_pair['target_binding_hours'])
    single_recovery = float(best_single['target_recovery_fraction'])
    pair_recovery = float(best_pair['target_recovery_fraction'])

    binding_improved = pair_binding < single_binding - 0.5
    recovery_improved = pair_recovery > single_recovery + recovery_tolerance
    not_worse_binding = pair_binding <= single_binding + 0.5
    not_worse_recovery = pair_recovery >= single_recovery - recovery_tolerance

    if (binding_improved and not_worse_recovery) or (recovery_improved and not_worse_binding):
        sites = str(best_pair['sites']).split(';')
        allocations = [float(value) for value in str(best_pair['allocations']).split(';')]
        return sites, allocations, 'two-site layout technically improves the same-capacity probe result'

    return single_sites, [1.0], 'two-site layout did not materially improve the same-capacity probe result'

def build_power_candidates(power_hint_mw: float, full: bool) -> list[float]:
    if full:
        multipliers = [0.25, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        fixed = [10.0, 25.0, 50.0, 75.0, 100.0, 150.0, 250.0, 500.0]
        step = 5.0
    else:
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        fixed = [25.0, 50.0, 100.0]
        step = 10.0
    values = fixed + [round_up(power_hint_mw * multiplier, step) for multiplier in multipliers]
    values = [value for value in values if 5.0 <= value <= 2000.0]
    return unique_sorted_positive(values)

def build_energy_candidates(power_mw: float, usable_energy_hint_mwh: float, model: BatteryModel, full: bool) -> list[float]:
    if full:
        durations = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]
        hint_multipliers = [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
        energy_step = 10.0
    else:
        durations = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
        hint_multipliers = [0.6, 0.8, 1.0, 1.25, 1.5]
        energy_step = 25.0
    duration_based = [power_mw * duration for duration in durations]
    denominator = model.state_of_health * model.soc_window_fraction
    nameplate_hint = usable_energy_hint_mwh / denominator
    hint_based = [round_up(nameplate_hint * multiplier, energy_step) for multiplier in hint_multipliers]
    values = duration_based + hint_based
    values = [value for value in values if 5.0 <= value <= 30000.0]
    return unique_sorted_positive(values)

def parse_boolean(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes', 'y'}
    if pd.isna(value):
        return False
    return bool(value)

def boolean_series(series: pd.Series) -> pd.Series:
    return series.map(parse_boolean).astype(bool)

def row_is_success(row: dict[str, Any]) -> bool:
    return parse_boolean(row.get('success', False))

def search_sizing_model(context: StudyContext, sites: Sequence[str], allocations: Sequence[float], model: BatteryModel, power_candidates: Sequence[float], recovery_target: float, success_mode: str, cache: TrialCache, full: bool, exhaustive: bool) -> pd.DataFrame:
    print(f"\n[sizing:{model.name}] Searching {len(power_candidates)} power ratings at sites {', '.join(sites)} ...")
    rows_by_key: dict[str, dict[str, Any]] = {}

    def evaluate(power: float, energy: float, position_text: str) -> dict[str, Any]:
        label = f'[sizing:{model.name} {position_text}] {power:.1f} MW / {energy:.1f} MWh'
        row, _, _ = run_battery_trial(context=context, sites=sites, allocations=allocations, total_power_mw=power, total_nameplate_energy_mwh=energy, model=model, recovery_target=recovery_target, success_mode=success_mode, cache=cache, progress_label=label)
        rows_by_key[str(row['trial_key'])] = dict(row)
        return row
    for power_index, power in enumerate(power_candidates, start=1):
        energies = build_energy_candidates(power_mw=power, usable_energy_hint_mwh=context.usable_energy_hint_mwh, model=model, full=full)
        if not energies:
            continue
        if exhaustive:
            for energy_index, energy in enumerate(energies, start=1):
                evaluate(power, energy, f'P {power_index}/{len(power_candidates)}, E {energy_index}/{len(energies)}')
            continue
        high_index = len(energies) - 1
        high_row = evaluate(power, energies[high_index], f'P {power_index}/{len(power_candidates)}, upper bound')
        if not row_is_success(high_row):
            continue
        low_index = -1
        while high_index - low_index > 1:
            middle_index = (low_index + high_index) // 2
            middle_row = evaluate(power, energies[middle_index], f'P {power_index}/{len(power_candidates)}, binary E {middle_index + 1}')
            if row_is_success(middle_row):
                high_index = middle_index
            else:
                low_index = middle_index
        evaluate(power, energies[high_index], f'P {power_index}/{len(power_candidates)}, minimum feasible')
    frame = pd.DataFrame(rows_by_key.values())
    if frame.empty:
        raise RuntimeError(f'No sizing trials were produced for model {model.name}.')
    frame = frame.sort_values(['power_mw', 'nameplate_energy_mwh'], ascending=[True, True]).reset_index(drop=True)
    return frame

def pareto_frontier(feasible: pd.DataFrame) -> pd.DataFrame:
    if feasible.empty:
        return feasible.copy()
    ordered = feasible.sort_values(['power_mw', 'nameplate_energy_mwh'], ascending=[True, True])
    keep_indices: list[int] = []
    best_energy = float('inf')
    for index, row in ordered.iterrows():
        energy = float(row['nameplate_energy_mwh'])
        if energy < best_energy - 1e-09:
            keep_indices.append(index)
            best_energy = energy
    return ordered.loc[keep_indices].sort_values('power_mw').reset_index(drop=True)

def choose_representative_points(trials: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    feasible = trials[boolean_series(trials['solve_ok']) & boolean_series(trials['success'])].copy()
    if feasible.empty:
        best_effort = trials.sort_values(['target_recovery_fraction', 'target_binding_hours'], ascending=[False, True], na_position='last').iloc[0]
        return (pd.DataFrame(), {'best_effort': best_effort})
    frontier = pareto_frontier(feasible)
    minimum_power = frontier.sort_values(['power_mw', 'nameplate_energy_mwh']).iloc[0]
    minimum_energy = frontier.sort_values(['nameplate_energy_mwh', 'power_mw']).iloc[0]
    p_reference = max(float(frontier['power_mw'].min()), 1e-09)
    e_reference = max(float(frontier['nameplate_energy_mwh'].min()), 1e-09)
    balance_score = frontier['power_mw'].astype(float) / p_reference + frontier['nameplate_energy_mwh'].astype(float) / e_reference
    balanced = frontier.loc[balance_score.idxmin()]
    points = {'minimum_power': minimum_power, 'minimum_energy': minimum_energy, 'balanced': balanced}
    print(f'\n[sizing:{model_name}] Pareto-feasible representative points:')
    for label, row in points.items():
        print(f"  {label:16s} {float(row['power_mw']):8.1f} MW / {float(row['nameplate_energy_mwh']):8.1f} MWh | recovery {float(row['target_recovery_pct']):7.2f}% | binding {int(round(float(row['target_binding_hours']))):3d} h")
    return (frontier, points)

def verify_lifetime_recommendation(context: StudyContext, sites: Sequence[str], allocations: Sequence[float], operational_balanced: pd.Series, eol_soh: float, engineering_margin: float, health_soc_min: float, health_soc_max: float, health_round_trip_efficiency: float, marginal_cost: float, recovery_target: float, success_mode: str, cache: TrialCache) -> tuple[dict[str, Any], BatteryModel]:
    eol_model = BatteryModel(name='lifetime_eol', soc_min=health_soc_min, soc_max=health_soc_max, round_trip_efficiency=health_round_trip_efficiency, state_of_health=eol_soh, marginal_cost_eur_per_mwh=marginal_cost)
    base_power = float(operational_balanced['power_mw'])
    base_energy = float(operational_balanced['nameplate_energy_mwh'])
    proposed_power = round_up(base_power * (1.0 + engineering_margin), 5.0)
    proposed_energy = round_up(base_energy / eol_soh * (1.0 + engineering_margin), 10.0)
    for attempt in range(1, 6):
        row, _, _ = run_battery_trial(context=context, sites=sites, allocations=allocations, total_power_mw=proposed_power, total_nameplate_energy_mwh=proposed_energy, model=eol_model, recovery_target=recovery_target, success_mode=success_mode, cache=cache, progress_label=f'[lifetime verification {attempt}/5] {proposed_power:.1f} MW / {proposed_energy:.1f} MWh at {eol_soh * 100:.0f}% SoH')
        if row_is_success(row):
            return (row, eol_model)
        proposed_power = round_up(proposed_power * 1.1, 5.0)
        proposed_energy = round_up(proposed_energy * 1.1, 10.0)
    raise RuntimeError('Lifetime-aware recommendation did not pass after five enlargement attempts. Re-run with --full or inspect the target line and search ranges.')

def export_context_tables(context: StudyContext) -> None:
    baseline_loading = gridkit.line_loading(context.baseline_network)
    target_rows: list[dict[str, Any]] = []
    for line in context.target_lines:
        row = context.baseline_network.lines.loc[line]
        series = baseline_loading[line]
        target_rows.append({'target_label': context.target_label, 'line': line, 'bus0': str(row['bus0']), 'bus1': str(row['bus1']), 'rating_mva': context.original_ratings_mva[line], 'baseline_binding_hours': int((series >= BINDING_THRESHOLD).sum()), 'baseline_max_loading_pct': float(series.max() * 100.0), 'baseline_dispatch_down_mwh': context.baseline_dispatch_down_mwh, 'target_relaxed_dispatch_down_mwh': context.relaxed_dispatch_down_mwh, 'target_attributable_dispatch_down_mwh': context.target_impact_mwh, 'power_search_hint_mw': context.power_hint_mw, 'usable_energy_search_hint_mwh': context.usable_energy_hint_mwh})
    pd.DataFrame(target_rows).to_csv(TABLE_DIR / '01_target_line_summary.csv', index=False)
    weather_names = weather_generator_names(context.baseline_network)
    baseline_energy = generator_dispatch_energy_mwh(context.baseline_network, weather_names)
    relaxed_energy = generator_dispatch_energy_mwh(context.relaxed_network, weather_names)
    affected_rows: list[dict[str, Any]] = []
    for generator in context.affected_generators:
        generator_row = context.baseline_network.generators.loc[generator]
        affected_rows.append({'generator': generator, 'bus': str(generator_row['bus']), 'carrier': str(generator_row['carrier']), 'p_nom_mw': float(generator_row['p_nom']), 'baseline_dispatch_mwh': float(baseline_energy.get(generator, 0.0)), 'target_relaxed_dispatch_mwh': float(relaxed_energy.get(generator, 0.0)), 'dispatch_gain_when_target_relaxed_mwh': float(relaxed_energy.get(generator, 0.0) - baseline_energy.get(generator, 0.0))})
    affected_columns = ['generator', 'bus', 'carrier', 'p_nom_mw', 'baseline_dispatch_mwh', 'target_relaxed_dispatch_mwh', 'dispatch_gain_when_target_relaxed_mwh']
    affected_frame = pd.DataFrame(affected_rows, columns=affected_columns)
    if not affected_frame.empty:
        affected_frame = affected_frame.sort_values('dispatch_gain_when_target_relaxed_mwh', ascending=False)
    affected_frame.to_csv(TABLE_DIR / '02_target_affected_generators.csv', index=False)
    context.excess_profile_mw.to_frame().to_csv(TABLE_DIR / '03_target_relaxed_excess_profile.csv', index_label='snapshot')

def recommendation_rows(sites: Sequence[str], allocations: Sequence[float], model_name: str, points: dict[str, pd.Series]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, point in points.items():
        row = point.to_dict()
        row.update({'recommendation_stage': model_name, 'recommendation_type': label, 'recommended_sites': ';'.join(map(str, sites)), 'site_allocations': ';'.join((f'{value:.6f}' for value in allocations))})
        rows.append(row)
    return rows

def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_site_screening(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    plot_frame = frame.head(15).sort_values('target_recovery_pct', ascending=True)
    plt.figure(figsize=(10, max(5, 0.42 * len(plot_frame) + 1.5)))
    plt.barh(plot_frame['site'], plot_frame['target_recovery_pct'])
    plt.axvline(100.0, linestyle='--', linewidth=1.0)
    plt.xlabel('Target-line counterfactual benefit recovered (%)')
    plt.ylabel('Candidate battery node')
    plt.title('Battery siting screen using one common probe battery')
    plt.grid(axis='x', alpha=0.3)
    save_figure(FIGURE_DIR / '01_single_site_screening.png')

def plot_sizing_trials(trials: pd.DataFrame, model_name: str, filename: str) -> None:
    if trials.empty:
        return
    successful = trials[boolean_series(trials['success'])]
    unsuccessful = trials[~boolean_series(trials['success'])]
    plt.figure(figsize=(9, 6))
    if not unsuccessful.empty:
        plt.scatter(unsuccessful['power_mw'], unsuccessful['nameplate_energy_mwh'], marker='x', label='Not feasible')
    if not successful.empty:
        plt.scatter(successful['power_mw'], successful['nameplate_energy_mwh'], marker='o', label='Feasible')
    plt.xlabel('Battery power rating (MW)')
    plt.ylabel('Battery nameplate energy (MWh)')
    plt.title(f'{model_name}: tested battery power/energy combinations')
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure(FIGURE_DIR / filename)

def plot_pareto_comparison(theoretical_frontier: pd.DataFrame, operational_frontier: pd.DataFrame, operational_model: BatteryModel) -> None:
    if theoretical_frontier.empty and operational_frontier.empty:
        return
    plt.figure(figsize=(9, 6))
    if not theoretical_frontier.empty:
        plt.plot(theoretical_frontier['power_mw'], theoretical_frontier['nameplate_energy_mwh'], marker='o', label='Theoretical 0–100% SoC, 100% RTE')
    if not operational_frontier.empty:
        plt.plot(operational_frontier['power_mw'], operational_frontier['nameplate_energy_mwh'], marker='o', label=f'Health-aware {operational_model.soc_min * 100:.0f}–{operational_model.soc_max * 100:.0f}% SoC, {operational_model.round_trip_efficiency * 100:.0f}% RTE')
    plt.xlabel('Battery power rating (MW)')
    plt.ylabel('Battery nameplate energy (MWh)')
    plt.title('Minimum feasible battery sizing frontier')
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure(FIGURE_DIR / '04_pareto_frontier_comparison.png')

def build_hourly_recommendation_table(context: StudyContext, recommended_network: Any, battery_names: Sequence[str], model: BatteryModel, nameplate_energy_mwh: float) -> pd.DataFrame:
    index = recommended_network.snapshots
    table = pd.DataFrame(index=index)
    baseline_loading = gridkit.line_loading(context.baseline_network)[context.target_lines].max(axis=1)
    relaxed_loading_by_original_rating = pd.DataFrame({line: context.relaxed_network.lines_t.p0[line].abs() / context.original_ratings_mva[line] for line in context.target_lines}, index=index)
    relaxed_loading = relaxed_loading_by_original_rating.max(axis=1)
    recommended_loading = gridkit.line_loading(recommended_network)[context.target_lines].max(axis=1)
    table['baseline_target_loading_pct'] = baseline_loading * 100.0
    table['target_relaxed_loading_pct'] = relaxed_loading * 100.0
    table['recommended_target_loading_pct'] = recommended_loading * 100.0
    p_store = recommended_network.storage_units_t.p_store[list(battery_names)].sum(axis=1)
    p_dispatch = recommended_network.storage_units_t.p_dispatch[list(battery_names)].sum(axis=1)
    model_soc = recommended_network.storage_units_t.state_of_charge[list(battery_names)].sum(axis=1)
    current_capacity = nameplate_energy_mwh * model.state_of_health
    physical_floor = current_capacity * model.soc_min
    physical_soc_pct = (model_soc + physical_floor) / current_capacity * 100.0
    table['battery_charge_mw'] = p_store
    table['battery_discharge_mw'] = p_dispatch
    table['battery_net_injection_mw'] = p_dispatch - p_store
    table['battery_model_usable_soc_mwh'] = model_soc
    table['battery_physical_soc_pct'] = physical_soc_pct
    table['baseline_dispatch_down_mw'] = hourly_dispatch_down_mw(context.baseline_network)
    table['recommended_dispatch_down_mw'] = hourly_dispatch_down_mw(recommended_network)
    table.index.name = 'snapshot'
    return table

def plot_final_line_loading(hourly: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.plot(hourly.index, hourly['baseline_target_loading_pct'], label='Baseline')
    plt.plot(hourly.index, hourly['recommended_target_loading_pct'], label='Lifetime-aware recommendation')
    plt.axhline(BINDING_THRESHOLD * 100.0, linestyle='--', linewidth=1.0)
    plt.xlabel('Snapshot')
    plt.ylabel('Maximum target line/corridor loading (%)')
    plt.title('Target transmission loading across all 168 hours')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    save_figure(FIGURE_DIR / '05_target_line_loading_recommendation.png')

def plot_final_soc(hourly: pd.DataFrame, model: BatteryModel) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.plot(hourly.index, hourly['battery_physical_soc_pct'])
    plt.axhline(model.soc_min * 100.0, linestyle='--', linewidth=1.0, label='SoC lower bound')
    plt.axhline(model.soc_max * 100.0, linestyle='--', linewidth=1.0, label='SoC upper bound')
    plt.xlabel('Snapshot')
    plt.ylabel('Physical state of charge (%)')
    plt.title('Lifetime-aware battery state of charge')
    plt.ylim(max(0.0, model.soc_min * 100.0 - 5.0), min(100.0, model.soc_max * 100.0 + 5.0))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    save_figure(FIGURE_DIR / '06_recommended_battery_soc.png')

def plot_final_power(hourly: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5.5))
    plt.plot(hourly.index, hourly['battery_charge_mw'], label='Charging')
    plt.plot(hourly.index, hourly['battery_discharge_mw'], label='Discharging')
    plt.xlabel('Snapshot')
    plt.ylabel('Battery power (MW)')
    plt.title('Lifetime-aware battery charge and discharge schedule')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    save_figure(FIGURE_DIR / '07_recommended_battery_power.png')

def plot_dispatch_down_comparison(context: StudyContext, recommendation_row: dict[str, Any]) -> None:
    labels = ['Baseline', 'Target relaxed', 'Recommended battery']
    values = [context.baseline_dispatch_down_mwh, context.relaxed_dispatch_down_mwh, float(recommendation_row['total_dispatch_down_mwh'])]
    plt.figure(figsize=(8, 5.5))
    bars = plt.bar(labels, values)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f'{value:,.0f} MWh', ha='center', va='bottom')
    plt.ylabel('Renewable dispatch-down over 168 hours (MWh)')
    plt.title('Dispatch-down comparison')
    plt.grid(axis='y', alpha=0.3)
    save_figure(FIGURE_DIR / '08_dispatch_down_comparison.png')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Battery siting and sizing for TPSA Hackathon 2026 Problem 3.1 Question 2.')
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO)
    parser.add_argument('--scope', default=DEFAULT_SCOPE)
    parser.add_argument('--target-line', default=None, help='Line ID from Question 1. If omitted, the script reads Q1 CSVs.')
    parser.add_argument('--target-mode', choices=['line', 'corridor'], default='line', help='Analyse only the selected line or all parallel lines with the same endpoints.')
    parser.add_argument('--candidate-mode', choices=['eligible', 'renewable', 'all'], default='eligible', help='Which buses are screened as battery sites.')
    parser.add_argument('--max-sites', type=int, choices=[1, 2], default=2, help='1 = single-site only; 2 = also test pairs among top single sites.')
    parser.add_argument('--pair-top-n', type=int, default=4, help='Number of top single sites used to construct pair tests.')
    parser.add_argument('--probe-power', type=float, default=None, help='Optional fixed siting-probe power in MW. Default is a moderate target-based value capped at 100 MW.')
    parser.add_argument('--probe-hours', type=float, default=4.0, help='Nameplate duration of the common siting probe, default 4 h.')
    parser.add_argument('--success-mode', choices=['benchmark', 'headroom', 'both'], default='benchmark', help='benchmark = recover the dispatch-down benefit of relaxing the target line (default); headroom = zero >=99.9% loading hours; both = require both. Binding hours are always reported separately.')
    parser.add_argument('--recovery-target', type=float, default=DEFAULT_RECOVERY_TARGET, help='Required fraction of target-line counterfactual benefit, default 0.99.')
    parser.add_argument('--soc-min', type=float, default=HEALTH_SOC_MIN, help='Physical minimum SoC for the health-aware model, default 0.10.')
    parser.add_argument('--soc-max', type=float, default=HEALTH_SOC_MAX, help='Physical maximum SoC for the health-aware model, default 0.90.')
    parser.add_argument('--round-trip-efficiency', type=float, default=HEALTH_ROUND_TRIP_EFFICIENCY, help='Health-aware battery round-trip efficiency, default 0.90.')
    parser.add_argument('--eol-soh', type=float, default=DEFAULT_EOL_SOH, help='End-of-life state of health used for final verification, default 0.80.')
    parser.add_argument('--engineering-margin', type=float, default=DEFAULT_ENGINEERING_MARGIN, help='Power and energy design margin applied to final recommendation.')
    parser.add_argument('--battery-throughput-cost', type=float, default=DEFAULT_BATTERY_MARGINAL_COST, help='StorageUnit marginal cost in EUR/MWh; default matches official example.')
    parser.add_argument('--full', action='store_true', help='Use denser MW and MWh candidate grids.')
    parser.add_argument('--exhaustive', action='store_true', help='Evaluate every grid point instead of adaptive binary search.')
    parser.add_argument('--smoke-test', action='store_true', help='Run only baseline, target-relaxed, and one battery trial to check the environment; skip the full search.')
    parser.add_argument('--no-resume', action='store_true', help='Ignore the saved trial cache and solve every requested case again.')
    arguments = parser.parse_args()
    if not 0.0 < arguments.recovery_target <= 2.0:
        parser.error('--recovery-target must be greater than 0 and no more than 2.')
    if not 0.0 <= arguments.soc_min < arguments.soc_max <= 1.0:
        parser.error('Require 0 <= --soc-min < --soc-max <= 1.')
    if not 0.0 < arguments.round_trip_efficiency <= 1.0:
        parser.error('--round-trip-efficiency must be in (0, 1].')
    if not 0.0 < arguments.eol_soh <= 1.0:
        parser.error('--eol-soh must be in (0, 1].')
    if arguments.engineering_margin < 0.0:
        parser.error('--engineering-margin cannot be negative.')
    if arguments.pair_top_n < 2:
        parser.error('--pair-top-n must be at least 2.')
    if arguments.probe_power is not None and arguments.probe_power <= 0.0:
        parser.error('--probe-power must be positive when provided.')
    if arguments.probe_hours <= 0.0:
        parser.error('--probe-hours must be positive.')
    if arguments.battery_throughput_cost < 0.0:
        parser.error('--battery-throughput-cost cannot be negative.')
    return arguments

def main() -> int:
    args = parse_arguments()
    ensure_output_directories()
    cache = TrialCache(CACHE_PATH, enabled=not args.no_resume)
    theoretical_model = BatteryModel(name='theoretical', soc_min=0.0, soc_max=1.0, round_trip_efficiency=1.0, state_of_health=1.0, marginal_cost_eur_per_mwh=args.battery_throughput_cost)
    operational_model = BatteryModel(name='health_aware', soc_min=args.soc_min, soc_max=args.soc_max, round_trip_efficiency=args.round_trip_efficiency, state_of_health=1.0, marginal_cost_eur_per_mwh=args.battery_throughput_cost)
    theoretical_model.validate()
    operational_model.validate()
    context = build_study_context(scenario=args.scenario, scope=args.scope, requested_line=args.target_line, target_mode=args.target_mode)
    export_context_tables(context)
    buses = candidate_buses(context.baseline_network, mode=args.candidate_mode, target_endpoints=context.target_endpoints)
    probe_power, probe_energy = make_probe_size(context, operational_model, requested_power_mw=args.probe_power, requested_hours=args.probe_hours)
    if args.smoke_test:
        smoke_site = context.target_endpoints[0] if context.target_endpoints[0] in buses else buses[0]
        smoke_row, _, _ = run_battery_trial(context=context, sites=[smoke_site], total_power_mw=probe_power, total_nameplate_energy_mwh=probe_energy, model=operational_model, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache, force_solve=True, progress_label=f'[smoke-test] {smoke_site}: {probe_power:.1f} MW / {probe_energy:.1f} MWh')
        pd.DataFrame([smoke_row]).to_csv(TABLE_DIR / 'smoke_test_result.csv', index=False)
        print('\n[smoke-test] Environment and battery solve completed successfully.')
        print(f"[smoke-test] Target recovery = {float(smoke_row['target_recovery_pct']):.2f}%; binding hours = {int(round(float(smoke_row['target_binding_hours'])))}")
        print('[smoke-test] This is only a technical check, not the final answer.')
        return 0
    single_results = screen_single_sites(context=context, buses=buses, probe_power_mw=probe_power, probe_energy_mwh=probe_energy, model=operational_model, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache)
    single_results.to_csv(TABLE_DIR / '04_single_site_screening.csv', index=False)
    plot_site_screening(single_results)
    pair_results = pd.DataFrame()
    if args.max_sites == 2:
        pair_results = screen_site_pairs(context=context, single_site_results=single_results, top_n=args.pair_top_n, probe_power_mw=probe_power, probe_energy_mwh=probe_energy, model=operational_model, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache)
        pair_results.to_csv(TABLE_DIR / '05_two_site_screening.csv', index=False)
    selected_sites, selected_allocations, architecture_reason = choose_architecture(single_results=single_results, pair_results=pair_results)
    print('\n[architecture] Selected battery location structure:')
    for site, share in zip(selected_sites, selected_allocations):
        print(f'  {site}: {share * 100:.1f}% of total MW and MWh')
    print(f'[architecture] Reason: {architecture_reason}')
    power_candidates = build_power_candidates(context.power_hint_mw, full=args.full)
    print(f'[sizing] Power candidates ({len(power_candidates)}): ' + ', '.join((f'{value:.0f}' for value in power_candidates)) + ' MW')
    theoretical_trials = search_sizing_model(context=context, sites=selected_sites, allocations=selected_allocations, model=theoretical_model, power_candidates=power_candidates, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache, full=args.full, exhaustive=args.exhaustive)
    theoretical_trials.to_csv(TABLE_DIR / '06_theoretical_sizing_trials.csv', index=False)
    operational_trials = search_sizing_model(context=context, sites=selected_sites, allocations=selected_allocations, model=operational_model, power_candidates=power_candidates, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache, full=args.full, exhaustive=args.exhaustive)
    operational_trials.to_csv(TABLE_DIR / '07_health_aware_sizing_trials.csv', index=False)
    plot_sizing_trials(theoretical_trials, 'Theoretical battery', '02_theoretical_sizing_trials.png')
    plot_sizing_trials(operational_trials, 'Health-aware battery', '03_health_aware_sizing_trials.png')
    theoretical_frontier, theoretical_points = choose_representative_points(theoretical_trials, 'theoretical')
    operational_frontier, operational_points = choose_representative_points(operational_trials, 'health-aware')
    theoretical_frontier.assign(model='theoretical').to_csv(TABLE_DIR / '08_theoretical_pareto_frontier.csv', index=False)
    operational_frontier.assign(model='health_aware').to_csv(TABLE_DIR / '09_health_aware_pareto_frontier.csv', index=False)
    plot_pareto_comparison(theoretical_frontier, operational_frontier, operational_model)
    if 'balanced' not in operational_points:
        best_effort = operational_points['best_effort']
        print('\n[warning] No fully feasible health-aware battery was found in the current search range.')
        print(f"Best tested point: {float(best_effort['power_mw']):.1f} MW / {float(best_effort['nameplate_energy_mwh']):.1f} MWh, target recovery {float(best_effort['target_recovery_pct']):.2f}%.")
        print('Re-run with --full, or inspect the selected target line.')
        return 2
    lifetime_row, lifetime_model = verify_lifetime_recommendation(context=context, sites=selected_sites, allocations=selected_allocations, operational_balanced=operational_points['balanced'], eol_soh=args.eol_soh, engineering_margin=args.engineering_margin, health_soc_min=args.soc_min, health_soc_max=args.soc_max, health_round_trip_efficiency=args.round_trip_efficiency, marginal_cost=args.battery_throughput_cost, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache)
    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(recommendation_rows(selected_sites, selected_allocations, 'theoretical', theoretical_points))
    summary_rows.extend(recommendation_rows(selected_sites, selected_allocations, 'health_aware', operational_points))
    lifetime_summary = dict(lifetime_row)
    lifetime_summary.update({'recommendation_stage': 'lifetime_aware_eol_verified', 'recommendation_type': 'recommended_installed_capacity', 'recommended_sites': ';'.join(selected_sites), 'site_allocations': ';'.join((f'{value:.6f}' for value in selected_allocations)), 'engineering_margin_pct': args.engineering_margin * 100.0, 'eol_soh_assumption_pct': args.eol_soh * 100.0, 'architecture_reason': architecture_reason})
    summary_rows.append(lifetime_summary)
    recommendation_summary = pd.DataFrame(summary_rows)
    recommendation_summary.to_csv(TABLE_DIR / '10_recommendation_summary.csv', index=False)
    final_row, final_network, battery_names = run_battery_trial(context=context, sites=selected_sites, allocations=selected_allocations, total_power_mw=float(lifetime_row['power_mw']), total_nameplate_energy_mwh=float(lifetime_row['nameplate_energy_mwh']), model=lifetime_model, recovery_target=args.recovery_target, success_mode=args.success_mode, cache=cache, force_solve=True, return_network=True, progress_label='[final] Re-solving the selected lifetime-aware recommendation for plots')
    if final_network is None:
        raise RuntimeError('Final recommendation network was not returned.')
    hourly = build_hourly_recommendation_table(context=context, recommended_network=final_network, battery_names=battery_names, model=lifetime_model, nameplate_energy_mwh=float(final_row['nameplate_energy_mwh']))
    hourly.to_csv(TABLE_DIR / '11_recommended_hourly_operation.csv')
    plot_final_line_loading(hourly)
    plot_final_soc(hourly, lifetime_model)
    plot_final_power(hourly)
    plot_dispatch_down_comparison(context, final_row)
    print('\n' + '=' * 78)
    print('FINAL RECOMMENDATION WITHIN THE SUPPLIED 168-HOUR SYNTHETIC SCENARIO')
    print('=' * 78)
    print(f'Target: {context.target_label}')
    print('Sites:')
    for site, share in zip(selected_sites, selected_allocations):
        print(f'  - {site}: {share * 100:.1f}% of installed power and energy')
    print(f"Installed rating: {float(final_row['power_mw']):,.1f} MW / {float(final_row['nameplate_energy_mwh']):,.1f} MWh")
    print(f'Operating assumptions at verification: {lifetime_model.soc_min * 100:.0f}–{lifetime_model.soc_max * 100:.0f}% SoC, {lifetime_model.round_trip_efficiency * 100:.0f}% round-trip efficiency, {lifetime_model.state_of_health * 100:.0f}% SoH')
    print(f"Target counterfactual benefit recovered: {float(final_row['target_recovery_pct']):,.2f}%")
    print(f"Target >=99.9% loading hours after battery: {int(round(float(final_row['target_binding_hours'])))}")
    print(f"Dispatch-down: {context.baseline_dispatch_down_mwh:,.1f} -> {float(final_row['total_dispatch_down_mwh']):,.1f} MWh")
    print(f"Battery throughput: {float(final_row['throughput_mwh']):,.1f} MWh; EFC/week (nameplate basis): {float(final_row['equivalent_full_cycles_nameplate']):,.3f}")
    print(f"Average absolute battery power: {float(final_row['average_absolute_power_mw']):,.1f} MW ({float(final_row['average_power_utilisation_pct']):,.1f}% of installed MW)")
    print(f"Active / idle hours: {int(round(float(final_row['active_hours'])))} / {int(round(float(final_row['idle_hours'])))}")
    print('Note: a line may still operate at its rating even after its dispatch-down impact has been removed; binding hours are therefore reported as a separate diagnostic.')
    print(f'\nTables:  {TABLE_DIR}')
    print(f'Figures: {FIGURE_DIR}')
    print('\nInterpretation note: this recommendation is valid only within the supplied synthetic WP2033 North-West week and the DC network model.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
