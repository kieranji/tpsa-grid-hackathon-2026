#!/usr/bin/env python3
"""TPSA Hackathon 2026 — Problem 3.1, Question 3.

Evaluates Dynamic Line Rating (DLR) sensitivity on the supplied 168-hour
WP2033 North-West model. The wind-dependent DLR profile is a transparent
proxy based on local wind availability because the kit does not provide the
weather and conductor measurements needed for an operational thermal rating.

DLR-only results answer the question. Q2 BESS coordination is a secondary
analysis and inherits Q2's battery assumptions and success criterion.
"""
from __future__ import annotations
import argparse
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from network_validation import scale_thermal_limits
SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
KIT_DIR = PROJECT_ROOT / 'data' / 'participant-kit'
GRIDKIT_PATH = KIT_DIR / 'gridkit.py'
if not GRIDKIT_PATH.exists():
    raise FileNotFoundError(f'Cannot find the official participant kit.\nExpected gridkit.py at: {GRIDKIT_PATH}\nMake sure the official participant-kit is copied to data/participant-kit.')
sys.path.insert(0, str(KIT_DIR))
import gridkit
gridkit.quiet()
DEFAULT_SCENARIO = 'WP2033'
DEFAULT_SCOPE = 'north-west'
BINDING_THRESHOLD = 0.999
# Counterfactual only: make branch ratings non-binding.
RELAX_RATING_MULTIPLIER = 1000.0
# DLR uplift levels are sensitivity cases, not operational ratings.
DEFAULT_MAX_UPLIFTS = (0.1, 0.2, 0.3, 0.4)
DEFAULT_WIND_THRESHOLD = 0.2
DEFAULT_WIND_EXPONENT = 1.0
# Conservative reduction applied to the wind-proxy uplift.
DEFAULT_FORECAST_DERATE = 0.2
DEFAULT_SENSOR_AVAILABILITY = 1.0
DEFAULT_NEIGHBOUR_HOPS = 2
DEFAULT_CABLE_X_PER_KM_THRESHOLD = 0.2
DEFAULT_DLR_BENEFIT_FRACTION = 0.95
DEFAULT_TARGET_RECOVERY = 0.99
DEFAULT_MAX_EFC_PER_DAY = 1.0
DEFAULT_BATTERY_THROUGHPUT_COST = 0.5
# Fallback Q2 health assumptions if the recommendation CSV lacks them.
DEFAULT_SOC_MIN = 0.1
DEFAULT_SOC_MAX = 0.9
DEFAULT_BATTERY_RTE = 0.9
DEFAULT_EOL_SOH = 0.8
DEFAULT_BATTERY_SCALES = (0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0, 1.1, 1.25, 1.5)
ENERGY_TOLERANCE_MWH = 0.001
# New outputs are isolated from the earlier Q3 run.
OUTPUT_ROOT = PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_3_final'
TABLE_DIR = OUTPUT_ROOT / 'tables'
FIGURE_DIR = OUTPUT_ROOT / 'figures'
DEFAULT_Q2_SUMMARY = PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_2_final' / 'tables' / '10_recommendation_summary.csv'
DEFAULT_Q2_TARGET = PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_2_final' / 'tables' / '01_target_line_summary.csv'
Q1_RESULT_CANDIDATES = [PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1_final' / 'tables' / '03_lines_ranked_by_recovered_energy.csv', PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1_final' / 'tables' / '02_line_constraint_and_rating_results.csv', PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1' / 'tables' / '03_lines_ranked_by_recovered_energy.csv', PROJECT_ROOT / 'results' / 'problem_3_1' / 'question_1' / 'tables' / '02_line_constraint_and_rating_results.csv', PROJECT_ROOT / 'results' / 'tables' / '03_lines_ranked_by_recovered_energy.csv', PROJECT_ROOT / 'results' / 'tables' / '02_line_constraint_and_rating_results.csv']

@dataclass(frozen=True)
class DLRConfig:
    mode: str
    max_uplift: float
    wind_threshold: float
    wind_exponent: float
    forecast_derate: float
    sensor_availability: float

    def validate(self) -> None:
        if self.mode not in {'constant', 'wind_proxy'}:
            raise ValueError(f'Unsupported DLR mode: {self.mode}')
        if not 0.0 <= self.max_uplift <= 2.0:
            raise ValueError('max_uplift must be between 0 and 2')
        if not 0.0 <= self.wind_threshold < 1.0:
            raise ValueError('wind_threshold must be in [0, 1)')
        if self.wind_exponent <= 0.0:
            raise ValueError('wind_exponent must be positive')
        if not 0.0 <= self.forecast_derate < 1.0:
            raise ValueError('forecast_derate must be in [0, 1)')
        if not 0.0 < self.sensor_availability <= 1.0:
            raise ValueError('sensor_availability must be in (0, 1]')

@dataclass(frozen=True)
class BatteryConfig:
    sites: tuple[str, ...]
    allocations: tuple[float, ...]
    power_mw: float
    nameplate_energy_mwh: float
    soc_min: float
    soc_max: float
    round_trip_efficiency: float
    state_of_health: float
    marginal_cost_eur_per_mwh: float
    success_mode: str
    recovery_target: float
    max_efc_per_day: float | None
    source: str

    @property
    def soc_window_fraction(self) -> float:
        return self.soc_max - self.soc_min

    @property
    def one_way_efficiency(self) -> float:
        return math.sqrt(self.round_trip_efficiency)

    @property
    def usable_energy_mwh(self) -> float:
        return self.nameplate_energy_mwh * self.state_of_health * self.soc_window_fraction

    def scaled(self, scale: float) -> 'BatteryConfig':
        return BatteryConfig(sites=self.sites, allocations=self.allocations, power_mw=self.power_mw * scale, nameplate_energy_mwh=self.nameplate_energy_mwh * scale, soc_min=self.soc_min, soc_max=self.soc_max, round_trip_efficiency=self.round_trip_efficiency, state_of_health=self.state_of_health, marginal_cost_eur_per_mwh=self.marginal_cost_eur_per_mwh, success_mode=self.success_mode, recovery_target=self.recovery_target, max_efc_per_day=self.max_efc_per_day, source=self.source)

    def validate(self) -> None:
        if not self.sites:
            raise ValueError('Battery recommendation contains no sites')
        if len(self.sites) != len(self.allocations):
            raise ValueError('Battery sites and allocations have different lengths')
        if any((value <= 0.0 for value in self.allocations)):
            raise ValueError('Every battery allocation must be positive')
        if not math.isclose(sum(self.allocations), 1.0, rel_tol=1e-06, abs_tol=1e-06):
            raise ValueError('Battery allocations must sum to 1')
        if self.power_mw <= 0.0 or self.nameplate_energy_mwh <= 0.0:
            raise ValueError('Battery MW and MWh must be positive')
        if not 0.0 <= self.soc_min < self.soc_max <= 1.0:
            raise ValueError('Battery SoC window is invalid')
        if not 0.0 < self.round_trip_efficiency <= 1.0:
            raise ValueError('Battery round-trip efficiency must be in (0, 1]')
        if not 0.0 < self.state_of_health <= 1.0:
            raise ValueError('Battery SoH must be in (0, 1]')
        if self.success_mode not in {'benchmark', 'headroom', 'both'}:
            raise ValueError(f'Unsupported Q2 success mode: {self.success_mode}')
        if not 0.0 < self.recovery_target <= 2.0:
            raise ValueError('Battery recovery target must be in (0, 2]')
        if self.max_efc_per_day is not None and self.max_efc_per_day <= 0.0:
            raise ValueError('max_efc_per_day must be positive when enabled')

@dataclass
class StudyContext:
    scenario: str
    scope: str
    selected_line: str
    target_mode: str
    target_lines: list[str]
    target_label: str
    target_endpoints: tuple[str, str]
    original_ratings_mva: dict[str, float]
    baseline_network: Any
    target_relaxed_network: Any
    all_relaxed_network: Any
    baseline_total_dispatch_down_mwh: float
    baseline_wind_dispatch_down_mwh: float
    baseline_solar_dispatch_down_mwh: float
    target_relaxed_total_dispatch_down_mwh: float
    target_relaxed_wind_dispatch_down_mwh: float
    all_relaxed_total_dispatch_down_mwh: float
    all_relaxed_wind_dispatch_down_mwh: float
    baseline_unserved_mwh: float
    target_impact_total_mwh: float
    target_impact_wind_mwh: float
    baseline_target_binding_hours: int
    baseline_target_max_loading_pct: float
    baseline_binding_lines: list[str]
    affected_generators: list[str]
    affected_target_gain_mwh: float
    baseline_affected_dispatch_mwh: float

def ensure_output_directories() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for raw in text.split(','):
        stripped = raw.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError('At least one numeric value is required')
    return sorted(set(values))

def normalise_allocations(values: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0 or np.any(array <= 0.0):
        raise ValueError('Battery allocations must be positive')
    return tuple((array / array.sum()).tolist())

def safe_float(value: Any, default: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default

def solve_checked(network: Any, label: str, extra_functionality: Any | None=None) -> None:
    kwargs: dict[str, Any] = {}
    if extra_functionality is not None:
        kwargs['extra_functionality'] = extra_functionality
    result = gridkit.solve(network, **kwargs)
    if isinstance(result, tuple) and len(result) >= 2:
        status = str(result[0]).lower()
        condition = str(result[1]).lower()
        if status != 'ok' or 'optimal' not in condition:
            raise RuntimeError(f'{label} solve failed: status={result[0]}, condition={result[1]}')
    if not len(network.generators_t.p.columns):
        raise RuntimeError(f'{label} solve produced no generator dispatch')

def snapshot_weights(network: Any, component: str='generators') -> pd.Series:
    weightings = network.snapshot_weightings
    if component in weightings.columns:
        return weightings[component].reindex(network.snapshots).astype(float)
    return pd.Series(1.0, index=network.snapshots, dtype=float)

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
    weights = snapshot_weights(network, 'generators')
    return dispatch[selected].mul(weights, axis=0).sum(axis=0)

def dispatch_down_by_carrier(network: Any) -> dict[str, float]:
    available = network.generators_t.p_max_pu
    dispatched = network.generators_t.p
    if available.empty or dispatched.empty:
        return {'total': 0.0, 'wind': 0.0, 'solar': 0.0}
    names = [name for name in available.columns if name in dispatched.columns]
    if not names:
        return {'total': 0.0, 'wind': 0.0, 'solar': 0.0}
    offered_mw = available[names].mul(network.generators.loc[names, 'p_nom'], axis=1)
    lost_mw = (offered_mw - dispatched[names].clip(lower=0.0)).clip(lower=0.0)
    weights = snapshot_weights(network, 'generators')
    lost_mwh = lost_mw.mul(weights, axis=0).sum(axis=0)
    carriers = network.generators.loc[names, 'carrier'].astype(str)
    grouped = lost_mwh.groupby(carriers).sum()
    return {'total': float(lost_mwh.sum()), 'wind': float(grouped.get('wind', 0.0)), 'solar': float(grouped.get('solar', 0.0))}

def total_unserved_mwh(network: Any) -> float:
    values = gridkit.unserved(network)
    return float(values.sum()) if len(values) else 0.0

def target_lines_from_mode(network: Any, selected_line: str, target_mode: str) -> list[str]:
    if selected_line not in network.lines.index:
        raise KeyError(f'Unknown line: {selected_line}')
    if target_mode == 'line':
        return [selected_line]
    row = network.lines.loc[selected_line]
    endpoints = frozenset((str(row['bus0']), str(row['bus1'])))
    members = [str(name) for name, candidate in network.lines.iterrows() if frozenset((str(candidate['bus0']), str(candidate['bus1']))) == endpoints]
    return sorted(members)

def _line_column(frame: pd.DataFrame) -> str | None:
    for candidate in ('line', 'name', 'line_name', 'Unnamed: 0'):
        if candidate in frame.columns:
            return candidate
    return None

def choose_target_line(network: Any, requested_line: str | None) -> tuple[str, str]:
    if requested_line:
        if requested_line not in network.lines.index:
            raise KeyError(f'Requested line {requested_line!r} is not in this network.')
        return (requested_line, 'manually specified with --target-line')
    if DEFAULT_Q2_TARGET.exists():
        frame = pd.read_csv(DEFAULT_Q2_TARGET)
        if 'line' in frame.columns:
            for value in frame['line'].dropna().astype(str):
                if value in network.lines.index:
                    return (value, f'read from {DEFAULT_Q2_TARGET}')
    for path in Q1_RESULT_CANDIDATES:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        line_column = _line_column(frame)
        if line_column is None:
            continue
        working = frame.copy()
        working[line_column] = working[line_column].astype(str)
        working = working[working[line_column].isin(network.lines.index)]
        if 'binding_hours' in working.columns:
            working = working[pd.to_numeric(working['binding_hours'], errors='coerce').fillna(0) > 0]
        if working.empty:
            continue
        sort_columns: list[str] = []
        ascending: list[bool] = []
        for candidate in ('saved_dispatch_down_plus_25_mwh', 'saved_mwh_plus_25', 'saved_mwh_if_line_unlimited', 'binding_hours'):
            if candidate in working.columns:
                sort_columns.append(candidate)
                ascending.append(False)
        if sort_columns:
            working = working.sort_values(sort_columns, ascending=ascending)
        value = str(working.iloc[0][line_column])
        return (value, f'selected from first-question result {path}')
    loading = network.lines_t.p0.abs().div(network.lines['s_nom'], axis=1)
    binding_hours = (loading >= BINDING_THRESHOLD).sum().sort_values(ascending=False)
    if len(binding_hours) and int(binding_hours.iloc[0]) > 0:
        return (str(binding_hours.index[0]), 'fallback: most binding baseline line')
    return (str(loading.max().sort_values(ascending=False).index[0]), 'fallback: highest maximum baseline loading')

def line_asset_screen(network: Any, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, line in network.lines.iterrows():
        length = safe_float(line.get('length'), 0.0)
        x = abs(safe_float(line.get('x'), float('nan')))
        if length > 0.0 and math.isfinite(x):
            x_per_km = x / length
            inferred_type = 'cable_like' if x_per_km < threshold else 'overhead_like'
        else:
            x_per_km = float('nan')
            inferred_type = 'unknown'
        rows.append({'line': str(name), 'bus0': str(line['bus0']), 'bus1': str(line['bus1']), 'rating_mva': float(line['s_nom']), 'length_km': length, 'x_ohm': x, 'x_per_km': x_per_km, 'inferred_asset_type': inferred_type, 'wind_cooling_dlr_eligible': inferred_type == 'overhead_like'})
    return pd.DataFrame(rows)

def target_loading_metrics(network: Any, target_lines: Sequence[str], applied_s_max_pu: pd.DataFrame | None=None) -> dict[str, float]:
    flows = network.lines_t.p0[list(target_lines)].abs()
    ratings = network.lines.loc[list(target_lines), 's_nom'].astype(float)
    static_loading = flows.div(ratings, axis=1)
    if applied_s_max_pu is None:
        dynamic_multiplier = pd.DataFrame(1.0, index=network.snapshots, columns=list(target_lines))
    else:
        dynamic_multiplier = applied_s_max_pu.reindex(index=network.snapshots, columns=list(target_lines)).fillna(1.0)
    dynamic_limits = dynamic_multiplier.mul(ratings, axis=1)
    dynamic_loading = flows.div(dynamic_limits.replace(0.0, np.nan))
    any_binding = (dynamic_loading >= BINDING_THRESHOLD).any(axis=1)
    line_hours = int((dynamic_loading >= BINDING_THRESHOLD).sum().sum())
    return {'target_binding_hours': float(int(any_binding.sum())), 'target_binding_line_hours': float(line_hours), 'target_max_dynamic_loading_pct': float(dynamic_loading.max().max() * 100.0), 'target_mean_max_dynamic_loading_pct': float(dynamic_loading.max(axis=1).mean() * 100.0), 'target_max_static_equivalent_loading_pct': float(static_loading.max().max() * 100.0), 'target_mean_dlr_multiplier': float(dynamic_multiplier.mean().mean()), 'target_max_dlr_multiplier': float(dynamic_multiplier.max().max())}

def build_study_context(scenario: str, scope: str, requested_line: str | None, target_mode: str) -> StudyContext:
    print('\n[baseline] Loading and solving the original network ...')
    baseline = gridkit.load(scenario, scope)
    solve_checked(baseline, 'baseline')
    selected_line, selection_reason = choose_target_line(baseline, requested_line)
    target_lines = target_lines_from_mode(baseline, selected_line, target_mode)
    selected_row = baseline.lines.loc[selected_line]
    endpoints = (str(selected_row['bus0']), str(selected_row['bus1']))
    target_label = selected_line if len(target_lines) == 1 else f'corridor {endpoints[0]} <-> {endpoints[1]}'
    original_ratings = {line: float(baseline.lines.at[line, 's_nom']) for line in target_lines}
    baseline_down = dispatch_down_by_carrier(baseline)
    baseline_unserved = total_unserved_mwh(baseline)
    baseline_metrics = target_loading_metrics(baseline, target_lines)
    baseline_loading = baseline.lines_t.p0.abs().div(baseline.lines['s_nom'], axis=1)
    baseline_binding = (baseline_loading >= BINDING_THRESHOLD).sum().loc[lambda series: series > 0].sort_values(ascending=False)
    print(f'[target] {target_label}')
    print(f'[target] Selection reason: {selection_reason}')
    if len(target_lines) > 1:
        print(f"[target] Corridor members: {', '.join(target_lines)}")
    print(f"[baseline] wind dispatch-down = {baseline_down['wind']:,.2f} MWh; total renewable dispatch-down = {baseline_down['total']:,.2f} MWh; target binding hours = {int(baseline_metrics['target_binding_hours'])}")
    print('[counterfactual] Relaxing only the target line/corridor ...')
    target_relaxed = gridkit.load(scenario, scope)
    for line, rating in original_ratings.items():
        gridkit.set_rating(target_relaxed, line, rating * RELAX_RATING_MULTIPLIER)
    solve_checked(target_relaxed, 'target-relaxed counterfactual')
    target_relaxed_down = dispatch_down_by_carrier(target_relaxed)
    weather_names = weather_generator_names(baseline)
    baseline_dispatch = generator_dispatch_energy_mwh(baseline, weather_names)
    target_relaxed_dispatch = generator_dispatch_energy_mwh(target_relaxed, weather_names)
    dispatch_gain = target_relaxed_dispatch.sub(baseline_dispatch, fill_value=0.0)
    positive_gain = dispatch_gain[dispatch_gain > ENERGY_TOLERANCE_MWH].sort_values(ascending=False)
    affected_generators = list(positive_gain.index.astype(str))
    affected_target_gain_mwh = float(positive_gain.sum())
    baseline_affected_dispatch_mwh = float(baseline_dispatch.reindex(affected_generators).fillna(0.0).sum())
    print(f'[counterfactual] {len(affected_generators)} weather-driven generators gain {affected_target_gain_mwh:,.2f} MWh in total')
    print('[counterfactual] Relaxing all line and transformer ratings ...')
    all_relaxed = gridkit.load(scenario, scope)
    scale_thermal_limits(all_relaxed, RELAX_RATING_MULTIPLIER)
    solve_checked(all_relaxed, 'all-ratings-relaxed counterfactual')
    all_relaxed_down = dispatch_down_by_carrier(all_relaxed)
    target_impact_total = max(0.0, baseline_down['total'] - target_relaxed_down['total'])
    target_impact_wind = max(0.0, baseline_down['wind'] - target_relaxed_down['wind'])
    print(f'[counterfactual] Target-attributable wind reduction = {target_impact_wind:,.2f} MWh')
    print(f"[counterfactual] All-network wind surplus floor = {all_relaxed_down['wind']:,.2f} MWh")
    return StudyContext(scenario=scenario, scope=scope, selected_line=selected_line, target_mode=target_mode, target_lines=target_lines, target_label=target_label, target_endpoints=endpoints, original_ratings_mva=original_ratings, baseline_network=baseline, target_relaxed_network=target_relaxed, all_relaxed_network=all_relaxed, baseline_total_dispatch_down_mwh=baseline_down['total'], baseline_wind_dispatch_down_mwh=baseline_down['wind'], baseline_solar_dispatch_down_mwh=baseline_down['solar'], target_relaxed_total_dispatch_down_mwh=target_relaxed_down['total'], target_relaxed_wind_dispatch_down_mwh=target_relaxed_down['wind'], all_relaxed_total_dispatch_down_mwh=all_relaxed_down['total'], all_relaxed_wind_dispatch_down_mwh=all_relaxed_down['wind'], baseline_unserved_mwh=baseline_unserved, target_impact_total_mwh=target_impact_total, target_impact_wind_mwh=target_impact_wind, baseline_target_binding_hours=int(baseline_metrics['target_binding_hours']), baseline_target_max_loading_pct=float(baseline_metrics['target_max_dynamic_loading_pct']), baseline_binding_lines=list(baseline_binding.index.astype(str)), affected_generators=affected_generators, affected_target_gain_mwh=affected_target_gain_mwh, baseline_affected_dispatch_mwh=baseline_affected_dispatch_mwh)

def choose_dlr_lines(context: StudyContext, asset_screen: pd.DataFrame, dlr_scope: str, allow_non_overhead: bool) -> list[str]:
    eligible = set(asset_screen.loc[asset_screen['wind_cooling_dlr_eligible'], 'line'].astype(str))
    if dlr_scope == 'target':
        candidates = list(context.target_lines)
    elif dlr_scope == 'binding-overhead':
        candidates = list(context.baseline_binding_lines)
    elif dlr_scope == 'all-overhead':
        candidates = list(context.baseline_network.lines.index.astype(str))
    else:
        raise ValueError(f'Unknown dlr_scope: {dlr_scope}')
    if allow_non_overhead:
        selected = [line for line in candidates if line in context.baseline_network.lines.index]
    else:
        selected = [line for line in candidates if line in eligible]
    excluded = [line for line in candidates if line not in selected]
    if excluded:
        print(f"[DLR] Excluded from wind-cooling DLR because they are not inferred overhead lines: {', '.join(excluded)}")
    if not selected:
        raise RuntimeError('No DLR-eligible overhead-like lines remain. Inspect 01_line_asset_screen.csv, or use --allow-non-overhead-dlr only as a clearly-labelled sensitivity test.')
    return selected

def build_network_graph(network: Any) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index.astype(str))
    for _, line in network.lines.iterrows():
        graph.add_edge(str(line['bus0']), str(line['bus1']))
    for _, transformer in network.transformers.iterrows():
        graph.add_edge(str(transformer['bus0']), str(transformer['bus1']))
    return graph

def local_wind_availability_for_line(network: Any, line_name: str, neighbour_hops: int) -> tuple[pd.Series, pd.DataFrame]:
    if line_name not in network.lines.index:
        raise KeyError(f'Unknown line: {line_name}')
    line = network.lines.loc[line_name]
    endpoints = (str(line['bus0']), str(line['bus1']))
    graph = build_network_graph(network)
    wind_generators = network.generators[network.generators['carrier'].astype(str) == 'wind'].copy()
    wind_generators = wind_generators[wind_generators.index.isin(network.generators_t.p_max_pu.columns)]
    if wind_generators.empty:
        raise RuntimeError('No wind generators with p_max_pu profiles were found')
    selected_rows: list[dict[str, Any]] = []
    for generator_name, generator in wind_generators.iterrows():
        bus = str(generator['bus'])
        distances: list[int] = []
        for endpoint in endpoints:
            try:
                distances.append(nx.shortest_path_length(graph, bus, endpoint))
            except nx.NetworkXNoPath:
                continue
        if not distances:
            continue
        hops = min(distances)
        if hops <= neighbour_hops:
            p_nom = float(generator['p_nom'])
            weight = p_nom / (1.0 + hops)
            selected_rows.append({'line': line_name, 'generator': str(generator_name), 'bus': bus, 'network_hops': hops, 'p_nom_mw': p_nom, 'proxy_weight': weight, 'selection': 'local'})
    if not selected_rows:
        for generator_name, generator in wind_generators.iterrows():
            selected_rows.append({'line': line_name, 'generator': str(generator_name), 'bus': str(generator['bus']), 'network_hops': float('nan'), 'p_nom_mw': float(generator['p_nom']), 'proxy_weight': float(generator['p_nom']), 'selection': 'all_wind_fallback'})
    metadata = pd.DataFrame(selected_rows)
    names = metadata['generator'].tolist()
    weights = metadata.set_index('generator')['proxy_weight'].astype(float)
    profiles = network.generators_t.p_max_pu[names].astype(float)
    weighted_profile = profiles.mul(weights, axis=1).sum(axis=1) / weights.sum()
    weighted_profile = weighted_profile.clip(lower=0.0, upper=1.0)
    weighted_profile.name = line_name
    return (weighted_profile, metadata)

def build_local_wind_proxy_matrix(network: Any, dlr_lines: Sequence[str], neighbour_hops: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles: dict[str, pd.Series] = {}
    metadata_frames: list[pd.DataFrame] = []
    for line in dlr_lines:
        profile, metadata = local_wind_availability_for_line(network, line, neighbour_hops)
        profiles[line] = profile
        metadata_frames.append(metadata)
    profile_frame = pd.DataFrame(profiles, index=network.snapshots)
    metadata_frame = pd.concat(metadata_frames, ignore_index=True)
    return (profile_frame, metadata_frame)

def stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return int(digest[:8], 16)

def sensor_availability_mask(snapshots: pd.Index, line_name: str, availability: float) -> pd.Series:
    if availability >= 1.0:
        return pd.Series(True, index=snapshots, name=line_name)
    rng = np.random.default_rng(stable_seed(f'Q3-DLR-{line_name}'))
    values = rng.random(len(snapshots)) < availability
    return pd.Series(values, index=snapshots, name=line_name)

def make_dlr_multipliers(wind_proxy: pd.DataFrame, config: DLRConfig) -> pd.DataFrame:
    config.validate()
    result = pd.DataFrame(1.0, index=wind_proxy.index, columns=wind_proxy.columns)
    for line in wind_proxy.columns:
        if config.mode == 'constant':
            uplift = pd.Series(config.max_uplift, index=wind_proxy.index)
        else:
            denominator = max(1e-09, 1.0 - config.wind_threshold)
            cooling_index = ((wind_proxy[line] - config.wind_threshold) / denominator).clip(lower=0.0, upper=1.0)
            cooling_index = cooling_index.pow(config.wind_exponent)
            uplift = config.max_uplift * cooling_index * (1.0 - config.forecast_derate)
        available = sensor_availability_mask(wind_proxy.index, line, config.sensor_availability)
        result[line] = 1.0 + uplift.where(available, 0.0)
    return result

def apply_dlr_to_network(network: Any, dlr_multipliers: pd.DataFrame) -> pd.DataFrame:
    static_s_max = network.lines.get('s_max_pu', pd.Series(1.0, index=network.lines.index)).astype(float)
    applied = pd.DataFrame(np.tile(static_s_max.to_numpy(), (len(network.snapshots), 1)), index=network.snapshots, columns=network.lines.index, dtype=float)
    existing = getattr(network.lines_t, 's_max_pu', pd.DataFrame())
    if isinstance(existing, pd.DataFrame) and (not existing.empty):
        existing = existing.reindex(index=network.snapshots)
        for line in existing.columns:
            if line in applied.columns:
                applied[line] = existing[line].fillna(static_s_max[line])
    for line in dlr_multipliers.columns:
        if line not in network.lines.index:
            raise KeyError(f'DLR line {line!r} is not in the network')
        applied[line] = static_s_max[line] * dlr_multipliers[line]
    network.lines_t.s_max_pu = applied
    return applied

def split_semicolon_text(value: Any) -> tuple[str, ...]:
    return tuple((part.strip() for part in str(value).split(';') if part.strip() and part.strip().lower() != 'nan'))

def load_q2_battery_recommendation(path: Path, throughput_cost: float, optional_max_efc_per_day: float | None) -> BatteryConfig | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    preferred = frame.copy()
    if 'recommendation_stage' in preferred.columns:
        exact = preferred[preferred['recommendation_stage'].astype(str) == 'lifetime_aware_eol_verified']
        if not exact.empty:
            preferred = exact
    if 'recommendation_type' in preferred.columns:
        exact = preferred[preferred['recommendation_type'].astype(str) == 'recommended_installed_capacity']
        if not exact.empty:
            preferred = exact
    row = preferred.iloc[-1]
    sites = split_semicolon_text(row.get('recommended_sites', row.get('sites', '')))
    if not sites:
        return None
    raw_allocations = [safe_float(part, 0.0) for part in str(row.get('site_allocations', row.get('allocations', ''))).split(';') if str(part).strip()]
    if len(raw_allocations) != len(sites) or any((value <= 0.0 for value in raw_allocations)):
        allocations = tuple([1.0 / len(sites)] * len(sites))
    else:
        allocations = normalise_allocations(raw_allocations)
    soc_min_pct = safe_float(row.get('soc_min_pct'), DEFAULT_SOC_MIN * 100.0)
    soc_max_pct = safe_float(row.get('soc_max_pct'), DEFAULT_SOC_MAX * 100.0)
    rte_pct = safe_float(row.get('round_trip_efficiency_pct'), DEFAULT_BATTERY_RTE * 100.0)
    soh_pct = safe_float(row.get('state_of_health_pct', row.get('eol_soh_assumption_pct')), DEFAULT_EOL_SOH * 100.0)
    success_mode = str(row.get('effective_success_mode', 'benchmark')).strip().lower()
    if success_mode not in {'benchmark', 'headroom', 'both'}:
        success_mode = 'benchmark'
    recovery_target = safe_float(row.get('recovery_target', DEFAULT_TARGET_RECOVERY), DEFAULT_TARGET_RECOVERY)
    config = BatteryConfig(sites=sites, allocations=allocations, power_mw=safe_float(row.get('power_mw'), 0.0), nameplate_energy_mwh=safe_float(row.get('nameplate_energy_mwh'), 0.0), soc_min=soc_min_pct / 100.0, soc_max=soc_max_pct / 100.0, round_trip_efficiency=rte_pct / 100.0, state_of_health=soh_pct / 100.0, marginal_cost_eur_per_mwh=throughput_cost, success_mode=success_mode, recovery_target=recovery_target, max_efc_per_day=optional_max_efc_per_day, source=str(path))
    config.validate()
    return config

def add_protected_battery(network: Any, config: BatteryConfig) -> list[str]:
    config.validate()
    if 'battery' not in network.carriers.index:
        network.add('Carrier', 'battery')
    names: list[str] = []
    for index, (site, share) in enumerate(zip(config.sites, config.allocations), start=1):
        if site not in network.buses.index:
            raise KeyError(f'Battery site {site!r} is not a network bus')
        power = config.power_mw * share
        nameplate_energy = config.nameplate_energy_mwh * share
        usable_energy = nameplate_energy * config.state_of_health * config.soc_window_fraction
        if power <= 0.0 or usable_energy <= 0.0:
            raise ValueError('Scaled battery leaves non-positive power or usable energy')
        name = f'Q3 protected battery {index}'
        network.add('StorageUnit', name, bus=site, p_nom=power, max_hours=usable_energy / power, efficiency_store=config.one_way_efficiency, efficiency_dispatch=config.one_way_efficiency, cyclic_state_of_charge=True, carrier='battery', marginal_cost=config.marginal_cost_eur_per_mwh)
        names.append(name)
    return names

def _component_dimension(variable: Any, excluded: set[str]) -> str:
    candidates = [dimension for dimension in variable.dims if dimension not in excluded]
    if len(candidates) != 1:
        raise RuntimeError(f'Cannot identify component dimension in variable dims {variable.dims}')
    return candidates[0]

def make_daily_battery_cycle_constraint(battery_names: Sequence[str], total_usable_energy_mwh: float, max_efc_per_day: float):

    def extra_functionality(network: Any, snapshots: pd.Index) -> None:
        model = network.model
        p_store = model.variables['StorageUnit-p_store']
        p_dispatch = model.variables['StorageUnit-p_dispatch']
        excluded = {'snapshot', 'scenario', 'period', 'investment_period'}
        component_dim = _component_dimension(p_store, excluded)
        coordinate_values = set((str(value) for value in p_store.coords[component_dim].values))
        present_names = [name for name in battery_names if name in coordinate_values]
        if not present_names:
            raise RuntimeError('Protected battery variables are missing from the model')
        charge = p_store.sel({component_dim: present_names})
        discharge = p_dispatch.sel({component_dim: present_names})
        snapshot_index = pd.Index(snapshots)
        datetime_index = pd.to_datetime(snapshot_index)
        dates = pd.Index(datetime_index.normalize())
        weights_series = snapshot_weights(network, 'stores').reindex(snapshot_index)
        weights = xr.DataArray(weights_series.to_numpy(dtype=float), coords={'snapshot': snapshot_index}, dims=['snapshot'])
        daily_limit = 2.0 * total_usable_energy_mwh * max_efc_per_day
        for day_number, day in enumerate(dates.unique()):
            day_snapshots = snapshot_index[dates == day]
            day_weights = weights.sel(snapshot=day_snapshots)
            day_throughput = ((charge.sel(snapshot=day_snapshots) + discharge.sel(snapshot=day_snapshots)) * day_weights).sum()
            model.add_constraints(day_throughput <= daily_limit, name=f'Q3-battery-daily-throughput-{day_number:02d}')
    return extra_functionality

def battery_metrics(network: Any, battery_names: Sequence[str], config: BatteryConfig) -> tuple[dict[str, float], pd.DataFrame]:
    if not battery_names:
        return ({'battery_power_mw': 0.0, 'battery_nameplate_energy_mwh': 0.0, 'battery_usable_energy_mwh': 0.0, 'battery_charge_mwh': 0.0, 'battery_discharge_mwh': 0.0, 'battery_throughput_mwh': 0.0, 'battery_efc_week': 0.0, 'battery_max_daily_efc': 0.0, 'battery_soc_min_pct': float('nan'), 'battery_soc_mean_pct': float('nan'), 'battery_soc_max_pct': float('nan'), 'simultaneous_charge_discharge_hours': 0.0}, pd.DataFrame(index=network.snapshots))
    p_store = network.storage_units_t.p_store[battery_names]
    p_dispatch = network.storage_units_t.p_dispatch[battery_names]
    soc = network.storage_units_t.state_of_charge[battery_names]
    weights = snapshot_weights(network, 'stores')
    charge_series = p_store.sum(axis=1)
    discharge_series = p_dispatch.sum(axis=1)
    throughput_series = charge_series + discharge_series
    charge_mwh = float((charge_series * weights).sum())
    discharge_mwh = float((discharge_series * weights).sum())
    throughput_mwh = charge_mwh + discharge_mwh
    usable_energy = config.usable_energy_mwh
    efc_week = throughput_mwh / (2.0 * usable_energy)
    dates = pd.to_datetime(network.snapshots).normalize()
    weighted_throughput = throughput_series * weights
    daily_throughput = weighted_throughput.groupby(dates).sum()
    daily_efc = daily_throughput / (2.0 * usable_energy)
    model_soc = soc.sum(axis=1)
    current_physical_capacity = config.nameplate_energy_mwh * config.state_of_health
    physical_floor = current_physical_capacity * config.soc_min
    physical_soc_pct = (model_soc + physical_floor) / current_physical_capacity * 100.0
    simultaneous = np.minimum(charge_series, discharge_series)
    hourly = pd.DataFrame({'battery_charge_mw': charge_series, 'battery_discharge_mw': discharge_series, 'battery_net_discharge_mw': discharge_series - charge_series, 'battery_model_soc_mwh': model_soc, 'battery_physical_soc_pct': physical_soc_pct, 'battery_hourly_throughput_mw': throughput_series}, index=network.snapshots)
    metrics = {'battery_power_mw': config.power_mw, 'battery_nameplate_energy_mwh': config.nameplate_energy_mwh, 'battery_usable_energy_mwh': usable_energy, 'battery_charge_mwh': charge_mwh, 'battery_discharge_mwh': discharge_mwh, 'battery_throughput_mwh': throughput_mwh, 'battery_efc_week': efc_week, 'battery_max_daily_efc': float(daily_efc.max()), 'battery_soc_min_pct': float(physical_soc_pct.min()), 'battery_soc_mean_pct': float(physical_soc_pct.mean()), 'battery_soc_max_pct': float(physical_soc_pct.max()), 'simultaneous_charge_discharge_hours': float(int((simultaneous > 0.0001).sum()))}
    return (metrics, hourly)

def run_scenario(context: StudyContext, label: str, dlr_multipliers: pd.DataFrame | None, dlr_lines: Sequence[str], dlr_mode: str, nominal_max_uplift: float, battery: BatteryConfig | None=None, return_network: bool=False) -> tuple[dict[str, Any], Any | None, pd.DataFrame]:
    print(f'[solve] {label}')
    network = gridkit.load(context.scenario, context.scope)
    if dlr_multipliers is None:
        applied_s_max_pu = pd.DataFrame(1.0, index=network.snapshots, columns=network.lines.index)
    else:
        applied_s_max_pu = apply_dlr_to_network(network, dlr_multipliers)
    battery_names: list[str] = []
    extra_functionality = None
    if battery is not None:
        battery_names = add_protected_battery(network, battery)
        if battery.max_efc_per_day is not None:
            extra_functionality = make_daily_battery_cycle_constraint(battery_names=battery_names, total_usable_energy_mwh=battery.usable_energy_mwh, max_efc_per_day=battery.max_efc_per_day)
    solve_checked(network, label, extra_functionality=extra_functionality)
    down = dispatch_down_by_carrier(network)
    unserved = total_unserved_mwh(network)
    if context.affected_generators:
        trial_affected_dispatch_mwh = float(generator_dispatch_energy_mwh(network, context.affected_generators).sum())
        affected_generator_gain_mwh = trial_affected_dispatch_mwh - context.baseline_affected_dispatch_mwh
    else:
        affected_generator_gain_mwh = 0.0
    if context.target_impact_total_mwh > ENERGY_TOLERANCE_MWH:
        total_recovery_fraction = (context.baseline_total_dispatch_down_mwh - down['total']) / context.target_impact_total_mwh
    else:
        total_recovery_fraction = float('nan')
    if context.affected_target_gain_mwh > ENERGY_TOLERANCE_MWH:
        affected_recovery_fraction = affected_generator_gain_mwh / context.affected_target_gain_mwh
    else:
        affected_recovery_fraction = total_recovery_fraction
    finite_recovery_fractions = [value for value in (total_recovery_fraction, affected_recovery_fraction) if math.isfinite(value)]
    target_recovery_fraction = min(finite_recovery_fractions) if finite_recovery_fractions else 0.0
    target_metrics = target_loading_metrics(network, context.target_lines, applied_s_max_pu)
    baseline_wind_constraint = max(0.0, context.baseline_wind_dispatch_down_mwh - context.all_relaxed_wind_dispatch_down_mwh)
    scenario_wind_constraint = max(0.0, down['wind'] - context.all_relaxed_wind_dispatch_down_mwh)
    wind_constraint_saved = baseline_wind_constraint - scenario_wind_constraint
    if baseline_wind_constraint > ENERGY_TOLERANCE_MWH:
        wind_constraint_reduction_pct = wind_constraint_saved / baseline_wind_constraint * 100.0
    else:
        wind_constraint_reduction_pct = float('nan')
    if dlr_multipliers is not None and len(dlr_lines):
        ratings = network.lines.loc[list(dlr_lines), 's_nom'].astype(float)
        uplift_mva = (dlr_multipliers[list(dlr_lines)] - 1.0).mul(ratings, axis=1)
        line_weights = snapshot_weights(network, 'generators')
        dlr_added_capacity_mva_hours = float(uplift_mva.mul(line_weights, axis=0).sum().sum())
        mean_dlr_multiplier = float(dlr_multipliers[list(dlr_lines)].mean().mean())
        max_dlr_multiplier = float(dlr_multipliers[list(dlr_lines)].max().max())
    else:
        dlr_added_capacity_mva_hours = 0.0
        mean_dlr_multiplier = 1.0
        max_dlr_multiplier = 1.0
    row: dict[str, Any] = {'scenario_label': label, 'scenario': context.scenario, 'scope': context.scope, 'target_label': context.target_label, 'target_lines': ';'.join(context.target_lines), 'dlr_mode': dlr_mode, 'nominal_max_uplift_pct': nominal_max_uplift * 100.0, 'dlr_line_count': len(dlr_lines) if dlr_multipliers is not None else 0, 'dlr_lines': ';'.join(dlr_lines) if dlr_multipliers is not None else '', 'mean_dlr_multiplier': mean_dlr_multiplier, 'max_dlr_multiplier': max_dlr_multiplier, 'dlr_added_capacity_mva_hours': dlr_added_capacity_mva_hours, 'total_dispatch_down_mwh': down['total'], 'wind_dispatch_down_mwh': down['wind'], 'solar_dispatch_down_mwh': down['solar'], 'wind_dispatch_down_saved_mwh': context.baseline_wind_dispatch_down_mwh - down['wind'], 'total_dispatch_down_saved_mwh': context.baseline_total_dispatch_down_mwh - down['total'], 'wind_constraint_component_mwh': scenario_wind_constraint, 'wind_constraint_saved_mwh': wind_constraint_saved, 'wind_constraint_reduction_pct': wind_constraint_reduction_pct, 'affected_generator_gain_mwh': affected_generator_gain_mwh, 'total_recovery_fraction': total_recovery_fraction, 'affected_recovery_fraction': affected_recovery_fraction, 'target_recovery_fraction': target_recovery_fraction, 'target_recovery_pct': target_recovery_fraction * 100.0, 'unserved_mwh': unserved, **target_metrics}
    battery_hourly = pd.DataFrame(index=network.snapshots)
    if battery is not None:
        battery_result, battery_hourly = battery_metrics(network, battery_names, battery)
        row.update(battery_result)
        row.update({'battery_sites': ';'.join(battery.sites), 'battery_allocations': ';'.join((f'{value:.6f}' for value in battery.allocations)), 'battery_soc_window': f'{battery.soc_min * 100:.1f}-{battery.soc_max * 100:.1f}%', 'battery_round_trip_efficiency_pct': battery.round_trip_efficiency * 100.0, 'battery_state_of_health_pct': battery.state_of_health * 100.0, 'battery_max_efc_per_day_limit': battery.max_efc_per_day if battery.max_efc_per_day is not None else float('nan'), 'battery_success_mode': battery.success_mode, 'battery_recovery_target': battery.recovery_target})
    else:
        row.update({'battery_power_mw': 0.0, 'battery_nameplate_energy_mwh': 0.0, 'battery_usable_energy_mwh': 0.0, 'battery_charge_mwh': 0.0, 'battery_discharge_mwh': 0.0, 'battery_throughput_mwh': 0.0, 'battery_efc_week': 0.0, 'battery_max_daily_efc': 0.0})
    unserved_tolerance = max(0.1, context.baseline_unserved_mwh * 0.001)
    unserved_ok = unserved <= context.baseline_unserved_mwh + unserved_tolerance
    recovery_target = battery.recovery_target if battery is not None else DEFAULT_TARGET_RECOVERY
    success_mode = battery.success_mode if battery is not None else 'benchmark'
    recovery_ok = bool(math.isfinite(target_recovery_fraction) and target_recovery_fraction + 1e-09 >= recovery_target)
    headroom_ok = int(round(target_metrics['target_binding_hours'])) == 0
    battery_cycle_ok = bool(battery is None or battery.max_efc_per_day is None or row['battery_max_daily_efc'] <= battery.max_efc_per_day + 1e-06)
    if success_mode == 'benchmark':
        criterion_ok = recovery_ok
    elif success_mode == 'headroom':
        criterion_ok = headroom_ok
    elif success_mode == 'both':
        criterion_ok = recovery_ok and headroom_ok
    else:
        raise ValueError(f'Unsupported success mode: {success_mode}')
    row['unserved_ok'] = bool(unserved_ok)
    row['target_recovery_ok'] = bool(recovery_ok)
    row['target_headroom_ok'] = bool(headroom_ok)
    row['battery_cycle_ok'] = bool(battery_cycle_ok)
    row['effective_success_mode'] = success_mode
    row['criterion_success'] = bool(criterion_ok and unserved_ok)
    row['combined_success'] = bool(criterion_ok and unserved_ok and battery_cycle_ok)
    hourly = pd.DataFrame(index=network.snapshots)
    for line in context.target_lines:
        hourly[f'flow_abs_mw__{line}'] = network.lines_t.p0[line].abs()
        hourly[f'static_rating_mva__{line}'] = network.lines.at[line, 's_nom']
        hourly[f's_max_pu__{line}'] = applied_s_max_pu[line]
        hourly[f'dynamic_rating_mva__{line}'] = network.lines.at[line, 's_nom'] * applied_s_max_pu[line]
        hourly[f'dynamic_loading_pct__{line}'] = network.lines_t.p0[line].abs() / hourly[f'dynamic_rating_mva__{line}'] * 100.0
    hourly['target_aggregate_abs_flow_mw'] = network.lines_t.p0[context.target_lines].abs().sum(axis=1)
    hourly['target_aggregate_dynamic_rating_mva'] = pd.DataFrame({line: network.lines.at[line, 's_nom'] * applied_s_max_pu[line] for line in context.target_lines}).sum(axis=1)
    hourly['target_max_dynamic_loading_pct'] = pd.DataFrame({line: network.lines_t.p0[line].abs() / (network.lines.at[line, 's_nom'] * applied_s_max_pu[line]) * 100.0 for line in context.target_lines}).max(axis=1)
    if not battery_hourly.empty:
        hourly = hourly.join(battery_hourly)
    return (row, network if return_network else None, hourly)

def choose_balanced_dlr_result(proxy_results: pd.DataFrame, benefit_fraction: float) -> tuple[pd.Series, str]:
    if proxy_results.empty:
        raise RuntimeError('No wind-proxy DLR results are available')
    working = proxy_results.sort_values('nominal_max_uplift_pct').copy()
    working = working[working['unserved_ok'].astype(bool)]
    if working.empty:
        raise RuntimeError('Every wind-proxy DLR scenario increases unserved energy')
    max_saved = float(working['wind_constraint_saved_mwh'].max())
    if max_saved > ENERGY_TOLERANCE_MWH:
        threshold = max_saved * benefit_fraction
        near_best = working[working['wind_constraint_saved_mwh'] >= threshold]
        if not near_best.empty:
            return (near_best.iloc[0], f'smallest uplift delivering at least {benefit_fraction:.0%} of the maximum measured wind-constraint benefit')
    return (working.iloc[0], 'no material wind-constraint benefit; smallest tested uplift retained')

def search_dlr_battery_scales(context: StudyContext, selected_dlr_multipliers: pd.DataFrame, dlr_lines: Sequence[str], nominal_uplift: float, base_battery: BatteryConfig, scales: Sequence[float]) -> tuple[pd.DataFrame, pd.Series | None, pd.DataFrame | None, pd.Series]:
    rows: list[dict[str, Any]] = []
    hourly_by_scale: dict[float, pd.DataFrame] = {}
    for scale in sorted(set((float(value) for value in scales))):
        battery = base_battery.scaled(scale)
        row, _, hourly = run_scenario(context=context, label=f'wind_proxy_DLR_plus_battery_scale_{scale:.4f}', dlr_multipliers=selected_dlr_multipliers, dlr_lines=dlr_lines, dlr_mode='wind_proxy_plus_protected_battery', nominal_max_uplift=nominal_uplift, battery=battery, return_network=False)
        row['battery_scale_vs_q2'] = scale
        rows.append(row)
        hourly_by_scale[scale] = hourly
    frame = pd.DataFrame(rows).sort_values('battery_scale_vs_q2')
    feasible = frame[frame['combined_success'].astype(bool)]
    if not feasible.empty:
        selected = feasible.iloc[0]
        selected_scale = float(selected['battery_scale_vs_q2'])
        return (frame, selected, hourly_by_scale[selected_scale], selected)
    best_effort = frame.sort_values(['target_recovery_pct', 'target_binding_hours', 'total_dispatch_down_mwh', 'battery_scale_vs_q2'], ascending=[False, True, True, True]).iloc[0]
    return (frame, None, None, best_effort)

def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=200, bbox_inches='tight')
    plt.close()

def plot_wind_proxy(proxy: pd.DataFrame, target_lines: Sequence[str]) -> None:
    columns = [line for line in target_lines if line in proxy.columns]
    if not columns:
        columns = list(proxy.columns[:min(5, len(proxy.columns))])
    plt.figure(figsize=(12, 5))
    for line in columns:
        plt.plot(proxy.index, proxy[line], label=line)
    plt.ylabel('Local wind availability proxy (p.u.)')
    plt.xlabel('Snapshot')
    plt.title('Local wind-availability proxy used for DLR')
    plt.grid(alpha=0.3)
    if len(columns) > 1:
        plt.legend(fontsize=8)
    save_figure('01_local_wind_proxy.png')

def plot_selected_dlr_multiplier(multipliers: pd.DataFrame, target_lines: Sequence[str]) -> None:
    columns = [line for line in target_lines if line in multipliers.columns]
    if not columns:
        columns = list(multipliers.columns[:min(5, len(multipliers.columns))])
    plt.figure(figsize=(12, 5))
    for line in columns:
        plt.plot(multipliers.index, multipliers[line], label=line)
    plt.ylabel('Dynamic rating multiplier (s_max_pu)')
    plt.xlabel('Snapshot')
    plt.title('Selected wind-proxy Dynamic Line Rating profile')
    plt.grid(alpha=0.3)
    if len(columns) > 1:
        plt.legend(fontsize=8)
    save_figure('02_selected_dlr_multiplier.png')

def plot_dlr_sensitivity(results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    for mode, group in results.groupby('dlr_mode'):
        ordered = group.sort_values('nominal_max_uplift_pct')
        plt.plot(ordered['nominal_max_uplift_pct'], ordered['wind_constraint_saved_mwh'], marker='o', label=mode)
    plt.xlabel('Nominal maximum rating uplift (%)')
    plt.ylabel('Wind constraint avoided over 168 hours (MWh)')
    plt.title('Dynamic Line Rating sensitivity')
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure('03_wind_constraint_avoided_vs_uplift.png')

def plot_binding_sensitivity(results: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 6))
    for mode, group in results.groupby('dlr_mode'):
        ordered = group.sort_values('nominal_max_uplift_pct')
        plt.plot(ordered['nominal_max_uplift_pct'], ordered['target_binding_hours'], marker='o', label=mode)
    plt.xlabel('Nominal maximum rating uplift (%)')
    plt.ylabel('Target binding hours')
    plt.title('Target-line binding hours under DLR')
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure('04_target_binding_hours_vs_uplift.png')

def plot_final_comparison(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    plot_frame = frame.copy()
    plt.figure(figsize=(11, 6))
    bars = plt.bar(plot_frame['display_name'], plot_frame['wind_dispatch_down_mwh'])
    for bar, value in zip(bars, plot_frame['wind_dispatch_down_mwh']):
        plt.annotate(f'{value:,.0f}', (bar.get_x() + bar.get_width() / 2.0, bar.get_height()), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)
    plt.ylabel('Wind dispatch-down over 168 hours (MWh)')
    plt.xlabel('Scenario')
    plt.title('DLR and protected-battery comparison')
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', alpha=0.3)
    save_figure('05_final_wind_dispatch_down_comparison.png')

def plot_final_target_operation(hourly: pd.DataFrame) -> None:
    if hourly.empty:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(hourly.index, hourly['target_aggregate_abs_flow_mw'], label='absolute target flow')
    plt.plot(hourly.index, hourly['target_aggregate_dynamic_rating_mva'], linestyle='--', label='dynamic rating')
    plt.ylabel('MW / MVA')
    plt.xlabel('Snapshot')
    plt.title('Final coordinated scenario: target flow and dynamic rating')
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure('06_final_target_flow_and_dynamic_rating.png')

def plot_final_battery_soc(hourly: pd.DataFrame) -> None:
    if 'battery_physical_soc_pct' not in hourly.columns:
        return
    plt.figure(figsize=(12, 5))
    plt.plot(hourly.index, hourly['battery_physical_soc_pct'])
    plt.ylabel('Physical battery SoC (%)')
    plt.xlabel('Snapshot')
    plt.title('Protected battery state of charge')
    plt.grid(alpha=0.3)
    save_figure('07_final_battery_soc.png')

def plot_final_battery_power(hourly: pd.DataFrame) -> None:
    required = {'battery_charge_mw', 'battery_discharge_mw'}
    if not required.issubset(hourly.columns):
        return
    plt.figure(figsize=(12, 5))
    plt.plot(hourly.index, hourly['battery_charge_mw'], label='charge')
    plt.plot(hourly.index, -hourly['battery_discharge_mw'], label='discharge (negative for display)')
    plt.axhline(0.0, linewidth=0.8)
    plt.ylabel('Battery power (MW)')
    plt.xlabel('Snapshot')
    plt.title('Protected battery operation with DLR')
    plt.grid(alpha=0.3)
    plt.legend()
    save_figure('08_final_battery_power.png')

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Analyse Dynamic Line Rating for TPSA Hackathon 2026 Problem 3.1 Q3, with optional health-aware battery coordination.')
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO)
    parser.add_argument('--scope', default=DEFAULT_SCOPE)
    parser.add_argument('--target-line', default=None)
    parser.add_argument('--target-mode', choices=('line', 'corridor'), default='line')
    parser.add_argument('--dlr-scope', choices=('target', 'binding-overhead', 'all-overhead'), default='target', help='Which inferred overhead lines receive DLR.')
    parser.add_argument('--max-uplifts', default=','.join((str(value) for value in DEFAULT_MAX_UPLIFTS)), help='Comma-separated maximum rating uplifts, e.g. 0.1,0.2,0.3,0.4.')
    parser.add_argument('--wind-threshold', type=float, default=DEFAULT_WIND_THRESHOLD)
    parser.add_argument('--wind-exponent', type=float, default=DEFAULT_WIND_EXPONENT)
    parser.add_argument('--forecast-derate', type=float, default=DEFAULT_FORECAST_DERATE)
    parser.add_argument('--sensor-availability', type=float, default=DEFAULT_SENSOR_AVAILABILITY)
    parser.add_argument('--neighbour-hops', type=int, default=DEFAULT_NEIGHBOUR_HOPS)
    parser.add_argument('--cable-x-per-km-threshold', type=float, default=DEFAULT_CABLE_X_PER_KM_THRESHOLD)
    parser.add_argument('--allow-non-overhead-dlr', action='store_true', help='Allow DLR on cable-like/unknown lines only as a labelled sensitivity.')
    parser.add_argument('--skip-constant-benchmark', action='store_true', help='Skip the ideal constant-uplift upper-bound solves.')
    parser.add_argument('--dlr-benefit-fraction', type=float, default=DEFAULT_DLR_BENEFIT_FRACTION)
    parser.add_argument('--battery-mode', choices=('auto', 'off', 'required'), default='auto')
    parser.add_argument('--q2-summary', type=Path, default=DEFAULT_Q2_SUMMARY)
    parser.add_argument('--battery-scales', default=','.join((str(value) for value in DEFAULT_BATTERY_SCALES)))
    parser.add_argument('--max-efc-per-day', type=float, default=None, help='Optional daily-EFC sensitivity. Disabled by default so Q3 uses the same battery operating constraints as Q2 final.')
    parser.add_argument('--battery-throughput-cost', type=float, default=DEFAULT_BATTERY_THROUGHPUT_COST)
    parser.add_argument('--smoke-test', action='store_true', help='Run baseline and one wind-proxy DLR case only.')
    return parser.parse_args()

def main() -> int:
    args = parse_arguments()
    ensure_output_directories()
    max_uplifts = parse_float_list(args.max_uplifts)
    battery_scales = parse_float_list(args.battery_scales)
    if args.neighbour_hops < 0:
        raise ValueError('--neighbour-hops cannot be negative')
    if not 0.0 < args.dlr_benefit_fraction <= 1.0:
        raise ValueError('--dlr-benefit-fraction must be in (0, 1]')
    context = build_study_context(scenario=args.scenario, scope=args.scope, requested_line=args.target_line, target_mode=args.target_mode)
    asset_screen = line_asset_screen(context.baseline_network, threshold=args.cable_x_per_km_threshold)
    asset_screen.to_csv(TABLE_DIR / '01_line_asset_screen.csv', index=False)
    dlr_lines = choose_dlr_lines(context=context, asset_screen=asset_screen, dlr_scope=args.dlr_scope, allow_non_overhead=args.allow_non_overhead_dlr)
    print(f'[DLR] Scope: {args.dlr_scope}')
    print(f"[DLR] Lines receiving DLR: {', '.join(dlr_lines)}")
    wind_proxy, wind_metadata = build_local_wind_proxy_matrix(context.baseline_network, dlr_lines, neighbour_hops=args.neighbour_hops)
    wind_proxy.to_csv(TABLE_DIR / '02_local_wind_proxy_hourly.csv', index_label='snapshot')
    wind_metadata.to_csv(TABLE_DIR / '03_local_wind_proxy_generators.csv', index=False)
    plot_wind_proxy(wind_proxy, context.target_lines)
    baseline_summary = pd.DataFrame([{'scenario': context.scenario, 'scope': context.scope, 'selected_line': context.selected_line, 'target_mode': context.target_mode, 'target_label': context.target_label, 'target_lines': ';'.join(context.target_lines), 'dlr_scope': args.dlr_scope, 'dlr_lines': ';'.join(dlr_lines), 'baseline_wind_dispatch_down_mwh': context.baseline_wind_dispatch_down_mwh, 'target_relaxed_wind_dispatch_down_mwh': context.target_relaxed_wind_dispatch_down_mwh, 'all_relaxed_wind_dispatch_down_mwh': context.all_relaxed_wind_dispatch_down_mwh, 'target_attributable_wind_mwh': context.target_impact_wind_mwh, 'baseline_wind_constraint_component_mwh': max(0.0, context.baseline_wind_dispatch_down_mwh - context.all_relaxed_wind_dispatch_down_mwh), 'baseline_target_binding_hours': context.baseline_target_binding_hours, 'baseline_target_max_loading_pct': context.baseline_target_max_loading_pct, 'wind_threshold': args.wind_threshold, 'wind_exponent': args.wind_exponent, 'forecast_derate': args.forecast_derate, 'sensor_availability': args.sensor_availability, 'neighbour_hops': args.neighbour_hops, 'cable_x_per_km_threshold': args.cable_x_per_km_threshold, 'model_limitation': 'DLR is a sensitivity proxy based on synthetic local wind availability; no conductor temperature, ambient weather, sag/tension or asset thermal model is available'}])
    baseline_summary.to_csv(TABLE_DIR / '04_study_and_baseline_summary.csv', index=False)
    if args.smoke_test:
        uplift = max_uplifts[0]
        config = DLRConfig(mode='wind_proxy', max_uplift=uplift, wind_threshold=args.wind_threshold, wind_exponent=args.wind_exponent, forecast_derate=args.forecast_derate, sensor_availability=args.sensor_availability)
        multipliers = make_dlr_multipliers(wind_proxy, config)
        row, _, hourly = run_scenario(context=context, label=f'smoke_wind_proxy_{uplift:.2f}', dlr_multipliers=multipliers, dlr_lines=dlr_lines, dlr_mode='wind_proxy', nominal_max_uplift=uplift, return_network=False)
        pd.DataFrame([row]).to_csv(TABLE_DIR / 'smoke_test_result.csv', index=False)
        hourly.to_csv(TABLE_DIR / 'smoke_test_hourly.csv', index_label='snapshot')
        print('\n[smoke-test] DLR time series and optimisation completed successfully.')
        return 0
    sensitivity_rows: list[dict[str, Any]] = []
    proxy_multipliers_by_uplift: dict[float, pd.DataFrame] = {}
    for uplift in max_uplifts:
        proxy_config = DLRConfig(mode='wind_proxy', max_uplift=uplift, wind_threshold=args.wind_threshold, wind_exponent=args.wind_exponent, forecast_derate=args.forecast_derate, sensor_availability=args.sensor_availability)
        proxy_multipliers = make_dlr_multipliers(wind_proxy, proxy_config)
        proxy_multipliers_by_uplift[uplift] = proxy_multipliers
        row, _, _ = run_scenario(context=context, label=f'wind_proxy_DLR_{uplift:.2f}', dlr_multipliers=proxy_multipliers, dlr_lines=dlr_lines, dlr_mode='wind_proxy', nominal_max_uplift=uplift)
        sensitivity_rows.append(row)
        if not args.skip_constant_benchmark:
            constant_config = DLRConfig(mode='constant', max_uplift=uplift, wind_threshold=args.wind_threshold, wind_exponent=args.wind_exponent, forecast_derate=0.0, sensor_availability=args.sensor_availability)
            constant_multipliers = make_dlr_multipliers(wind_proxy, constant_config)
            row, _, _ = run_scenario(context=context, label=f'constant_uplift_benchmark_{uplift:.2f}', dlr_multipliers=constant_multipliers, dlr_lines=dlr_lines, dlr_mode='constant_uplift_benchmark', nominal_max_uplift=uplift)
            sensitivity_rows.append(row)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLE_DIR / '05_dlr_sensitivity_results.csv', index=False)
    plot_dlr_sensitivity(sensitivity)
    plot_binding_sensitivity(sensitivity)
    proxy_results = sensitivity[sensitivity['dlr_mode'] == 'wind_proxy'].copy()
    selected_dlr_row, dlr_selection_reason = choose_balanced_dlr_result(proxy_results, benefit_fraction=args.dlr_benefit_fraction)
    selected_uplift = float(selected_dlr_row['nominal_max_uplift_pct']) / 100.0
    selected_multipliers = proxy_multipliers_by_uplift[selected_uplift]
    selected_multipliers.to_csv(TABLE_DIR / '06_selected_dlr_multipliers.csv', index_label='snapshot')
    plot_selected_dlr_multiplier(selected_multipliers, context.target_lines)
    selected_dlr_final_row, _, selected_dlr_hourly = run_scenario(context=context, label='selected_wind_proxy_DLR', dlr_multipliers=selected_multipliers, dlr_lines=dlr_lines, dlr_mode='wind_proxy_selected', nominal_max_uplift=selected_uplift, return_network=False)
    selected_dlr_hourly.to_csv(TABLE_DIR / '07_selected_dlr_hourly_operation.csv', index_label='snapshot')
    print(f'\n[selection] Selected nominal DLR uplift = {selected_uplift * 100:.1f}%')
    print(f"[selection] Actual maximum multiplier after derating = {float(selected_dlr_row['max_dlr_multiplier']):.4f}")
    print(f"[selection] Wind constraint avoided = {float(selected_dlr_row['wind_constraint_saved_mwh']):,.2f} MWh")
    battery_config: BatteryConfig | None = None
    if args.battery_mode != 'off':
        battery_config = load_q2_battery_recommendation(path=args.q2_summary, throughput_cost=args.battery_throughput_cost, optional_max_efc_per_day=args.max_efc_per_day)
        if battery_config is None and args.battery_mode == 'required':
            raise FileNotFoundError(f"Battery mode is 'required', but no usable Question 2 recommendation was found at {args.q2_summary}.")
        if battery_config is None:
            print('[battery] No Question 2 recommendation found; continuing with DLR-only analysis.')
    final_rows: list[dict[str, Any]] = []
    final_rows.append({'display_name': 'Baseline', 'scenario_label': 'baseline', 'wind_dispatch_down_mwh': context.baseline_wind_dispatch_down_mwh, 'total_dispatch_down_mwh': context.baseline_total_dispatch_down_mwh, 'target_binding_hours': context.baseline_target_binding_hours, 'battery_power_mw': 0.0, 'battery_nameplate_energy_mwh': 0.0, 'battery_efc_week': 0.0})
    selected_dlr_display = dict(selected_dlr_final_row)
    selected_dlr_display['display_name'] = 'Selected DLR'
    final_rows.append(selected_dlr_display)
    final_hourly = selected_dlr_hourly
    selected_combined_row: pd.Series | None = None
    if battery_config is not None:
        battery_config.validate()
        pd.DataFrame([{'source': battery_config.source, 'sites': ';'.join(battery_config.sites), 'allocations': ';'.join((f'{value:.6f}' for value in battery_config.allocations)), 'power_mw': battery_config.power_mw, 'nameplate_energy_mwh': battery_config.nameplate_energy_mwh, 'usable_energy_mwh': battery_config.usable_energy_mwh, 'soc_min_pct': battery_config.soc_min * 100.0, 'soc_max_pct': battery_config.soc_max * 100.0, 'round_trip_efficiency_pct': battery_config.round_trip_efficiency * 100.0, 'state_of_health_pct': battery_config.state_of_health * 100.0, 'success_mode_inherited_from_q2': battery_config.success_mode, 'recovery_target_inherited_from_q2': battery_config.recovery_target, 'optional_max_efc_per_day': battery_config.max_efc_per_day}]).to_csv(TABLE_DIR / '08_q2_battery_input.csv', index=False)
        battery_only_row, _, battery_only_hourly = run_scenario(context=context, label='q2_protected_battery_only', dlr_multipliers=None, dlr_lines=[], dlr_mode='none', nominal_max_uplift=0.0, battery=battery_config)
        battery_only_row['battery_scale_vs_q2'] = 1.0
        battery_only_display = dict(battery_only_row)
        battery_only_display['display_name'] = 'Protected battery'
        final_rows.append(battery_only_display)
        full_combined_row, _, full_combined_hourly = run_scenario(context=context, label='selected_DLR_plus_full_q2_battery', dlr_multipliers=selected_multipliers, dlr_lines=dlr_lines, dlr_mode='wind_proxy_plus_protected_battery', nominal_max_uplift=selected_uplift, battery=battery_config)
        full_combined_row['battery_scale_vs_q2'] = 1.0
        scale_results, selected_combined_row, selected_combined_hourly, best_effort_row = search_dlr_battery_scales(context=context, selected_dlr_multipliers=selected_multipliers, dlr_lines=dlr_lines, nominal_uplift=selected_uplift, base_battery=battery_config, scales=battery_scales)
        scale_results.to_csv(TABLE_DIR / '09_dlr_battery_scale_search.csv', index=False)
        if selected_combined_row is not None and selected_combined_hourly is not None:
            selected_combined_display = selected_combined_row.to_dict()
            selected_combined_display['display_name'] = 'DLR + feasible downsized battery'
            selected_combined_display['recommendation_status'] = 'FEASIBLE_DOWNSIZED_BESS'
            final_rows.append(selected_combined_display)
            final_hourly = selected_combined_hourly
        else:
            best_effort_display = best_effort_row.to_dict()
            best_effort_display['display_name'] = 'DLR + best tested infeasible battery'
            best_effort_display['recommendation_status'] = 'NO_FEASIBLE_DOWNSIZED_BESS_BEST_EFFORT_ONLY'
            final_rows.append(best_effort_display)
        wear_rows = [{'scenario': 'battery_only', 'battery_scale_vs_q2': 1.0, 'throughput_mwh': battery_only_row['battery_throughput_mwh'], 'efc_week': battery_only_row['battery_efc_week'], 'max_daily_efc': battery_only_row['battery_max_daily_efc'], 'soc_min_pct': battery_only_row['battery_soc_min_pct'], 'soc_mean_pct': battery_only_row['battery_soc_mean_pct'], 'soc_max_pct': battery_only_row['battery_soc_max_pct'], 'wind_dispatch_down_mwh': battery_only_row['wind_dispatch_down_mwh']}, {'scenario': 'selected_DLR_plus_same_full_battery', 'battery_scale_vs_q2': 1.0, 'throughput_mwh': full_combined_row['battery_throughput_mwh'], 'efc_week': full_combined_row['battery_efc_week'], 'max_daily_efc': full_combined_row['battery_max_daily_efc'], 'soc_min_pct': full_combined_row['battery_soc_min_pct'], 'soc_mean_pct': full_combined_row['battery_soc_mean_pct'], 'soc_max_pct': full_combined_row['battery_soc_max_pct'], 'wind_dispatch_down_mwh': full_combined_row['wind_dispatch_down_mwh']}]
        if selected_combined_row is not None:
            wear_rows.append({'scenario': 'selected_DLR_plus_minimum_feasible_scaled_battery', 'battery_scale_vs_q2': selected_combined_row['battery_scale_vs_q2'], 'throughput_mwh': selected_combined_row['battery_throughput_mwh'], 'efc_week': selected_combined_row['battery_efc_week'], 'max_daily_efc': selected_combined_row['battery_max_daily_efc'], 'soc_min_pct': selected_combined_row['battery_soc_min_pct'], 'soc_mean_pct': selected_combined_row['battery_soc_mean_pct'], 'soc_max_pct': selected_combined_row['battery_soc_max_pct'], 'wind_dispatch_down_mwh': selected_combined_row['wind_dispatch_down_mwh']})
        pd.DataFrame(wear_rows).to_csv(TABLE_DIR / '10_battery_health_and_wear_comparison.csv', index=False)
    final_comparison = pd.DataFrame(final_rows)
    final_comparison.to_csv(TABLE_DIR / '11_final_scenario_comparison.csv', index=False)
    plot_final_comparison(final_comparison)
    final_hourly.to_csv(TABLE_DIR / '12_final_hourly_operation.csv', index_label='snapshot')
    plot_final_target_operation(final_hourly)
    plot_final_battery_soc(final_hourly)
    plot_final_battery_power(final_hourly)
    print('\n' + '=' * 82)
    print('QUESTION 3 RESULT WITHIN THE SUPPLIED 168-HOUR SYNTHETIC SCENARIO')
    print('=' * 82)
    print(f'Target: {context.target_label}')
    print(f'DLR scope: {args.dlr_scope}')
    print(f"DLR lines: {', '.join(dlr_lines)}")
    print(f'Selected nominal maximum uplift: {selected_uplift * 100:.1f}%')
    print(f'Forecast/safety derating: {args.forecast_derate * 100:.1f}% of the proxy-derived uplift')
    print(f"Wind dispatch-down: {context.baseline_wind_dispatch_down_mwh:,.1f} -> {float(selected_dlr_final_row['wind_dispatch_down_mwh']):,.1f} MWh")
    print(f"Wind constraint avoided by selected DLR: {float(selected_dlr_final_row['wind_constraint_saved_mwh']):,.1f} MWh")
    print(f"Target binding hours: {context.baseline_target_binding_hours} -> {int(round(float(selected_dlr_final_row['target_binding_hours'])))}")
    print(f'DLR selection reason: {dlr_selection_reason}')
    if battery_config is not None:
        print('\nProtected battery coordination:')
        print(f"  Q2 sites: {', '.join(battery_config.sites)}")
        print(f'  Q2 installed battery: {battery_config.power_mw:,.1f} MW / {battery_config.nameplate_energy_mwh:,.1f} MWh')
        print(f'  Inherited success mode: {battery_config.success_mode}')
        print(f'  SoC / RTE / SoH: {battery_config.soc_min * 100:.0f}-{battery_config.soc_max * 100:.0f}% / {battery_config.round_trip_efficiency * 100:.0f}% / {battery_config.state_of_health * 100:.0f}%')
        if selected_combined_row is not None:
            selected_scale = float(selected_combined_row['battery_scale_vs_q2'])
            print(f"  Minimum tested feasible BESS with DLR: {selected_scale:.3f} x Q2 = {float(selected_combined_row['battery_power_mw']):,.1f} MW / {float(selected_combined_row['battery_nameplate_energy_mwh']):,.1f} MWh")
            print(f"  EFC/week: {float(selected_combined_row['battery_efc_week']):.3f}")
        else:
            print('  NO FEASIBLE DOWNSIZED BESS FOUND in the tested scale range.')
            print(f"  Best tested infeasible scale: {float(best_effort_row['battery_scale_vs_q2']):.3f} x Q2; recovery={float(best_effort_row['target_recovery_pct']):.2f}%, binding={int(round(float(best_effort_row['target_binding_hours'])))} h.")
    print(f'\nTables:  {TABLE_DIR}')
    print(f'Figures: {FIGURE_DIR}')
    print('\nInterpretation note: the DLR profile is a wind-availability proxy, not a conductor thermal model. Replace it with measured/forecast weather, conductor temperature, sag/tension and asset data before making an operational recommendation.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
