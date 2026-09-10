"""End-to-end national constraint-group workflow for Problem 3.2."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import platform
import subprocess
from typing import Any
import warnings

import numpy as np
import pandas as pd

from problem_3_1.q5_shift_factors.core import (
    branches,
    bus_components,
    component_adjust_reference,
    mean_load_by_bus,
    ptdf,
)
from . import __version__
from .core import (
    constraint_memberships,
    directional_modes,
    directional_relief,
    lpf_sample_validation,
    merge_similar_modes,
    nonlocal_similar_pairs,
    passive_flows,
    passive_ratings,
    reference_sensitivity,
    response_zones,
    snapshot_hours,
)
from .reporting import write_outputs


DEFAULT_CONFIG: dict[str, Any] = {
    "scenario": "WP2033",
    "scope": "all-island",
    "primary_reference": "load",
    "comparison_reference": "uniform",
    "renewable_carriers": ["wind", "solar", "hydro", "biomass"],
    "near_congestion_threshold_pu": 0.90,
    "binding_threshold_pu": 0.999,
    "constraint_group_threshold": 0.05,
    "mode_merge_jaccard_threshold": 0.80,
    "mode_merge_cosine_threshold": 0.98,
    "maximum_zones_per_component": 8,
    "nonlocal_minimum_distance_km": 150.0,
    "nonlocal_minimum_cosine_similarity": 0.95,
    "nonlocal_pair_limit": 100,
    "lpf_validation_bus_count": 6,
    "lpf_delta_mw": 1.0,
    "lpf_tolerance": 1e-6,
    "q5_result_csv": "results/problem_3_1/question_5_final/03_target_line_ranking.csv",
    "q5_target_line": "5041-17010-2",
    "q3_dlr_csv": "results/problem_3_1/question_3_final/tables/06_selected_dlr_multipliers.csv",
    "q6_result_dir": "results/problem_3_1/question_6_final_20260910_v2",
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        override = json.load(handle)
    if not isinstance(override, dict):
        raise ValueError("Problem 3.2 configuration must be a JSON object")
    return _deep_update(DEFAULT_CONFIG, override)


def validate_config(config: dict[str, Any]) -> None:
    if str(config["scope"]) != "all-island":
        raise ValueError("Problem 3.2 national run must use the official all-island scope")
    near = float(config["near_congestion_threshold_pu"])
    binding = float(config["binding_threshold_pu"])
    if not 0 < near < binding <= 1:
        raise ValueError("Require 0 < near threshold < binding threshold <= 1")
    for key in ("constraint_group_threshold", "mode_merge_jaccard_threshold", "mode_merge_cosine_threshold"):
        if not 0 <= float(config[key]) <= 1:
            raise ValueError(f"{key} must be between zero and one")
    carriers = config["renewable_carriers"]
    if not isinstance(carriers, list) or not carriers or not all(str(value).strip() for value in carriers):
        raise ValueError("renewable_carriers must be a non-empty list")
    if int(config["maximum_zones_per_component"]) < 1:
        raise ValueError("maximum_zones_per_component must be positive")
    if int(config["lpf_validation_bus_count"]) < 1:
        raise ValueError("lpf_validation_bus_count must be positive")
    if float(config["lpf_delta_mw"]) <= 0 or float(config["lpf_tolerance"]) <= 0:
        raise ValueError("LPF delta and tolerance must be positive")


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gridkit(kit_dir: Path):
    module_path = kit_dir / "gridkit.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Official participant kit missing: {module_path}")
    spec = importlib.util.spec_from_file_location("_q32_gridkit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _package_versions() -> dict[str, str]:
    import pypsa
    import scipy
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pypsa": pypsa.__version__,
    }


def _validation_buses(
    network: Any,
    modes: pd.DataFrame,
    relief: pd.DataFrame,
    components: pd.Series,
    count: int,
) -> list[str]:
    demand = mean_load_by_bus(network).sort_values(ascending=False)
    component_demand = demand.groupby(components).sum()
    candidates: list[str] = []
    for mode in modes.head(count * 2)["mode_id"].astype(str):
        for bus in (str(relief.loc[mode].idxmax()), str(relief.loc[mode].idxmin())):
            component = int(components.loc[bus])
            if float(component_demand.get(component, 0.0)) > 0 and bus not in candidates:
                candidates.append(bus)
            if len(candidates) >= count:
                return candidates
    for bus in demand.index.astype(str):
        component = int(components.loc[bus])
        if float(component_demand.get(component, 0.0)) > 0 and bus not in candidates:
            candidates.append(bus)
        if len(candidates) >= count:
            break
    return candidates


def _north_west_consistency(
    root: Path,
    gridkit: Any,
    config: dict[str, Any],
    national_adjusted: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _path(root, config["q5_result_csv"])
    if not path.exists():
        return pd.DataFrame(), {"path": str(path), "exists": False}
    reported = pd.read_csv(path)
    target = str(config["q5_target_line"])
    reported = reported.loc[reported["monitored_line"].astype(str).eq(target)].copy()
    north_west = gridkit.load(config["scenario"], "north-west")
    nw_frame = branches(north_west)
    nw_adjusted = component_adjust_reference(north_west, ptdf(north_west, nw_frame), "load", nw_frame)
    rows = []
    for row in reported.itertuples(index=False):
        bus = str(row.nearest_substation)
        recalculated = float(nw_adjusted.at[target, bus])
        national = (
            float(national_adjusted.at[target, bus])
            if target in national_adjusted.index and bus in national_adjusted.columns
            else np.nan
        )
        rows.append(
            {
                "wind_farm": str(row.wind_farm),
                "nearest_substation": bus,
                "target_line": target,
                "q5_reported_north_west_ptdf": float(row.primary_shift_factor),
                "q32_recalculated_north_west_ptdf": recalculated,
                "north_west_absolute_error": abs(recalculated - float(row.primary_shift_factor)),
                "q32_all_island_ptdf": national,
                "all_island_minus_north_west_ptdf": national - recalculated if np.isfinite(national) else np.nan,
                "interpretation": "North-West equality is a code-consistency check; the all-island value may differ because model scope and topology differ",
            }
        )
    return pd.DataFrame(rows), {
        "path": str(path),
        "exists": True,
        "sha256": _sha256(path),
        "rows": int(len(reported)),
    }


def _dlr_comparison(
    root: Path,
    solved: Any,
    branch_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _path(root, config["q3_dlr_csv"])
    audit = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return pd.DataFrame(), audit
    raw = pd.read_csv(path)
    if "snapshot" not in raw.columns:
        raise ValueError(f"Q3 DLR file has no snapshot column: {path}")
    raw["snapshot"] = pd.to_datetime(raw["snapshot"])
    raw = raw.set_index("snapshot")
    audit.update({"sha256": _sha256(path), "columns": list(raw.columns), "rows": len(raw)})
    flows = passive_flows(solved, branch_frame)
    static_ratings = passive_ratings(solved, branch_frame)
    hours = snapshot_hours(solved)
    near = float(config["near_congestion_threshold_pu"])
    binding = float(config["binding_threshold_pu"])
    rows = []
    for branch in raw.columns.astype(str):
        if branch not in branch_frame.index:
            continue
        multipliers = pd.to_numeric(raw[branch], errors="coerce").reindex(solved.snapshots).fillna(1.0).clip(lower=1.0)
        cases = {
            "static": static_ratings[branch],
            "q3_dlr_rating_only": static_ratings[branch] * multipliers,
        }
        for sign in (1, -1):
            directional = flows[branch] * sign > 1e-12
            for case, rating in cases.items():
                loading = flows[branch].abs() / rating
                event = directional & loading.ge(near)
                severity = ((loading - near) / (1.0 - near)).clip(0.0, 1.0)
                rows.append(
                    {
                        "branch": branch,
                        "flow_sign": sign,
                        "rating_case": case,
                        "near_congestion_hours": float(hours.loc[event].sum()),
                        "binding_hours": float(hours.loc[event & loading.ge(binding)].sum()),
                        "event_weighted_severity_hours": float((severity * hours).where(event, 0.0).sum()),
                        "maximum_loading_pu": float(loading.loc[directional].max()) if directional.any() else 0.0,
                        "mean_rating_multiplier": 1.0 if case == "static" else float(multipliers.mean()),
                        "maximum_rating_multiplier": 1.0 if case == "static" else float(multipliers.max()),
                        "dispatch_treatment": "same solved flow trace; rating-only event reclassification",
                        "ptdf_change": 0.0,
                    }
                )
    return pd.DataFrame(rows), audit


def _q6_crosswalk(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = _path(root, config["q6_result_dir"])
    scenario_path = directory / "02_scenario_summary.csv"
    impacts_path = directory / "06_generator_status_impacts.csv"
    audit = {"directory": str(directory), "exists": scenario_path.exists() and impacts_path.exists()}
    if not audit["exists"]:
        return pd.DataFrame(), audit
    scenarios = pd.read_csv(scenario_path)
    impacts = pd.read_csv(impacts_path)
    neutral = scenarios.set_index("scenario").loc["network_neutral_static"]
    priority = scenarios.set_index("scenario").loc["network_priority_static"]
    rows = [
        {
            "metric": "priority_minus_neutral_total_dispatch_down_mwh",
            "value": float(priority["wind_dispatch_down_mwh"] - neutral["wind_dispatch_down_mwh"]),
            "source_file": str(scenario_path),
            "source_locator": "scenario keys network_priority_static minus network_neutral_static; column wind_dispatch_down_mwh",
            "data_class": "MODEL_DERIVED",
            "economic_treatment": "technical metric; not BESS developer revenue",
        },
        {
            "metric": "priority_farms_protected_energy_mwh",
            "value": float(-impacts.loc[impacts["additional_dispatch_down_mwh"] < 0, "additional_dispatch_down_mwh"].sum()),
            "source_file": str(impacts_path),
            "source_locator": "sum of negative additional_dispatch_down_mwh",
            "data_class": "MODEL_DERIVED under SCENARIO_CHOICE status labels",
            "economic_treatment": "wind-owner allocation transfer; not BESS developer revenue",
        },
        {
            "metric": "non_priority_farms_additional_burden_mwh",
            "value": float(impacts.loc[impacts["additional_dispatch_down_mwh"] > 0, "additional_dispatch_down_mwh"].sum()),
            "source_file": str(impacts_path),
            "source_locator": "sum of positive additional_dispatch_down_mwh",
            "data_class": "MODEL_DERIVED under SCENARIO_CHOICE status labels",
            "economic_treatment": "wind-owner allocation transfer; not BESS developer revenue",
        },
    ]
    audit.update({
        "scenario_sha256": _sha256(scenario_path),
        "impacts_sha256": _sha256(impacts_path),
    })
    return pd.DataFrame(rows), audit


def _renewable_crosswalk(
    network: Any,
    config: dict[str, Any],
    memberships: pd.DataFrame,
    zones: pd.DataFrame,
) -> pd.DataFrame:
    """Map every configured RES generator to its official bus, groups, and zone."""
    allowed = {str(value).lower() for value in config["renewable_carriers"]}
    selected = network.generators.loc[
        network.generators["carrier"].astype(str).str.lower().isin(allowed)
    ].copy()
    zone_by_bus = zones.set_index("bus")["response_zone_id"]
    grouped = {
        str(bus): frame.copy()
        for bus, frame in memberships.groupby("bus")
    }
    records = []
    for resource, row in selected.iterrows():
        bus = str(row["bus"])
        member_rows = grouped.get(bus)
        if member_rows is None or member_rows.empty:
            records.append(
                {
                    "renewable_resource": str(resource),
                    "carrier": str(row["carrier"]),
                    "installed_capacity_mw": float(row["p_nom"]),
                    "official_connection_bus": bus,
                    "response_zone_id": str(zone_by_bus.loc[bus]),
                    "constraint_group_count": 0,
                    "mode_id": None,
                    "branch": None,
                    "flow_sign": None,
                    "normalized_positive_relief": None,
                    "relief_mw_per_mw_curtailment": None,
                    "mapping_method": "official_model_generator_bus_assignment",
                }
            )
            continue
        group_count = int(member_rows["mode_id"].nunique())
        for member in member_rows.itertuples(index=False):
            records.append(
                {
                    "renewable_resource": str(resource),
                    "carrier": str(row["carrier"]),
                    "installed_capacity_mw": float(row["p_nom"]),
                    "official_connection_bus": bus,
                    "response_zone_id": str(zone_by_bus.loc[bus]),
                    "constraint_group_count": group_count,
                    "mode_id": str(member.mode_id),
                    "branch": str(member.branch),
                    "flow_sign": int(member.flow_sign),
                    "normalized_positive_relief": float(member.normalized_positive_relief),
                    "relief_mw_per_mw_curtailment": float(member.relief_mw_per_mw_curtailment_or_charge),
                    "mapping_method": "official_model_generator_bus_assignment",
                }
            )
    return pd.DataFrame(records)


def _electrical_similarity_matrix(
    features: pd.DataFrame, components: pd.Series
) -> pd.DataFrame:
    """Return cosine similarity within components; undefined entries remain blank."""
    values = features.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1)
    denominator = np.outer(norms, norms)
    similarity = np.divide(
        values @ values.T,
        denominator,
        out=np.full((len(values), len(values)), np.nan),
        where=denominator > 1e-12,
    )
    component_values = components.reindex(features.index).to_numpy(dtype=int)
    similarity[component_values[:, None] != component_values[None, :]] = np.nan
    return pd.DataFrame(similarity, index=features.index, columns=features.index)


def _scope_gaps(dlr_audit: dict[str, Any], q6_audit: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("active_power_dc_approximation", True, "PTDF and LPF cover active-power redistribution only"),
        ("voltage_and_reactive_power", False, "requires AC power-flow and voltage-limit study"),
        ("transient_and_frequency_stability", False, "requires dynamic models and contingencies"),
        ("protection_and_fault_level", False, "requires protection and short-circuit study"),
        ("n_minus_one_outages", False, "base topology only; contingency screening is not implemented"),
        ("market_bidding_and_prices", False, "electrical grouping does not estimate market revenue"),
        ("dlr_national_coverage", False, f"Q3 evidence covers only {len(dlr_audit.get('columns', []))} supplied branch column(s)"),
        ("priority_status_observation", False, "Q6 statuses are explicitly labelled SCENARIO_CHOICE"),
        ("q6_link_available", bool(q6_audit.get("exists")), "policy-allocation outputs are a crosswalk, not a clustering feature"),
        ("bus_geography", True, "official model bus coordinates used; no independent substation geocoding"),
    ]
    return pd.DataFrame(rows, columns=["topic", "implemented", "scope_or_required_extension"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPSA Problem 3.2 nationwide constraint groups")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def main(root: Path) -> int:
    args = parse_args()
    config_path = args.config or root / "configs" / "problem_3_2" / "constraint_groups.json"
    config = load_config(config_path if config_path.exists() else None)
    if args.scenario:
        config["scenario"] = args.scenario
    validate_config(config)
    for name in ("pypsa", "linopy", "highspy"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)

    kit_dir = root / "data" / "participant-kit"
    gridkit = _load_gridkit(kit_dir)
    network_path = kit_dir / "networks" / f"{config['scenario']}_{config['scope']}.nc"
    if not network_path.exists():
        raise FileNotFoundError(network_path)
    network = gridkit.load(config["scenario"], config["scope"])
    branch_frame = branches(network)
    components = bus_components(network, branch_frame)
    base = ptdf(network, branch_frame)
    load_adjusted = component_adjust_reference(network, base, config["primary_reference"], branch_frame)
    uniform_adjusted = component_adjust_reference(network, base, config["comparison_reference"], branch_frame)

    print("=== PROBLEM 3.2 INPUT CHECK ===", flush=True)
    print(f"Official network: {config['scenario']} {config['scope']}")
    print(f"Buses: {len(network.buses)} | lines: {len(network.lines)} | transformers: {len(network.transformers)}")
    print(f"Passive branches: {len(branch_frame)} | passive AC components: {components.nunique()}")
    print(f"Snapshots available: {len(network.snapshots)} | network SHA256: {_sha256(network_path)}")
    if not (args.run or args.smoke):
        print("CHECK PASSED: official national source, topology, references, and PTDF are valid.")
        return 0

    solved = network.copy()
    if args.smoke:
        solved.set_snapshots(solved.snapshots[: min(24, len(solved.snapshots))])
    gridkit.quiet()
    status, condition = gridkit.solve(solved)
    if str(status).lower() != "ok" or str(condition).lower() != "optimal":
        raise RuntimeError(f"Official dispatch failed: {status}, {condition}")
    gridkit.freeze_dispatch(solved)
    solved.lpf(solved.snapshots)
    modes, event_weights = directional_modes(
        solved,
        branch_frame,
        near_threshold_pu=float(config["near_congestion_threshold_pu"]),
        binding_threshold_pu=float(config["binding_threshold_pu"]),
    )
    if modes.empty:
        raise RuntimeError("No directional near-congestion modes were found at the configured threshold")
    relief_load = directional_relief(load_adjusted, modes)
    relief_uniform = directional_relief(uniform_adjusted, modes)
    memberships = constraint_memberships(
        modes, relief_load, float(config["constraint_group_threshold"]), "load_weighted_component_local"
    )
    merged_modes, mode_pairs = merge_similar_modes(
        modes,
        relief_load,
        memberships,
        jaccard_threshold=float(config["mode_merge_jaccard_threshold"]),
        cosine_threshold=float(config["mode_merge_cosine_threshold"]),
    )
    zones, features, zone_summary = response_zones(
        components,
        modes,
        relief_load,
        event_weights,
        maximum_zones_per_component=int(config["maximum_zones_per_component"]),
    )
    nonlocal_pairs = nonlocal_similar_pairs(
        network.buses,
        components,
        zones,
        features,
        minimum_distance_km=float(config["nonlocal_minimum_distance_km"]),
        minimum_cosine_similarity=float(config["nonlocal_minimum_cosine_similarity"]),
        limit=int(config["nonlocal_pair_limit"]),
    )
    reference_audit = reference_sensitivity(
        modes,
        relief_load,
        relief_uniform,
        components,
        float(config["constraint_group_threshold"]),
        "load_weighted_component_local",
        "uniform_component_local",
    )
    validation_buses = _validation_buses(
        network, modes, relief_load, components, int(config["lpf_validation_bus_count"])
    )
    lpf_validation = lpf_sample_validation(
        network,
        branch_frame,
        load_adjusted,
        modes,
        validation_buses,
        reference=config["primary_reference"],
        delta_mw=float(config["lpf_delta_mw"]),
    )
    lpf_validation["tolerance"] = float(config["lpf_tolerance"])
    lpf_validation["passed"] = lpf_validation["absolute_error"] <= float(config["lpf_tolerance"])
    if lpf_validation.empty or not bool(lpf_validation["passed"].all()):
        raise RuntimeError("National LPF finite-difference validation failed")

    north_west, q5_audit = _north_west_consistency(root, gridkit, config, load_adjusted)
    if len(north_west) and float(north_west["north_west_absolute_error"].max()) > float(config["lpf_tolerance"]):
        raise RuntimeError("Q5 North-West consistency validation failed")
    dlr_comparison, dlr_audit = _dlr_comparison(root, solved, branch_frame, config)
    q6_crosswalk, q6_audit = _q6_crosswalk(root, config)
    scope_gaps = _scope_gaps(dlr_audit, q6_audit)
    renewable_crosswalk = _renewable_crosswalk(network, config, memberships, zones)
    similarity_matrix = _electrical_similarity_matrix(features, components)

    code_digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code_digest.update(path.name.encode())
        code_digest.update(path.read_bytes())
    overlap = memberships.groupby("bus")["mode_id"].nunique()
    input_audit = {
        "official_network": {
            "path": str(network_path),
            "sha256": _sha256(network_path),
            "data_class": "OFFICIAL_INPUT",
            "loader": f"gridkit.load({config['scenario']!r}, {config['scope']!r})",
        },
        "q5_consistency_source": q5_audit,
        "q3_dlr_source": dlr_audit,
        "q6_crosswalk_source": q6_audit,
        "balancing_references": {
            "primary": "load weighted and normalized independently inside each passive AC component",
            "comparison": "uniform and normalized independently inside each passive AC component",
        },
        "data_class_rules": {
            "network_topology_and_profiles": "OFFICIAL_INPUT",
            "dispatch_ptdf_groups_and_zones": "MODEL_DERIVED",
            "thresholds_and_status_assignments": "SCENARIO_CHOICE",
            "cost_or_price_values": "not used by Problem 3.2 grouping",
        },
    }
    manifest = {
        "q32_model_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "config": config,
        "network": {
            "scenario": config["scenario"],
            "scope": config["scope"],
            "source": str(network_path),
            "source_sha256": _sha256(network_path),
            "buses": int(len(network.buses)),
            "lines": int(len(network.lines)),
            "transformers": int(len(network.transformers)),
            "links": int(len(network.links)),
            "passive_branches": int(len(branch_frame)),
            "passive_ac_components": int(components.nunique()),
            "snapshots": int(len(solved.snapshots)),
        },
        "results": {
            "directional_mode_count": int(len(modes)),
            "overlapping_membership_count": int(len(memberships)),
            "buses_in_any_constraint_group": int(memberships["bus"].nunique()),
            "maximum_groups_per_bus": int(overlap.max()),
            "merged_constraint_group_count": int(merged_modes["merged_constraint_group_id"].nunique()),
            "response_zone_count": int(zones["response_zone_id"].nunique()),
            "nonlocal_similar_pair_count": int(len(nonlocal_pairs)),
            "renewable_resource_count": int(renewable_crosswalk["renewable_resource"].nunique()),
            "renewable_resources_in_any_constraint_group": int(
                renewable_crosswalk.loc[renewable_crosswalk["mode_id"].notna(), "renewable_resource"].nunique()
            ),
        },
        "numerical_checks": {
            "dispatch_status": str(status),
            "dispatch_condition": str(condition),
            "all_lpf_checks_passed": bool(lpf_validation["passed"].all()),
            "maximum_lpf_absolute_error": float(lpf_validation["absolute_error"].max()),
            "maximum_north_west_q5_recalculation_error": float(north_west["north_west_absolute_error"].max()) if len(north_west) else None,
            "maximum_centered_reference_response_change": float(reference_audit["maximum_absolute_centered_response_change"].max()),
            "all_buses_have_exactly_one_response_zone": bool(len(zones) == len(network.buses) and zones["bus"].nunique() == len(network.buses)),
        },
        "validation_buses": validation_buses,
        "packages": _package_versions(),
        "code_sha256": code_digest.hexdigest(),
        "git": {
            "branch": _git(root, "branch", "--show-current"),
            "commit": _git(root, "rev-parse", "HEAD"),
            "dirty": bool(_git(root, "status", "--porcelain")),
        },
    }
    output = args.out or root / "results" / "problem_3_2" / (
        "constraint_groups_smoke" if args.smoke else "constraint_groups_final"
    )
    write_outputs(
        output,
        manifest=manifest,
        input_audit=input_audit,
        modes=modes,
        memberships=memberships,
        merged_modes=merged_modes,
        mode_pairs=mode_pairs,
        zones=zones,
        zone_summary=zone_summary,
        nonlocal_pairs=nonlocal_pairs,
        reference_audit=reference_audit,
        lpf_validation=lpf_validation,
        north_west_consistency=north_west,
        dlr_comparison=dlr_comparison,
        q6_crosswalk=q6_crosswalk,
        event_weights=event_weights,
        relief_load=relief_load,
        relief_uniform=relief_uniform,
        scope_gaps=scope_gaps,
        renewable_crosswalk=renewable_crosswalk,
        similarity_matrix=similarity_matrix,
        buses=network.buses,
        make_plots=not args.no_plots,
    )
    print("=== PROBLEM 3.2 RESULT ===")
    print(f"Dispatch: {status}, {condition} | hours: {len(solved.snapshots)}")
    print(f"Directional modes: {len(modes)} | merged groups: {merged_modes['merged_constraint_group_id'].nunique()}")
    print(f"Response zones: {zones['response_zone_id'].nunique()} | memberships: {len(memberships)}")
    print(f"Maximum LPF error: {lpf_validation['absolute_error'].max():.3e}")
    print(f"Results: {output}")
    return 0
