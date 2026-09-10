"""Command-line workflow for Q5 wind-farm shift-factor analysis."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd

from . import __version__
from .core import (
    component_adjust_reference,
    connected_component_count,
    factor_tables,
    largest_load_bus,
    maximum_reference_residual,
    operational_context,
    rating_invariance,
    reference_weights,
    validate_with_lpf,
    wind_factor_matrix,
    wind_farms,
)
from .reporting import write_outputs


DEFAULT_CONFIG: dict[str, Any] = {
    "scenario": "WP2033",
    "scope": "north-west",
    "monitored_lines": ["5041-17010-2"],
    "primary_reference": "load",
    "wind_carriers": ["wind"],
    "constraint_group_threshold": 0.05,
    "binding_tolerance_pu": 1e-6,
    "finite_difference_delta_mw": 1.0,
    "finite_difference_tolerance": 1e-7,
    "dlr_rating_multiplier_check": 1.25,
    "q1_results_csv": "results/problem_3_1/question_1_final/tables/02_line_constraint_and_rating_results.csv",
    "q3_dlr_csv": "results/problem_3_1/question_3_final/tables/06_selected_dlr_multipliers.csv",
    "q4_portfolio_config": "configs/problem_3_1/question_4_distributed_bess_strict_tie_break.json",
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
    """Load a partial JSON config over validated defaults."""
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        override = json.load(handle)
    if not isinstance(override, dict):
        raise ValueError("Q5 configuration must be a JSON object")
    return _deep_update(DEFAULT_CONFIG, override)


def validate_config(config: dict[str, Any]) -> None:
    """Reject ambiguous or physically invalid Q5 settings before calculation."""
    if not str(config["scenario"]).strip() or not str(config["scope"]).strip():
        raise ValueError("Scenario and scope must be non-empty")
    monitored = config["monitored_lines"]
    if not isinstance(monitored, list) or not monitored or not all(
        isinstance(value, str) and value.strip() for value in monitored
    ):
        raise ValueError("monitored_lines must be a non-empty list of exact branch IDs")
    if len(set(monitored)) != len(monitored):
        raise ValueError("monitored_lines contains duplicate branch IDs")
    carriers = config["wind_carriers"]
    if not isinstance(carriers, list) or not carriers:
        raise ValueError("wind_carriers must be a non-empty list")
    if float(config["constraint_group_threshold"]) < 0:
        raise ValueError("constraint_group_threshold must be non-negative")
    if not 0 <= float(config["binding_tolerance_pu"]) < 0.1:
        raise ValueError("binding_tolerance_pu must be between zero and 0.1")
    if float(config["finite_difference_delta_mw"]) <= 0:
        raise ValueError("finite_difference_delta_mw must be positive")
    if float(config["finite_difference_tolerance"]) <= 0:
        raise ValueError("finite_difference_tolerance must be positive")
    if float(config["dlr_rating_multiplier_check"]) <= 0:
        raise ValueError("dlr_rating_multiplier_check must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gridkit(kit_dir: Path):
    module_path = kit_dir / "gridkit.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing {module_path}; the official participant kit is required"
        )
    spec = importlib.util.spec_from_file_location("_q5_gridkit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import the official grid kit from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_network(
    root: Path,
    config: dict[str, Any],
    network_path: Path | None,
    kit_dir: Path | None,
):
    import pypsa

    kit = (kit_dir or root / "data" / "participant-kit").resolve()
    gridkit = _load_gridkit(kit)
    if network_path is not None:
        source = network_path.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        network = pypsa.Network(source)
        origin = str(source)
    else:
        network = gridkit.load(config["scenario"], config["scope"])
        source = kit / "networks" / f"{config['scenario']}_{config['scope']}.nc"
        origin = f"gridkit.load({config['scenario']!r}, {config['scope']!r})"
    return network, gridkit, source, origin


def _path_from_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_q1_evidence(
    root: Path, config: dict[str, Any], monitored_lines: list[str]
) -> tuple[pd.DataFrame, Path]:
    path = _path_from_root(root, config["q1_results_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Q1 target-line evidence is missing: {path}. Run or restore Q1 results first."
        )
    frame = pd.read_csv(path)
    if "line" not in frame.columns:
        raise ValueError(f"Q1 evidence lacks a line column: {path}")
    selected = frame.loc[frame["line"].astype(str).isin(monitored_lines)].copy()
    missing = sorted(set(monitored_lines) - set(selected["line"].astype(str)))
    if missing:
        raise ValueError(f"Monitored lines are absent from the Q1 evidence: {missing}")
    return selected, path


def _read_dlr_evidence(path: Path, monitored_lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() else None,
    }
    if not path.exists():
        return result
    frame = pd.read_csv(path)
    result["rows"] = int(len(frame))
    statistics: dict[str, Any] = {}
    for line in monitored_lines:
        values: pd.Series | None = None
        if line in frame.columns:
            values = pd.to_numeric(frame[line], errors="coerce")
        elif {"line", "multiplier"}.issubset(frame.columns):
            values = pd.to_numeric(
                frame.loc[frame["line"].astype(str).eq(line), "multiplier"],
                errors="coerce",
            )
        if values is not None:
            values = values.dropna()
            if len(values):
                statistics[line] = {
                    "count": int(len(values)),
                    "mean": float(values.mean()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
    result["monitored_line_statistics"] = statistics
    return result


def _reference_label(reference: str, single_slack_bus: str) -> str:
    if reference == "load":
        return "load_weighted"
    if reference == "uniform":
        return "uniform"
    if reference == single_slack_bus:
        return "single_slack"
    return "primary_custom_bus"


def _build_factor_frame(
    branch_frame: pd.DataFrame,
    farms: pd.DataFrame,
    matrices: dict[str, pd.DataFrame],
    monitored_lines: list[str],
    primary_label: str,
    single_slack_bus: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for line in monitored_lines:
        branch = branch_frame.loc[line]
        for farm, farm_row in farms.iterrows():
            record: dict[str, Any] = {
                "wind_farm": farm,
                "nearest_substation": farm_row["nearest_substation"],
                "mapping_method": farm_row["mapping_method"],
                "installed_capacity_mw": float(farm_row["installed_capacity_mw"]),
                "monitored_line": line,
                "component": branch["kind"],
                "line_bus0": branch["bus0"],
                "line_bus1": branch["bus1"],
                "positive_flow_orientation": f"{branch['bus0']} to {branch['bus1']}",
                "single_slack_bus": single_slack_bus,
            }
            for label, matrix in matrices.items():
                record[f"shift_factor_{label}"] = float(matrix.at[farm, line])
            record["primary_reference"] = primary_label
            record["primary_shift_factor"] = record[f"shift_factor_{primary_label}"]
            values = [
                float(record[column])
                for column in record
                if column.startswith("shift_factor_") and np.isfinite(record[column])
            ]
            record["reference_range_min"] = float(min(values))
            record["reference_range_max"] = float(max(values))
            record["reference_range_width"] = float(max(values) - min(values))
            rows.append(record)
    return pd.DataFrame(rows)


def _rank_directional_effects(
    factors: pd.DataFrame,
    context: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    merged = factors.merge(
        context[
            [
                "monitored_line",
                "dominant_binding_flow_sign",
                "dominant_binding_flow_direction",
                "binding_hours",
            ]
        ],
        on="monitored_line",
        how="left",
        validate="many_to_one",
    )
    merged["absolute_primary_shift_factor"] = merged["primary_shift_factor"].abs()
    merged["rank_abs_shift_factor"] = (
        merged.groupby("monitored_line")["absolute_primary_shift_factor"]
        .transform(lambda values: values.round(12).rank(method="dense", ascending=False))
        .astype(int)
    )
    merged["flow_change_mw_per_mw_injection"] = merged["primary_shift_factor"]
    merged["flow_change_mw_per_mw_curtailment"] = -merged["primary_shift_factor"]
    merged["relief_mw_per_mw_curtailment"] = (
        merged["dominant_binding_flow_sign"] * merged["primary_shift_factor"]
    )
    merged["relief_mw_if_fully_curtailed"] = (
        merged["relief_mw_per_mw_curtailment"].clip(lower=0.0)
        * merged["installed_capacity_mw"]
    )
    merged["absolute_constraint_group_member"] = (
        merged["absolute_primary_shift_factor"] >= float(threshold)
    )
    merged["directional_constraint_group_member"] = (
        merged["relief_mw_per_mw_curtailment"] >= float(threshold)
    )
    merged["directional_effect_of_curtailment"] = np.select(
        [
            merged["relief_mw_per_mw_curtailment"] > 1e-10,
            merged["relief_mw_per_mw_curtailment"] < -1e-10,
        ],
        ["relieves observed constraint", "worsens observed constraint"],
        default="negligible",
    )
    return merged.sort_values(
        ["monitored_line", "rank_abs_shift_factor", "wind_farm"]
    ).reset_index(drop=True)


def _q4_crosswalk(
    root: Path,
    config: dict[str, Any],
    monitored_lines: list[str],
    primary_matrix: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = _path_from_root(root, config["q4_portfolio_config"])
    if not path.exists():
        raise FileNotFoundError(f"Q4 portfolio config is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        q4_config = json.load(handle)
    portfolios = q4_config.get("designs", {}).get("bess_portfolios", [])
    site_order: list[str] = []
    for portfolio in portfolios:
        for site in portfolio.get("sites", []):
            site = str(site)
            if site not in site_order:
                site_order.append(site)
    rows: list[dict[str, Any]] = []
    context_by_line = context.set_index("monitored_line")
    for line in monitored_lines:
        sign = int(context_by_line.at[line, "dominant_binding_flow_sign"])
        for site in site_order:
            if site not in primary_matrix.columns:
                raise ValueError(f"Q4 site is not an exact Q5 network bus ID: {site}")
            factor = float(primary_matrix.at[line, site])
            rows.append(
                {
                    "record_type": "site",
                    "label": site,
                    "sites": site,
                    "allocations": "1.000000",
                    "site_count": 1,
                    "monitored_line": line,
                    "primary_shift_factor": factor,
                    "flow_change_mw_per_mw_bess_discharge": factor,
                    "flow_change_mw_per_mw_bess_charge": -factor,
                    "relief_mw_per_mw_bess_discharge": -sign * factor,
                    "relief_mw_per_mw_bess_charge": sign * factor,
                }
            )
        for portfolio in portfolios:
            sites = [str(site) for site in portfolio.get("sites", [])]
            allocations = [float(value) for value in portfolio.get("allocations", [])]
            if len(sites) != len(allocations) or not np.isclose(sum(allocations), 1.0):
                raise ValueError(f"Invalid Q4 portfolio allocation: {portfolio}")
            factor = float(
                sum(primary_matrix.at[line, site] * share for site, share in zip(sites, allocations))
            )
            rows.append(
                {
                    "record_type": "portfolio",
                    "label": str(portfolio.get("label", ";".join(sites))),
                    "sites": ";".join(sites),
                    "allocations": ";".join(f"{value:.6f}" for value in allocations),
                    "site_count": len(sites),
                    "monitored_line": line,
                    "primary_shift_factor": factor,
                    "flow_change_mw_per_mw_bess_discharge": factor,
                    "flow_change_mw_per_mw_bess_charge": -factor,
                    "relief_mw_per_mw_bess_discharge": -sign * factor,
                    "relief_mw_per_mw_bess_charge": sign * factor,
                }
            )
    audit = {
        "path": str(path),
        "sha256": _sha256(path),
        "site_count": len(site_order),
        "portfolio_count": len(portfolios),
        "sites": site_order,
        "battery_sizes_mw_mwh": q4_config.get("designs", {}).get("bess_mw_mwh", []),
    }
    return pd.DataFrame(rows), audit


def _package_versions() -> dict[str, str]:
    import matplotlib
    import pypsa

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "pypsa": pypsa.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TPSA Problem 3.1 Q5 wind-farm shift-factor analysis"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--scope")
    parser.add_argument("--network", type=Path)
    parser.add_argument("--kit-dir", type=Path)
    parser.add_argument(
        "--target-lines",
        help="Comma-separated exact line or transformer IDs; defaults to Q1/Q3 target",
    )
    parser.add_argument(
        "--reference",
        help="Primary balancing reference: load, uniform, or an exact bus ID",
    )
    parser.add_argument("--single-slack-bus")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Validate inputs only; default")
    mode.add_argument("--smoke", action="store_true", help="Run 24 hours and three farms")
    mode.add_argument("--run", action="store_true", help="Run the full Q5 analysis")
    return parser.parse_args()


def main(root: Path) -> int:
    args = parse_args()
    config_path = args.config or root / "configs" / "problem_3_1" / "question_5.json"
    config = load_config(config_path if config_path.exists() else None)
    if args.config is not None and not args.config.exists():
        raise FileNotFoundError(args.config)
    if args.scenario:
        config["scenario"] = args.scenario
    if args.scope:
        config["scope"] = args.scope
    if args.reference:
        config["primary_reference"] = args.reference
    if args.target_lines:
        config["monitored_lines"] = [
            value.strip() for value in args.target_lines.split(",") if value.strip()
        ]
    validate_config(config)
    monitored_lines = list(dict.fromkeys(map(str, config["monitored_lines"])))
    network, gridkit, network_source, network_origin = _load_network(
        root, config, args.network, args.kit_dir
    )
    frame_components = connected_component_count(network)
    farms = wind_farms(network, config["wind_carriers"])
    if args.smoke:
        farms = farms.iloc[: min(3, len(farms))].copy()
    single_slack_bus = args.single_slack_bus or largest_load_bus(network)
    primary_reference = str(config["primary_reference"])
    primary_label = _reference_label(primary_reference, single_slack_bus)
    references: dict[str, str] = {
        "load_weighted": "load",
        "uniform": "uniform",
        "single_slack": single_slack_bus,
    }
    if primary_label not in references:
        references[primary_label] = primary_reference
    branch_frame, farm_matrices, base_ptdf = factor_tables(
        network, monitored_lines, farms, references
    )
    if primary_label not in farm_matrices:
        raise RuntimeError(f"Primary reference label was not calculated: {primary_label}")
    q1_evidence, q1_path = _load_q1_evidence(root, config, monitored_lines)

    print("=== Q5 INPUT CHECK ===", flush=True)
    print(f"Network: {config['scenario']} {config['scope']} | {network_origin}")
    print(
        f"Buses: {len(network.buses)} | passive branches: {len(branch_frame)} | "
        f"connected components: {frame_components}"
    )
    print(
        f"Snapshots: {len(network.snapshots)} | wind farms: {len(farms)} | "
        f"monitored lines: {', '.join(monitored_lines)}"
    )
    print(
        f"Primary reference: {primary_label} ({primary_reference}) | "
        f"single-slack comparison bus: {single_slack_bus}"
    )
    for _, row in q1_evidence.iterrows():
        print(
            f"Q1 evidence {row['line']}: binding_hours={int(row['binding_hours'])}, "
            f"saved_dispatch_down_plus_25_mwh={float(row['saved_dispatch_down_plus_25_mwh']):.3f}"
        )
    if not (args.run or args.smoke):
        print("CHECK PASSED: inputs, exact IDs, topology, PTDF, and Q1 linkage are valid.")
        return 0

    solved = network.copy()
    if args.smoke:
        solved.set_snapshots(solved.snapshots[: min(24, len(solved.snapshots))])
    gridkit.quiet()
    status, condition = gridkit.solve(solved)
    if str(status).lower() != "ok" or str(condition).lower() != "optimal":
        raise RuntimeError(f"Official network dispatch failed: {status}, {condition}")
    gridkit.freeze_dispatch(solved)
    solved.lpf(solved.snapshots)
    context = operational_context(
        solved, monitored_lines, float(config["binding_tolerance_pu"])
    )
    factors = _build_factor_frame(
        branch_frame,
        farms,
        farm_matrices,
        monitored_lines,
        primary_label,
        single_slack_bus,
    )
    ranking = _rank_directional_effects(
        factors,
        context,
        float(config["constraint_group_threshold"]),
    )
    primary_weights = reference_weights(network, primary_reference)
    validation = validate_with_lpf(
        network,
        monitored_lines,
        farms,
        farm_matrices[primary_label],
        primary_reference,
        float(config["finite_difference_delta_mw"]),
        float(config["finite_difference_tolerance"]),
    )
    if validation.empty or not bool(validation["passed"].all()):
        failed = validation.loc[~validation["passed"]]
        raise RuntimeError(
            f"LPF finite-difference validation failed for {len(failed)} rows"
        )
    invariance = rating_invariance(
        network,
        monitored_lines,
        farms,
        primary_reference,
        float(config["dlr_rating_multiplier_check"]),
    )
    if not bool(invariance["ptdf_unchanged"].all()):
        raise RuntimeError("Rating-only DLR unexpectedly changed the PTDF")
    primary_adjusted = component_adjust_reference(
        network, base_ptdf, primary_reference, branch_frame
    )
    all_branch = farms[["nearest_substation"]].join(
        wind_factor_matrix(primary_adjusted, farms)
    )
    q4_crosswalk, q4_audit = _q4_crosswalk(
        root, config, monitored_lines, primary_adjusted, context
    )
    dlr_path = _path_from_root(root, config["q3_dlr_csv"])
    dlr_audit = _read_dlr_evidence(dlr_path, monitored_lines)
    q1_records = q1_evidence.replace({np.nan: None}).to_dict("records")
    input_audit = {
        "network_source": str(network_source),
        "network_origin": network_origin,
        "network_sha256": _sha256(network_source) if network_source.exists() else None,
        "wind_farm_mapping_method": "official_model_generator_bus_assignment",
        "q1_evidence": {
            "path": str(q1_path),
            "sha256": _sha256(q1_path),
            "rows": q1_records,
        },
        "q3_dlr_evidence": dlr_audit,
        "q4_portfolios": q4_audit,
        "assumptions": [
            "The model generator bus is treated as the nearest substation because no independent farm coordinates are supplied.",
            "The primary shift factor balances each injection over mean load.",
            "Positive flow follows each branch's bus0-to-bus1 orientation.",
            "Directional relief uses the observed binding-flow sign from the solved official network.",
            "DLR is tested as a rating-only change; topology and reactance remain fixed.",
        ],
    }
    weighted_zero = maximum_reference_residual(
        network, primary_adjusted, primary_reference, branch_frame
    )
    rank_correlations: dict[str, dict[str, float]] = {}
    for line in monitored_lines:
        comparison = pd.DataFrame(
            {
                label: matrix[line].abs().round(12)
                for label, matrix in farm_matrices.items()
            }
        )
        rank_correlations[line] = (
            comparison.rank(method="average", ascending=False)
            .corr(method="spearman")
            .round(12)
            .to_dict()
        )
    manifest = {
        "q5_model_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "config": config,
        "network": {
            "scenario": config["scenario"],
            "scope": config["scope"],
            "origin": network_origin,
            "source": str(network_source),
            "source_sha256": _sha256(network_source) if network_source.exists() else None,
            "buses": int(len(network.buses)),
            "lines": int(len(network.lines)),
            "transformers": int(len(network.transformers)),
            "passive_branches": int(len(branch_frame)),
            "connected_components": int(frame_components),
            "snapshots": int(len(solved.snapshots)),
            "wind_farms": int(len(farms)),
        },
        "references": {
            "primary_label": primary_label,
            "primary_reference": primary_reference,
            "single_slack_bus": single_slack_bus,
            "available": references,
            "absolute_rank_spearman_correlations": rank_correlations,
        },
        "numerical_checks": {
            "maximum_uniform_ptdf_row_sum": float(base_ptdf.sum(axis=1).abs().max()),
            "maximum_primary_reference_weighted_mean": float(weighted_zero),
            "maximum_lpf_finite_difference_error": float(validation["absolute_error"].max()),
            "all_lpf_checks_passed": bool(validation["passed"].all()),
            "maximum_rating_only_dlr_ptdf_change": float(
                invariance["maximum_absolute_shift_factor_change"].max()
            ),
        },
        "packages": _package_versions(),
        "input_audit": input_audit,
    }
    output = args.out or root / "results" / "problem_3_1" / (
        "question_5_smoke" if args.smoke else "question_5_final"
    )
    write_outputs(
        output,
        manifest,
        farms,
        factors,
        ranking,
        all_branch,
        validation,
        invariance,
        q4_crosswalk,
        context,
        smoke=args.smoke,
        make_plots=not args.no_plots,
    )
    print("=== Q5 RESULT ===")
    print(f"Dispatch: {status}, {condition}")
    print(
        f"LPF validation: {len(validation)}/{len(validation)} passed | "
        f"max error {validation['absolute_error'].max():.3e}"
    )
    for _, row in context.iterrows():
        print(
            f"{row['monitored_line']}: {int(row['binding_hours'])} binding hours, "
            f"direction {row['dominant_binding_flow_direction']}"
        )
    for line, group in ranking.groupby("monitored_line"):
        top = group.iloc[0]
        members = int(group["directional_constraint_group_member"].sum())
        print(
            f"{line}: highest absolute factor {top['wind_farm']} "
            f"({top['primary_shift_factor']:+.6f}); directional group members {members}"
        )
    print(f"Results: {output}")
    return 0
