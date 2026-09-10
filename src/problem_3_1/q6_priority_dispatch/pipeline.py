"""Command-line workflow for Problem 3.1 Q6."""
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
import random
import subprocess
import sys
from typing import Any
import warnings

import numpy as np
import pandas as pd

from . import __version__
from .core import (
    ScenarioResult,
    compare_policy_results,
    solve_lexicographic,
    status_assignment,
    target_line_hotspots,
    validate_assignment,
    wind_ids,
)
from .economics import (
    collect_q4_sensitivity,
    q4_case_rows,
    wind_value_distribution,
)
from .reporting import write_outputs


DEFAULT_CONFIG: dict[str, Any] = {
    "scenario": "WP2033",
    "scope": "north-west",
    "wind_carriers": ["wind"],
    "binding_threshold": 0.999,
    "hotspot_threshold": 0.999,
    "lexicographic_tolerance_mwh": 1e-5,
    "solver_time_limit_s": 180.0,
    "priority_assignment": {
        "mode": "illustrative",
        "priority_generators": [
            "Ardnagappary wind",
            "Binbane wind",
            "Croaghonagh wind",
            "Drumkeen wind",
        ],
        "csv": None,
        "require_verified": False,
        "source": "illustrative_q4_site_priority_scenario",
        "assumption_note": (
            "Q4 candidate-site wind farms are marked priority only to test the mechanism; "
            "this is not verified legal status."
        ),
    },
    "counterfactuals": {
        "relaxed_rating_multiplier": 1000.0,
        "permutation_count": 4,
        "permutation_seed": 42,
        "leave_one_out": True,
        "line_relaxation_count": 3,
        "line_relaxation_multiplier": 1000.0,
    },
    "q3_dlr_csv": (
        "results/problem_3_1/question_3_final/tables/"
        "06_selected_dlr_multipliers.csv"
    ),
    "q4_economic_bridge": {
        "size_run": "results/problem_3_1/question_4_economic_downsizing_168h_20260910",
        "distributed_run": (
            "results/problem_3_1/"
            "question_4_distributed_bess_strict_45mw_168h_20260910"
        ),
        "size_pairs_mw_mwh": [
            [22.5, 110.0],
            [45.0, 220.0],
            [67.5, 330.0],
            [90.0, 440.0],
        ],
        "annual_hours": 8760.0,
        "energy_value_eur_mwh": 70.0,
        "years": 20,
        "real_discount_rate": 0.07,
    },
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"Unknown Q6 configuration key: {key}")
        if isinstance(value, dict) and isinstance(result[key], dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
    else:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError("Q6 configuration must be a JSON object")
        config = _deep_update(DEFAULT_CONFIG, value)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if not str(config["scenario"]).strip() or not str(config["scope"]).strip():
        raise ValueError("scenario and scope must be non-empty")
    if not config["wind_carriers"]:
        raise ValueError("wind_carriers must be non-empty")
    for key in ("binding_threshold", "hotspot_threshold"):
        if not 0 < float(config[key]) <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if float(config["lexicographic_tolerance_mwh"]) <= 0:
        raise ValueError("lexicographic_tolerance_mwh must be positive")
    if float(config["solver_time_limit_s"]) <= 0:
        raise ValueError("solver_time_limit_s must be positive")
    assignment = config["priority_assignment"]
    if assignment["mode"] not in {"illustrative", "csv"}:
        raise ValueError("priority_assignment.mode must be illustrative or csv")
    if assignment["mode"] == "csv" and not assignment["csv"]:
        raise ValueError("priority_assignment.csv is required in csv mode")
    counter = config["counterfactuals"]
    if float(counter["relaxed_rating_multiplier"]) <= 1:
        raise ValueError("relaxed_rating_multiplier must exceed one")
    if int(counter["permutation_count"]) < 0:
        raise ValueError("permutation_count must be non-negative")
    if int(counter["line_relaxation_count"]) < 0:
        raise ValueError("line_relaxation_count must be non-negative")
    bridge = config["q4_economic_bridge"]
    if float(bridge["annual_hours"]) <= 0 or float(bridge["energy_value_eur_mwh"]) < 0:
        raise ValueError("Invalid Q4 economic bridge hours or energy value")
    if int(bridge["years"]) < 1 or not 0 <= float(bridge["real_discount_rate"]) < 1:
        raise ValueError("Invalid economic bridge horizon or discount rate")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gridkit(kit_dir: Path):
    module_path = kit_dir / "gridkit.py"
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("_q6_gridkit", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_network(root: Path, config: dict[str, Any], max_hours: int | None):
    kit_dir = root / "data" / "participant-kit"
    gridkit = _load_gridkit(kit_dir)
    network = gridkit.load(config["scenario"], config["scope"])
    if max_hours is not None:
        network = network.copy()
        network.set_snapshots(network.snapshots[:max_hours])
    source = kit_dir / "networks" / f"{config['scenario']}_{config['scope']}.nc"
    return network, gridkit, source


def _assignment_for(
    root: Path,
    network: Any,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    settings = config["priority_assignment"]
    if settings["mode"] == "csv":
        path = Path(settings["csv"])
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(path)
        assignment = pd.read_csv(path)
        source_audit = {
            "path": str(path),
            "sha256": _sha256(path),
            "mode": "csv",
        }
    else:
        assignment = status_assignment(
            network,
            settings["priority_generators"],
            source=settings["source"],
            verified=False,
            assumption_note=settings["assumption_note"],
            carriers=config["wind_carriers"],
        )
        source_audit = {
            "path": None,
            "sha256": None,
            "mode": "illustrative",
        }
    validate_assignment(
        network,
        assignment,
        require_verified=bool(settings["require_verified"]),
        carriers=config["wind_carriers"],
    )
    source_audit["verified_rows"] = int(assignment["verified"].astype(bool).sum())
    source_audit["total_rows"] = int(len(assignment))
    return assignment, source_audit


def _apply_dlr(root: Path, network: Any, path_value: str) -> tuple[Any, Path, str]:
    from q4_techno_economic.grid import apply_dlr_multiplier_csv

    path = Path(path_value)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        raise FileNotFoundError(path)
    changed = network.copy()
    _, digest = apply_dlr_multiplier_csv(changed, path, root)
    return changed, path, digest


def _solve(
    network: Any,
    assignment: pd.DataFrame,
    config: dict[str, Any],
    *,
    policy: str,
    label: str,
    relaxed_rating_multiplier: float | None = None,
) -> ScenarioResult:
    return solve_lexicographic(
        network,
        assignment,
        policy=policy,
        label=label,
        wind_carriers=config["wind_carriers"],
        binding_threshold=float(config["binding_threshold"]),
        lexicographic_tolerance_mwh=float(
            config["lexicographic_tolerance_mwh"]
        ),
        relaxed_rating_multiplier=relaxed_rating_multiplier,
        solver_time_limit_s=float(config["solver_time_limit_s"]),
    )


def _combine_results(
    results: list[ScenarioResult],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scenarios = pd.DataFrame([result.summary for result in results])
    generator = []
    system = []
    branch = []
    for result in results:
        for frame, target in (
            (result.generator_hourly, generator),
            (result.system_hourly, system),
            (result.branch_hourly, branch),
        ):
            copy_frame = frame.copy()
            copy_frame.insert(0, "scenario", result.label)
            target.append(copy_frame)
    return (
        scenarios,
        pd.concat(generator, ignore_index=True),
        pd.concat(system, ignore_index=True),
        pd.concat(branch, ignore_index=True),
    )


def _swapped_assignment(assignment: pd.DataFrame) -> pd.DataFrame:
    changed = assignment.copy()
    changed["priority_status"] = changed["priority_status"].map(
        {"priority": "non_priority", "non_priority": "priority"}
    )
    changed["status_source"] = "status_swapped_counterfactual"
    changed["verified"] = False
    changed["data_class"] = "SCENARIO_CHOICE"
    changed["assumption_note"] = "Every illustrative status label is reversed."
    return changed


def _leave_one_out(
    network: Any,
    assignment: pd.DataFrame,
    config: dict[str, Any],
    full_priority: ScenarioResult,
    limit: int | None,
) -> pd.DataFrame:
    if not config["counterfactuals"]["leave_one_out"]:
        return pd.DataFrame()
    priority_names = assignment.loc[
        assignment["priority_status"].eq("priority"), "generator"
    ].astype(str).tolist()
    if limit is not None:
        priority_names = priority_names[:limit]
    rows = []
    for generator in priority_names:
        changed = assignment.copy()
        changed.loc[
            changed["generator"].astype(str).eq(generator), "priority_status"
        ] = "non_priority"
        result = _solve(
            network,
            changed,
            config,
            policy="priority",
            label=f"leave_out_{generator}",
        )
        rows.append(
            {
                "generator_changed_to_non_priority": generator,
                "full_priority_dispatch_down_mwh": full_priority.summary[
                    "wind_dispatch_down_mwh"
                ],
                "counterfactual_dispatch_down_mwh": result.summary[
                    "wind_dispatch_down_mwh"
                ],
                "dispatch_down_reduction_mwh": (
                    full_priority.summary["wind_dispatch_down_mwh"]
                    - result.summary["wind_dispatch_down_mwh"]
                ),
                "physical_cost_change_eur": (
                    result.summary["physical_generator_cost_eur"]
                    - full_priority.summary["physical_generator_cost_eur"]
                ),
                "non_additive": True,
            }
        )
    return pd.DataFrame(rows)


def _permutations(
    network: Any,
    assignment: pd.DataFrame,
    config: dict[str, Any],
    count: int,
) -> pd.DataFrame:
    farms = assignment["generator"].astype(str).tolist()
    priority_count = int(assignment["priority_status"].eq("priority").sum())
    rng = random.Random(int(config["counterfactuals"]["permutation_seed"]))
    rows = []
    for number in range(count):
        selected = sorted(rng.sample(farms, priority_count))
        changed = status_assignment(
            network,
            selected,
            source=f"fixed_seed_permutation_{number + 1}",
            verified=False,
            assumption_note="Random fixed-count status permutation for sensitivity.",
            carriers=config["wind_carriers"],
        )
        result = _solve(
            network,
            changed,
            config,
            policy="priority",
            label=f"permutation_{number + 1}",
        )
        rows.append(
            {
                "permutation": number + 1,
                "seed": int(config["counterfactuals"]["permutation_seed"]),
                "priority_generators": ";".join(selected),
                "priority_capacity_mw": float(
                    changed.loc[
                        changed["priority_status"].eq("priority"),
                        "installed_capacity_mw",
                    ].sum()
                ),
                "wind_dispatch_down_mwh": result.summary["wind_dispatch_down_mwh"],
                "physical_generator_cost_eur": result.summary[
                    "physical_generator_cost_eur"
                ],
                "unserved_mwh": result.summary["unserved_mwh"],
            }
        )
    return pd.DataFrame(rows)


def _line_relaxation_attribution(
    network: Any,
    assignment: pd.DataFrame,
    config: dict[str, Any],
    neutral: ScenarioResult,
    priority: ScenarioResult,
    count: int,
) -> pd.DataFrame:
    if count <= 0:
        return pd.DataFrame()
    base_inefficiency = (
        priority.summary["wind_dispatch_down_mwh"]
        - neutral.summary["wind_dispatch_down_mwh"]
    )
    candidates = (
        priority.branch_hourly.groupby(
            ["component", "branch", "bus0", "bus1"], as_index=False
        )
        .agg(
            binding_hours=("binding", "sum"),
            maximum_loading_pu=("loading_pu", "max"),
        )
        .sort_values(
            ["binding_hours", "maximum_loading_pu", "branch"],
            ascending=[False, False, True],
        )
        .head(count)
    )
    rows = []
    multiplier = float(
        config["counterfactuals"]["line_relaxation_multiplier"]
    )
    for _, candidate in candidates.iterrows():
        branch = str(candidate["branch"])
        changed = network.copy()
        if branch in changed.lines.index:
            changed.lines.at[branch, "s_nom"] *= multiplier
        elif branch in changed.transformers.index:
            changed.transformers.at[branch, "s_nom"] *= multiplier
        else:
            raise KeyError(branch)
        neutral_changed = _solve(
            changed,
            assignment,
            config,
            policy="neutral",
            label=f"line_relaxed_neutral_{branch}",
        )
        priority_changed = _solve(
            changed,
            assignment,
            config,
            policy="priority",
            label=f"line_relaxed_priority_{branch}",
        )
        changed_inefficiency = (
            priority_changed.summary["wind_dispatch_down_mwh"]
            - neutral_changed.summary["wind_dispatch_down_mwh"]
        )
        rows.append(
            {
                "branch": branch,
                "component": candidate["component"],
                "bus0": candidate["bus0"],
                "bus1": candidate["bus1"],
                "base_binding_hours": int(candidate["binding_hours"]),
                "base_maximum_loading_pu": float(candidate["maximum_loading_pu"]),
                "rating_multiplier": multiplier,
                "base_status_inefficiency_mwh": float(base_inefficiency),
                "relaxed_status_inefficiency_mwh": float(changed_inefficiency),
                "status_inefficiency_reduction_when_relaxed_mwh": float(
                    base_inefficiency - changed_inefficiency
                ),
                "non_additive": True,
            }
        )
    return pd.DataFrame(rows)


def _q4_bridge(
    root: Path,
    config: dict[str, Any],
    impacts: pd.DataFrame,
    block_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[Path]]:
    bridge = config["q4_economic_bridge"]
    size_run = Path(bridge["size_run"])
    distributed_run = Path(bridge["distributed_run"])
    cases, sources = q4_case_rows(
        root,
        size_run,
        distributed_run,
        [tuple(map(float, pair)) for pair in bridge["size_pairs_mw_mwh"]],
    )
    wind_value, audit = wind_value_distribution(
        impacts,
        block_hours=block_hours,
        annual_hours=float(bridge["annual_hours"]),
        energy_value_eur_mwh=float(bridge["energy_value_eur_mwh"]),
        discount_rate=float(bridge["real_discount_rate"]),
        years=int(bridge["years"]),
    )
    selected = set(cases.loc[cases["case_id"].ne("NO_BUILD"), "case_id"].astype(str))
    sensitivity = collect_q4_sensitivity(
        root,
        [size_run, distributed_run],
        selected,
    )
    cases["q6_priority_minus_neutral_dispatch_down_mwh"] = float(
        -wind_value["dispatched_energy_change_mwh"].sum()
    )
    cases["q6_aggregate_wind_value_pv_eur"] = float(
        wind_value["pv_energy_value_transfer_eur"].sum()
    )
    cases["q6_aggregate_wind_value_included_in_project_npv"] = False
    cases["developer_project_revenue_definition"] = (
        "Q4 energy sales plus configured grid-service payment only"
    )
    cases["system_value_definition"] = (
        "Q4 physical dispatch-cost saving, reported outside project cash flow"
    )
    return cases, wind_value, sensitivity, audit, sources


def _provenance(
    config: dict[str, Any],
    network_source: Path,
    dlr_path: Path,
    q4_sources: list[Path],
) -> pd.DataFrame:
    rows = [
        {
            "conclusion_id": "Q6-C01",
            "metric": "priority-minus-neutral total wind dispatch-down",
            "output_file": "02_scenario_summary.csv",
            "row_locator": (
                "scenario in {network_neutral_static,network_priority_static}"
            ),
            "column": "wind_dispatch_down_mwh",
            "upstream_source": str(network_source),
            "source_locator": "generators.p_nom; generators_t.p_max_pu; network constraints",
            "formula": "priority dispatch-down minus neutral dispatch-down",
            "implemented_by": "q6_priority_dispatch.core.solve_lexicographic",
            "data_class": "MODEL_DERIVED",
        },
        {
            "conclusion_id": "Q6-C02",
            "metric": "farm energy protected or burdened",
            "output_file": "06_generator_status_impacts.csv",
            "row_locator": "generator exact ID",
            "column": "additional_dispatch_down_mwh",
            "upstream_source": "03_generator_hourly_dispatch.csv",
            "source_locator": "same generator and snapshot under both policies",
            "formula": "priority dispatch-down minus neutral dispatch-down",
            "implemented_by": "q6_priority_dispatch.core.compare_policy_results",
            "data_class": "MODEL_DERIVED",
        },
        {
            "conclusion_id": "Q6-C03",
            "metric": "priority status assignment",
            "output_file": "01_priority_status_assignment.csv",
            "row_locator": "generator exact ID",
            "column": "priority_status",
            "upstream_source": "configs/problem_3_1/question_6.json",
            "source_locator": "/priority_assignment/priority_generators",
            "formula": "explicit membership; all others non-priority",
            "implemented_by": "q6_priority_dispatch.core.status_assignment",
            "data_class": "SCENARIO_CHOICE",
        },
        {
            "conclusion_id": "Q6-C04",
            "metric": "DLR interaction",
            "output_file": "10_static_vs_dlr_priority_comparison.csv",
            "row_locator": "rating_case in {static,q3_dlr}",
            "column": "priority_minus_neutral_dispatch_down_mwh",
            "upstream_source": str(dlr_path),
            "source_locator": "snapshot and exact line multiplier columns",
            "formula": "same policy comparison after timestamp-aligned rating multipliers",
            "implemented_by": "q4_techno_economic.grid.apply_dlr_multiplier_csv",
            "data_class": "MODEL_DERIVED",
        },
        {
            "conclusion_id": "Q6-C05",
            "metric": "Q4 project NPV and system saving remain separate",
            "output_file": "12_q4_economic_bridge.csv",
            "row_locator": "case_id",
            "column": "npv_eur; lifetime_system_dispatch_cost_saving_eur",
            "upstream_source": ";".join(map(str, q4_sources)),
            "source_locator": "01_project_ranking.csv exact case_id rows",
            "formula": "values copied without adding Q6 or system value to project NPV",
            "implemented_by": "q6_priority_dispatch.economics.q4_case_rows",
            "data_class": "MODEL_DERIVED plus COMMERCIAL_ASSUMPTION",
        },
    ]
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TPSA Problem 3.1 Q6 priority-status counterfactual"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scenario")
    parser.add_argument("--scope")
    parser.add_argument("--out", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def _git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def main(root: Path) -> int:
    args = parse_args()
    config_path = (
        args.config
        or root / "configs" / "problem_3_1" / "question_6.json"
    )
    if args.config is not None and not args.config.exists():
        raise FileNotFoundError(args.config)
    config = load_config(config_path if config_path.exists() else None)
    if args.scenario:
        config["scenario"] = args.scenario
    if args.scope:
        config["scope"] = args.scope
    validate_config(config)

    for logger_name in ("pypsa", "linopy", "highspy"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)

    max_hours = 24 if args.smoke else None
    network, gridkit, network_source = _load_network(
        root,
        config,
        max_hours,
    )
    gridkit.quiet()
    assignment, assignment_audit = _assignment_for(root, network, config)
    official_status_columns = [
        str(column)
        for column in network.generators.columns
        if "priority" in str(column).lower()
    ]
    dlr_network, dlr_path, dlr_hash = _apply_dlr(
        root,
        network,
        config["q3_dlr_csv"],
    )

    q4_settings = config["q4_economic_bridge"]
    q4_size_path = root / q4_settings["size_run"] / "01_project_ranking.csv"
    q4_distributed_path = (
        root / q4_settings["distributed_run"] / "01_project_ranking.csv"
    )
    for path in (network_source, dlr_path, q4_size_path, q4_distributed_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("=== Q6 INPUT CHECK ===", flush=True)
    print(
        f"Network: {config['scenario']} {config['scope']} | "
        f"{len(network.buses)} buses, {len(network.lines)} lines, "
        f"{len(network.transformers)} transformers"
    )
    print(
        f"Snapshots: {len(network.snapshots)} | wind farms: "
        f"{len(wind_ids(network, config['wind_carriers']))}"
    )
    print(
        f"Priority scenario: "
        f"{int(assignment['priority_status'].eq('priority').sum())} priority / "
        f"{int(assignment['priority_status'].eq('non_priority').sum())} non-priority"
    )
    print(
        "Verified priority columns in official generator table: "
        + (", ".join(official_status_columns) if official_status_columns else "none")
    )
    print(
        "STATUS LABEL: SCENARIO_CHOICE"
        if not bool(assignment["verified"].all())
        else "STATUS LABEL: VERIFIED INPUT"
    )
    if not (args.run or args.smoke):
        print(
            "CHECK PASSED: exact generator IDs, DLR timestamps, Q4 linkage, "
            "and status provenance are valid."
        )
        return 0

    neutral_static = _solve(
        network,
        assignment,
        config,
        policy="neutral",
        label="network_neutral_static",
    )
    priority_static = _solve(
        network,
        assignment,
        config,
        policy="priority",
        label="network_priority_static",
    )
    relaxed = float(
        config["counterfactuals"]["relaxed_rating_multiplier"]
    )
    copper_neutral = _solve(
        network,
        assignment,
        config,
        policy="neutral",
        label="copperplate_neutral",
        relaxed_rating_multiplier=relaxed,
    )
    copper_priority = _solve(
        network,
        assignment,
        config,
        policy="priority",
        label="copperplate_priority",
        relaxed_rating_multiplier=relaxed,
    )
    swapped = _swapped_assignment(assignment)
    swapped_priority = _solve(
        network,
        swapped,
        config,
        policy="priority",
        label="status_swapped",
    )
    neutral_dlr = _solve(
        dlr_network,
        assignment,
        config,
        policy="neutral",
        label="network_neutral_dlr",
    )
    priority_dlr = _solve(
        dlr_network,
        assignment,
        config,
        policy="priority",
        label="network_priority_dlr",
    )
    main_results = [
        neutral_static,
        priority_static,
        copper_neutral,
        copper_priority,
        swapped_priority,
        neutral_dlr,
        priority_dlr,
    ]
    scenarios, generator_hourly, system_hourly, branch_hourly = _combine_results(
        main_results
    )
    _, impacts = compare_policy_results(neutral_static, priority_static)
    hotspots = target_line_hotspots(
        priority_static,
        assignment,
        reference="load",
        minimum_loading_pu=float(config["hotspot_threshold"]),
    )

    limit = 1 if args.smoke else None
    leave_one_out = _leave_one_out(
        network,
        assignment,
        config,
        priority_static,
        limit,
    )
    permutation_count = (
        min(1, int(config["counterfactuals"]["permutation_count"]))
        if args.smoke
        else int(config["counterfactuals"]["permutation_count"])
    )
    permutations = _permutations(
        network,
        assignment,
        config,
        permutation_count,
    )
    line_count = (
        min(1, int(config["counterfactuals"]["line_relaxation_count"]))
        if args.smoke
        else int(config["counterfactuals"]["line_relaxation_count"])
    )
    line_attribution = _line_relaxation_attribution(
        network,
        assignment,
        config,
        neutral_static,
        priority_static,
        line_count,
    )

    dlr_comparison = pd.DataFrame(
        [
            {
                "rating_case": "static",
                "neutral_dispatch_down_mwh": neutral_static.summary[
                    "wind_dispatch_down_mwh"
                ],
                "priority_dispatch_down_mwh": priority_static.summary[
                    "wind_dispatch_down_mwh"
                ],
                "priority_minus_neutral_dispatch_down_mwh": (
                    priority_static.summary["wind_dispatch_down_mwh"]
                    - neutral_static.summary["wind_dispatch_down_mwh"]
                ),
                "neutral_physical_cost_eur": neutral_static.summary[
                    "physical_generator_cost_eur"
                ],
                "priority_physical_cost_eur": priority_static.summary[
                    "physical_generator_cost_eur"
                ],
            },
            {
                "rating_case": "q3_dlr",
                "neutral_dispatch_down_mwh": neutral_dlr.summary[
                    "wind_dispatch_down_mwh"
                ],
                "priority_dispatch_down_mwh": priority_dlr.summary[
                    "wind_dispatch_down_mwh"
                ],
                "priority_minus_neutral_dispatch_down_mwh": (
                    priority_dlr.summary["wind_dispatch_down_mwh"]
                    - neutral_dlr.summary["wind_dispatch_down_mwh"]
                ),
                "neutral_physical_cost_eur": neutral_dlr.summary[
                    "physical_generator_cost_eur"
                ],
                "priority_physical_cost_eur": priority_dlr.summary[
                    "physical_generator_cost_eur"
                ],
            },
        ]
    )
    (
        q4_bridge,
        wind_value,
        sensitivity,
        economic_audit,
        q4_sources,
    ) = _q4_bridge(
        root,
        config,
        impacts,
        float(priority_static.summary["hours"]),
    )
    provenance = _provenance(
        config,
        network_source,
        dlr_path,
        q4_sources,
    )

    code_hasher = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        code_hasher.update(source.name.encode())
        code_hasher.update(source.read_bytes())
    input_audit = {
        "network": {
            "path": str(network_source),
            "sha256": _sha256(network_source),
            "meta": getattr(network, "meta", {}),
            "buses": int(len(network.buses)),
            "lines": int(len(network.lines)),
            "transformers": int(len(network.transformers)),
            "generators": int(len(network.generators)),
            "snapshots": int(len(network.snapshots)),
        },
        "official_priority_status_columns": official_status_columns,
        "priority_assignment": assignment_audit,
        "q3_dlr": {
            "path": str(dlr_path),
            "sha256": dlr_hash,
        },
        "q4_sources": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in q4_sources
        ],
        "assumptions": [
            "Priority statuses are scenario choices unless every assignment row is verified.",
            "Priority is implemented by exact lexicographic stages, not a price penalty.",
            "Physical generator cost uses the original marginal-cost objective.",
            "The copperplate comparison multiplies passive ratings and does not remove topology.",
            "Q6 wind-owner energy value is not booked as stand-alone BESS revenue.",
        ],
    }
    manifest = {
        "q6_model_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "smoke" if args.smoke else "full",
        "config": config,
        "input_audit": input_audit,
        "economic_bridge": economic_audit,
        "code_sha256": code_hasher.hexdigest(),
        "git": {
            "branch": _git_value(root, "branch", "--show-current"),
            "commit": _git_value(root, "rev-parse", "HEAD"),
            "dirty": bool(_git_value(root, "status", "--porcelain")),
        },
        "packages": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "numerical_checks": {
            "all_solver_status_ok": bool(
                scenarios["solver_status"].astype(str).str.lower().eq("ok").all()
            ),
            "maximum_component_balance_error_mw": float(
                scenarios["maximum_component_balance_error_mw"].max()
            ),
            "maximum_passive_loading_pu": float(
                scenarios.loc[
                    scenarios["relaxed_rating_multiplier"].isna(),
                    "maximum_passive_loading_pu",
                ].max()
            ),
            "maximum_unserved_mwh": float(scenarios["unserved_mwh"].max()),
            "status_penalty_ever_in_physical_cost": bool(
                scenarios["status_penalty_in_physical_cost"].any()
            ),
        },
    }

    output = args.out or root / "results" / "problem_3_1" / (
        "question_6_smoke" if args.smoke else "question_6_final"
    )
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; choose a new --out directory")
    write_outputs(
        output,
        manifest=manifest,
        assignment=assignment,
        scenarios=scenarios,
        generator_hourly=generator_hourly,
        system_hourly=system_hourly,
        branch_hourly=branch_hourly,
        impacts=impacts,
        hotspots=hotspots,
        leave_one_out=leave_one_out,
        permutations=permutations,
        dlr_comparison=dlr_comparison,
        line_attribution=line_attribution,
        q4_bridge=q4_bridge,
        wind_value=wind_value,
        sensitivity=sensitivity,
        provenance=provenance,
    )

    total_delta = (
        priority_static.summary["wind_dispatch_down_mwh"]
        - neutral_static.summary["wind_dispatch_down_mwh"]
    )
    copper_delta = (
        copper_priority.summary["wind_dispatch_down_mwh"]
        - copper_neutral.summary["wind_dispatch_down_mwh"]
    )
    print("=== Q6 RESULT ===")
    print(f"Priority-minus-neutral dispatch-down: {total_delta:+,.6f} MWh")
    print(
        "Network-specific difference-in-differences: "
        f"{total_delta - copper_delta:+,.6f} MWh"
    )
    print(
        f"Generator impacts reconcile to "
        f"{float(impacts['additional_dispatch_down_mwh'].sum()):+,.6f} MWh"
    )
    print(
        f"Maximum component balance error: "
        f"{manifest['numerical_checks']['maximum_component_balance_error_mw']:.3e} MW"
    )
    print(f"Results: {output}")
    return 0
