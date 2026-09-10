"""Exact lexicographic wind-priority dispatch and Q6 diagnostics.

The official project input does not contain legal priority-status labels or an
operational priority-dispatch rule.  This module therefore implements an
explicit counterfactual.  Status coefficients never enter reported physical
operating cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from q5_shift_factors.core import (
    branch_rating_series,
    branches,
    bus_components,
    component_adjust_reference,
    ptdf,
)


EPSILON = 1e-9


@dataclass
class ScenarioResult:
    """Complete result of one dispatch-policy counterfactual."""

    label: str
    policy: str
    network: Any
    summary: dict[str, Any]
    generator_hourly: pd.DataFrame
    system_hourly: pd.DataFrame
    branch_hourly: pd.DataFrame


def dense_generator_attribute(network: Any, attribute: str) -> pd.DataFrame:
    """Broadcast a static generator attribute and overwrite dynamic columns."""
    static = network.generators
    if attribute not in static:
        raise ValueError(f"Missing generators.{attribute}")
    values = np.tile(
        static[attribute].astype(float).to_numpy(),
        (len(network.snapshots), 1),
    )
    result = pd.DataFrame(
        values,
        index=network.snapshots,
        columns=static.index,
        dtype=float,
    )
    dynamic = getattr(network.generators_t, attribute, pd.DataFrame())
    if isinstance(dynamic, pd.DataFrame) and not dynamic.empty:
        for column in dynamic.columns.intersection(result.columns):
            result[column] = dynamic[column].reindex(network.snapshots)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite generators.{attribute}")
    return result


def dense_load(network: Any) -> pd.DataFrame:
    """Return chronological MW demand for every load."""
    static = network.loads
    base = (
        static["p_set"].astype(float)
        if "p_set" in static
        else pd.Series(0.0, index=static.index)
    )
    result = pd.DataFrame(
        np.tile(base.to_numpy(), (len(network.snapshots), 1)),
        index=network.snapshots,
        columns=static.index,
        dtype=float,
    )
    dynamic = network.loads_t.p_set
    if isinstance(dynamic, pd.DataFrame) and not dynamic.empty:
        for column in dynamic.columns.intersection(result.columns):
            result[column] = dynamic[column].reindex(network.snapshots)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite load profile")
    return result


def duration_hours(network: Any) -> pd.Series:
    """Return physical snapshot durations used in lexicographic energy stages."""
    weightings = network.snapshot_weightings
    column = "generators" if "generators" in weightings else weightings.columns[0]
    values = weightings[column].reindex(network.snapshots).astype(float)
    if not np.isfinite(values).all() or bool((values <= 0).any()):
        raise ValueError("Snapshot durations must be finite and positive")
    return values


def wind_ids(network: Any, carriers: Iterable[str] = ("wind",)) -> list[str]:
    """Return exact generator IDs for the configured wind carriers."""
    allowed = {str(value).strip().lower() for value in carriers}
    selected = network.generators.index[
        network.generators["carrier"].astype(str).str.lower().isin(allowed)
    ]
    if not len(selected):
        raise ValueError(f"No wind generators for carriers {sorted(allowed)}")
    return [str(value) for value in selected]


def shedding_ids(network: Any) -> list[str]:
    """Return explicit high-penalty feasibility generators."""
    carrier = network.generators["carrier"].astype(str).str.lower()
    names = network.generators.index.astype(str)
    mask = carrier.str.contains("load shedding", regex=False) | names.str.lower().str.startswith("shed ")
    return [str(value) for value in network.generators.index[mask]]


def status_assignment(
    network: Any,
    priority_generators: Iterable[str],
    *,
    source: str,
    verified: bool,
    assumption_note: str,
    carriers: Iterable[str] = ("wind",),
) -> pd.DataFrame:
    """Build and validate a complete wind status assignment."""
    farms = wind_ids(network, carriers)
    priority = {str(value) for value in priority_generators}
    unknown = sorted(priority - set(farms))
    if unknown:
        raise ValueError(f"Priority assignments are not exact wind IDs: {unknown}")
    rows = []
    for generator in farms:
        rows.append(
            {
                "generator": generator,
                "bus": str(network.generators.at[generator, "bus"]),
                "installed_capacity_mw": float(network.generators.at[generator, "p_nom"]),
                "priority_status": "priority" if generator in priority else "non_priority",
                "status_source": str(source),
                "verified": bool(verified),
                "data_class": "OFFICIAL_INPUT" if verified else "SCENARIO_CHOICE",
                "assumption_note": str(assumption_note),
            }
        )
    return pd.DataFrame(rows)


def validate_assignment(
    network: Any,
    assignment: pd.DataFrame,
    *,
    require_verified: bool = False,
    carriers: Iterable[str] = ("wind",),
) -> None:
    """Reject missing, duplicated, invalid, or falsely observed assignments."""
    required = {
        "generator",
        "bus",
        "priority_status",
        "status_source",
        "verified",
        "assumption_note",
    }
    missing_columns = sorted(required - set(assignment.columns))
    if missing_columns:
        raise ValueError(f"Status assignment is missing columns: {missing_columns}")
    if assignment["generator"].astype(str).duplicated().any():
        raise ValueError("Status assignment contains duplicate generators")
    expected = set(wind_ids(network, carriers))
    actual = set(assignment["generator"].astype(str))
    if actual != expected:
        raise ValueError(
            f"Status assignment must cover every wind generator exactly; "
            f"missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
        )
    allowed = {"priority", "non_priority"}
    invalid = set(assignment["priority_status"].astype(str)) - allowed
    if invalid:
        raise ValueError(f"Invalid priority_status values: {sorted(invalid)}")
    generator_bus = network.generators["bus"].astype(str)
    for _, row in assignment.iterrows():
        if str(row["bus"]) != generator_bus.loc[str(row["generator"])]:
            raise ValueError(f"Status bus does not match model bus for {row['generator']}")
    if require_verified and not bool(assignment["verified"].astype(bool).all()):
        raise ValueError(
            "Observed mode requires a verified source for every priority status; "
            "the official project network contains no such field"
        )


def _component_variable_sum(variable: Any, names: list[str]) -> Any:
    dimensions = [name for name in variable.dims if name != "snapshot"]
    if len(dimensions) != 1:
        raise RuntimeError(f"Unsupported generator variable dimensions: {variable.dims}")
    return variable.sel({dimensions[0]: names}).sum(dimensions[0])


def _energy_expression(variable: Any, names: list[str], weights: Any) -> Any:
    return (_component_variable_sum(variable, names) * weights).sum()


def _stage_solve(model: Any, expression: Any, sense: str, name: str, tolerance: float) -> dict[str, Any]:
    """Solve one exact lexicographic stage and fix its optimum."""
    if sense not in {"min", "max"}:
        raise ValueError("Stage sense must be min or max")
    objective = expression if sense == "min" else -expression
    model.add_objective(objective, overwrite=True)
    status, condition = model.solve(
        solver_name="highs",
        io_api="direct",
        output_flag=False,
        log_to_console=False,
    )
    if str(status).lower() != "ok" or "optimal" not in str(condition).lower():
        raise RuntimeError(f"Lexicographic stage {name} failed: {status}/{condition}")
    optimum = float(expression.solution)
    if sense == "min":
        model.add_constraints(
            expression <= optimum + tolerance,
            name=f"Q6-fix-{name}",
        )
    else:
        model.add_constraints(
            expression >= optimum - tolerance,
            name=f"Q6-fix-{name}",
        )
    return {
        "stage": name,
        "sense": sense,
        "optimum_mwh": optimum,
        "fix_tolerance_mwh": float(tolerance),
    }


def _branch_hourly(network: Any, binding_threshold: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for table, component in (("lines", "Line"), ("transformers", "Transformer")):
        static = getattr(network, table)
        if static.empty:
            continue
        flow = getattr(network, f"{table}_t").p0.reindex(
            index=network.snapshots, columns=static.index
        )
        for branch in static.index:
            rating = branch_rating_series(network, str(branch))
            local = pd.DataFrame(
                {
                    "snapshot": network.snapshots,
                    "component": component,
                    "branch": str(branch),
                    "bus0": str(static.at[branch, "bus0"]),
                    "bus1": str(static.at[branch, "bus1"]),
                    "flow_mw_bus0_to_bus1": flow[branch].to_numpy(dtype=float),
                    "rating_mva": rating.to_numpy(dtype=float),
                }
            )
            local["loading_pu"] = (
                local["flow_mw_bus0_to_bus1"].abs()
                / local["rating_mva"].replace(0.0, np.nan)
            )
            local["binding"] = local["loading_pu"] >= float(binding_threshold)
            local["flow_direction_sign"] = np.sign(
                local["flow_mw_bus0_to_bus1"]
            ).astype(int)
            rows.append(local)
    if not rows:
        return pd.DataFrame(
            columns=[
                "snapshot",
                "component",
                "branch",
                "bus0",
                "bus1",
                "flow_mw_bus0_to_bus1",
                "rating_mva",
                "loading_pu",
                "binding",
                "flow_direction_sign",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _generator_hourly(
    network: Any,
    assignment: pd.DataFrame,
    wind_carriers: Iterable[str],
) -> pd.DataFrame:
    farms = wind_ids(network, wind_carriers)
    availability = dense_generator_attribute(network, "p_max_pu")[farms].mul(
        network.generators.loc[farms, "p_nom"].astype(float),
        axis=1,
    )
    dispatch = network.generators_t.p[farms].reindex(network.snapshots)
    status = assignment.set_index("generator")
    rows = []
    for generator in farms:
        local = pd.DataFrame(
            {
                "snapshot": network.snapshots,
                "generator": generator,
                "bus": str(network.generators.at[generator, "bus"]),
                "priority_status": status.at[generator, "priority_status"],
                "available_mw": availability[generator].to_numpy(dtype=float),
                "dispatched_mw": dispatch[generator].to_numpy(dtype=float),
            }
        )
        local["dispatch_down_mw"] = (
            local["available_mw"] - local["dispatched_mw"]
        ).clip(lower=0.0)
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


def _system_hourly(
    network: Any,
    generator_hourly: pd.DataFrame,
) -> pd.DataFrame:
    durations = duration_hours(network)
    all_dispatch = network.generators_t.p.reindex(
        index=network.snapshots,
        columns=network.generators.index,
    )
    costs = dense_generator_attribute(network, "marginal_cost")
    shed = shedding_ids(network)
    grouped = generator_hourly.groupby("snapshot", sort=False)[
        ["available_mw", "dispatched_mw", "dispatch_down_mw"]
    ].sum()
    result = pd.DataFrame(index=network.snapshots)
    result.index.name = "snapshot"
    result["duration_h"] = durations
    result["wind_available_mw"] = grouped["available_mw"].reindex(result.index)
    result["wind_dispatched_mw"] = grouped["dispatched_mw"].reindex(result.index)
    result["wind_dispatch_down_mw"] = grouped["dispatch_down_mw"].reindex(result.index)
    result["unserved_mw"] = (
        all_dispatch[shed].sum(axis=1) if shed else 0.0
    )
    result["physical_generator_cost_eur"] = (
        all_dispatch * costs
    ).sum(axis=1) * durations
    return result.reset_index()


def component_balance_error(network: Any) -> float:
    """Return maximum lossless AC-component energy-balance residual in MW."""
    frame = branches(network)
    components = bus_components(network, frame)
    generation = network.generators_t.p.reindex(columns=network.generators.index)
    loads = dense_load(network)
    errors: list[float] = []
    for component in sorted(components.unique()):
        buses = set(components.index[components.eq(component)].astype(str))
        gen_ids = network.generators.index[
            network.generators["bus"].astype(str).isin(buses)
        ]
        load_ids = network.loads.index[
            network.loads["bus"].astype(str).isin(buses)
        ]
        residual = generation[gen_ids].sum(axis=1) - loads[load_ids].sum(axis=1)
        errors.append(float(residual.abs().max()))
    return max(errors, default=0.0)


def solve_lexicographic(
    network: Any,
    assignment: pd.DataFrame,
    *,
    policy: str,
    label: str,
    wind_carriers: Iterable[str] = ("wind",),
    binding_threshold: float = 0.999,
    lexicographic_tolerance_mwh: float = 1e-5,
    relaxed_rating_multiplier: float | None = None,
    solver_time_limit_s: float = 180.0,
) -> ScenarioResult:
    """Solve neutral or status-priority dispatch with exact objective stages."""
    if policy not in {"neutral", "priority"}:
        raise ValueError("policy must be neutral or priority")
    validate_assignment(network, assignment, carriers=wind_carriers)
    solved = network.copy()
    if relaxed_rating_multiplier is not None:
        if relaxed_rating_multiplier <= 1:
            raise ValueError("relaxed_rating_multiplier must exceed one")
        solved.lines.loc[:, "s_nom"] *= float(relaxed_rating_multiplier)
        if len(solved.transformers):
            solved.transformers.loc[:, "s_nom"] *= float(relaxed_rating_multiplier)
    model = solved.optimize.create_model()
    physical_objective = model.objective.expression
    generator_variable = model.variables["Generator-p"]

    import xarray as xr

    weights = xr.DataArray(
        duration_hours(solved).to_numpy(dtype=float),
        coords={"snapshot": solved.snapshots},
        dims=["snapshot"],
    )
    stages: list[dict[str, Any]] = []
    shed = shedding_ids(solved)
    if shed:
        stages.append(
            _stage_solve(
                model,
                _energy_expression(generator_variable, shed, weights),
                "min",
                "unserved-energy",
                lexicographic_tolerance_mwh,
            )
        )
    assignment_indexed = assignment.set_index("generator")
    farms = wind_ids(solved, wind_carriers)
    priority = [
        generator
        for generator in farms
        if assignment_indexed.at[generator, "priority_status"] == "priority"
    ]
    if policy == "priority" and priority:
        stages.append(
            _stage_solve(
                model,
                _energy_expression(generator_variable, priority, weights),
                "max",
                "priority-wind-energy",
                lexicographic_tolerance_mwh,
            )
        )
    stages.append(
        _stage_solve(
            model,
            _energy_expression(generator_variable, farms, weights),
            "max",
            "total-wind-energy",
            lexicographic_tolerance_mwh,
        )
    )
    model.add_objective(physical_objective, overwrite=True)
    status, condition = solved.optimize.solve_model(
        solver_name="highs",
        solver_options={
            "output_flag": False,
            "log_to_console": False,
            "time_limit": float(solver_time_limit_s),
        },
    )
    if str(status).lower() != "ok" or "optimal" not in str(condition).lower():
        raise RuntimeError(f"Final physical-cost dispatch failed: {status}/{condition}")

    generator_hourly = _generator_hourly(solved, assignment, wind_carriers)
    system_hourly = _system_hourly(solved, generator_hourly)
    branch_hourly = _branch_hourly(solved, binding_threshold)
    durations_by_snapshot = duration_hours(solved)
    generator_hourly["duration_h"] = generator_hourly["snapshot"].map(
        durations_by_snapshot
    )
    generator_hourly["dispatch_down_mwh"] = (
        generator_hourly["dispatch_down_mw"] * generator_hourly["duration_h"]
    )
    by_status = generator_hourly.groupby("priority_status")["dispatch_down_mwh"].sum()
    unserved_mwh = float(
        (
            system_hourly["unserved_mw"]
            * system_hourly["duration_h"]
        ).sum()
    )
    maximum_loading = (
        float(branch_hourly["loading_pu"].max())
        if len(branch_hourly)
        else 0.0
    )
    summary = {
        "scenario": label,
        "policy": policy,
        "snapshots": int(len(solved.snapshots)),
        "hours": float(duration_hours(solved).sum()),
        "priority_generator_count": int(len(priority)),
        "non_priority_generator_count": int(len(farms) - len(priority)),
        "wind_available_mwh": float(
            (
                system_hourly["wind_available_mw"]
                * system_hourly["duration_h"]
            ).sum()
        ),
        "wind_dispatched_mwh": float(
            (
                system_hourly["wind_dispatched_mw"]
                * system_hourly["duration_h"]
            ).sum()
        ),
        "wind_dispatch_down_mwh": float(generator_hourly["dispatch_down_mwh"].sum()),
        "priority_dispatch_down_mwh": float(by_status.get("priority", 0.0)),
        "non_priority_dispatch_down_mwh": float(by_status.get("non_priority", 0.0)),
        "unserved_mwh": unserved_mwh,
        "physical_generator_cost_eur": float(
            system_hourly["physical_generator_cost_eur"].sum()
        ),
        "binding_branch_hours": int(branch_hourly["binding"].sum()),
        "maximum_passive_loading_pu": maximum_loading,
        "maximum_component_balance_error_mw": component_balance_error(solved),
        "relaxed_rating_multiplier": (
            None if relaxed_rating_multiplier is None else float(relaxed_rating_multiplier)
        ),
        "lexicographic_stages": stages,
        "status_penalty_in_physical_cost": False,
        "solver_status": str(status),
        "solver_condition": str(condition),
    }
    if unserved_mwh > 1e-5:
        raise RuntimeError(f"{label} has {unserved_mwh:.6f} MWh unserved energy")
    if maximum_loading > 1.0 + 2e-5:
        raise RuntimeError(f"{label} exceeds a passive branch rating")
    if summary["maximum_component_balance_error_mw"] > 2e-4:
        raise RuntimeError(f"{label} violates component power balance")
    return ScenarioResult(
        label,
        policy,
        solved,
        summary,
        generator_hourly,
        system_hourly,
        branch_hourly,
    )


def compare_policy_results(
    neutral: ScenarioResult,
    priority: ScenarioResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return system and farm-level priority-minus-neutral differences."""
    if not neutral.network.snapshots.equals(priority.network.snapshots):
        raise ValueError("Policy results use different snapshots")
    system = pd.DataFrame([neutral.summary, priority.summary])
    common_numeric = [
        column
        for column in system.columns
        if pd.api.types.is_numeric_dtype(system[column])
    ]
    delta = {
        "scenario": f"{priority.label}_minus_{neutral.label}",
        "policy": "priority_minus_neutral",
    }
    for column in common_numeric:
        delta[column] = float(system.iloc[1][column]) - float(system.iloc[0][column])
    system = pd.concat([system, pd.DataFrame([delta])], ignore_index=True)

    keys = ["snapshot", "generator", "bus", "priority_status", "duration_h"]
    left = neutral.generator_hourly[keys + ["dispatch_down_mwh"]].rename(
        columns={"dispatch_down_mwh": "neutral_dispatch_down_mwh"}
    )
    right = priority.generator_hourly[
        keys + ["available_mw", "dispatched_mw", "dispatch_down_mwh"]
    ].rename(
        columns={
            "available_mw": "priority_available_mw",
            "dispatched_mw": "priority_dispatched_mw",
            "dispatch_down_mwh": "priority_dispatch_down_mwh",
        }
    )
    hourly = left.merge(
        right,
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    hourly["additional_dispatch_down_mwh"] = (
        hourly["priority_dispatch_down_mwh"]
        - hourly["neutral_dispatch_down_mwh"]
    )
    farm = (
        hourly.groupby(["generator", "bus", "priority_status"], as_index=False)
        .agg(
            neutral_dispatch_down_mwh=("neutral_dispatch_down_mwh", "sum"),
            priority_dispatch_down_mwh=("priority_dispatch_down_mwh", "sum"),
            additional_dispatch_down_mwh=("additional_dispatch_down_mwh", "sum"),
        )
    )
    farm["interpretation"] = np.select(
        [
            farm["additional_dispatch_down_mwh"] > 1e-6,
            farm["additional_dispatch_down_mwh"] < -1e-6,
        ],
        [
            "additional burden under priority policy",
            "energy protected under priority policy",
        ],
        default="no material allocation change",
    )
    return system, farm.sort_values(
        ["additional_dispatch_down_mwh", "generator"],
        ascending=[False, True],
    ).reset_index(drop=True)


def target_line_hotspots(
    priority_result: ScenarioResult,
    assignment: pd.DataFrame,
    *,
    reference: str = "load",
    minimum_loading_pu: float = 0.999,
) -> pd.DataFrame:
    """Explain binding branch-hours using directional PTDF relief."""
    network = priority_result.network
    frame = branches(network)
    matrix = component_adjust_reference(
        network,
        ptdf(network, frame),
        reference,
        frame,
    )
    status = assignment.set_index("generator")
    farms = wind_ids(network)
    rows: list[dict[str, Any]] = []
    active = priority_result.branch_hourly.loc[
        priority_result.branch_hourly["loading_pu"] >= float(minimum_loading_pu)
    ]
    for _, branch_hour in active.iterrows():
        branch = str(branch_hour["branch"])
        sign = int(branch_hour["flow_direction_sign"])
        if sign == 0:
            continue
        effects = []
        for generator in farms:
            bus = str(network.generators.at[generator, "bus"])
            factor = float(matrix.at[branch, bus])
            effects.append(
                {
                    "generator": generator,
                    "bus": bus,
                    "priority_status": status.at[generator, "priority_status"],
                    "shift_factor": factor,
                    "relief_mw_per_mw_curtailment": sign * factor,
                }
            )
        effect_frame = pd.DataFrame(effects)
        best_priority = effect_frame.loc[
            effect_frame["priority_status"].eq("priority")
        ].sort_values("relief_mw_per_mw_curtailment", ascending=False)
        best_nonpriority = effect_frame.loc[
            effect_frame["priority_status"].eq("non_priority")
        ].sort_values("relief_mw_per_mw_curtailment", ascending=False)
        if best_priority.empty or best_nonpriority.empty:
            continue
        p_row = best_priority.iloc[0]
        n_row = best_nonpriority.iloc[0]
        rows.append(
            {
                "snapshot": branch_hour["snapshot"],
                "branch": branch,
                "bus0": branch_hour["bus0"],
                "bus1": branch_hour["bus1"],
                "flow_mw_bus0_to_bus1": float(branch_hour["flow_mw_bus0_to_bus1"]),
                "rating_mva": float(branch_hour["rating_mva"]),
                "loading_pu": float(branch_hour["loading_pu"]),
                "flow_direction_sign": sign,
                "highest_relief_priority_generator": p_row["generator"],
                "priority_relief_mw_per_mw": float(
                    p_row["relief_mw_per_mw_curtailment"]
                ),
                "highest_relief_nonpriority_generator": n_row["generator"],
                "nonpriority_relief_mw_per_mw": float(
                    n_row["relief_mw_per_mw_curtailment"]
                ),
                "relief_advantage_priority_minus_nonpriority": float(
                    p_row["relief_mw_per_mw_curtailment"]
                    - n_row["relief_mw_per_mw_curtailment"]
                ),
                "dominated_dispatch_pair_possible": bool(
                    p_row["relief_mw_per_mw_curtailment"]
                    > n_row["relief_mw_per_mw_curtailment"] + 1e-8
                ),
            }
        )
    return pd.DataFrame(rows)
