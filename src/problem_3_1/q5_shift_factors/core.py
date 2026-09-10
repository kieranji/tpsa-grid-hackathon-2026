"""Core PTDF, wind-farm mapping, and independent LPF validation for Q5."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


EPSILON = 1e-12


def branches(network: Any) -> pd.DataFrame:
    """Return every passive AC branch using PyPSA's effective reactance."""
    network.calculate_dependent_values()
    rows: list[pd.DataFrame] = []
    for kind, frame in (("Line", network.lines), ("Transformer", network.transformers)):
        if frame.empty:
            continue
        x = frame["x_pu_eff"].to_numpy(dtype=float)
        if (~np.isfinite(x)).any() or (np.abs(x) <= EPSILON).any():
            bad = frame.index[(~np.isfinite(x)) | (np.abs(x) <= EPSILON)].tolist()
            raise ValueError(f"Invalid effective reactance on {kind} branches: {bad}")
        phase_shift = (
            np.radians(frame["phase_shift"].to_numpy(dtype=float))
            if "phase_shift" in frame.columns
            else np.zeros(len(frame), dtype=float)
        )
        rows.append(
            pd.DataFrame(
                {
                    "branch": frame.index.astype(str),
                    "kind": kind,
                    "bus0": frame["bus0"].astype(str).to_numpy(),
                    "bus1": frame["bus1"].astype(str).to_numpy(),
                    "x_pu_eff": x,
                    "susceptance": 1.0 / x,
                    "phase_shift_rad": phase_shift,
                    "s_nom_mva": frame["s_nom"].to_numpy(dtype=float),
                }
            )
        )
    if not rows:
        raise ValueError("The network has no passive AC lines or transformers")
    result = pd.concat(rows, ignore_index=True)
    duplicate = result["branch"].duplicated(keep=False)
    if duplicate.any():
        names = sorted(result.loc[duplicate, "branch"].unique())
        raise ValueError(f"Line and transformer identifiers must be unique: {names}")
    return result.set_index("branch")


def incidence(
    network: Any, branch_frame: pd.DataFrame | None = None
) -> tuple[np.ndarray, pd.Index, pd.Index]:
    """Build the bus-by-branch incidence matrix with positive bus0 orientation."""
    frame = branches(network) if branch_frame is None else branch_frame
    buses = pd.Index(network.buses.index.astype(str), name="bus")
    position = {bus: idx for idx, bus in enumerate(buses)}
    matrix = np.zeros((len(buses), len(frame)), dtype=float)
    for column, (_, row) in enumerate(frame.iterrows()):
        try:
            matrix[position[str(row["bus0"])], column] += 1.0
            matrix[position[str(row["bus1"])], column] -= 1.0
        except KeyError as exc:
            raise ValueError(f"Branch endpoint is not a network bus: {exc.args[0]}") from exc
    return matrix, buses, frame.index


def bus_components(
    network: Any, branch_frame: pd.DataFrame | None = None
) -> pd.Series:
    """Map every bus to a deterministic passive-AC connected-component ID."""
    frame = branches(network) if branch_frame is None else branch_frame
    buses = [str(bus) for bus in network.buses.index]
    parent = {bus: bus for bus in buses}

    def find(bus: str) -> str:
        while parent[bus] != bus:
            parent[bus] = parent[parent[bus]]
            bus = parent[bus]
        return bus

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for _, row in frame.iterrows():
        union(str(row["bus0"]), str(row["bus1"]))
    roots = {bus: find(bus) for bus in buses}
    ordered_roots = {
        root: number
        for number, root in enumerate(sorted(set(roots.values()), key=str))
    }
    return pd.Series(
        {bus: ordered_roots[root] for bus, root in roots.items()},
        name="ac_component",
        dtype=int,
    )


def connected_component_count(
    network: Any, branch_frame: pd.DataFrame | None = None
) -> int:
    """Count passive AC connected components, including isolated buses."""
    return int(bus_components(network, branch_frame).nunique())

def ptdf(
    network: Any, branch_frame: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return branch-by-bus PTDF values under the uniform pseudoinverse reference."""
    frame = branches(network) if branch_frame is None else branch_frame
    incidence_matrix, buses, edges = incidence(network, frame)
    susceptance = frame["susceptance"].to_numpy(dtype=float)
    laplacian = (incidence_matrix * susceptance) @ incidence_matrix.T
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    values = (susceptance[:, None] * incidence_matrix.T) @ inverse
    return pd.DataFrame(values, index=edges, columns=buses)


def mean_load_by_bus(network: Any) -> pd.Series:
    """Return non-negative mean demand by bus using dynamic profiles when present."""
    buses = pd.Index(network.buses.index.astype(str), name="bus")
    result = pd.Series(0.0, index=buses, dtype=float)
    if network.loads.empty:
        return result
    demand = network.loads["p_set"].astype(float).copy()
    dynamic = network.loads_t.p_set
    if isinstance(dynamic, pd.DataFrame) and not dynamic.empty:
        for name in dynamic.columns:
            demand.loc[name] = float(dynamic[name].mean())
    for name, value in demand.items():
        bus = str(network.loads.at[name, "bus"])
        result.loc[bus] += max(float(value), 0.0)
    return result


def reference_weights(network: Any, reference: str) -> pd.Series:
    """Return bus weights for load, uniform, or exact single-bus balancing."""
    buses = pd.Index(network.buses.index.astype(str), name="bus")
    if reference == "uniform":
        return pd.Series(1.0 / len(buses), index=buses, dtype=float)
    if reference == "load":
        weights = mean_load_by_bus(network)
        total = float(weights.sum())
        if total <= EPSILON:
            raise ValueError("A load-weighted reference requires positive demand")
        return weights / total
    if reference not in buses:
        raise ValueError(
            f"Reference must be 'load', 'uniform', or an exact bus ID; received {reference!r}"
        )
    weights = pd.Series(0.0, index=buses, dtype=float)
    weights.loc[reference] = 1.0
    return weights


def largest_load_bus(network: Any) -> str:
    """Return the exact bus ID carrying the greatest mean demand."""
    demand = mean_load_by_bus(network)
    if float(demand.sum()) <= EPSILON:
        raise ValueError("Cannot select a single slack bus because the network has no load")
    return str(demand.idxmax())


def adjust_reference(matrix: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Change a uniform-reference PTDF to the supplied distributed reference."""
    aligned = weights.reindex(matrix.columns).fillna(0.0).astype(float)
    total = float(aligned.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"Reference weights must sum to one, not {total}")
    base = matrix.to_numpy(dtype=float) @ aligned.to_numpy(dtype=float)
    values = matrix.to_numpy(dtype=float) - base[:, None]
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)



def reference_weights_for_source(
    network: Any,
    reference: str,
    source_bus: str,
    branch_frame: pd.DataFrame | None = None,
) -> pd.Series:
    """Return a balanced reference restricted to the source bus's AC component."""
    frame = branches(network) if branch_frame is None else branch_frame
    components = bus_components(network, frame)
    source_bus = str(source_bus)
    if source_bus not in components.index:
        raise ValueError(f"Source bus is absent from the network: {source_bus}")
    component = int(components.loc[source_bus])
    local_buses = components.index[components.eq(component)]
    weights = pd.Series(0.0, index=components.index, dtype=float)
    if reference in ("load", "uniform"):
        raw = reference_weights(network, reference).reindex(local_buses).fillna(0.0)
        total = float(raw.sum())
        if total <= EPSILON:
            raise ValueError(
                f"Reference {reference!r} has no balancing weight in AC component {component}"
            )
        weights.loc[local_buses] = raw / total
        return weights
    if reference not in components.index:
        raise ValueError(f"Single-slack bus is absent from the network: {reference}")
    if int(components.loc[reference]) != component:
        raise ValueError(
            f"Source bus {source_bus!r} and single-slack bus {reference!r} "
            "are in different passive AC components"
        )
    weights.loc[reference] = 1.0
    return weights


def component_adjust_reference(
    network: Any,
    matrix: pd.DataFrame,
    reference: str,
    branch_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply a reference independently within each passive AC component.

    A transfer cannot be balanced through a disconnected component. Load and
    uniform references are therefore normalized separately inside each component.
    An exact single-slack result is undefined outside the slack bus's component.
    """
    frame = branches(network) if branch_frame is None else branch_frame
    components = bus_components(network, frame)
    exact_slack = reference not in ("load", "uniform")
    fill = np.nan if exact_slack else 0.0
    result = pd.DataFrame(
        fill,
        index=matrix.index,
        columns=matrix.columns,
        dtype=float,
    )
    for component in sorted(components.unique()):
        local_buses = components.index[components.eq(component)]
        local_branches = frame.index[
            frame["bus0"].map(components).astype(int).eq(int(component))
        ]
        if not len(local_branches):
            continue
        if exact_slack and reference not in local_buses:
            continue
        weights = reference_weights_for_source(
            network,
            reference,
            str(local_buses[0]),
            frame,
        )
        block = matrix.loc[local_branches, local_buses]
        base = (
            matrix.loc[local_branches].to_numpy(dtype=float)
            @ weights.reindex(matrix.columns).fillna(0.0).to_numpy(dtype=float)
        )
        result.loc[local_branches, local_buses] = block.to_numpy(dtype=float) - base[:, None]
        if not exact_slack:
            outside = matrix.columns.difference(local_buses)
            result.loc[local_branches, outside] = 0.0
    return result


def maximum_reference_residual(
    network: Any,
    adjusted_matrix: pd.DataFrame,
    reference: str,
    branch_frame: pd.DataFrame | None = None,
) -> float:
    """Return the largest component-local weighted mean after reference adjustment."""
    frame = branches(network) if branch_frame is None else branch_frame
    components = bus_components(network, frame)
    residuals: list[float] = []
    for component in sorted(components.unique()):
        local_buses = components.index[components.eq(component)]
        local_branches = frame.index[
            frame["bus0"].map(components).astype(int).eq(int(component))
        ]
        if not len(local_branches):
            continue
        if reference not in ("load", "uniform") and reference not in local_buses:
            continue
        weights = reference_weights_for_source(
            network,
            reference,
            str(local_buses[0]),
            frame,
        )
        values = (
            adjusted_matrix.loc[local_branches].fillna(0.0).to_numpy(dtype=float)
            @ weights.reindex(adjusted_matrix.columns).fillna(0.0).to_numpy(dtype=float)
        )
        residuals.extend(np.abs(values).tolist())
    return float(max(residuals, default=0.0))
def wind_farms(network: Any, carriers: Iterable[str] = ("wind",)) -> pd.DataFrame:
    """Map each modeled wind farm to its model-assigned connection substation."""
    allowed = {str(value).strip().lower() for value in carriers}
    generators = network.generators.copy()
    keep = generators["carrier"].astype(str).str.lower().isin(allowed)
    selected = generators.loc[keep].copy()
    if selected.empty:
        raise ValueError(f"No wind generators found for carriers {sorted(allowed)}")
    missing = sorted(set(selected["bus"].astype(str)) - set(network.buses.index.astype(str)))
    if missing:
        raise ValueError(f"Wind generators reference unknown buses: {missing}")
    rows = []
    for name, row in selected.iterrows():
        bus = str(row["bus"])
        bus_row = network.buses.loc[bus]
        rows.append(
            {
                "wind_farm": str(name),
                "nearest_substation": bus,
                "mapping_method": "official_model_generator_bus_assignment",
                "carrier": str(row["carrier"]),
                "installed_capacity_mw": float(row["p_nom"]),
                "substation_voltage_kv": float(bus_row.get("v_nom", np.nan)),
                "longitude": float(bus_row.get("x", np.nan)),
                "latitude": float(bus_row.get("y", np.nan)),
            }
        )
    return pd.DataFrame(rows).set_index("wind_farm")


def wind_factor_matrix(
    adjusted_matrix: pd.DataFrame, farm_table: pd.DataFrame
) -> pd.DataFrame:
    """Select PTDF columns at each farm's connection bus and retain farm labels."""
    bus_ids = farm_table["nearest_substation"].astype(str).tolist()
    missing = sorted(set(bus_ids) - set(adjusted_matrix.columns.astype(str)))
    if missing:
        raise ValueError(f"Farm connection buses are absent from the PTDF: {missing}")
    values = adjusted_matrix.reindex(columns=bus_ids).to_numpy(dtype=float).T
    return pd.DataFrame(values, index=farm_table.index, columns=adjusted_matrix.index)


def factor_tables(
    network: Any,
    monitored_lines: Iterable[str],
    farm_table: pd.DataFrame,
    references: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Calculate base PTDF, all reference-adjusted matrices, and farm factors."""
    frame = branches(network)
    base = ptdf(network, frame)
    monitored = [str(line) for line in monitored_lines]
    missing = sorted(set(monitored) - set(frame.index.astype(str)))
    if missing:
        raise ValueError(f"Monitored branches are absent from the network: {missing}")
    adjusted: dict[str, pd.DataFrame] = {}
    farm_matrices: dict[str, pd.DataFrame] = {}
    for label, reference in references.items():
        matrix = component_adjust_reference(network, base, reference, frame)
        adjusted[label] = matrix
        farm_matrices[label] = wind_factor_matrix(matrix, farm_table)
    return frame, farm_matrices, base


def _flow_vector(network: Any, snapshot: Any) -> pd.Series:
    line_flow = (
        network.lines_t.p0.loc[snapshot]
        if not network.lines.empty
        else pd.Series(dtype=float)
    )
    transformer_flow = (
        network.transformers_t.p0.loc[snapshot]
        if not network.transformers.empty
        else pd.Series(dtype=float)
    )
    return pd.concat([line_flow, transformer_flow]).astype(float)


def _zero_dispatch_inputs(network: Any) -> None:
    for plural in ("generators", "loads", "storage_units", "stores", "links"):
        frame = getattr(network, plural, None)
        if isinstance(frame, pd.DataFrame) and "p_set" in frame.columns:
            frame.loc[:, "p_set"] = 0.0
        dynamic = getattr(network, f"{plural}_t", None)
        values = getattr(dynamic, "p_set", None) if dynamic is not None else None
        if isinstance(values, pd.DataFrame) and not values.empty:
            values.loc[:, :] = 0.0


def lpf_transfer_delta(
    network: Any,
    source_bus: str,
    weights: pd.Series,
    delta_mw: float = 1.0,
) -> pd.Series:
    """Independently measure a balanced transfer with two PyPSA linear power flows."""
    if delta_mw <= 0:
        raise ValueError("Finite-difference delta must be positive")
    probe = network.copy()
    _zero_dispatch_inputs(probe)
    snapshot = probe.snapshots[0]
    if not probe.generators["control"].astype(str).eq("Slack").any():
        probe.add(
            "Generator",
            "__q5_validation_slack__",
            bus=str(weights.idxmax()),
            control="Slack",
            p_set=0.0,
            p_nom=0.0,
        )
    source_name = "__q5_validation_source__"
    probe.add(
        "Generator",
        source_name,
        bus=str(source_bus),
        control="PQ",
        p_set=0.0,
        p_nom=max(delta_mw, 1.0),
    )
    sink_names: dict[str, str] = {}
    for number, (bus, weight) in enumerate(weights.items()):
        if float(weight) <= EPSILON:
            continue
        name = f"__q5_validation_sink_{number}__"
        probe.add("Load", name, bus=str(bus), p_set=0.0)
        sink_names[name] = str(bus)
    probe.lpf(snapshots=[snapshot])
    baseline = _flow_vector(probe, snapshot)
    probe.generators.at[source_name, "p_set"] = float(delta_mw)
    for name, bus in sink_names.items():
        probe.loads.at[name, "p_set"] = float(delta_mw * weights.loc[bus])
    probe.lpf(snapshots=[snapshot])
    perturbed = _flow_vector(probe, snapshot)
    return (perturbed - baseline).reindex(baseline.index) / float(delta_mw)


def validate_with_lpf(
    network: Any,
    monitored_lines: Iterable[str],
    farm_table: pd.DataFrame,
    expected_factors: pd.DataFrame,
    reference: str,
    delta_mw: float,
    tolerance: float,
) -> pd.DataFrame:
    """Validate every farm's primary-reference factor against PyPSA LPF."""
    monitored = [str(line) for line in monitored_lines]
    frame = branches(network)
    rows: list[dict[str, Any]] = []
    for farm, row in farm_table.iterrows():
        source_bus = str(row["nearest_substation"])
        weights = reference_weights_for_source(
            network,
            reference,
            source_bus,
            frame,
        )
        measured = lpf_transfer_delta(
            network,
            source_bus,
            weights,
            delta_mw=delta_mw,
        )
        for line in monitored:
            expected = float(expected_factors.at[farm, line])
            actual = float(measured.loc[line])
            error = actual - expected
            rows.append(
                {
                    "wind_farm": farm,
                    "nearest_substation": source_bus,
                    "source_ac_component": int(bus_components(network, frame).loc[source_bus]),
                    "monitored_line": line,
                    "delta_mw": float(delta_mw),
                    "analytical_shift_factor": expected,
                    "lpf_finite_difference_shift_factor": actual,
                    "error": error,
                    "absolute_error": abs(error),
                    "tolerance": float(tolerance),
                    "passed": bool(abs(error) <= tolerance),
                }
            )
    return pd.DataFrame(rows)

def rating_invariance(
    network: Any,
    monitored_lines: Iterable[str],
    farm_table: pd.DataFrame,
    reference: str,
    multiplier: float,
) -> pd.DataFrame:
    """Show that rating-only DLR changes headroom, not network PTDF values."""
    if multiplier <= 0:
        raise ValueError("Rating multiplier must be positive")
    monitored = [str(line) for line in monitored_lines]
    original_frame = branches(network)
    original = component_adjust_reference(
        network, ptdf(network, original_frame), reference, original_frame
    )
    modified = network.copy()
    for line in monitored:
        if line in modified.lines.index:
            modified.lines.at[line, "s_nom"] *= float(multiplier)
        elif line in modified.transformers.index:
            modified.transformers.at[line, "s_nom"] *= float(multiplier)
        else:
            raise KeyError(line)
    changed_frame = branches(modified)
    changed = component_adjust_reference(
        modified, ptdf(modified, changed_frame), reference, changed_frame
    )
    original_farms = wind_factor_matrix(original, farm_table)
    changed_farms = wind_factor_matrix(changed, farm_table)
    rows = []
    for line in monitored:
        delta = changed_farms[line] - original_farms[line]
        rows.append(
            {
                "monitored_line": line,
                "rating_multiplier_tested": float(multiplier),
                "maximum_absolute_shift_factor_change": float(delta.abs().max()),
                "mean_absolute_shift_factor_change": float(delta.abs().mean()),
                "ptdf_unchanged": bool(float(delta.abs().max()) <= 1e-12),
                "interpretation": "DLR changes thermal capacity and headroom, not topology or reactance",
            }
        )
    return pd.DataFrame(rows)


def branch_rating_series(network: Any, branch: str) -> pd.Series:
    """Return the effective thermal rating for each snapshot."""
    if branch in network.lines.index:
        frame, dynamic = network.lines, network.lines_t.s_max_pu
    elif branch in network.transformers.index:
        frame, dynamic = network.transformers, network.transformers_t.s_max_pu
    else:
        raise KeyError(branch)
    static_factor = float(frame.at[branch, "s_max_pu"]) if "s_max_pu" in frame.columns else 1.0
    factors = pd.Series(static_factor, index=network.snapshots, dtype=float)
    if isinstance(dynamic, pd.DataFrame) and branch in dynamic.columns:
        factors = dynamic[branch].reindex(network.snapshots).fillna(static_factor).astype(float)
    return factors * float(frame.at[branch, "s_nom"])


def operational_context(
    solved_network: Any,
    monitored_lines: Iterable[str],
    binding_tolerance_pu: float,
) -> pd.DataFrame:
    """Summarize observed direction and binding hours after dispatch and LPF."""
    rows = []
    for line in [str(value) for value in monitored_lines]:
        if line in solved_network.lines.index:
            static = solved_network.lines.loc[line]
            flow = solved_network.lines_t.p0[line].astype(float)
            kind = "Line"
        elif line in solved_network.transformers.index:
            static = solved_network.transformers.loc[line]
            flow = solved_network.transformers_t.p0[line].astype(float)
            kind = "Transformer"
        else:
            raise KeyError(line)
        rating = branch_rating_series(solved_network, line)
        loading = flow.abs() / rating.replace(0.0, np.nan)
        binding = loading >= (1.0 - float(binding_tolerance_pu))
        relevant = flow.loc[binding] if binding.any() else flow.loc[[flow.abs().idxmax()]]
        sign = int(np.sign(float(relevant.median())))
        peak = loading.idxmax()
        if sign > 0:
            direction = f"{static['bus0']} to {static['bus1']}"
        elif sign < 0:
            direction = f"{static['bus1']} to {static['bus0']}"
        else:
            direction = "no dominant direction"
        rows.append(
            {
                "monitored_line": line,
                "component": kind,
                "bus0": str(static["bus0"]),
                "bus1": str(static["bus1"]),
                "nominal_rating_mva": float(static["s_nom"]),
                "binding_hours": int(binding.sum()),
                "positive_binding_hours": int((flow.loc[binding] > 0).sum()),
                "negative_binding_hours": int((flow.loc[binding] < 0).sum()),
                "dominant_binding_flow_sign": sign,
                "dominant_binding_flow_direction": direction,
                "maximum_absolute_flow_mw": float(flow.abs().max()),
                "maximum_loading_pu": float(loading.max()),
                "peak_snapshot": str(peak),
                "flow_at_peak_mw_bus0_to_bus1": float(flow.loc[peak]),
                "rating_at_peak_mva": float(rating.loc[peak]),
            }
        )
    return pd.DataFrame(rows)
