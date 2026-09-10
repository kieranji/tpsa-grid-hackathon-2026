"""Electrical core for nationwide, overlapping constraint groups.

Constraint groups are directional branch operating modes. Electrical response zones
are mutually exclusive bus clusters built from reference-invariant response
*differences*. Curtailment relief values remain reference dependent and are always
reported with their balancing convention.
"""
from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from problem_3_1.q5_shift_factors.core import (
    branch_rating_series,
    branches,
    bus_components,
    lpf_transfer_delta,
    reference_weights_for_source,
)

EPSILON = 1e-12


def snapshot_hours(network: Any) -> pd.Series:
    """Return objective snapshot weights as physical hours."""
    weights = network.snapshot_weightings
    if isinstance(weights, pd.DataFrame):
        column = "objective" if "objective" in weights.columns else weights.columns[0]
        result = weights[column]
    else:
        result = weights
    result = pd.Series(result, index=network.snapshots, dtype=float)
    if (result <= 0).any() or (~np.isfinite(result)).any():
        raise ValueError("Snapshot hours must be finite and positive")
    return result


def passive_flows(network: Any, branch_frame: pd.DataFrame) -> pd.DataFrame:
    """Return snapshot-by-branch bus0-oriented active power flows."""
    pieces: list[pd.DataFrame] = []
    if not network.lines.empty:
        pieces.append(network.lines_t.p0.reindex(index=network.snapshots, columns=network.lines.index))
    if not network.transformers.empty:
        pieces.append(
            network.transformers_t.p0.reindex(
                index=network.snapshots, columns=network.transformers.index
            )
        )
    if not pieces:
        raise ValueError("No passive branch flow results are available")
    result = pd.concat(pieces, axis=1).reindex(columns=branch_frame.index).astype(float)
    if result.isna().any().any():
        raise ValueError("Passive branch flow results contain missing values")
    return result


def passive_ratings(network: Any, branch_frame: pd.DataFrame) -> pd.DataFrame:
    """Return snapshot-by-branch effective thermal ratings."""
    data = {
        str(branch): branch_rating_series(network, str(branch)).reindex(network.snapshots)
        for branch in branch_frame.index
    }
    result = pd.DataFrame(data, index=network.snapshots, dtype=float)
    if (result <= 0).any().any() or (~np.isfinite(result)).any().any():
        raise ValueError("Passive branch ratings must be finite and positive")
    return result


def directional_modes(
    network: Any,
    branch_frame: pd.DataFrame,
    *,
    near_threshold_pu: float,
    binding_threshold_pu: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect positive and negative near-congestion modes and event weights."""
    if not 0 < near_threshold_pu < binding_threshold_pu <= 1.0:
        raise ValueError("Require 0 < near threshold < binding threshold <= 1")
    flows = passive_flows(network, branch_frame)
    ratings = passive_ratings(network, branch_frame)
    loading = flows.abs() / ratings
    hours = snapshot_hours(network)
    components = bus_components(network, branch_frame)
    rows: list[dict[str, Any]] = []
    weights: dict[str, pd.Series] = {}
    for branch, static in branch_frame.iterrows():
        for sign in (1, -1):
            directional = flows[branch] * sign > EPSILON
            event = directional & loading[branch].ge(near_threshold_pu)
            if not bool(event.any()):
                continue
            severity = ((loading[branch] - near_threshold_pu) / (1.0 - near_threshold_pu)).clip(0.0, 1.0)
            event_weight = (severity * hours).where(event, 0.0)
            if float(event_weight.sum()) <= EPSILON:
                event_weight = hours.where(event, 0.0) * EPSILON
            mode_id = f"{branch}::{sign:+d}"
            weights[mode_id] = event_weight
            if sign > 0:
                direction = f"{static['bus0']} to {static['bus1']}"
            else:
                direction = f"{static['bus1']} to {static['bus0']}"
            rows.append(
                {
                    "mode_id": mode_id,
                    "branch": str(branch),
                    "branch_kind": str(static["kind"]),
                    "bus0": str(static["bus0"]),
                    "bus1": str(static["bus1"]),
                    "flow_sign": int(sign),
                    "flow_direction": direction,
                    "ac_component": int(components.loc[str(static["bus0"])]),
                    "near_congestion_hours": float(hours.loc[event].sum()),
                    "binding_hours": float(
                        hours.loc[event & loading[branch].ge(binding_threshold_pu)].sum()
                    ),
                    "event_count": int(event.sum()),
                    "event_weighted_severity_hours": float(event_weight.sum()),
                    "maximum_loading_pu": float(loading.loc[event, branch].max()),
                    "mean_rating_mva_during_events": float(ratings.loc[event, branch].mean()),
                }
            )
    modes = pd.DataFrame(rows)
    if modes.empty:
        columns = ["mode_id", "branch", "flow_sign", "ac_component"]
        return pd.DataFrame(columns=columns), pd.DataFrame(index=network.snapshots)
    modes = modes.sort_values(
        ["event_weighted_severity_hours", "mode_id"], ascending=[False, True]
    ).reset_index(drop=True)
    event_weights = pd.DataFrame(weights, index=network.snapshots).reindex(columns=modes["mode_id"])
    return modes, event_weights


def directional_relief(
    adjusted_ptdf: pd.DataFrame, modes: pd.DataFrame
) -> pd.DataFrame:
    """Return MW branch relief per MW curtailment or charging for each mode."""
    if modes.empty:
        return pd.DataFrame(columns=adjusted_ptdf.columns, dtype=float)
    rows = []
    for row in modes.itertuples(index=False):
        if row.branch not in adjusted_ptdf.index:
            raise KeyError(row.branch)
        rows.append(adjusted_ptdf.loc[row.branch].astype(float) * int(row.flow_sign))
    result = pd.DataFrame(rows, index=modes["mode_id"].astype(str), columns=adjusted_ptdf.columns)
    result.index.name = "mode_id"
    return result


def normalized_positive_relief(relief: pd.DataFrame) -> pd.DataFrame:
    """Scale positive directional relief to one independently for each mode."""
    positive = relief.clip(lower=0.0)
    maxima = positive.max(axis=1)
    return positive.div(maxima.where(maxima > EPSILON), axis=0).fillna(0.0)


def constraint_memberships(
    modes: pd.DataFrame,
    relief: pd.DataFrame,
    threshold: float,
    reference_label: str,
) -> pd.DataFrame:
    """Return the long-form overlapping constraint-group membership table."""
    if not 0 <= threshold <= 1:
        raise ValueError("Constraint-group threshold must be between zero and one")
    normalized = normalized_positive_relief(relief)
    mode_meta = modes.set_index("mode_id")
    records: list[dict[str, Any]] = []
    for mode_id in relief.index:
        members = normalized.loc[mode_id].ge(threshold) & normalized.loc[mode_id].gt(0.0)
        for bus in relief.columns[members]:
            records.append(
                {
                    "mode_id": str(mode_id),
                    "branch": str(mode_meta.at[mode_id, "branch"]),
                    "flow_sign": int(mode_meta.at[mode_id, "flow_sign"]),
                    "ac_component": int(mode_meta.at[mode_id, "ac_component"]),
                    "bus": str(bus),
                    "relief_mw_per_mw_curtailment_or_charge": float(relief.at[mode_id, bus]),
                    "normalized_positive_relief": float(normalized.at[mode_id, bus]),
                    "membership_threshold": float(threshold),
                    "balancing_reference": str(reference_label),
                    "membership_type": "overlapping_directional_constraint_group",
                }
            )
    return pd.DataFrame(records)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > EPSILON else 0.0


def merge_similar_modes(
    modes: pd.DataFrame,
    relief: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    jaccard_threshold: float,
    cosine_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge redundant modes using a deterministic complete-link threshold."""
    member_sets = {
        str(mode): set(group["bus"].astype(str))
        for mode, group in memberships.groupby("mode_id")
    }
    meta = modes.set_index("mode_id")
    ids = modes["mode_id"].astype(str).tolist()
    positive = relief.clip(lower=0.0)
    pair_records: list[dict[str, Any]] = []
    qualifies: dict[tuple[str, str], bool] = {}
    similarity: dict[tuple[str, str], tuple[float, float]] = {}
    for left, right in combinations(ids, 2):
        if int(meta.at[left, "ac_component"]) != int(meta.at[right, "ac_component"]):
            continue
        left_members = member_sets.get(left, set())
        right_members = member_sets.get(right, set())
        union_size = len(left_members | right_members)
        jaccard = len(left_members & right_members) / union_size if union_size else 0.0
        cosine = _cosine(positive.loc[left].to_numpy(), positive.loc[right].to_numpy())
        key = tuple(sorted((left, right)))
        passes = jaccard >= jaccard_threshold and cosine >= cosine_threshold
        qualifies[key] = passes
        similarity[key] = (float(jaccard), float(cosine))
        pair_records.append(
            {
                "left_mode_id": left,
                "right_mode_id": right,
                "ac_component": int(meta.at[left, "ac_component"]),
                "membership_jaccard": float(jaccard),
                "positive_relief_cosine": float(cosine),
                "pair_passes_thresholds": bool(passes),
            }
        )

    clusters: list[list[str]] = []
    for mode in ids:
        compatible: list[tuple[float, float, int]] = []
        for number, cluster in enumerate(clusters):
            if int(meta.at[mode, "ac_component"]) != int(meta.at[cluster[0], "ac_component"]):
                continue
            keys = [tuple(sorted((mode, other))) for other in cluster]
            if all(qualifies.get(key, False) for key in keys):
                values = [similarity[key] for key in keys]
                compatible.append(
                    (
                        min(value[1] for value in values),
                        min(value[0] for value in values),
                        number,
                    )
                )
        if compatible:
            _, _, selected = max(compatible, key=lambda value: (value[0], value[1], -value[2]))
            clusters[selected].append(mode)
        else:
            clusters.append([mode])

    clusters = sorted(clusters, key=lambda values: min(values))
    group_by_mode: dict[str, str] = {}
    for number, cluster in enumerate(clusters, 1):
        group = f"MCG-{number:04d}"
        for mode in cluster:
            group_by_mode[mode] = group
    mapping = modes.copy()
    mapping["merged_constraint_group_id"] = mapping["mode_id"].astype(str).map(group_by_mode)
    mapping["merged_mode_count"] = mapping.groupby("merged_constraint_group_id")["mode_id"].transform("size")
    pair_frame = pd.DataFrame(pair_records)
    if len(pair_frame):
        pair_frame["merged"] = [
            bool(
                row.pair_passes_thresholds
                and group_by_mode[str(row.left_mode_id)] == group_by_mode[str(row.right_mode_id)]
            )
            for row in pair_frame.itertuples(index=False)
        ]
    return mapping, pair_frame


def _silhouette(distance: np.ndarray, labels: np.ndarray) -> float:
    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0
    scores = []
    for idx, label in enumerate(labels):
        same = np.flatnonzero(labels == label)
        same = same[same != idx]
        if len(same) == 0:
            scores.append(0.0)
            continue
        a = float(distance[idx, same].mean())
        b = min(float(distance[idx, labels == other].mean()) for other in unique if other != label)
        scores.append((b - a) / max(a, b, EPSILON))
    return float(np.mean(scores))


def response_zones(
    components: pd.Series,
    modes: pd.DataFrame,
    relief: pd.DataFrame,
    event_weights: pd.DataFrame,
    *,
    maximum_zones_per_component: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cluster buses into exclusive zones using reference-invariant response differences."""
    if maximum_zones_per_component < 1:
        raise ValueError("maximum_zones_per_component must be positive")
    feature_blocks: list[pd.DataFrame] = []
    bus_records: list[dict[str, Any]] = []
    zone_records: list[dict[str, Any]] = []
    mode_meta = modes.set_index("mode_id") if not modes.empty else pd.DataFrame()
    for component in sorted(components.unique()):
        buses = components.index[components.eq(component)].astype(str)
        local_modes = [] if modes.empty else [
            mode for mode in relief.index
            if int(mode_meta.at[mode, "ac_component"]) == int(component)
        ]
        if local_modes:
            block = relief.loc[local_modes, buses].T.astype(float)
            block = block - block.mean(axis=0)
            severity = event_weights[local_modes].sum(axis=0).astype(float)
            severity = severity / max(float(severity.sum()), EPSILON)
            block = block.mul(np.sqrt(severity), axis=1)
        else:
            block = pd.DataFrame(0.0, index=buses, columns=["NO_ACTIVE_CONSTRAINT_MODE"])
        feature_blocks.append(block.reindex(columns=relief.index, fill_value=0.0))
        values = block.to_numpy(dtype=float)
        n_bus = len(buses)
        distance = cdist(values, values, metric="euclidean")
        best_labels = np.ones(n_bus, dtype=int)
        best_score = 0.0
        candidate_scores = {1: 0.0}
        if n_bus >= 3 and float(np.ptp(values, axis=0).max(initial=0.0)) > EPSILON:
            tree = linkage(values, method="average", metric="euclidean")
            max_k = min(maximum_zones_per_component, n_bus - 1, max(2, int(round(np.sqrt(n_bus)))))
            for k in range(2, max_k + 1):
                labels = fcluster(tree, k, criterion="maxclust").astype(int)
                score = _silhouette(distance, labels)
                candidate_scores[k] = score
                if score > best_score + 1e-12:
                    best_score, best_labels = score, labels
        if best_score < 0.05:
            best_labels = np.ones(n_bus, dtype=int)
            best_score = 0.0
        relabel = {old: new for new, old in enumerate(sorted(np.unique(best_labels)), 1)}
        best_labels = np.array([relabel[value] for value in best_labels], dtype=int)
        for bus, label, norm in zip(buses, best_labels, np.linalg.norm(values, axis=1)):
            zone_id = f"C{int(component):03d}-Z{int(label):03d}"
            bus_records.append(
                {
                    "bus": str(bus),
                    "ac_component": int(component),
                    "response_zone_id": zone_id,
                    "component_zone_count": int(len(np.unique(best_labels))),
                    "component_silhouette": float(best_score),
                    "feature_norm": float(norm),
                    "active_mode_count": int(len(local_modes)),
                    "clustering_basis": "centered_event_weighted_directional_ptdf",
                }
            )
        for label in sorted(np.unique(best_labels)):
            selected = [str(bus) for bus, value in zip(buses, best_labels) if value == label]
            zone_records.append(
                {
                    "response_zone_id": f"C{int(component):03d}-Z{int(label):03d}",
                    "ac_component": int(component),
                    "bus_count": int(len(selected)),
                    "buses": ";".join(selected),
                    "component_silhouette": float(best_score),
                    "candidate_silhouettes": ";".join(
                        f"{key}:{value:.6f}" for key, value in sorted(candidate_scores.items())
                    ),
                }
            )
    bus_zones = pd.DataFrame(bus_records)
    features = pd.concat(feature_blocks, axis=0).reindex(index=components.index.astype(str), columns=relief.index, fill_value=0.0)
    features.index.name = "bus"
    return bus_zones, features, pd.DataFrame(zone_records)


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in kilometres."""
    radius = 6371.0088
    phi1, phi2 = np.radians([lat1, lat2])
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    value = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2.0 * radius * np.arcsin(np.sqrt(min(1.0, max(0.0, value)))))


def nonlocal_similar_pairs(
    buses: pd.DataFrame,
    components: pd.Series,
    zones: pd.DataFrame,
    features: pd.DataFrame,
    *,
    minimum_distance_km: float,
    minimum_cosine_similarity: float,
    limit: int,
) -> pd.DataFrame:
    """Find geographically distant buses with similar electrical responses."""
    zone_by_bus = zones.set_index("bus")["response_zone_id"]
    records: list[dict[str, Any]] = []
    for component in sorted(components.unique()):
        local = components.index[components.eq(component)].astype(str).tolist()
        for left, right in combinations(local, 2):
            coords = [
                float(buses.at[left, "x"]), float(buses.at[left, "y"]),
                float(buses.at[right, "x"]), float(buses.at[right, "y"]),
            ]
            if not all(np.isfinite(coords)):
                continue
            geographic = haversine_km(*coords)
            if geographic < minimum_distance_km:
                continue
            left_vector = features.loc[left].to_numpy(dtype=float)
            right_vector = features.loc[right].to_numpy(dtype=float)
            cosine = _cosine(left_vector, right_vector)
            if cosine < minimum_cosine_similarity:
                continue
            records.append(
                {
                    "bus_a": left,
                    "bus_b": right,
                    "ac_component": int(component),
                    "response_zone_a": str(zone_by_bus.loc[left]),
                    "response_zone_b": str(zone_by_bus.loc[right]),
                    "same_response_zone": bool(zone_by_bus.loc[left] == zone_by_bus.loc[right]),
                    "geographic_distance_km": float(geographic),
                    "electrical_response_cosine": float(cosine),
                    "electrical_response_distance": float(np.linalg.norm(left_vector - right_vector)),
                }
            )
    if not records:
        return pd.DataFrame(columns=[
            "bus_a", "bus_b", "ac_component", "response_zone_a", "response_zone_b",
            "same_response_zone", "geographic_distance_km",
            "electrical_response_cosine", "electrical_response_distance",
        ])
    return pd.DataFrame(records).sort_values(
        ["electrical_response_cosine", "geographic_distance_km"], ascending=[False, False]
    ).head(limit).reset_index(drop=True)


def reference_sensitivity(
    modes: pd.DataFrame,
    relief_a: pd.DataFrame,
    relief_b: pd.DataFrame,
    components: pd.Series,
    threshold: float,
    label_a: str,
    label_b: str,
) -> pd.DataFrame:
    """Compare reference-dependent membership and invariant response differences."""
    norm_a = normalized_positive_relief(relief_a)
    norm_b = normalized_positive_relief(relief_b)
    meta = modes.set_index("mode_id")
    rows = []
    for mode in relief_a.index:
        members_a = set(norm_a.columns[(norm_a.loc[mode] >= threshold) & (norm_a.loc[mode] > 0)])
        members_b = set(norm_b.columns[(norm_b.loc[mode] >= threshold) & (norm_b.loc[mode] > 0)])
        union = members_a | members_b
        component = int(meta.at[mode, "ac_component"])
        local_buses = components.index[components.eq(component)].astype(str)
        raw_delta = relief_a.loc[mode, local_buses] - relief_b.loc[mode, local_buses]
        centered_a = relief_a.loc[mode, local_buses] - relief_a.loc[mode, local_buses].mean()
        centered_b = relief_b.loc[mode, local_buses] - relief_b.loc[mode, local_buses].mean()
        rows.append(
            {
                "mode_id": str(mode),
                "branch": str(meta.at[mode, "branch"]),
                "flow_sign": int(meta.at[mode, "flow_sign"]),
                "reference_a": label_a,
                "reference_b": label_b,
                "member_count_a": int(len(members_a)),
                "member_count_b": int(len(members_b)),
                "membership_jaccard": float(len(members_a & members_b) / len(union)) if union else 1.0,
                "maximum_absolute_raw_relief_change": float(raw_delta.abs().max()),
                "maximum_absolute_centered_response_change": float((centered_a - centered_b).abs().max()),
                "interpretation": "membership is reference dependent; pairwise response differences are invariant",
            }
        )
    return pd.DataFrame(rows)


def lpf_sample_validation(
    network: Any,
    branch_frame: pd.DataFrame,
    adjusted_ptdf: pd.DataFrame,
    modes: pd.DataFrame,
    source_buses: Iterable[str],
    *,
    reference: str,
    delta_mw: float,
) -> pd.DataFrame:
    """Validate sampled national PTDF entries with independent PyPSA LPF probes."""
    rows = []
    unique_branches = modes.drop_duplicates("branch")["branch"].astype(str).tolist()
    for source in [str(value) for value in source_buses]:
        weights = reference_weights_for_source(network, reference, source, branch_frame)
        measured = lpf_transfer_delta(network, source, weights, delta_mw=delta_mw)
        for branch in unique_branches:
            expected = float(adjusted_ptdf.at[branch, source])
            actual = float(measured.at[branch])
            rows.append(
                {
                    "source_bus": source,
                    "branch": branch,
                    "reference": reference,
                    "delta_mw": float(delta_mw),
                    "analytical_ptdf": expected,
                    "lpf_finite_difference_ptdf": actual,
                    "error": actual - expected,
                    "absolute_error": abs(actual - expected),
                }
            )
    return pd.DataFrame(rows)
