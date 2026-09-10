"""Electrical-input identity and source-preserving North-West replay validation.

Replay proves reduction fidelity for a specified dispatch. It does not establish
grid connection permission, intervention feasibility, or merchant revenues.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

COMPONENT_TABLES = {
    "Bus": "buses", "Carrier": "carriers", "Generator": "generators",
    "Load": "loads", "StorageUnit": "storage_units", "Store": "stores",
    "ShuntImpedance": "shunt_impedances", "Line": "lines",
    "Transformer": "transformers", "Link": "links",
}
# Exact transmission buses of the official native STATIONS list, plus Firlough.
NW_BUSES = set("5042 5041 1701 17010 17061 1761 1341 5191 1571 2870 2871 "
               "28710 28712 2801 28019 4091 51911 2321 4071 3581 35861 "
               "35862 3591 5361 4991 1631 2671 4981 49861 1931 4371 4041 "
               "40461 40462 5241 5251 2601".split())
NW_BUSES.add("star:5042-5041-50421-1")


def scale_thermal_limits(network, multiplier: float, tables=("lines", "transformers")) -> None:
    """Change only allowed flow; transformer nominal-base impedance stays fixed."""
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Thermal multiplier must be finite and positive")
    for table in tables:
        static = getattr(network, table)
        if static.empty:
            continue
        static.loc[:, "s_max_pu"] = static["s_max_pu"].astype(float) * multiplier
        dynamic = getattr(network, table + "_t").s_max_pu
        if not dynamic.empty:
            getattr(network, table + "_t").s_max_pu = dynamic * multiplier


def network_input_fingerprint(network) -> str:
    """Hash loaded component tables, dynamic profiles, weights and metadata.

    Call before solving. Including all loaded values intentionally errs toward
    invalidating a cache; derived or saved result changes cannot cause a false hit.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(network.meta, sort_keys=True, default=str).encode())
    digest.update(network.snapshot_weightings.to_csv().encode())
    for table in COMPONENT_TABLES.values():
        digest.update(table.encode())
        static = getattr(network, table)
        digest.update(static.sort_index().sort_index(axis=1).to_csv().encode())
        dynamic = getattr(network, table + "_t", {})
        for name in sorted(dynamic):
            frame = dynamic[name]
            if len(frame.columns):
                digest.update(name.encode())
                digest.update(frame.sort_index(axis=1).to_csv().encode())
    return digest.hexdigest()


def passive_components(network) -> list[set[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(network.buses.index)
    for table in ("lines", "transformers"):
        frame = getattr(network, table)
        graph.add_edges_from(zip(frame.bus0, frame.bus1))
    return list(nx.connected_components(graph))


def source_bus_extract(full, buses=None):
    """Retain exact internal components; replace every crossing port by its flow.

    The full network must already be solved. All exchanges and internal one-port
    dispatch are replayed at every snapshot, so no economic slack is introduced.
    """
    selected = set(full.buses.index) & (NW_BUSES if buses is None else set(buses))
    if not selected:
        raise ValueError("No selected buses occur in the source network")
    # Detach the model only on a shallow wrapper; preserve the caller's solver
    # and use PyPSA's component/snapshot-copy route to make independent tables.
    wrapper = copy.copy(full)
    del wrapper.model
    n = wrapper.copy(snapshots=full.snapshots)
    boundary = []
    for component in ("Line", "Transformer", "Link"):
        table = COMPONENT_TABLES[component]
        static, dynamic = getattr(full, table), getattr(full, table + "_t")
        ports = [col for col in static if col.startswith("bus") and col[3:].isdigit()]
        remove = []
        for name, row in static.iterrows():
            active = [port for port in ports if pd.notna(row[port]) and str(row[port])]
            inside = [port for port in active if row[port] in selected]
            if len(inside) == len(active):
                continue
            remove.append(name)
            for port in inside:
                attr = "p" + port[3:]
                series = getattr(dynamic, attr)[name].reindex(full.snapshots)
                if not np.isfinite(series).all():
                    raise ValueError(f"Unsolved boundary {component}/{name}/{port}")
                boundary.append((component, name, port, row[port], -series))
        if remove:
            n.remove(component, remove)
    for component in ("Generator", "Load", "StorageUnit", "Store", "ShuntImpedance"):
        static = getattr(n, COMPONENT_TABLES[component])
        remove = static.index[~static.bus.isin(selected)]
        if len(remove):
            n.remove(component, remove)
    n.remove("Bus", n.buses.index.difference(list(selected)))
    # Freeze optimized dispatch for LPF, which reads p_set rather than p.
    for table in ("generators", "storage_units", "stores"):
        frame = getattr(n, table)
        if not frame.empty:
            source_p = getattr(full, table + "_t").p.reindex(columns=frame.index)
            if not np.isfinite(source_p.to_numpy()).all():
                raise ValueError(f"Unsolved internal {table}")
            getattr(n, table + "_t").p_set = source_p.copy()
    if len(n.links):
        n.links_t.p_set = full.links_t.p0[n.links.index].copy()
    rows, profiles = [], {}
    if boundary and "boundary_replay" not in n.carriers.index:
        n.add("Carrier", "boundary_replay")
    for i, (component, name, port, bus, injection) in enumerate(boundary):
        label = f"REPLAY_BOUNDARY_{i:03d}"
        nominal = max(1.0, float(injection.abs().max()))
        n.add("Generator", label, bus=bus, carrier="boundary_replay", p_nom=nominal,
              p_set=injection, p_min_pu=injection / nominal,
              p_max_pu=injection / nominal, marginal_cost=0.0)
        profiles[label] = injection
        rows.append({"generator": label, "component": component, "branch": name,
                     "inside_port": port, "inside_bus": bus,
                     "min_injection_mw": float(injection.min()),
                     "max_injection_mw": float(injection.max())})
    # Select an angle reference; it does not authorize extra real-power injection.
    n.generators.loc[:, "control"] = "PQ"
    n.buses.loc[:, "control"] = "PQ"
    for connected in passive_components(n):
        generators = n.generators.index[n.generators.bus.isin(connected)]
        if len(generators):
            n.generators.loc[generators[0], "control"] = "Slack"
    n.meta = {**full.meta, "scope": "north-west-source-bus-replay",
              "boundary_policy": "all_ports_fixed_to_hourly_full_grid_dispatch",
              "economic_or_connection_feasibility_validated": False}
    return n, pd.DataFrame(rows), pd.DataFrame(profiles, index=full.snapshots)


def validate_replay(full, extract, tolerance_mw=1e-4):
    before = extract.generators_t.p_set.copy()
    extract.lpf()
    rows = []
    for table in ("lines", "transformers"):
        names = getattr(extract, table).index
        for name in names:
            difference = getattr(extract, table + "_t").p0[name] - getattr(full, table + "_t").p0[name]
            rows.append({"component": table, "branch": name,
                         "max_absolute_flow_error_mw": float(difference.abs().max())})
    table = pd.DataFrame(rows)
    max_error = float(table.max_absolute_flow_error_mw.max()) if len(table) else 0.0
    # A missed boundary/injection can be hidden by LPF balancing at the slack.
    slack_adjustment = float((extract.generators_t.p - before).abs().max().max())
    passed = max_error <= tolerance_mw and slack_adjustment <= tolerance_mw
    return {"passed": bool(passed), "tolerance_mw": tolerance_mw,
            "max_internal_flow_error_mw": max_error,
            "max_implicit_slack_adjustment_mw": slack_adjustment,
            "source_snapshots": len(full.snapshots), "buses": len(extract.buses),
            "lines": len(extract.lines), "transformers": len(extract.transformers),
            "connected_components": len(passive_components(extract)),
            "claim": "fixed-dispatch electrical replay only",
            "site_connection_and_merchant_dispatch_validated": False}, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--network", type=Path, default=root / "data/participant-kit/networks/WP2033_all-island.nc")
    parser.add_argument("--output", type=Path, default=root / "results/network_validation/WP2033")
    parser.add_argument("--hours", type=int, default=168)
    parser.add_argument("--tolerance-mw", type=float, default=1e-4)
    args = parser.parse_args()
    if args.hours < 1 or args.tolerance_mw <= 0:
        parser.error("hours and tolerance must be positive")
    import pypsa
    full = pypsa.Network(args.network)
    full.set_snapshots(full.snapshots[:args.hours])
    identity = network_input_fingerprint(full)
    status, condition = full.optimize(solver_name="highs", solver_options={"output_flag": False}, progress=False)
    if status != "ok" or condition != "optimal":
        raise RuntimeError(f"Source optimization failed: {status}/{condition}")
    extract, crosswalk, profiles = source_bus_extract(full)
    summary, errors = validate_replay(full, extract, args.tolerance_mw)
    summary.update({"source_network": str(args.network), "source_file_sha256": hashlib.sha256(args.network.read_bytes()).hexdigest(),
                    "source_input_sha256": identity, "source_objective": float(full.objective),
                    "contains_firlough": "2601" in extract.buses.index,
                    "contains_srananagh_110_and_220": {"5041", "5042"}.issubset(extract.buses.index)})
    args.output.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(args.output / "boundary_crosswalk.csv", index=False)
    profiles.to_csv(args.output / "hourly_boundary_injections_mw.csv", index_label="snapshot")
    errors.to_csv(args.output / "internal_flow_errors.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    extract.export_to_netcdf(args.output / "fixed_dispatch_replay.nc")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
