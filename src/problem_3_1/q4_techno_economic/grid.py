"""PyPSA adapter: preserve the supplied topology and original system-cost objective.

This is operator-dispatched, fixed-investment siting. External market prices are
used in settlement, NOT substituted for the official generator costs. Therefore
NPV is conditional on this dispatch policy; it is not a strategic SEM bidding model.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from .types import Design, Dispatch

WIND_NAME = "Q4_NEW_WIND"
BATTERY_NAME = "Q4_NEW_BESS"


def dense_attribute(n, table: str, attr: str) -> pd.DataFrame:
    """Static default broadcast, explicitly overwritten by time-series columns."""
    static = getattr(n, table)
    result = pd.DataFrame(np.tile(static[attr].to_numpy(), (len(n.snapshots), 1)),
                          index=n.snapshots, columns=static.index, dtype=float)
    dynamic = getattr(getattr(n, table + "_t"), attr)
    for col in dynamic.columns.intersection(static.index):
        result[col] = dynamic[col].reindex(n.snapshots).to_numpy()
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"Missing/nonfinite {table}.{attr}; no silent profile filling")
    return result


def aligned_csv(path: Path, snapshots: pd.Index, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "snapshot" not in frame or not set(columns).issubset(frame.columns):
        raise ValueError(f"{path}: required columns: snapshot, {', '.join(columns)}")
    # Naive kit timestamps are treated as UTC labels. Never align by row position.
    frame["snapshot"] = pd.to_datetime(frame["snapshot"], utc=True, errors="raise")
    if frame["snapshot"].duplicated().any():
        raise ValueError(f"{path}: duplicate timestamps")
    target = pd.to_datetime(snapshots, utc=True)
    frame = frame.set_index("snapshot").reindex(target)[columns].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError(f"{path}: missing/nonfinite values for the selected snapshots")
    return frame


def apply_dlr_multiplier_csv(n, path: Path, root: Path) -> tuple[pd.DataFrame, str]:
    source = Path(path)
    if not source.is_absolute():
        source = root / source
    if not source.is_file():
        raise FileNotFoundError(source)
    raw = pd.read_csv(source)
    if "snapshot" not in raw.columns:
        raise ValueError(f"{source}: DLR multiplier CSV requires a snapshot column")
    columns = [str(c) for c in raw.columns if c != "snapshot"]
    if not columns:
        raise ValueError(f"{source}: DLR multiplier CSV has no line columns")
    missing = set(columns) - set(map(str, n.lines.index))
    if missing:
        raise ValueError(f"{source}: DLR lines not in network: {sorted(missing)}")
    raw["snapshot"] = pd.to_datetime(raw["snapshot"], utc=True, errors="raise")
    if raw["snapshot"].duplicated().any():
        raise ValueError(f"{source}: duplicate DLR timestamps")
    target = pd.to_datetime(n.snapshots, utc=True)
    multipliers = raw.set_index("snapshot").reindex(target)[columns].astype(float)
    if not np.isfinite(multipliers.to_numpy()).all() or np.any(multipliers.to_numpy() < 1.0):
        raise ValueError(f"{source}: DLR multipliers must be finite and at least 1.0")
    static = n.lines.get("s_max_pu", pd.Series(1.0, index=n.lines.index)).astype(float)
    applied = pd.DataFrame(np.tile(static.to_numpy(), (len(n.snapshots), 1)),
                           index=n.snapshots, columns=n.lines.index, dtype=float)
    existing = getattr(n.lines_t, "s_max_pu", pd.DataFrame())
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        existing = existing.reindex(index=n.snapshots)
        for line in existing.columns.intersection(applied.columns):
            applied[line] = existing[line].fillna(static[line])
    for line in columns:
        applied[line] = static[line] * multipliers[line].to_numpy()
    n.lines_t.s_max_pu = applied
    return multipliers.set_axis(n.snapshots), hashlib.sha256(source.read_bytes()).hexdigest()


def variable_sum(variable, names):
    """Works with PyPSA's component dimension 'name' and older component labels."""
    dims = [d for d in variable.dims if d != "snapshot"]
    if len(dims) != 1:
        raise RuntimeError(f"Unsupported variable dimensions: {variable.dims}")
    return variable.sel({dims[0]: list(names)}).sum(dims[0])


def validate_dispatch(result: Dispatch, design: Design, available: float, cfg: dict) -> None:
    tol = 2e-4  # absolute MW/MWh numerical reporting tolerance
    arrays = (result.project_wind_mw, result.battery_charge_mw,
              result.battery_discharge_mw, result.battery_soc_mwh, result.duration_h)
    if any(not np.isfinite(a).all() for a in arrays):
        raise RuntimeError("Solver returned nonfinite dispatch")
    c, d, soc, dt = (result.battery_charge_mw, result.battery_discharge_mw,
                     result.battery_soc_mwh, result.duration_h)
    if np.min(c) < -tol or np.min(d) < -tol:
        raise RuntimeError("Negative gross charging/discharging")
    if np.max(np.minimum(c, d)) > tol:
        raise RuntimeError("Simultaneous battery charging and discharging detected")
    if available > 0 and design.battery_mw:
        eta = np.sqrt(cfg["battery"]["round_trip_efficiency"])
        residual = soc - np.roll(soc, 1) - eta * c * dt + d * dt / eta
        if np.max(np.abs(residual)) > tol:
            raise RuntimeError("Battery cyclic energy conservation failed")
        if np.min(soc) < cfg["battery"]["soc_min"] * available - tol or np.max(soc) > cfg["battery"]["soc_max"] * available + tol:
            raise RuntimeError("Battery SoC limit violated")
        p = min(design.battery_mw, cfg["battery"]["max_c_rate_per_hour"] * available)
        if max(c.max(), d.max()) > p + tol:
            raise RuntimeError("Battery AC power limit violated")
    net = result.project_wind_mw + d - c
    if net.max() > design.export_mw + tol or net.min() < -design.import_mw - tol:
        raise RuntimeError("Project connection limit violated")
    if result.metrics["max_passive_loading_pu"] > 1 + 1e-5:
        raise RuntimeError("Passive thermal rating exceeded after solve")


class PyPSABackend:
    def __init__(self, root: Path, config: dict, network_path: Path | None = None,
                 kit_dir: Path | None = None, max_hours: int | None = None):
        try:
            import pypsa
        except ImportError as exc:
            raise ImportError("Activate .venv, then run: python -m pip install -r requirements-q4.txt") from exc
        for logger in ("pypsa", "linopy", "highspy"):
            logging.getLogger(logger).setLevel(logging.ERROR)
        self.config, self.root = config, root
        if network_path:
            path = Path(network_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            n = pypsa.Network(path)
            network_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            origin = str(path.resolve())
        else:
            kit_dir = kit_dir or root / "data" / "participant-kit"
            module_path = Path(kit_dir) / "gridkit.py"
            if not module_path.exists():
                raise FileNotFoundError(f"Missing {module_path}; use --kit-dir or --network your_validated_model.nc")
            sys.path.insert(0, str(module_path.parent))
            spec = importlib.util.spec_from_file_location("q4_official_gridkit", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            n = module.load(config["grid"]["scenario"], config["grid"]["scope"])
            origin = str(module_path.resolve()) + ":load"
            # Hash loaded values below as well: modified loaders must invalidate cache.
            network_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
        if isinstance(n.snapshots, pd.MultiIndex):
            raise ValueError("V1 requires one chronological snapshot index, not multi-investment periods")
        if not n.snapshots.is_unique or not n.snapshots.is_monotonic_increasing:
            raise ValueError("Snapshots must be unique and chronological")
        if max_hours:
            n = n.copy()
            n.set_snapshots(n.snapshots[:max_hours])
        if len(n.snapshots) < 2:
            raise ValueError("Need at least two chronological snapshots")
        self.n = n
        self.dlr_multipliers = None
        self.dlr_source_sha256 = None
        if config["grid"]["dlr_multiplier_csv"]:
            self.dlr_multipliers, self.dlr_source_sha256 = apply_dlr_multiplier_csv(
                n, Path(config["grid"]["dlr_multiplier_csv"]), root)
        self.snapshots = n.snapshots.copy()
        sw = n.snapshot_weightings
        self.duration_h = sw["stores"].to_numpy(float)
        if not np.isfinite(self.duration_h).all() or np.any(self.duration_h <= 0):
            raise ValueError("Invalid physical snapshot durations")
        if isinstance(n.snapshots, pd.DatetimeIndex):
            gaps = np.diff(n.snapshots.asi8) / 3.6e12
            if not np.allclose(gaps, self.duration_h[:-1]):
                raise ValueError("Snapshot gaps and physical-hour weights disagree; use a chronological contiguous time series")
        for col in ("generators", "objective"):
            if not np.allclose(sw[col], self.duration_h):
                raise ValueError("V1 expects physical-hour weights. Keep representative-year multipliers in annualisation, not SoC weights.")
        for table, capacity in (("generators", "p_nom"), ("storage_units", "p_nom"),
                ("stores", "e_nom"), ("links", "p_nom"), ("lines", "s_nom"), ("transformers", "s_nom")):
            f = getattr(n, table)
            col = capacity + "_extendable"
            if col in f and f[col].fillna(False).astype(bool).any():
                raise ValueError(f"{table}: baseline has extendable assets; export a fixed-capacity validated network first")
            if table in ("generators", "storage_units", "links"):
                dynamic = getattr(getattr(n, table + "_t"), "p_set")
                if dynamic.notna().any().any() or ("p_set" in f and f["p_set"].notna().any()):
                    raise ValueError(f"{table}.p_set contains fixed dispatch. Use a fresh unsolved gridkit network, not freeze_dispatch output.")
        self.wind_ids = n.generators.index[n.generators.carrier.astype(str).str.lower().str.contains("wind", regex=False)]
        renewable = n.generators.carrier.astype(str).str.lower().str.contains("wind|solar", regex=True)
        if not renewable.any():
            raise ValueError("No wind/solar carriers. Check generator aggregation before Q4")
        # Official kit warns that missing variable-resource profiles silently become 1.
        for g in n.generators.index[renewable]:
            if g not in n.generators_t.p_max_pu:
                raise ValueError(f"Missing explicit wind/solar hourly profile for {g}")
        profiles = dense_attribute(n, "generators", "p_max_pu")
        if np.any(profiles[self.wind_ids].to_numpy() < 0) or np.any(profiles[self.wind_ids].to_numpy() > 1 + 1e-8):
            raise ValueError("Wind capacity factors outside [0,1]")
        if not len(self.wind_ids) or n.generators.loc[self.wind_ids, "p_nom"].sum() <= 0:
            raise ValueError("Need at least one existing wind profile or a supported custom resource case")
        self.common_profile = profiles[self.wind_ids].mul(n.generators.loc[self.wind_ids, "p_nom"]).sum(axis=1) / n.generators.loc[self.wind_ids, "p_nom"].sum()
        # Do not assume a 15-node or a 16-node representation; use exact loaded IDs.
        s = n.buses.copy()
        s["bus"] = s.index.astype(str)
        name_col = next((col for col in ("station_name", "station", "label", "name") if col in s), None)
        s["display_name"] = s[name_col].fillna(s["bus"]) if name_col else s["bus"]
        excluded = set(map(str, config["grid"]["excluded_buses"]))
        artificial = (s["bus"] + " " + s["display_name"].astype(str)).str.lower().str.contains("boundary|external|equivalent|slack", regex=True)
        s["eligible"] = (s.v_nom >= config["grid"]["minimum_bus_kv"]) & ~s["bus"].isin(excluded) & ~artificial
        if config["grid"]["wind_profile_mode"] == "local":
            s["eligible"] &= s.index.isin(n.generators.loc[self.wind_ids, "bus"])
        self.sites = s
        if not s.eligible.any():
            raise ValueError("No eligible sites; inspect bus names, voltage filter and exclusions")
        h = hashlib.sha256(network_hash.encode())
        for table in ("buses", "generators", "loads", "lines", "transformers", "links", "storage_units", "stores"):
            h.update(getattr(n, table).to_csv().encode())
            for attr, frame in getattr(n, table + "_t").items():
                if not frame.empty:
                    h.update(attr.encode() + frame.to_csv().encode())
        h.update(sw.to_csv().encode())
        self.metadata = {"backend": "pypsa", "pypsa_version": pypsa.__version__, "origin": origin,
            "network_hash": h.hexdigest(), "network_meta": getattr(n, "meta", {}),
            "buses": len(n.buses), "lines": len(n.lines), "transformers": len(n.transformers),
            "links": len(n.links), "snapshots": len(n.snapshots),
            "scope_validation_status": "NOT certified by Q4; use your separate 15/16/all-island audit",
            "dispatch_policy": "official system-cost optimum, same-bus wind pro-rata, battery exclusive charge/discharge",
            "economic_interpretation": "NPV conditional on operator dispatch, not merchant dispatch optimum",
            "dlr_label": config["grid"]["dlr_label"] if self.dlr_multipliers is not None else "none",
            "dlr_lines": list(self.dlr_multipliers.columns) if self.dlr_multipliers is not None else [],
            "dlr_mean_multiplier": float(self.dlr_multipliers.mean().mean()) if self.dlr_multipliers is not None else 1.0,
            "dlr_max_multiplier": float(self.dlr_multipliers.max().max()) if self.dlr_multipliers is not None else 1.0,
            "dlr_source_sha256": self.dlr_source_sha256}
        self._cache = {}

    def profile(self, bus: str) -> np.ndarray:
        mode = self.config["grid"]["wind_profile_mode"]
        if mode == "csv":
            path = Path(self.config["grid"]["wind_profile_csv"])
            if not path.is_absolute(): path = self.root / path
            p = aligned_csv(path, self.snapshots, [str(bus)])[str(bus)].to_numpy()
        elif mode == "local":
            ids = self.wind_ids[self.n.generators.loc[self.wind_ids, "bus"].eq(bus)]
            if not len(ids): raise ValueError(f"No local wind profile at {bus}; do not guess wind resource")
            p = (dense_attribute(self.n, "generators", "p_max_pu")[ids]
                .mul(self.n.generators.loc[ids, "p_nom"]).sum(axis=1)
                / self.n.generators.loc[ids, "p_nom"].sum()).to_numpy()
        else:
            p = self.common_profile.to_numpy()
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("New wind profile must be finite and within [0,1]")
        return p

    def solve(self, design: Design, available_energy_mwh: float, wind_factor: float = 1.0,
              relax_passive_limits: bool = False) -> Dispatch:
        key = (design, float(available_energy_mwh), float(wind_factor), relax_passive_limits)
        if key in self._cache: return self._cache[key]
        if design.bus not in self.n.buses.index: raise ValueError(f"No exact bus ID {design.bus!r}")
        n, cfg = self.n.copy(), self.config
        if WIND_NAME in n.generators.index or any(str(name).startswith(BATTERY_NAME) for name in n.storage_units.index):
            raise ValueError("Q4 reserved component names already occur in the supplied network")
        if relax_passive_limits:
            for table in ("lines", "transformers"):
                f = getattr(n, table)
                f.loc[:, "s_nom"] = f.s_nom * cfg["grid"]["relaxed_rating_multiplier"]
        if design.wind_mw:
            profile = self.profile(design.bus) * wind_factor
            mc = float(n.generators.loc[self.wind_ids, "marginal_cost"].median())
            if "wind" not in n.carriers.index: n.add("Carrier", "wind")
            n.add("Generator", WIND_NAME, bus=design.bus, carrier="wind",
                  p_nom=design.wind_mw, marginal_cost=mc, p_max_pu=pd.Series(profile, index=n.snapshots))
        battery_active = design.battery_mw > 0 and available_energy_mwh > 1e-9
        battery_names, battery_powers = [], {}
        if battery_active:
            eta = np.sqrt(cfg["battery"]["round_trip_efficiency"])
            if "battery" not in n.carriers.index: n.add("Carrier", "battery")
            for idx, (site, share) in enumerate(design.placements, 1):
                site_energy = available_energy_mwh * share
                site_power = min(design.battery_mw * share, cfg["battery"]["max_c_rate_per_hour"] * site_energy)
                name = BATTERY_NAME if len(design.placements) == 1 else f"{BATTERY_NAME}_{idx}"
                n.add("StorageUnit", name, bus=site, carrier="battery", p_nom=site_power,
                      max_hours=site_energy / site_power, efficiency_store=eta, efficiency_dispatch=eta,
                      cyclic_state_of_charge=True, standing_loss=0.0,
                      marginal_cost=cfg["battery"]["dispatch_wear_penalty_eur_mwh"])
                battery_names.append(name)
                battery_powers[name] = site_power
        m = n.optimize.create_model()
        import xarray as xr
        def series(values):
            return xr.DataArray(np.asarray(values), coords={"snapshot": n.snapshots}, dims=["snapshot"])
        gen = m.variables["Generator-p"]
        # Explicit equal-priority allocation among co-located wind generators.
        # Different buses remain governed by network constraints and system dispatch.
        winds = n.generators.index[n.generators.carrier.astype(str).str.lower().str.contains("wind", regex=False)]
        availability = dense_attribute(n, "generators", "p_max_pu").mul(n.generators.p_nom)
        for bus, group in n.generators.loc[winds].groupby("bus"):
            ids = list(group.index)
            if len(ids) < 2: continue
            total = availability[ids].sum(axis=1).to_numpy()
            total_var = variable_sum(gen, ids)
            for idx, name in enumerate(ids[:-1]):
                share = np.divide(availability[name].to_numpy(), total, out=np.zeros_like(total), where=total > 1e-10)
                m.add_constraints(variable_sum(gen, [name]) == total_var * series(share),
                                  name=f"Q4-wind-pro-rata-{str(bus)}-{idx}")
        project_by_bus = {}
        if design.wind_mw:
            project_by_bus[design.bus] = variable_sum(gen, [WIND_NAME])
        if battery_active:
            mode = m.add_variables(coords=[n.snapshots.rename("snapshot")], binary=True, name="Q4-portfolio-discharge-mode")
            for idx, (name, (site, share)) in enumerate(zip(battery_names, design.placements), 1):
                charge_site = variable_sum(m.variables["StorageUnit-p_store"], [name])
                discharge_site = variable_sum(m.variables["StorageUnit-p_dispatch"], [name])
                soc_site = variable_sum(m.variables["StorageUnit-state_of_charge"], [name])
                site_energy = available_energy_mwh * share
                m.add_constraints(soc_site >= cfg["battery"]["soc_min"] * site_energy, name=f"Q4-SoC-min-{idx}")
                m.add_constraints(soc_site <= cfg["battery"]["soc_max"] * site_energy, name=f"Q4-SoC-max-{idx}")
                m.add_constraints(discharge_site <= battery_powers[name] * mode, name=f"Q4-discharge-exclusive-{idx}")
                m.add_constraints(charge_site <= battery_powers[name] * (1 - mode), name=f"Q4-charge-exclusive-{idx}")
                site_net = discharge_site - charge_site
                project_by_bus[site] = project_by_bus.get(site, 0) + site_net
        for idx, (site, project_net) in enumerate(project_by_bus.items(), 1):
            if len(design.placements) > 1:
                share = dict(design.placements)[site]
                export_limit = import_limit = design.battery_mw * share
            else:
                export_limit, import_limit = design.export_mw, design.import_mw
            m.add_constraints(project_net <= export_limit, name=f"Q4-PCC-export-{idx}")
            m.add_constraints(project_net >= -import_limit, name=f"Q4-PCC-import-{idx}")
        status, condition = n.optimize.solve_model(solver_name="highs",
            solver_options={"output_flag": False, "mip_rel_gap": cfg["grid"]["mip_gap"],
                            "time_limit": cfg["grid"]["solver_time_limit_s"]})
        if str(status) != "ok" or str(condition) != "optimal":
            raise RuntimeError(f"{design.id}: solver {status}/{condition}; no guessed result")
        z = np.zeros(len(n.snapshots))
        wind = n.generators_t.p[WIND_NAME].to_numpy() if design.wind_mw else z.copy()
        if battery_active:
            charge_frame = n.storage_units_t.p_store[battery_names]
            discharge_frame = n.storage_units_t.p_dispatch[battery_names]
            soc_frame = n.storage_units_t.state_of_charge[battery_names]
            charge = charge_frame.sum(axis=1).to_numpy()
            discharge = discharge_frame.sum(axis=1).to_numpy()
            soc = soc_frame.sum(axis=1).to_numpy()
            site_dispatch = pd.DataFrame(index=n.snapshots)
            for name, (site, _) in zip(battery_names, design.placements):
                site_dispatch[f"charge_mw__{site}"] = charge_frame[name]
                site_dispatch[f"discharge_mw__{site}"] = discharge_frame[name]
                site_dispatch[f"soc_mwh__{site}"] = soc_frame[name]
        else:
            charge = z.copy(); discharge = z.copy(); soc = z.copy()
            site_dispatch = pd.DataFrame(index=n.snapshots)
        renew = n.generators.index[n.generators.carrier.astype(str).str.lower().str.contains("wind|solar", regex=True)]
        offered = availability[renew].sum(axis=1).to_numpy()
        taken = n.generators_t.p[renew].sum(axis=1).to_numpy()
        shed = [g for g in n.generators.index if str(g).lower().startswith("shed ") or "shedding" in str(n.generators.at[g, "carrier"]).lower()]
        unserved = float(n.generators_t.p[shed].sum(axis=1).to_numpy() @ self.duration_h) if shed else 0.0
        costs = dense_attribute(n, "generators", "marginal_cost")
        gen_cost = float((n.generators_t.p * costs).sum(axis=1).to_numpy() @ self.duration_h)
        branch_rows = []
        for table, component in (("lines", "Line"), ("transformers", "Transformer")):
            f = getattr(n, table)
            if f.empty: continue
            limit = dense_attribute(n, table, "s_max_pu").mul(f.s_nom)
            flow = getattr(n, table + "_t").p0.reindex(columns=f.index)
            ratio = flow.abs() / limit.replace(0, np.nan)
            if ((limit <= 0) & (flow.abs() > 1e-5)).any().any():
                raise RuntimeError("Nonzero flow on zero-rated branch")
            for name in f.index:
                load = ratio[name].fillna(0).to_numpy()
                branch_rows.append({"component": component, "branch": str(name),
                    "bus0": str(f.at[name, "bus0"]), "bus1": str(f.at[name, "bus1"]),
                    "peak_loading_pu": float(load.max()),
                    "binding_hours": float((load >= cfg["grid"]["binding_threshold"]) @ self.duration_h)})
        branches = pd.DataFrame(branch_rows, columns=["component", "branch", "bus0", "bus1", "peak_loading_pu", "binding_hours"])
        peak = float(branches.peak_loading_pu.max()) if len(branches) else 0.0
        if relax_passive_limits and peak >= cfg["grid"]["binding_threshold"]:
            raise RuntimeError("Counterfactual ratings still bind; increase relaxed_rating_multiplier")
        metrics = {"renewable_available_mwh": float(offered @ self.duration_h),
            "renewable_dispatched_mwh": float(taken @ self.duration_h),
            "renewable_dispatch_down_mwh": float(np.maximum(offered - taken, 0) @ self.duration_h),
            "project_wind_available_mwh": float(availability[WIND_NAME].to_numpy() @ self.duration_h) if design.wind_mw else 0.0,
            "battery_loss_mwh": float((charge - discharge) @ self.duration_h),
            "unserved_mwh": unserved, "generator_dispatch_cost_eur": gen_cost,
            "max_passive_loading_pu": peak,
            "binding_branch_hours": float(branches.binding_hours.sum()) if len(branches) else 0.0}
        result = Dispatch(n.snapshots.copy(), self.duration_h.copy(), wind, charge, discharge, soc, metrics, branches, site_dispatch)
        validate_dispatch(result, design, available_energy_mwh, cfg)
        self._cache[key] = result
        return result
