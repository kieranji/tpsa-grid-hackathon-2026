"""A two-bus synthetic MILP for reproducible offline tests, NOT the Irish network.

The main pipeline uses this backend only with --demo. It exercises chronology,
network bottlenecks, pro-rata wind, shared meters, binary battery dispatch and
annual degradation without requiring PyPSA or downloaded network files.
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from .types import Design, Dispatch
from .grid import validate_dispatch

class DemoBackend:
    def __init__(self, config: dict, max_hours: int | None = None):
        self.config = config
        self.snapshots = pd.date_range("2033-01-01", periods=max_hours or 48, freq="h", name="snapshot")
        self.duration_h = np.ones(len(self.snapshots))
        self.sites = pd.DataFrame({"bus": ["DEMO_UPSTREAM", "DEMO_DOWNSTREAM"],
            "display_name": ["Synthetic wind-side bus", "Synthetic load-side bus"],
            "eligible": [True, True], "v_nom": [110.0, 110.0], "x": [0.0, 1.0], "y": [0.0, 0.0]}).set_index("bus", drop=False)
        self.metadata = {"backend": "synthetic_two_bus_demo_NOT_IRELAND", "buses": 2,
            "snapshots": len(self.snapshots), "network_hash": hashlib.sha256(str(list(self.snapshots)).encode()).hexdigest(),
            "scope_validation_status": "synthetic demo only", "dispatch_policy": "system_cost"}
        self._cache = {}

    def solve(self, design: Design, available_energy_mwh: float, wind_factor: float = 1.0,
              relax_passive_limits: bool = False) -> Dispatch:
        key = (design, available_energy_mwh, wind_factor, relax_passive_limits)
        if key in self._cache:
            return self._cache[key]
        if design.bus not in self.sites.index:
            raise ValueError(f"Unknown demo bus {design.bus}")
        for site, _ in design.placements:
            if site not in self.sites.index:
                raise ValueError(f"Unknown demo battery bus {site}")
        t = len(self.snapshots)
        hour = np.arange(t) % 24
        cf = 0.05 + 0.9 * (1 + np.cos(2 * np.pi * hour / 24)) / 2
        old_available = 60 * cf
        new_available = design.wind_mw * wind_factor * cf
        load_a = np.full(t, 5.0)
        load_b = 35 + 8 * np.sin(2 * np.pi * (hour - 9) / 24)
        cfg = self.config
        eta = np.sqrt(cfg["battery"]["round_trip_efficiency"])
        placements = design.placements if design.battery_mw and available_energy_mwh > 1e-9 else ()
        site_power = np.asarray([
            min(design.battery_mw * share,
                available_energy_mwh * share * cfg["battery"]["max_c_rate_per_hour"])
            for _, share in placements], dtype=float)
        site_energy = np.asarray([available_energy_mwh * share for _, share in placements], dtype=float)
        limit = 10 * (cfg["grid"]["relaxed_rating_multiplier"] if relax_passive_limits else 1)
        block_count = 4 + 3 * len(placements) + (1 if placements else 0)
        blocks = np.arange(block_count * t).reshape(block_count, t)
        w0_i, w_i, gas_i, flow_i = blocks[:4]
        cursor = 4
        charge_i = blocks[cursor:cursor + len(placements)]; cursor += len(placements)
        discharge_i = blocks[cursor:cursor + len(placements)]; cursor += len(placements)
        soc_i = blocks[cursor:cursor + len(placements)]; cursor += len(placements)
        mode_i = blocks[cursor] if placements else None
        lb = np.zeros(block_count * t)
        ub = np.full(block_count * t, np.inf)
        ub[w0_i] = old_available
        ub[w_i] = new_available
        ub[gas_i] = 300
        lb[flow_i], ub[flow_i] = -limit, limit
        integrality = np.zeros(block_count * t, dtype=int)
        for idx in range(len(placements)):
            ub[charge_i[idx]], ub[discharge_i[idx]] = site_power[idx], site_power[idx]
            lb[soc_i[idx]] = site_energy[idx] * cfg["battery"]["soc_min"]
            ub[soc_i[idx]] = site_energy[idx] * cfg["battery"]["soc_max"]
        if placements:
            ub[mode_i] = 1
            integrality[mode_i] = 1
        objective = np.zeros(block_count * t)
        objective[w0_i], objective[w_i], objective[gas_i] = -1, -1, 90
        for idx in range(len(placements)):
            objective[discharge_i[idx]] = cfg["battery"]["dispatch_wear_penalty_eur_mwh"]
        rows, lower, upper = [], [], []

        def add(coefs, lo, hi):
            rows.append(coefs); lower.append(lo); upper.append(hi)

        same_bus = design.bus == "DEMO_UPSTREAM"
        for k in range(t):
            balances = {
                "DEMO_UPSTREAM": {w0_i[k]: 1, flow_i[k]: -1},
                "DEMO_DOWNSTREAM": {gas_i[k]: 1, flow_i[k]: 1},
            }
            if design.wind_mw:
                balances[design.bus][w_i[k]] = balances[design.bus].get(w_i[k], 0) + 1
            for idx, (site, _) in enumerate(placements):
                balances[site][charge_i[idx, k]] = -1
                balances[site][discharge_i[idx, k]] = 1
            add(balances["DEMO_UPSTREAM"], load_a[k], load_a[k])
            add(balances["DEMO_DOWNSTREAM"], load_b[k], load_b[k])
            for idx, (site, share) in enumerate(placements):
                add({soc_i[idx, k]: 1, soc_i[idx, (k - 1) % t]: -1,
                     charge_i[idx, k]: -eta, discharge_i[idx, k]: 1 / eta}, 0, 0)
                add({discharge_i[idx, k]: 1, mode_i[k]: -site_power[idx]}, -np.inf, 0)
                add({charge_i[idx, k]: 1, mode_i[k]: site_power[idx]}, -np.inf, site_power[idx])
                project = {discharge_i[idx, k]: 1, charge_i[idx, k]: -1}
                if design.wind_mw and site == design.bus:
                    project[w_i[k]] = 1
                if len(placements) > 1:
                    export_limit = import_limit = design.battery_mw * share
                else:
                    export_limit, import_limit = design.export_mw, design.import_mw
                add(project, -import_limit, export_limit)
            if design.wind_mw and not placements:
                add({w_i[k]: 1}, -design.import_mw, design.export_mw)
            if same_bus and design.wind_mw:
                add({w0_i[k]: design.wind_mw * wind_factor, w_i[k]: -60}, 0, 0)
        mat = lil_matrix((len(rows), block_count * t), dtype=float)
        for row, coeff in enumerate(rows):
            for col, val in coeff.items():
                mat[row, col] = val
        sol = milp(objective, integrality=integrality, bounds=Bounds(lb, ub),
                   constraints=LinearConstraint(mat.tocsr(), np.asarray(lower), np.asarray(upper)),
                   options={"mip_rel_gap": cfg["grid"]["mip_gap"],
                            "time_limit": cfg["grid"]["solver_time_limit_s"]})
        if not sol.success:
            raise RuntimeError(f"Demo MILP failed: {sol.message}")
        w0, wind, gas, flow = (sol.x[w0_i], sol.x[w_i], sol.x[gas_i], sol.x[flow_i])
        if placements:
            charge_by_site = np.vstack([sol.x[index] for index in charge_i])
            discharge_by_site = np.vstack([sol.x[index] for index in discharge_i])
            soc_by_site = np.vstack([sol.x[index] for index in soc_i])
            charge = charge_by_site.sum(axis=0)
            discharge = discharge_by_site.sum(axis=0)
            soc = soc_by_site.sum(axis=0)
            site_dispatch = pd.DataFrame(index=self.snapshots)
            for idx, (site, _) in enumerate(placements):
                site_dispatch[f"charge_mw__{site}"] = charge_by_site[idx]
                site_dispatch[f"discharge_mw__{site}"] = discharge_by_site[idx]
                site_dispatch[f"soc_mwh__{site}"] = soc_by_site[idx]
        else:
            charge = np.zeros(t); discharge = np.zeros(t); soc = np.zeros(t)
            site_dispatch = pd.DataFrame(index=self.snapshots)
        ratio = np.abs(flow) / limit
        branches = pd.DataFrame([{"component": "Line", "branch": "DEMO_LINE",
            "bus0": "DEMO_UPSTREAM", "bus1": "DEMO_DOWNSTREAM",
            "peak_loading_pu": float(ratio.max()),
            "binding_hours": float(np.sum(ratio >= cfg["grid"]["binding_threshold"]))}])
        metrics = {"renewable_available_mwh": float(np.sum(old_available + new_available)),
            "renewable_dispatched_mwh": float(np.sum(w0 + wind)),
            "renewable_dispatch_down_mwh": float(np.sum(old_available + new_available - w0 - wind)),
            "project_wind_available_mwh": float(np.sum(new_available)),
            "battery_loss_mwh": float(np.sum(charge - discharge)), "unserved_mwh": 0.0,
            "generator_dispatch_cost_eur": float(np.sum(90 * gas - w0 - wind)),
            "max_passive_loading_pu": float(ratio.max()),
            "binding_branch_hours": float(branches.binding_hours.sum())}
        result = Dispatch(self.snapshots, self.duration_h, wind, charge, discharge, soc,
                          metrics, branches, site_dispatch)
        validate_dispatch(result, design, available_energy_mwh, cfg)
        self._cache[key] = result
        return result
