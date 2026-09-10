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
        if key in self._cache: return self._cache[key]
        if design.bus not in self.sites.index:
            raise ValueError(f"Unknown demo bus {design.bus}")
        t = len(self.snapshots)
        hour = np.arange(t) % 24
        cf = 0.05 + 0.9 * (1 + np.cos(2 * np.pi * hour / 24)) / 2
        old_available = 60 * cf
        new_available = design.wind_mw * wind_factor * cf
        load_a = np.full(t, 5.0)
        load_b = 35 + 8 * np.sin(2 * np.pi * (hour - 9) / 24)
        cfg = self.config
        eta = np.sqrt(cfg["battery"]["round_trip_efficiency"])
        power = min(design.battery_mw, available_energy_mwh * cfg["battery"]["max_c_rate_per_hour"])
        limit = 10 * (cfg["grid"]["relaxed_rating_multiplier"] if relax_passive_limits else 1)
        # Order per block: existing wind, project wind, gas, A->B flow, charge,
        # discharge, internal SoC, binary discharge mode.
        ix = np.arange(8 * t).reshape(8, t)
        lb = np.zeros(8 * t)
        ub = np.full(8 * t, np.inf)
        ub[ix[0]] = old_available
        ub[ix[1]] = new_available
        ub[ix[2]] = 300
        lb[ix[3]], ub[ix[3]] = -limit, limit
        ub[ix[4]], ub[ix[5]] = power, power
        lb[ix[6]] = available_energy_mwh * cfg["battery"]["soc_min"]
        ub[ix[6]] = available_energy_mwh * cfg["battery"]["soc_max"]
        ub[ix[7]] = 1 if power else 0
        integ = np.zeros(8 * t, dtype=int)
        integ[ix[7]] = 1
        objective = np.zeros(8 * t)
        objective[ix[0]], objective[ix[1]], objective[ix[2]] = -1, -1, 90
        objective[ix[5]] = cfg["battery"]["dispatch_wear_penalty_eur_mwh"]
        same_bus = design.bus == "DEMO_UPSTREAM"
        rows, lower, upper = [], [], []
        def add(coefs, lo, hi):
            rows.append(coefs); lower.append(lo); upper.append(hi)
        for k in range(t):
            a = {ix[0, k]: 1, ix[3, k]: -1}
            b = {ix[2, k]: 1, ix[3, k]: 1}
            local = a if same_bus else b
            local.update({ix[1, k]: 1, ix[4, k]: -1, ix[5, k]: 1})
            add(a, load_a[k], load_a[k]); add(b, load_b[k], load_b[k])
            add({ix[6, k]: 1, ix[6, (k-1) % t]: -1, ix[4, k]: -eta, ix[5, k]: 1/eta}, 0, 0)
            add({ix[5, k]: 1, ix[7, k]: -power}, -np.inf, 0)
            add({ix[4, k]: 1, ix[7, k]: power}, -np.inf, power)
            add({ix[1, k]: 1, ix[5, k]: 1, ix[4, k]: -1}, -design.import_mw, design.export_mw)
            if same_bus and design.wind_mw:
                add({ix[0, k]: design.wind_mw * wind_factor, ix[1, k]: -60}, 0, 0)
        mat = lil_matrix((len(rows), 8 * t), dtype=float)
        for row, coeff in enumerate(rows):
            for col, val in coeff.items(): mat[row, col] = val
        sol = milp(objective, integrality=integ, bounds=Bounds(lb, ub),
                   constraints=LinearConstraint(mat.tocsr(), np.asarray(lower), np.asarray(upper)),
                   options={"mip_rel_gap": cfg["grid"]["mip_gap"], "time_limit": cfg["grid"]["solver_time_limit_s"]})
        if not sol.success: raise RuntimeError(f"Demo MILP failed: {sol.message}")
        w0, w, g, f, c, d, soc, mode = sol.x.reshape(8, t)
        ratio = np.abs(f) / limit
        branches = pd.DataFrame([{"component": "Line", "branch": "DEMO_LINE",
            "bus0": "DEMO_UPSTREAM", "bus1": "DEMO_DOWNSTREAM",
            "peak_loading_pu": float(ratio.max()),
            "binding_hours": float(np.sum(ratio >= cfg["grid"]["binding_threshold"]))}])
        metrics = {"renewable_available_mwh": float(np.sum(old_available + new_available)),
            "renewable_dispatched_mwh": float(np.sum(w0 + w)),
            "renewable_dispatch_down_mwh": float(np.sum(old_available + new_available - w0 - w)),
            "project_wind_available_mwh": float(np.sum(new_available)),
            "battery_loss_mwh": float(np.sum(c-d)), "unserved_mwh": 0.0,
            "generator_dispatch_cost_eur": float(np.sum(90*g-w0-w)),
            "max_passive_loading_pu": float(ratio.max()),
            "binding_branch_hours": float(branches.binding_hours.sum())}
        result = Dispatch(self.snapshots, self.duration_h, w, c, d, soc, metrics, branches)
        validate_dispatch(result, design, available_energy_mwh, cfg)
        self._cache[key] = result
        return result
