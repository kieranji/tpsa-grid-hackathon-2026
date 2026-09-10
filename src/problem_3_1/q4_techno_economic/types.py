"""Backend-independent data contracts. All electrical powers are MW, energy MWh."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Protocol
import hashlib
import json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Design:
    bus: str
    wind_mw: float = 0.0
    battery_mw: float = 0.0
    battery_mwh: float = 0.0  # installed DC nameplate, NOT usable energy
    maintenance: str = "replace"
    connection_export_mw: float | None = None
    connection_import_mw: float | None = None
    battery_placements: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        for x in (self.wind_mw, self.battery_mw, self.battery_mwh):
            if not np.isfinite(x) or x < 0:
                raise ValueError("Plant MW and MWh must be finite and non-negative")
        if (self.battery_mw == 0) != (self.battery_mwh == 0):
            raise ValueError("Battery MW and MWh must both be zero or both positive")
        if self.maintenance not in ("replace", "run_down", "augment"):
            raise ValueError("maintenance must be replace, run_down or augment")
        for x in (self.connection_export_mw, self.connection_import_mw):
            if x is not None and (not np.isfinite(x) or x < 0):
                raise ValueError("Connection limits must be finite and non-negative")
        if self.battery_placements and not self.battery_mw:
            raise ValueError("battery_placements require a positive battery size")
        placements = self.placements
        if placements:
            sites = [site for site, _ in placements]
            weights = np.asarray([share for _, share in placements], dtype=float)
            if any(not site for site in sites) or len(set(sites)) != len(sites):
                raise ValueError("Battery placement buses must be non-empty and unique")
            if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
                raise ValueError("Battery placement shares must be positive and sum to one")
            if self.wind_mw and (len(placements) != 1 or placements[0][0] != self.bus):
                raise ValueError("V1 hybrid projects require the battery at the wind bus")
            if len(placements) > 1 and (self.connection_export_mw is not None or self.connection_import_mw is not None):
                raise ValueError("Portfolio connection overrides must be modelled per site, not as one aggregate limit")

    @property
    def technology(self) -> str:
        if self.wind_mw > 0 and self.battery_mw > 0:
            return "hybrid"
        return "wind" if self.wind_mw > 0 else "bess" if self.battery_mw > 0 else "no_build"

    @property
    def export_mw(self) -> float:
        # Hybrid shares an export connection sized to its wind plant, unless overridden.
        default = self.wind_mw if self.wind_mw else self.battery_mw
        return default if self.connection_export_mw is None else self.connection_export_mw

    @property
    def import_mw(self) -> float:
        return self.battery_mw if self.connection_import_mw is None else self.connection_import_mw

    @property
    def placements(self) -> tuple[tuple[str, float], ...]:
        if not self.battery_mw:
            return ()
        if self.battery_placements:
            return tuple((str(site), float(share)) for site, share in self.battery_placements)
        return ((self.bus, 1.0),)

    @property
    def site_count(self) -> int:
        return len(self.placements) if self.placements else (0 if self.technology == "no_build" else 1)

    @property
    def sites_label(self) -> str:
        return ";".join(site for site, _ in self.placements) if self.placements else self.bus

    @property
    def allocations_label(self) -> str:
        return ";".join(f"{share:.6f}" for _, share in self.placements)

    @property
    def id(self) -> str:
        digest = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.technology}_{digest}"

@dataclass
class Dispatch:
    """One chronological operating block; no annualisation is embedded in these arrays.

    project_wind_mw is an explicit same-bus pro-rata allocation in the PyPSA
    backend, not an inference from a degenerate new-generator dispatch solution.
    Metrics use wind+solar for weather-dependent renewable dispatch-down.
    """
    snapshots: pd.Index
    duration_h: np.ndarray
    project_wind_mw: np.ndarray
    battery_charge_mw: np.ndarray
    battery_discharge_mw: np.ndarray
    battery_soc_mwh: np.ndarray
    metrics: dict[str, Any]
    branches: pd.DataFrame
    site_dispatch: pd.DataFrame | None = None

    @property
    def hours(self) -> float:
        return float(self.duration_h.sum())

    def hourly_frame(self) -> pd.DataFrame:
        net = self.project_wind_mw + self.battery_discharge_mw - self.battery_charge_mw
        frame = pd.DataFrame({"snapshot": self.snapshots, "duration_h": self.duration_h,
            "project_wind_mw": self.project_wind_mw,
            "battery_charge_mw": self.battery_charge_mw,
            "battery_discharge_mw": self.battery_discharge_mw,
            "battery_soc_mwh": self.battery_soc_mwh,
            "pcc_export_mw": np.maximum(net, 0),
            "pcc_import_mw": np.maximum(-net, 0)})
        if self.site_dispatch is not None and not self.site_dispatch.empty:
            aligned = self.site_dispatch.reindex(self.snapshots)
            for column in aligned.columns:
                frame[column] = aligned[column].to_numpy()
        return frame

class Backend(Protocol):
    snapshots: pd.Index
    duration_h: np.ndarray
    sites: pd.DataFrame
    metadata: dict[str, Any]
    def solve(self, design: Design, available_energy_mwh: float, wind_factor: float = 1.0,
              relax_passive_limits: bool = False) -> Dispatch: ...
