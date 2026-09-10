"""Input validation. Numeric defaults are DEMONSTRATION assumptions, not quotations."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np

DEFAULTS = {
 "schema_version": 1,
 "assumptions_status": "illustrative_not_an_irish_market_forecast",
 "currency_basis": "real EUR, 2026 purchasing power; pre-tax unlevered",
 "years": 20,
 "real_discount_rate": 0.07,
 "annualisation": {"mode": "repeat_block", "hours_per_year": 8760.0},
 "grid": {"scenario": "WP2033", "scope": "north-west", "minimum_bus_kv": 110,
          "excluded_buses": [], "wind_profile_mode": "common", "wind_profile_csv": None,
          "relaxed_rating_multiplier": 1000.0, "counterfactual_first_year": True,
          "binding_threshold": 0.999, "max_unserved_mwh": 0.001,
          "mip_gap": 1e-6, "solver_time_limit_s": 180.0,
          "dlr_multiplier_csv": None, "dlr_label": "none"},
 "designs": {"wind_mw": [25, 50], "bess_mw_mwh": [[25, 100], [50, 200]],
             "hybrid_wind_bess_mw_mwh": [[50, 25, 100]], "maintenance_policies": ["replace"],
             "bess_portfolios": []},
 "battery": {"round_trip_efficiency": 0.90, "soc_min": 0.10, "soc_max": 0.90,
             "calendar_fade_per_year": 0.008, "cycle_fade_per_efc": 0.00003333333333333333,
             "replacement_soh": 0.80, "retire_soh": 0.60,
             "augmentation_trigger_soh": 0.90, "augmentation_target_soh": 1.00,
             "max_c_rate_per_hour": 1.0,
             "dispatch_wear_penalty_eur_mwh": 0.5},
 "wind": {"annual_output_degradation": 0.003},
 "costs": {"battery_energy_eur_kwh": 140.0, "battery_power_eur_kw": 250.0,
           "battery_other_bop_eur_kw": 80.0, "wind_eur_kw": 1300.0,
           "shared_site_fixed_eur": 300000.0, "connection_fixed_eur": 250000.0,
           "connection_eur_kw": 60.0, "contingency_fraction": 0.10,
           "battery_fixed_om_eur_kw_year": 7.0, "wind_fixed_om_eur_kw_year": 35.0,
           "site_rent_admin_eur_year": 30000.0, "insurance_fraction_initial_capex_year": 0.0025,
           "battery_variable_om_eur_mwh_discharged": 0.5,
           "wind_variable_om_eur_mwh_generated": 0.0,
           "real_om_escalation": 0.0,
           "replacement_energy_cost_fraction": 1.10,
           "augmentation_energy_cost_fraction": 1.15,
           "real_cell_cost_escalation": 0.0,
           "inverter_replacement_years": [12], "inverter_replacement_fraction_power_capex": 0.40,
           "wind_overhaul_years": [15], "wind_overhaul_fraction_wind_capex": 0.03,
           "decommission_fraction_initial_capex": 0.03,
           "residual_fraction_initial_capex": 0.01,
           "site_overrides": {}},
 "prices": {"mode": "flat", "flat_eur_mwh": 70.0, "csv": None,
            "import_adder_eur_mwh": 0.0, "export_fee_eur_mwh": 0.0,
            "real_escalation": 0.0,
            "grid_service_payment_eur_kw_year": 0.0},
 "sensitivity": {"enabled": True, "capex_multipliers": [0.8, 1.0, 1.2],
                 "om_multipliers": [0.8, 1.0, 1.2], "price_multipliers": [0.7, 1.0, 1.3],
                 "discount_rates": [0.04, 0.07, 0.10]}
}

def _merge(base, update, trail=""):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key {trail + key!r}; check spelling")
        if key == "site_overrides":
            base[key] = copy.deepcopy(value)
        elif isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{trail + key} must be an object")
            _merge(base[key], value, trail + key + ".")
        else:
            base[key] = copy.deepcopy(value)
    return base

def load_config(path: Path | None = None) -> dict:
    c = copy.deepcopy(DEFAULTS)
    if path is not None:
        with Path(path).open(encoding="utf-8") as f:
            _merge(c, json.load(f))
    validate_config(c)
    return c

def validate_config(c: dict) -> None:
    if c["schema_version"] != 1:
        raise ValueError("Unsupported config schema")
    if not isinstance(c["years"], int) or not 1 <= c["years"] <= 60:
        raise ValueError("years must be an integer in [1,60]")
    if not 0 <= c["real_discount_rate"] < 1:
        raise ValueError("Use a fractional real discount rate, e.g. 0.07, not 7")
    b = c["battery"]
    if not 0 < b["round_trip_efficiency"] <= 1:
        raise ValueError("round_trip_efficiency must be in (0,1]")
    if not 0 <= b["soc_min"] < b["soc_max"] <= 1:
        raise ValueError("Require 0 <= soc_min < soc_max <= 1")
    if not 0 < b["retire_soh"] < b["replacement_soh"] < 1:
        raise ValueError("Require 0 < retire_soh < replacement_soh < 1")
    if not 0 < b["augmentation_trigger_soh"] < b["augmentation_target_soh"] <= 1:
        raise ValueError("Invalid augmentation trigger/target")
    for k in ("calendar_fade_per_year", "cycle_fade_per_efc", "dispatch_wear_penalty_eur_mwh"):
        if not np.isfinite(b[k]) or b[k] < 0:
            raise ValueError(f"battery.{k} must be finite and nonnegative")
    if not 0 < b["max_c_rate_per_hour"] <= 10:
        raise ValueError("Invalid max C-rate")
    if not 0 <= c["wind"]["annual_output_degradation"] < 0.20:
        raise ValueError("Invalid wind degradation")
    a = c["annualisation"]
    if a["mode"] not in ("repeat_block", "full_year") or not 0 < a["hours_per_year"] <= 8784:
        raise ValueError("Invalid annualisation mode/hours")
    if c["prices"]["mode"] not in ("flat", "csv"):
        raise ValueError("prices.mode must be flat or csv")
    if c["prices"]["mode"] == "csv" and not c["prices"]["csv"]:
        raise ValueError("prices.csv is required for csv mode")
    if c["grid"]["wind_profile_mode"] not in ("common", "local", "csv"):
        raise ValueError("wind_profile_mode must be common, local or csv")
    if c["grid"]["wind_profile_mode"] == "csv" and not c["grid"]["wind_profile_csv"]:
        raise ValueError("grid.wind_profile_csv is required")
    if c["prices"]["grid_service_payment_eur_kw_year"] < 0:
        raise ValueError("Service payment must be nonnegative")
    for section in ("costs", "prices"):
        for k, v in c[section].items():
            if isinstance(v, (float, int)) and not np.isfinite(v):
                raise ValueError(f"Nonfinite {section}.{k}")
    for k, v in c["costs"].items():
        if isinstance(v, (int, float)) and "escalation" not in k and v < 0:
            raise ValueError(f"Negative costs.{k}")
    for k in ("real_om_escalation", "real_cell_cost_escalation"):
        if not -1 < c["costs"][k] < 1:
            raise ValueError(f"Invalid costs.{k}")
    if not -1 < c["prices"]["real_escalation"] < 1:
        raise ValueError("Invalid price escalation")
    allowed_overrides = {"connection_fixed_eur", "connection_eur_kw", "site_rent_admin_eur_year",
                         "shared_site_fixed_eur"}
    for bus, overrides in c["costs"]["site_overrides"].items():
        if set(overrides) - allowed_overrides:
            raise ValueError(f"Unsupported site cost overrides at {bus}")
        if any(not np.isfinite(v) or v < 0 for v in overrides.values()):
            raise ValueError(f"Invalid site costs at {bus}")
    if not c["designs"]["maintenance_policies"]:
        raise ValueError("Select at least one maintenance policy")
    g = c["grid"]
    for key in ("minimum_bus_kv", "max_unserved_mwh", "mip_gap", "solver_time_limit_s", "relaxed_rating_multiplier", "binding_threshold"):
        if not np.isfinite(g[key]):
            raise ValueError(f"Nonfinite grid.{key}")
    if g["minimum_bus_kv"] < 0 or g["max_unserved_mwh"] < 0 or g["solver_time_limit_s"] <= 0:
        raise ValueError("Invalid grid voltage, unserved energy tolerance or solver time limit")
    if not 0 <= g["mip_gap"] < 1 or not 0 < g["binding_threshold"] <= 1 or g["relaxed_rating_multiplier"] <= 1:
        raise ValueError("Invalid solver gap, binding threshold or relaxed rating multiplier")
    if g["dlr_multiplier_csv"] is not None and not str(g["dlr_multiplier_csv"]).strip():
        raise ValueError("grid.dlr_multiplier_csv must be null or a non-empty path")
    if not isinstance(g["dlr_label"], str) or not g["dlr_label"].strip():
        raise ValueError("grid.dlr_label must be a non-empty string")
    for key in ("inverter_replacement_years", "wind_overhaul_years"):
        if any(not isinstance(y, int) or isinstance(y, bool) or y < 1 for y in c["costs"][key]):
            raise ValueError(f"costs.{key} must contain positive integer project years")
    sens = c["sensitivity"]
    for key in ("capex_multipliers", "om_multipliers", "price_multipliers"):
        if any(not np.isfinite(v) or v < 0 for v in sens[key]):
            raise ValueError(f"sensitivity.{key} must be finite and nonnegative")
    if any(not np.isfinite(v) or not 0 <= v < 1 for v in sens["discount_rates"]):
        raise ValueError("sensitivity.discount_rates must be fractional real rates in [0,1)")
    for pol in c["designs"]["maintenance_policies"]:
        if pol not in ("replace", "run_down", "augment"):
            raise ValueError(f"Invalid maintenance policy {pol}")
    for val in c["designs"]["wind_mw"]:
        if not np.isfinite(val) or val <= 0: raise ValueError("wind_mw sizes must be positive")
    for row in c["designs"]["bess_mw_mwh"]:
        if len(row) != 2 or any(not np.isfinite(v) or v <= 0 for v in row):
            raise ValueError("Each BESS size must be [positive MW, positive MWh]")
    for row in c["designs"]["hybrid_wind_bess_mw_mwh"]:
        if len(row) != 3 or any(not np.isfinite(v) or v <= 0 for v in row):
            raise ValueError("Each hybrid size must be [wind MW, battery MW, battery MWh]")
    for portfolio in c["designs"]["bess_portfolios"]:
        if not isinstance(portfolio, dict) or set(portfolio) - {"sites", "allocations", "label"}:
            raise ValueError("Each BESS portfolio may contain only sites, allocations and label")
        sites = portfolio.get("sites", [])
        allocations = portfolio.get("allocations", [])
        if not sites or len(sites) != len(set(map(str, sites))):
            raise ValueError("BESS portfolio sites must be non-empty and unique")
        if len(allocations) != len(sites) or any(not np.isfinite(v) or v <= 0 for v in allocations):
            raise ValueError("BESS portfolio allocations must be positive and match sites")
        if not np.isclose(sum(allocations), 1.0):
            raise ValueError("BESS portfolio allocations must sum to one")
        if "label" in portfolio and (not isinstance(portfolio["label"], str) or not portfolio["label"].strip()):
            raise ValueError("BESS portfolio label must be a non-empty string")

def annual_factor(c: dict, block_hours: float) -> float:
    if block_hours <= 0:
        raise ValueError("Operating block must have positive duration")
    if c["annualisation"]["mode"] == "full_year":
        if not np.isclose(block_hours, c["annualisation"]["hours_per_year"], atol=1e-6):
            raise ValueError("full_year mode requires the full stated chronological year; do not relabel 168 h")
        return 1.0
    return c["annualisation"]["hours_per_year"] / block_hours
