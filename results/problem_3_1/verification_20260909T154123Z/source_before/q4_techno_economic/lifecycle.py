"""Annual re-dispatch, cohort battery ageing and explicit cash accounting.

Calendar and cycling fade are simple linear scenario parameters, NOT calibrated
chemistry predictions. Capacity is piecewise constant within each model year;
ageing and maintenance occur at year end. Every subsequent year is re-solved.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .types import Design, Dispatch, Backend
from .config import annual_factor
from .finance import summarise_cashflow

@dataclass
class Cohort:
    nominal_mwh: float
    soh: float = 1.0

class BatteryFleet:
    def __init__(self, nominal_mwh: float, parameters: dict, policy: str):
        self.original = nominal_mwh
        self.parameters = parameters
        self.policy = policy
        self.cohorts = [Cohort(nominal_mwh)] if nominal_mwh else []

    @property
    def installed_mwh(self) -> float:
        return sum(c.nominal_mwh for c in self.cohorts)

    @property
    def available_mwh(self) -> float:
        # Out-of-service cohorts contribute no usable capacity or power.
        return sum(c.nominal_mwh * c.soh for c in self.cohorts
                   if c.soh >= self.parameters["retire_soh"])

    def age(self, internal_discharge_mwh: float) -> dict:
        if not np.isfinite(internal_discharge_mwh) or internal_discharge_mwh < -1e-8:
            raise ValueError("Internal discharged energy must be nonnegative")
        active_total = self.available_mwh
        efc = internal_discharge_mwh / self.original if self.original else 0.0
        for c in self.cohorts:
            share = c.nominal_mwh * c.soh / active_total if active_total > 0 and c.soh >= self.parameters["retire_soh"] else 0.0
            cohort_efc = internal_discharge_mwh * share / c.nominal_mwh
            c.soh = max(0.0, c.soh - self.parameters["calendar_fade_per_year"]
                        - self.parameters["cycle_fade_per_efc"] * cohort_efc)
        return {"efc": efc, "post_age_available_mwh": self.available_mwh}

    def maintain(self, another_year: bool) -> tuple[float, float]:
        """Returns replaced and newly added nameplate MWh. No final-year replacement."""
        if not another_year or self.original == 0:
            return 0.0, 0.0
        ratio = self.available_mwh / self.original
        if self.policy == "replace" and ratio < self.parameters["replacement_soh"]:
            self.cohorts = [Cohort(self.original)]
            return self.original, 0.0
        if self.policy == "augment" and ratio < self.parameters["augmentation_trigger_soh"]:
            added = max(0.0, self.original * self.parameters["augmentation_target_soh"] - self.available_mwh)
            self.cohorts.append(Cohort(added))
            return 0.0, added
        return 0.0, 0.0


def costs_for_site(config: dict, bus: str) -> dict:
    costs = dict(config["costs"])
    costs.update(costs["site_overrides"].get(str(bus), {}))
    return costs


def capital_cost(design: Design, c: dict) -> dict:
    if design.technology == "no_build":
        return {"initial_capex_eur": 0.0, "battery_energy_capex_eur": 0.0,
                "battery_power_capex_eur": 0.0, "wind_capex_eur": 0.0}
    energy = 1000 * design.battery_mwh * c["battery_energy_eur_kwh"]
    power = 1000 * design.battery_mw * c["battery_power_eur_kw"]
    bop = 1000 * design.battery_mw * c["battery_other_bop_eur_kw"]
    wind = 1000 * design.wind_mw * c["wind_eur_kw"]
    connection = c["connection_fixed_eur"] + 1000 * max(design.export_mw, design.import_mw) * c["connection_eur_kw"]
    before = energy + power + bop + wind + connection + c["shared_site_fixed_eur"]
    return {"initial_capex_eur": before * (1 + c["contingency_fraction"]),
            "battery_energy_capex_eur": energy, "battery_power_capex_eur": power,
            "battery_other_bop_capex_eur": bop, "wind_capex_eur": wind,
            "connection_capex_eur": connection, "shared_site_capex_eur": c["shared_site_fixed_eur"],
            "contingency_eur": before * c["contingency_fraction"]}


def settle_dispatch(result: Dispatch, prices: np.ndarray, design: Design, config: dict) -> dict:
    """One common-meter settlement: internal wind-to-battery transfers are not sold.

    External purchases are NOT free, even when renewable dispatch-down is reduced.
    The grid-service payment is a hypothetical availability contract (default 0),
    not inferred DS3/FASS revenue and not an independently stacked reserve service.
    """
    dt = result.duration_h
    if np.asarray(prices).shape != dt.shape or not np.isfinite(prices).all():
        raise ValueError("Price array must be finite and align exactly with snapshots")
    n = result.project_wind_mw + result.battery_discharge_mw - result.battery_charge_mw
    export, imports = np.maximum(n, 0), np.maximum(-n, 0)
    p = config["prices"]
    return {
        "export_mwh": float(export @ dt), "import_mwh": float(imports @ dt),
        "energy_sales_eur": float((export * (prices - p["export_fee_eur_mwh"])) @ dt),
        "energy_purchase_eur": float((imports * (prices + p["import_adder_eur_mwh"])) @ dt),
        "wind_generated_mwh": float(result.project_wind_mw @ dt),
        "battery_charge_mwh": float(result.battery_charge_mw @ dt),
        "battery_discharge_mwh": float(result.battery_discharge_mw @ dt),
    }


CASH_COLUMNS = ["initial_capex_eur", "energy_sales_eur", "grid_service_eur", "energy_purchase_eur",
    "fixed_om_eur", "variable_om_eur", "replacement_eur", "augmentation_eur",
    "inverter_replacement_eur", "wind_overhaul_eur", "decommission_eur", "residual_eur"]
ENERGY_COLUMNS = ["export_mwh", "import_mwh", "wind_generated_mwh", "battery_charge_mwh",
                  "battery_discharge_mwh", "net_renewable_gain_proxy_mwh", "efc", "support_eligible_kw_year"]


def simulate_lifetime(design: Design, backend: Backend, config: dict, base: Dispatch,
                      prices: np.ndarray, progress=None) -> tuple[pd.DataFrame, dict, Dispatch]:
    c = costs_for_site(config, design.bus)
    cap = capital_cost(design, c)
    capex = cap["initial_capex_eur"]
    fleet = BatteryFleet(design.battery_mwh, config["battery"], design.maintenance)
    factor = annual_factor(config, base.hours)
    r = config["real_discount_rate"]
    zero = {k: 0.0 for k in CASH_COLUMNS + ENERGY_COLUMNS}
    zero.update(year=0, initial_capex_eur=capex, net_cashflow_eur=-capex,
                available_battery_mwh=design.battery_mwh, installed_battery_mwh=design.battery_mwh,
                post_age_available_battery_mwh=design.battery_mwh, replaced_mwh=0.0, augmented_mwh=0.0)
    rows = [zero]
    first_result = None
    for year in range(1, config["years"] + 1):
        available = fleet.available_mwh
        installed = fleet.installed_mwh
        wind_factor = (1 - config["wind"]["annual_output_degradation"]) ** (year - 1)
        result = backend.solve(design, available, wind_factor)
        if first_result is None:
            first_result = result
        if result.metrics["unserved_mwh"] > config["grid"]["max_unserved_mwh"]:
            raise RuntimeError(f"{design.id}: load shedding in year {year}; no investable NPV will be reported")
        year_prices = prices * (1 + config["prices"]["real_escalation"]) ** (year - 1)
        row = {k: 0.0 for k in CASH_COLUMNS + ENERGY_COLUMNS}
        row.update({k: v * factor for k, v in settle_dispatch(result, year_prices, design, config).items()})
        row.update(year=year, available_battery_mwh=available, installed_battery_mwh=installed,
                   operating_block_hours=result.hours, annualisation_factor=factor,
                   annual_dispatch_down_mwh=result.metrics["renewable_dispatch_down_mwh"] * factor,
                   # Not traced green energy: subtract project battery losses from fleet RE gain.
                   net_renewable_gain_proxy_mwh=(result.metrics["renewable_dispatched_mwh"]
                      - base.metrics["renewable_dispatched_mwh"]
                      - result.metrics["battery_loss_mwh"]) * factor,
                   model_dispatch_cost_saving_eur=(base.metrics["generator_dispatch_cost_eur"]
                      - result.metrics["generator_dispatch_cost_eur"]) * factor)
        # Revenue is based on PCC sales only. Model system cost savings above are
        # deliberately EXCLUDED from cash flow; they are not developer income.
        om_scale = (1 + c["real_om_escalation"]) ** (year - 1)
        row["insurance_component_eur"] = om_scale * capex * c["insurance_fraction_initial_capex_year"]
        row["fixed_om_eur"] = om_scale * (
            design.battery_mw * 1000 * c["battery_fixed_om_eur_kw_year"]
            + design.wind_mw * 1000 * c["wind_fixed_om_eur_kw_year"]
            + c["site_rent_admin_eur_year"] + capex * c["insurance_fraction_initial_capex_year"])
        row["variable_om_eur"] = om_scale * (
            row["battery_discharge_mwh"] * c["battery_variable_om_eur_mwh_discharged"]
            + row["wind_generated_mwh"] * c["wind_variable_om_eur_mwh_generated"])
        # Capacity-payment exposure derates with energy capacity and power cap.
        power_available = min(design.battery_mw, available * config["battery"]["max_c_rate_per_hour"])
        energy_ratio = min(1.0, available / design.battery_mwh) if design.battery_mwh else 0.0
        row["support_eligible_kw_year"] = 1000 * power_available * energy_ratio
        row["grid_service_eur"] = row["support_eligible_kw_year"] * config["prices"]["grid_service_payment_eur_kw_year"]
        # Internal DC discharge / DC original nameplate is a consistent EFC.
        # By cyclic SoC, this equals average internal charge/discharge throughput.
        internal_discharge = row["battery_discharge_mwh"] / np.sqrt(config["battery"]["round_trip_efficiency"])
        ageing = fleet.age(internal_discharge)
        replaced, augmented = fleet.maintain(year < config["years"])
        row.update(efc=ageing["efc"], post_age_available_battery_mwh=ageing["post_age_available_mwh"],
                   replaced_mwh=replaced, augmented_mwh=augmented)
        cell_scale = (1 + c["real_cell_cost_escalation"]) ** year
        row["replacement_eur"] = 1000 * replaced * c["battery_energy_eur_kwh"] * c["replacement_energy_cost_fraction"] * cell_scale
        row["augmentation_eur"] = 1000 * augmented * c["battery_energy_eur_kwh"] * c["augmentation_energy_cost_fraction"] * cell_scale
        if year < config["years"] and year in c["inverter_replacement_years"] and fleet.available_mwh > 0:
            row["inverter_replacement_eur"] = cap["battery_power_capex_eur"] * c["inverter_replacement_fraction_power_capex"]
        if year < config["years"] and year in c["wind_overhaul_years"]:
            row["wind_overhaul_eur"] = cap["wind_capex_eur"] * c["wind_overhaul_fraction_wind_capex"]
        if year == config["years"]:
            row["decommission_eur"] = capex * c["decommission_fraction_initial_capex"]
            row["residual_eur"] = capex * c["residual_fraction_initial_capex"]
        row["net_cashflow_eur"] = (row["energy_sales_eur"] + row["grid_service_eur"] + row["residual_eur"]
             - sum(row[k] for k in CASH_COLUMNS if k not in ("energy_sales_eur", "grid_service_eur", "residual_eur")))
        rows.append(row)
        if progress:
            progress(year, result)
    frame = pd.DataFrame(rows).fillna(0.0)
    frame["discount_factor"] = (1 + r) ** -frame["year"]
    frame["discounted_cashflow_eur"] = frame["net_cashflow_eur"] * frame["discount_factor"]
    frame["cumulative_cashflow_eur"] = frame["net_cashflow_eur"].cumsum()
    frame["cumulative_discounted_cashflow_eur"] = frame["discounted_cashflow_eur"].cumsum()
    summary = summarise_cashflow(frame, r, design.technology)
    summary.update(cap)
    return frame, summary, first_result
