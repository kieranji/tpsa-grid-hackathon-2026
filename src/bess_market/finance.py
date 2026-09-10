"""Auditable, pretax, unlevered real-EUR BESS cash flows.

This module does not forecast prices, assume contract awards, or apply an implicit
degradation multiplier. Annual dispatch and accredited duration must come from the
caller after accounting for SOC limits, efficiency, availability and ageing.
Year zero contains construction CAPEX; operating cash flows occur at year end.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Iterable

import numpy as np


CAPACITY_REFERENCE_PRICES = {
    "2026_2027_T1": {
        "eur_per_derated_mw_year": 53978.87,
        "publication_date": "2026-08-26",
        "source_url": "https://www.sem-o.com/sites/semo/files/2026-08/Final%20Capacity%20Auction%20Results%20report%20FCAR2627T-1.pdf",
    },
    "2029_2030_T4": {
        "eur_per_derated_mw_year": 135499.99,
        "publication_date": "2026-05-05",
        "source_url": "https://www.sem-o.com/sites/semo/files/2026-05/2029_2030%20T-4%20Final%20Capacity%20Auction%20Results%20Report%20FCAR2930T-4.pdf",
    },
}
DERATING_SOURCE_URL = "https://www.sem-o.com/sites/semo/files/2025-08/IAIP2930T-4.pdf"
REFERENCE_DURATIONS_HOURS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
REFERENCE_DERATING_FACTORS = (0.036, 0.072, 0.107, 0.142, 0.210, 0.276, 0.417, 0.537, 0.665)


def _finite(name: str, value: float, minimum: float | None = None) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number, not bool")
    number = float(value)
    if not isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return number


def _integer(name: str, value: int, minimum: int) -> int:
    number = _finite(name, value, minimum)
    if number != int(number):
        raise ValueError(f"{name} must be an integer")
    return int(number)


def reference_storage_derating(duration_hours: float, power_mw: float = 85.0) -> float:
    """Interpolate the official 2029/30 *initial* 80 < MW <= 90 storage curve.

    This is a historical qualification reference, not project accreditation or a
    forecast for 2033. Duration is deliverable AC energy / accredited export MW,
    not DC/nameplate MWh divided by inverter MW. Values outside the documented
    duration and power range are rejected rather than silently extrapolated.
    """
    power = _finite("power_mw", power_mw)
    duration = _finite("duration_hours", duration_hours)
    if not 80.0 < power <= 90.0:
        raise ValueError("Reference curve supports only 80 < power_mw <= 90; obtain the applicable official curve")
    if not REFERENCE_DURATIONS_HOURS[0] <= duration <= REFERENCE_DURATIONS_HOURS[-1]:
        raise ValueError("Reference curve supports only 0.5 <= duration_hours <= 12; no extrapolation")
    return float(np.interp(duration, REFERENCE_DURATIONS_HOURS, REFERENCE_DERATING_FACTORS))


def capacity_cashflow(
    year: int,
    *,
    power_mw: float,
    duration_hours: float,
    price_eur_per_derated_mw_year: float,
    award_fraction: float,
    start_year: int,
    term_years: int,
    difference_charges_eur: float = 0.0,
    other_costs_eur: float = 0.0,
) -> dict:
    """Conditional capacity cash flow with explicit award, term and deductions.

    The fraction is a contracted fraction of reference qualified capacity, not an
    undisclosed probability. A project with no award should use zero. Call for
    each year using that year's physically supportable/accredited duration.
    Explicit incurred costs remain payable even outside the revenue term.
    """
    year = _integer("year", year, 1)
    start = _integer("start_year", start_year, 1)
    term = _integer("term_years", term_years, 0)
    fraction = _finite("award_fraction", award_fraction, 0.0)
    if fraction > 1:
        raise ValueError("award_fraction must be <= 1")
    price = _finite("price_eur_per_derated_mw_year", price_eur_per_derated_mw_year, 0.0)
    difference = _finite("difference_charges_eur", difference_charges_eur, 0.0)
    costs = _finite("other_costs_eur", other_costs_eur, 0.0)
    factor = reference_storage_derating(duration_hours, power_mw)
    active = start <= year < start + term
    qualified_mw = float(power_mw) * factor
    awarded_mw = qualified_mw * fraction if active else 0.0
    gross = awarded_mw * price
    return {
        "year": year,
        "contract_active": active,
        "reference_derating_factor": factor,
        "reference_qualified_mw": qualified_mw,
        "awarded_derated_mw": awarded_mw,
        "gross_capacity_revenue_eur": gross,
        "capacity_difference_charges_eur": difference,
        "other_capacity_costs_eur": costs,
        "net_capacity_cashflow_eur": gross - difference - costs,
        "price_is_guaranteed": False,
        "basis": "caller_assumed_award_and_price_with_historical_reference_derating",
    }


@dataclass(frozen=True)
class AnnualOperatingInputs:
    """One operating year's realized or scenario cash flows, in constant EUR.

    Energy-sale proceeds and purchases can be negative at negative energy prices.
    Purchases include charging energy; house-load costs are only the incremental
    costs not already included there. System benefits are informational only.
    Service activation/recharge must be reflected in energy and throughput inputs.
    Fees are explicit cash costs; the caller chooses the contractual fee basis.
    """
    year: int
    energy_sales_eur: float = 0.0
    energy_purchase_eur: float = 0.0
    balancing_net_eur: float = 0.0
    services_revenue_eur: float = 0.0
    capacity_revenue_eur: float = 0.0
    contracted_other_net_revenue_eur: float = 0.0
    capacity_difference_charges_eur: float = 0.0
    service_nonperformance_cost_eur: float = 0.0
    fees_eur: float = 0.0
    fixed_opex_eur: float = 0.0
    discharge_throughput_mwh: float = 0.0
    variable_opex_eur_per_mwh: float = 0.0
    generator_tuos_eur: float = 0.0
    house_load_grid_cost_eur: float = 0.0
    other_grid_cost_eur: float = 0.0
    replacement_capex_eur: float = 0.0
    augmentation_capex_eur: float = 0.0
    decommissioning_cost_eur: float = 0.0
    residual_value_eur: float = 0.0
    fixed_opex_includes_replacement_and_augmentation: bool = False
    system_benefit_eur: float = 0.0
    avoided_curtailment_value_eur: float = 0.0


_SIGNED_FIELDS = {
    "energy_sales_eur", "energy_purchase_eur", "balancing_net_eur",
    "contracted_other_net_revenue_eur", "system_benefit_eur",
    "avoided_curtailment_value_eur",
}


def annual_cashflow(inputs: AnnualOperatingInputs) -> dict:
    """Cash ledger; avoided curtailment/system benefits never become project cash."""
    if not isinstance(inputs, AnnualOperatingInputs):
        raise TypeError("inputs must be AnnualOperatingInputs")
    row = asdict(inputs)
    row["year"] = _integer("year", inputs.year, 1)
    inclusive = row["fixed_opex_includes_replacement_and_augmentation"]
    if not isinstance(inclusive, bool):
        raise ValueError("fixed_opex_includes_replacement_and_augmentation must be bool")
    for name, value in row.items():
        if name not in {"year", "fixed_opex_includes_replacement_and_augmentation"}:
            row[name] = _finite(name, value, None if name in _SIGNED_FIELDS else 0.0)
    if inclusive and (row["replacement_capex_eur"] > 0 or row["augmentation_capex_eur"] > 0):
        raise ValueError("Replacement/augmentation cash CAPEX duplicates the inclusive fixed OPEX assumption")
    energy_margin = row["energy_sales_eur"] - row["energy_purchase_eur"] + row["balancing_net_eur"]
    project_margin = (
        energy_margin + row["services_revenue_eur"] + row["capacity_revenue_eur"]
        + row["contracted_other_net_revenue_eur"]
    )
    variable_opex = row["discharge_throughput_mwh"] * row["variable_opex_eur_per_mwh"]
    operating_cost = (
        row["capacity_difference_charges_eur"] + row["service_nonperformance_cost_eur"]
        + row["fees_eur"] + row["fixed_opex_eur"] + variable_opex
        + row["generator_tuos_eur"] + row["house_load_grid_cost_eur"] + row["other_grid_cost_eur"]
    )
    maintenance = row["replacement_capex_eur"] + row["augmentation_capex_eur"]
    operating_net = project_margin - operating_cost
    row.update({
        "net_energy_margin_eur": energy_margin,
        "project_margin_before_operating_cost_eur": project_margin,
        "variable_opex_eur": variable_opex,
        "operating_cost_eur": operating_cost,
        "net_operating_cashflow_eur": operating_net,
        "maintenance_capex_eur": maintenance,
        "net_cashflow_eur": operating_net - maintenance - row["decommissioning_cost_eur"] + row["residual_value_eur"],
        "system_benefit_included_in_cashflow": False,
        "avoided_curtailment_value_included_in_cashflow": False,
        "degradation_applied_by_finance_layer": False,
    })
    return row


def _cashflow_array(cashflows: Iterable[float]) -> np.ndarray:
    cf = np.asarray(list(cashflows), dtype=float)
    if cf.ndim != 1 or not len(cf) or not np.isfinite(cf).all():
        raise ValueError("cashflows must be a nonempty finite 1-D sequence")
    return cf


def npv(cashflows: Iterable[float], discount_rate: float) -> float:
    """Year-zero-inclusive NPV; discount rate is real and must exceed -100%."""
    cf = _cashflow_array(cashflows)
    rate = _finite("discount_rate", discount_rate)
    if rate <= -1:
        raise ValueError("discount_rate must be > -1")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = float(np.dot(cf, (1 + rate) ** -np.arange(len(cf), dtype=float)))
        except FloatingPointError as exc:
            raise ValueError("Discount factors overflow; use numerically meaningful rates/horizon") from exc
    if not isfinite(result):
        raise ValueError("NPV is nonfinite")
    return result


def irr_summary(cashflows: Iterable[float]) -> dict:
    """All real IRRs > -100%; return None instead of selecting an ambiguous root."""
    cf = _cashflow_array(cashflows)
    if np.all(cf == 0):
        return {"irr": None, "irr_status": "indeterminate_all_zero", "irr_roots": []}
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return {"irr": None, "irr_status": "no_real_irr", "irr_roots": []}
    scaled = cf / np.max(np.abs(cf))
    roots = np.polynomial.polynomial.polyroots(scaled)
    rates = []
    for root in roots:
        if abs(root.imag) > 1e-7 * max(1.0, abs(root.real)) or root.real <= 0:
            continue
        rate = 1.0 / root.real - 1.0
        if not isfinite(rate) or rate <= -1:
            continue
        try:
            residual = npv(scaled, rate)
        except ValueError:
            continue
        if abs(residual) <= 1e-5 * max(1.0, np.abs(scaled).sum()):
            if not any(abs(rate - previous) < 1e-6 for previous in rates):
                rates.append(float(rate))
    rates.sort()
    return {
        "irr": rates[0] if len(rates) == 1 else None,
        "irr_status": "unique" if len(rates) == 1 else "multiple_roots" if rates else "no_real_irr",
        "irr_roots": rates,
    }


def payback_years(
    cashflows: Iterable[float], discount_rate: float = 0.0, *, sustained: bool = False,
) -> float | None:
    """First or sustained cumulative recovery, with within-year linear interpolation."""
    cf = _cashflow_array(cashflows)
    npv(cf, discount_rate)  # Validate rate and finite discount factors.
    discounted = cf / (1 + discount_rate) ** np.arange(len(cf), dtype=float)
    cumulative = np.cumsum(discounted)
    if cumulative[0] >= 0 and (not sustained or np.all(cumulative >= 0)):
        return 0.0
    for year in range(1, len(cf)):
        if cumulative[year - 1] < 0 <= cumulative[year]:
            if sustained and np.any(cumulative[year:] < -1e-8):
                continue
            return float(year - 1 - cumulative[year - 1] / discounted[year])
    return None


def break_even_annual_net_contract(
    cashflows: Iterable[float],
    discount_rate: float,
    *,
    start_year: int = 1,
    term_years: int | None = None,
) -> dict:
    """Level additional *net* payment required to bring project NPV to zero.

    No tax, fees or dispatch opportunity cost is deducted from this supplemental
    amount: those must already have been accounted for to call it net. A payment
    can only run within the cash-flow horizon. A profitable project requires zero.
    """
    cf = _cashflow_array(cashflows)
    present_value = npv(cf, discount_rate)
    start = _integer("start_year", start_year, 1)
    term = len(cf) - start if term_years is None else _integer("term_years", term_years, 1)
    if term < 1 or start + term > len(cf):
        raise ValueError("Supplemental contract must have positive term within the project horizon")
    years = np.arange(start, start + term, dtype=float)
    factor = float(np.sum((1 + discount_rate) ** -years))
    annual = max(0.0, -present_value) / factor
    return {
        "additional_annual_net_contract_eur": annual,
        "contract_start_year": start,
        "contract_term_years": term,
        "contract_annuity_pv_factor": factor,
        "project_npv_before_contract_eur": present_value,
        "project_npv_after_contract_eur": present_value + annual * factor,
        "contract_is_assumed_not_awarded": True,
    }


def project_cashflows(
    upfront_capex_eur: float,
    annual_inputs: Iterable[AnnualOperatingInputs],
    discount_rate: float,
) -> dict:
    """Build lifecycle cash flows from explicitly supplied contiguous operating years."""
    capex = _finite("upfront_capex_eur", upfront_capex_eur, 0.0)
    rows = [annual_cashflow(inputs) for inputs in annual_inputs]
    if not rows or [row["year"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("annual_inputs must cover years 1..N exactly, in order")
    cashflows = [-capex] + [row["net_cashflow_eur"] for row in rows]
    summary = {
        "upfront_capex_eur": capex,
        "discount_rate": float(discount_rate),
        "operating_years": len(rows),
        "cashflows_eur": cashflows,
        "annual_ledger": rows,
        "npv_eur": npv(cashflows, discount_rate),
        **irr_summary(cashflows),
        "simple_payback_years": payback_years(cashflows),
        "discounted_payback_years": payback_years(cashflows, discount_rate),
        "sustained_simple_payback_years": payback_years(cashflows, sustained=True),
        "sustained_discounted_payback_years": payback_years(cashflows, discount_rate, sustained=True),
        "lifetime_net_cashflow_eur": float(sum(cashflows)),
        "lifetime_maintenance_capex_eur": float(sum(row["maintenance_capex_eur"] for row in rows)),
        "lifetime_discharge_throughput_mwh": float(sum(row["discharge_throughput_mwh"] for row in rows)),
        "cashflow_basis": "pretax_unlevered_constant_real_eur_year_end",
        "implicit_degradation_or_revenue_growth": False,
        "system_benefits_included_in_project_cashflow": False,
        "additional_break_even_net_contract": break_even_annual_net_contract(cashflows, discount_rate),
    }
    return summary

