"""Discounted cash-flow mathematics, with explicit timing and IRR ambiguity.

All cash flows occur at year end, except the year-0 initial investment.
Amounts use a single real-EUR base year. No tax, debt or depreciation tax shield.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def npv(cashflows, rate: float) -> float:
    cf = np.asarray(cashflows, dtype=float)
    if cf.ndim != 1 or not np.isfinite(cf).all() or not np.isfinite(rate) or rate <= -1:
        raise ValueError("NPV requires finite 1-D cash flows and rate > -1")
    return float(np.dot(cf, (1 + rate) ** -np.arange(len(cf), dtype=float)))


def irr_roots(cashflows) -> list[float]:
    """All real IRRs greater than -100%; never silently choose one of many roots."""
    cf = np.asarray(cashflows, dtype=float)
    if cf.ndim != 1 or not np.isfinite(cf).all():
        raise ValueError("IRR requires finite 1-D cash flows")
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return []
    # NPV = sum(CF_y * x**y), x = 1/(1+r). cf is in ascending order.
    roots = np.polynomial.polynomial.polyroots(cf)
    result = []
    for z in roots:
        if abs(z.imag) > 1e-7 * max(1, abs(z.real)) or z.real <= 0:
            continue
        rate = 1 / z.real - 1
        if np.isfinite(rate) and abs(npv(cf, rate)) < 1e-5 * max(1, np.abs(cf).sum()):
            if not any(abs(rate - v) < 1e-6 for v in result):
                result.append(float(rate))
    return sorted(result)


def payback(cashflows, rate: float = 0.0, sustained: bool = False) -> float:
    """First (or sustained) cumulative crossing. NaN means no payback in horizon.

    Linear within-year interpolation is a reporting convention; annual accounting
    does not imply that the timing within each operating year was modelled.
    A later replacement can undo a first crossing; sustained=True flags this.
    """
    cf = np.asarray(cashflows, dtype=float)
    if len(cf) == 0 or not np.isfinite(cf).all() or rate <= -1:
        raise ValueError("Invalid payback inputs")
    discounted = cf / (1 + rate) ** np.arange(len(cf))
    cumulative = np.cumsum(discounted)
    if cumulative[0] >= 0 and (not sustained or np.all(cumulative >= -1e-8)):
        return 0.0
    for y in range(1, len(cf)):
        if cumulative[y - 1] < 0 <= cumulative[y]:
            if sustained and np.any(cumulative[y:] < -1e-8):
                continue
            return float(y - 1 + (-cumulative[y - 1]) / discounted[y])
    return float("nan")


def capital_recovery_factor(rate: float, years: int) -> float:
    if years < 1 or rate <= -1:
        raise ValueError("Invalid annuity parameters")
    return 1 / years if abs(rate) < 1e-12 else rate / (1 - (1 + rate) ** -years)


def pareto_flags(benefit, return_value) -> np.ndarray:
    """Both dimensions are maximised; ties do not dominate one another."""
    a, b = np.asarray(benefit, float), np.asarray(return_value, float)
    keep = np.isfinite(a) & np.isfinite(b)
    for i in np.flatnonzero(keep.copy()):
        dominating = (a >= a[i] - 1e-8) & (b >= b[i] - 1e-8)
        strictly = (a > a[i] + 1e-8) | (b > b[i] + 1e-8)
        if np.any(dominating & strictly):
            keep[i] = False
    return keep


def summarise_cashflow(frame: pd.DataFrame, rate: float, technology: str) -> dict:
    cf = frame["net_cashflow_eur"].to_numpy(float)
    discount = (1 + rate) ** -frame["year"].to_numpy(float)
    roots = irr_roots(cf)
    cash_npv = float(np.dot(cf, discount))
    # No amortisation/depreciation or hypothetical wear charge in this numerator.
    cost_columns = ["initial_capex_eur", "fixed_om_eur", "variable_om_eur",
                    "energy_purchase_eur", "replacement_eur", "augmentation_eur",
                    "inverter_replacement_eur", "wind_overhaul_eur", "decommission_eur"]
    costs = frame[cost_columns].sum(axis=1) - frame["residual_eur"]
    pv_cost = float(np.dot(costs, discount))
    def levelised(column):
        energy = float(np.dot(frame[column], discount))
        return pv_cost / energy if energy > 1e-9 else np.nan
    support_exposure = float(np.dot(frame["support_eligible_kw_year"], discount))
    out = {
        "npv_eur": cash_npv,
        "irr": roots[0] if len(roots) == 1 else np.nan,
        "irr_status": "unique" if len(roots) == 1 else "multiple" if len(roots) > 1 else "no_real_irr",
        "all_irrs": ";".join(f"{r:.10g}" for r in roots),
        "simple_payback_years": payback(cf),
        "discounted_payback_years": payback(cf, rate),
        "sustained_discounted_payback_years": payback(cf, rate, True),
        "lifecycle_cost_pv_eur": pv_cost,
        "equivalent_annual_cost_eur": pv_cost * capital_recovery_factor(rate, len(cf) - 1),
        "lcos_eur_mwh": levelised("battery_discharge_mwh") if technology == "bess" else np.nan,
        "delivered_lcoe_eur_mwh": levelised("export_mwh") if technology == "wind" else np.nan,
        "hybrid_levelised_export_cost_eur_mwh": levelised("export_mwh") if technology == "hybrid" else np.nan,
        "additional_break_even_payment_eur_kw_year": max(0, -cash_npv) / support_exposure if support_exposure > 0 else np.nan,
        "initial_capex_eur": float(frame["initial_capex_eur"].sum()),
        "undiscounted_lifetime_replacements_eur": float(frame[["replacement_eur", "augmentation_eur", "inverter_replacement_eur", "wind_overhaul_eur"]].sum().sum()),
        "year_1_net_cashflow_eur": float(cf[1]),
        "lifetime_net_renewable_gain_mwh": float(frame["net_renewable_gain_proxy_mwh"].sum()),
        "lifetime_battery_discharge_mwh": float(frame["battery_discharge_mwh"].sum()),
        "lifetime_efc": float(frame["efc"].sum()),
    }
    return out
