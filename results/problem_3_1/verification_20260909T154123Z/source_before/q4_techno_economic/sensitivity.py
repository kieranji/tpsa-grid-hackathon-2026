"""One-at-a-time financial revaluation of FIXED physical operating paths.

This is not dispatch re-optimisation under alternative market prices. For altered
fade/efficiency/life assumptions, run the pipeline with a different config/output.
"""
from __future__ import annotations
import pandas as pd
from .finance import npv


def financial_sensitivity(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    cf = frame.net_cashflow_eur.to_numpy(float)
    s = config["sensitivity"]
    base_rate = config["real_discount_rate"]
    records = []
    vectors = {
        "capex_multiplier": -frame[["initial_capex_eur", "replacement_eur", "augmentation_eur",
            "inverter_replacement_eur", "wind_overhaul_eur", "decommission_eur"]].sum(axis=1).to_numpy()
            + frame.residual_eur.to_numpy() - frame.insurance_component_eur.to_numpy(),
        "om_multiplier": -frame[["fixed_om_eur", "variable_om_eur"]].sum(axis=1).to_numpy(),
        # Settlement includes fixed adders; this sensitivity scales the entire bill,
        # not only its energy-price component. Labelled explicitly in output.
        "energy_settlement_multiplier": (frame.energy_sales_eur - frame.energy_purchase_eur).to_numpy(),
    }
    values = {"capex_multiplier": s["capex_multipliers"], "om_multiplier": s["om_multipliers"],
              "energy_settlement_multiplier": s["price_multipliers"]}
    for parameter, vector in vectors.items():
        for value in values[parameter]:
            records.append({"parameter": parameter, "value": value,
                "npv_eur": npv(cf + (value - 1) * vector, base_rate),
                "method": "fixed_dispatch_and_replacement_schedule_financial_revaluation"})
    for rate in s["discount_rates"]:
        records.append({"parameter": "real_discount_rate", "value": rate,
            "npv_eur": npv(cf, rate), "method": "fixed_dispatch_and_replacement_schedule_financial_revaluation"})
    return pd.DataFrame(records)
