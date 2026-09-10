"""Specification tests for the independent merchant BESS cash ledger."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bess_market.finance import (
    AnnualOperatingInputs,
    annual_cashflow,
    break_even_annual_net_contract,
    capacity_cashflow,
    irr_summary,
    npv,
    payback_years,
    project_cashflows,
    reference_storage_derating,
)


class CapacityTests(unittest.TestCase):
    def test_duration_reference_and_scope(self):
        self.assertEqual(reference_storage_derating(4.0), 0.276)
        self.assertAlmostEqual(reference_storage_derating(3.5), 0.243)
        self.assertEqual(reference_storage_derating(2.0, 90.0), 0.142)
        for duration, power in [(4, 80), (4, 91), (0.49, 85), (12.1, 85), (float("nan"), 85)]:
            with self.subTest(duration=duration, power=power), self.assertRaises(ValueError):
                reference_storage_derating(duration, power)

    def test_award_fraction_term_and_actual_deductions(self):
        params = dict(power_mw=85, duration_hours=4, price_eur_per_derated_mw_year=100,
                      award_fraction=0.5, start_year=2, term_years=2)
        self.assertEqual(capacity_cashflow(1, **params)["gross_capacity_revenue_eur"], 0)
        award = capacity_cashflow(2, **params, difference_charges_eur=50, other_costs_eur=10)
        self.assertAlmostEqual(award["awarded_derated_mw"], 11.73)
        self.assertAlmostEqual(award["net_capacity_cashflow_eur"], 1113)
        self.assertGreater(capacity_cashflow(3, **params)["gross_capacity_revenue_eur"], 0)
        self.assertEqual(capacity_cashflow(4, **params)["gross_capacity_revenue_eur"], 0)
        self.assertEqual(capacity_cashflow(4, **params, other_costs_eur=10)["net_capacity_cashflow_eur"], -10)
        self.assertFalse(award["price_is_guaranteed"])
        params["award_fraction"] = 0
        self.assertEqual(capacity_cashflow(2, **params)["net_capacity_cashflow_eur"], 0)
        params["award_fraction"] = 1.1
        with self.assertRaises(ValueError):
            capacity_cashflow(2, **params)

    def test_deliverable_duration_reduction_changes_accredited_mw(self):
        params = dict(power_mw=85, price_eur_per_derated_mw_year=100,
                      award_fraction=1, start_year=1, term_years=20)
        beginning = capacity_cashflow(1, duration_hours=4, **params)
        aged = capacity_cashflow(10, duration_hours=3, **params)
        self.assertAlmostEqual(aged["gross_capacity_revenue_eur"] / beginning["gross_capacity_revenue_eur"],
                               0.210 / 0.276)


class LedgerTests(unittest.TestCase):
    def test_complete_cash_ledger_excludes_system_benefits(self):
        row = annual_cashflow(AnnualOperatingInputs(
            year=1, energy_sales_eur=1000, energy_purchase_eur=400, balancing_net_eur=-30,
            services_revenue_eur=200, capacity_revenue_eur=100, contracted_other_net_revenue_eur=50,
            capacity_difference_charges_eur=25, service_nonperformance_cost_eur=5,
            fees_eur=20, fixed_opex_eur=30, discharge_throughput_mwh=100,
            variable_opex_eur_per_mwh=0.2, generator_tuos_eur=10,
            house_load_grid_cost_eur=4, other_grid_cost_eur=6,
            replacement_capex_eur=150, augmentation_capex_eur=50,
            decommissioning_cost_eur=30, residual_value_eur=20,
            system_benefit_eur=1000000, avoided_curtailment_value_eur=900000,
        ))
        self.assertEqual(row["net_energy_margin_eur"], 570)
        self.assertEqual(row["net_operating_cashflow_eur"], 800)
        self.assertEqual(row["net_cashflow_eur"], 590)
        self.assertFalse(row["system_benefit_included_in_cashflow"])
        self.assertFalse(row["avoided_curtailment_value_included_in_cashflow"])

    def test_negative_price_charging_is_cash_credit(self):
        row = annual_cashflow(AnnualOperatingInputs(year=1, energy_sales_eur=10, energy_purchase_eur=-20))
        self.assertEqual(row["net_cashflow_eur"], 30)

    def test_inclusive_fixed_opex_rejects_duplicate_replacement_or_augmentation(self):
        for capex in [{"replacement_capex_eur": 1}, {"augmentation_capex_eur": 1}]:
            with self.subTest(capex=capex), self.assertRaisesRegex(ValueError, "duplicates"):
                annual_cashflow(AnnualOperatingInputs(
                    year=1, fixed_opex_eur=10,
                    fixed_opex_includes_replacement_and_augmentation=True, **capex,
                ))
        row = annual_cashflow(AnnualOperatingInputs(
            year=1, fixed_opex_eur=10, fixed_opex_includes_replacement_and_augmentation=True,
        ))
        self.assertEqual(row["net_cashflow_eur"], -10)

    def test_explicit_annual_inputs_are_not_faded_again(self):
        result = project_cashflows(100, [
            AnnualOperatingInputs(1, energy_sales_eur=80, discharge_throughput_mwh=50),
            AnnualOperatingInputs(2, energy_sales_eur=40, discharge_throughput_mwh=25),
        ], 0.0)
        self.assertEqual(result["cashflows_eur"], [-100, 80, 40])
        self.assertEqual(result["lifetime_discharge_throughput_mwh"], 75)
        self.assertEqual(result["npv_eur"], 20)
        self.assertFalse(result["implicit_degradation_or_revenue_growth"])
        json.dumps(result, allow_nan=False)  # Machine-readable even when IRR/payback is absent.

    def test_incomplete_or_negative_cost_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            project_cashflows(100, [AnnualOperatingInputs(2)], 0.1)
        with self.assertRaises(ValueError):
            annual_cashflow(AnnualOperatingInputs(1, generator_tuos_eur=-1))
        with self.assertRaises(ValueError):
            annual_cashflow(AnnualOperatingInputs(1, fees_eur=float("inf")))


class DiscountedCashflowTests(unittest.TestCase):
    def test_npv_and_known_irr(self):
        cashflows = [-100, 60, 60]
        self.assertAlmostEqual(npv(cashflows, 0.1), -100 + 60 / 1.1 + 60 / 1.1**2)
        irr = irr_summary(cashflows)
        self.assertEqual(irr["irr_status"], "unique")
        self.assertAlmostEqual(npv(cashflows, irr["irr"]), 0, places=7)
        self.assertAlmostEqual(payback_years(cashflows), 1 + 40 / 60)
        self.assertAlmostEqual(payback_years(cashflows, 0.1), 1 + (100 - 60 / 1.1) / (60 / 1.1**2))

    def test_multiple_no_and_indeterminate_irr(self):
        result = irr_summary([-100, 230, -132])
        self.assertEqual(result["irr_status"], "multiple_roots")
        self.assertIsNone(result["irr"])
        np.testing.assert_allclose(result["irr_roots"], [0.1, 0.2], atol=1e-8)
        self.assertEqual(irr_summary([-100, -1])["irr_status"], "no_real_irr")
        self.assertEqual(irr_summary([0, 0])["irr_status"], "indeterminate_all_zero")
        self.assertIsNone(payback_years([-100, 1, 1]))

    def test_sustained_payback_handles_later_replacement(self):
        cashflows = [-100, 120, -80, 100]
        self.assertAlmostEqual(payback_years(cashflows), 100 / 120)
        self.assertAlmostEqual(payback_years(cashflows, sustained=True), 2.6)

    def test_break_even_contract_matches_npv_for_full_or_short_term(self):
        cf = [-100] + [3.0] * 20
        for start, term in [(1, 20), (5, 7), (20, 1)]:
            with self.subTest(start=start, term=term):
                contract = break_even_annual_net_contract(cf, 0.07, start_year=start, term_years=term)
                supported = list(cf)
                for year in range(start, start + term):
                    supported[year] += contract["additional_annual_net_contract_eur"]
                self.assertAlmostEqual(npv(supported, 0.07), 0.0, places=8)
        self.assertEqual(break_even_annual_net_contract([-100, 200], 0.1)["additional_annual_net_contract_eur"], 0)
        with self.assertRaises(ValueError):
            break_even_annual_net_contract(cf, 0.07, start_year=20, term_years=2)

    def test_invalid_financial_inputs(self):
        for cashflows, rate in [([], 0.1), ([1, float("nan")], 0.1), ([1], -1), ([1], float("inf"))]:
            with self.subTest(cashflows=cashflows, rate=rate), self.assertRaises(ValueError):
                npv(cashflows, rate)


if __name__ == "__main__":
    unittest.main()

