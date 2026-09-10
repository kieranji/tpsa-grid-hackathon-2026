"""Independent integration checks for money units and conditional lifecycle cash."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bess_market.dispatch import Battery
from bess_market.pipeline import installed_cost, lifecycle, load_real_prices


def summary(*, contracted: bool, annual_scale: float = 1.0):
    """Distinct ledgers make a stale contracted dispatch after expiry detectable."""
    return {
        "annual_equivalent_multiplier": annual_scale,
        "energy_sales_eur": 1000.0 if contracted else 1700.0,
        "energy_purchases_eur": 400.0 if contracted else 600.0,
        "service_gross_eur": 200.0 if contracted else 0.0,
        "service_deductions_eur": 30.0 if contracted else 0.0,
        "capacity_difference_charge_eur": 25.0 if contracted else 0.0,
        "execution_fees_eur": 12.0 if contracted else 20.0,
        "import_adder_eur": 8.0 if contracted else 10.0,
        "discharge_mwh": 90.0 if contracted else 150.0,
        # Deliberately inconsistent: pipeline must reconstruct gross components,
        # not append a pre-netted value and subtract charges a second time.
        "net_trading_and_services_eur": -99999999.0,
    }


class PipelineReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "configs/bess_market_public_benchmarks.json").read_text())
        cls.battery = Battery(**cls.config["battery"])

    def load_fixture(self, dates, indices):
        """Contiguous half-hour prices with explicit market dates, as in source."""
        with TemporaryDirectory() as directory:
            path = Path(directory)
            prices = pd.DataFrame({
                "timestamp_utc": pd.date_range("2026-08-31T21:30:00Z", periods=len(dates), freq="30min"),
                "duration_hours": 0.5,
                "price_eur_mwh": [(-1 if i % 2 else 1) * 100.0 for i in range(len(dates))],
                "market_date": dates,
            })
            prices.to_csv(path / "prices.csv", index=False)
            pd.DataFrame(indices, columns=["month", "cpi_dec2023_100"]).to_csv(path / "cpi.csv", index=False)
            return load_real_prices(path / "prices.csv", path / "cpi.csv")

    def test_unpublished_cpi_carry_is_explicit_and_preserves_negative_prices_and_duration(self):
        actual, current = self.load_fixture(
            ["2026-08-31", "2026-09-01"], [("2026-07", 106.7), ("2026-08", 107.4)])
        self.assertEqual(actual.cpi_last_observed_proxy.tolist(), [False, True])
        np.testing.assert_allclose(actual.cpi_index_used, [107.4, 107.4])
        np.testing.assert_allclose(actual.price_nominal_eur_mwh, [100.0, -100.0])
        np.testing.assert_allclose(actual.price_eur_mwh, [100 * 100.7 / 107.4, -100 * 100.7 / 107.4])
        self.assertEqual(actual.duration_hours.sum(), 1.0)
        self.assertAlmostEqual(current, 100.7 / 107.4)

    def test_missing_published_cpi_is_not_replaced_with_latest_observation(self):
        with self.assertRaisesRegex(ValueError, "Missing published historical CPI"):
            self.load_fixture(["2026-07-01"], [("2026-06", 106.6), ("2026-08", 107.4)])

    def test_cpi_uses_auction_delivery_month_and_historical_observation(self):
        actual, _ = self.load_fixture(
            ["2026-08-31", "2026-09-01"], [("2026-08", 107.4), ("2026-09", 108.0)])
        # Both period starts are August UTC; the second belongs to September's auction day.
        self.assertEqual(actual.timestamp_utc.dt.month.tolist(), [8, 8])
        np.testing.assert_allclose(actual.cpi_index_used, [107.4, 108.0])
        self.assertFalse(actual.cpi_last_observed_proxy.any())

    def test_config_cost_uses_usable_ac_energy_and_named_irish_allowances(self):
        b = self.battery
        self.assertAlmostEqual(b.round_trip_efficiency, 0.85)
        self.assertAlmostEqual(b.deliverable_ac_mwh, 228.64470254086362)
        self.assertAlmostEqual(b.deliverable_ac_mwh / b.power_mw, 2.689937676951337)
        costs = installed_cost(self.config, b, "reference_cost", 100.7 / 107.4)
        # Independently evaluated source arithmetic, with 15% Irish sensitivity
        # and EUR5m connection allowance; neither is claimed to be a vendor quote.
        self.assertAlmostEqual(costs["installed_plant_eur"], 70924028.97291689, places=5)
        self.assertAlmostEqual(costs["upfront_capex_eur"], 75924028.97291689, places=5)
        self.assertEqual(costs["connection_allowance_eur"], 5000000)
        self.assertAlmostEqual(costs["annual_gtuos_proxy_eur"], 894754.6606145252, places=6)
        # Temporary capability reduction must not imply a smaller purchased plant.
        unavailable = installed_cost(self.config, replace(b, capability_fraction=.5), "reference_cost", 100.7 / 107.4)
        self.assertEqual(costs["upfront_capex_eur"], unavailable["upfront_capex_eur"])

    def test_four_hour_benchmark_identity_and_money_factor_scope(self):
        c = deepcopy(self.config)
        c["cost_scenarios"]["reference_cost"]["ireland_installation_multiplier"] = 1.0
        c["cost_scenarios"]["reference_cost"]["connection_allowance_eur2024"] = 0.0
        b = replace(self.battery, energy_dc_mwh=340.0, soc_min=0.0, soc_max=1.0, round_trip_efficiency=1.0)
        one = installed_cost(c, b, "reference_cost", 1.0)
        half = installed_cost(c, b, "reference_cost", .5)
        self.assertAlmostEqual(one["installed_plant_eur"], 340000 * 257 / 1.082, places=6)
        self.assertEqual(one["upfront_capex_eur"], half["upfront_capex_eur"])
        self.assertEqual(one["annual_fixed_opex_including_augmentation_eur"], half["annual_fixed_opex_including_augmentation_eur"])
        self.assertAlmostEqual(one["annual_gtuos_proxy_eur"] / 2, half["annual_gtuos_proxy_eur"])

    def run_lifecycle(self, term, scale=1.0):
        case = next(row for row in self.config["market_cases"] if row["name"] == "stack_t4_services4")
        cost = {
            "upfront_capex_eur": 10000.0,
            "annual_fixed_opex_including_augmentation_eur": 100.0,
            "annual_gtuos_proxy_eur": 50.0,
            "annual_house_load_incremental_cost_eur": 10.0,
        }
        return lifecycle(self.config, self.battery, summary(contracted=True, annual_scale=scale),
                         summary(contracted=False, annual_scale=scale), case, cost, term, 1.0)

    def test_one_year_term_releases_dispatch_and_stops_both_revenues_and_clawback(self):
        result = self.run_lifecycle(1)
        rows = result["annual_ledger"]
        self.assertEqual(len(rows), 15)
        self.assertGreater(rows[0]["capacity_revenue_eur"], 0)
        self.assertEqual(rows[0]["services_revenue_eur"], 200)
        self.assertEqual(rows[0]["energy_sales_eur"], 1000)
        for row in rows[1:]:
            self.assertEqual(row["capacity_revenue_eur"], 0)
            self.assertEqual(row["services_revenue_eur"], 0)
            self.assertEqual(row["service_nonperformance_cost_eur"], 0)
            self.assertEqual(row["capacity_difference_charges_eur"], 0)
            self.assertEqual(row["energy_sales_eur"], 1700)
            self.assertEqual(row["energy_purchase_eur"], 600)
            self.assertEqual(row["discharge_throughput_mwh"], 150)
        self.assertEqual(result["secured_capacity_and_services_eur"], 0)

    def test_fifteen_year_continuation_is_explicit_and_zero_term_is_energy_only(self):
        full = self.run_lifecycle(15)
        none = self.run_lifecycle(0)
        self.assertEqual(full["conditional_contract_years"], 15)
        for row in full["annual_ledger"]:
            self.assertGreater(row["capacity_revenue_eur"], 0)
            self.assertEqual(row["services_revenue_eur"], 200)
            self.assertEqual(row["energy_sales_eur"], 1000)
        for row in none["annual_ledger"]:
            self.assertEqual(row["capacity_revenue_eur"], 0)
            self.assertEqual(row["services_revenue_eur"], 0)
            self.assertEqual(row["energy_sales_eur"], 1700)

    def test_annualization_applies_to_observed_flows_once_not_annual_award_or_fixed_cost(self):
        full = self.run_lifecycle(15)
        double_observed = self.run_lifecycle(15, scale=2.0)
        a, b = full["annual_ledger"][0], double_observed["annual_ledger"][0]
        self.assertEqual(a["capacity_revenue_eur"], b["capacity_revenue_eur"])
        self.assertEqual(a["fixed_opex_eur"], b["fixed_opex_eur"])
        self.assertEqual(a["generator_tuos_eur"], b["generator_tuos_eur"])
        expected = (1000 - 400 + 200 - 30 - 25 - 12 - 8) * 2 + b["capacity_revenue_eur"] - 100 - 50 - 10
        self.assertAlmostEqual(b["net_operating_cashflow_eur"], expected)

    def test_inclusive_fom_does_not_add_replacement_augmentation_or_non_cash_wear(self):
        self.assertTrue(self.config["cost_basis"]["fixed_opex_includes_augmentation"])
        costs = installed_cost(self.config, self.battery, "reference_cost", 1.0)
        self.assertAlmostEqual(costs["annual_fixed_opex_including_augmentation_eur"],
                               costs["installed_plant_eur"] * .04)
        result = self.run_lifecycle(15)
        self.assertEqual(result["lifetime_maintenance_capex_eur"], 0)
        self.assertEqual(result["cashflows_eur"][0], -10000)
        for row in result["annual_ledger"]:
            self.assertTrue(row["fixed_opex_includes_replacement_and_augmentation"])
            self.assertEqual(row["fixed_opex_eur"], 100)
            self.assertEqual(row["augmentation_capex_eur"], 0)
            self.assertEqual(row["replacement_capex_eur"], 0)
            self.assertEqual(row["variable_opex_eur"], 0)
            self.assertFalse(row["degradation_applied_by_finance_layer"])
        self.assertEqual(sum(row["decommissioning_cost_eur"] for row in result["annual_ledger"]), 300)
        self.assertEqual(result["annual_ledger"][-1]["decommissioning_cost_eur"], 300)


if __name__ == "__main__":
    unittest.main()
