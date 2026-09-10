from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "problem_3_1"))

from q4_techno_economic.comparison import (
    _same_basis, capacity_dlr_comparison, distributed_comparison,
)
from q4_techno_economic.config import load_config, validate_config
from q4_techno_economic.demo import DemoBackend
from q4_techno_economic.finance import npv, summarise_cashflow
from q4_techno_economic.grid import apply_dlr_multiplier_csv
from q4_techno_economic.lifecycle import capital_cost, simulate_lifetime
from q4_techno_economic.pipeline import smoke_designs_for
from q4_techno_economic.reporting import build_comparisons
from q4_techno_economic.types import Design


class DesignPortfolioTests(unittest.TestCase):
    def test_multi_site_design_contract(self):
        design = Design(
            "A",
            battery_mw=45,
            battery_mwh=220,
            battery_placements=(("A", 0.6), ("B", 0.4)),
        )
        self.assertEqual(design.technology, "bess")
        self.assertEqual(design.site_count, 2)
        self.assertEqual(design.sites_label, "A;B")
        self.assertEqual(design.allocations_label, "0.600000;0.400000")

    def test_invalid_portfolio_rejected(self):
        with self.assertRaises(ValueError):
            Design("A", battery_mw=45, battery_mwh=220,
                   battery_placements=(("A", 0.6), ("B", 0.3)))
        with self.assertRaises(ValueError):
            Design("A", wind_mw=50, battery_mw=45, battery_mwh=220,
                   battery_placements=(("A", 0.5), ("B", 0.5)))


class ConfigAndCostTests(unittest.TestCase):
    def test_portfolio_config_validation(self):
        config = load_config()
        config["designs"]["bess_portfolios"] = [
            {"label": "two", "sites": ["A", "B"], "allocations": [0.5, 0.5]}
        ]
        validate_config(config)
        bad = copy.deepcopy(config)
        bad["designs"]["bess_portfolios"][0]["allocations"] = [0.7, 0.2]
        with self.assertRaises(ValueError):
            validate_config(bad)

    def test_distributed_sites_pay_extra_fixed_site_and_connection_cost(self):
        config = load_config()
        single = Design("A", battery_mw=45, battery_mwh=220)
        distributed = Design(
            "A", battery_mw=45, battery_mwh=220,
            battery_placements=(("A", 0.5), ("B", 0.5)),
        )
        one = capital_cost(single, config)
        two = capital_cost(distributed, config)
        fixed_increment = (
            config["costs"]["shared_site_fixed_eur"]
            + config["costs"]["connection_fixed_eur"]
        ) * (1 + config["costs"]["contingency_fraction"])
        self.assertEqual(one["site_count"], 1)
        self.assertEqual(two["site_count"], 2)
        self.assertAlmostEqual(two["initial_capex_eur"] - one["initial_capex_eur"],
                               fixed_increment, places=6)

    def test_smoke_keeps_explicit_portfolio_sites(self):
        config = load_config()
        config["designs"].update(
            wind_mw=[],
            hybrid_wind_bess_mw_mwh=[],
            bess_mw_mwh=[[45.0, 220.0]],
            maintenance_policies=["replace"],
            bess_portfolios=[
                {"label": "one", "sites": ["Croaghonagh"], "allocations": [1.0]},
                {"label": "two", "sites": ["Croaghonagh", "Binbane"],
                 "allocations": [0.5, 0.5]},
            ],
        )
        sites, designs = smoke_designs_for(
            ["Ardnagappary", "Binbane", "Croaghonagh"], config
        )
        self.assertEqual(sites, ["Croaghonagh", "Binbane"])
        self.assertEqual(len(designs), 1)
        self.assertEqual(designs[0].site_count, 2)


class DispatchAndFinanceTests(unittest.TestCase):
    def test_demo_backend_jointly_dispatches_two_sites(self):
        config = load_config()
        backend = DemoBackend(config, max_hours=24)
        design = Design(
            "DEMO_UPSTREAM", battery_mw=45, battery_mwh=220,
            battery_placements=(("DEMO_UPSTREAM", 0.5), ("DEMO_DOWNSTREAM", 0.5)),
        )
        result = backend.solve(design, 220)
        self.assertIn("charge_mw__DEMO_UPSTREAM", result.site_dispatch)
        self.assertIn("discharge_mw__DEMO_DOWNSTREAM", result.site_dispatch)
        np.testing.assert_allclose(
            result.site_dispatch.filter(regex=r"^charge_mw__").sum(axis=1),
            result.battery_charge_mw,
        )
        self.assertLessEqual(result.metrics["max_passive_loading_pu"], 1.00001)

    def test_system_saving_is_not_project_cashflow(self):
        config = load_config()
        config["years"] = 1
        config["sensitivity"]["enabled"] = False
        backend = DemoBackend(config, max_hours=24)
        baseline = backend.solve(Design("DEMO_UPSTREAM"), 0.0)
        design = Design(
            "DEMO_UPSTREAM", battery_mw=45, battery_mwh=220,
            battery_placements=(("DEMO_UPSTREAM", 0.5), ("DEMO_DOWNSTREAM", 0.5)),
        )
        prices = np.full(len(backend.snapshots), 70.0)
        cash, summary, _ = simulate_lifetime(
            design, backend, config, baseline, prices
        )
        self.assertFalse(summary["system_saving_included_in_project_cashflow"])
        self.assertAlmostEqual(
            summary["npv_eur"],
            npv(cash["net_cashflow_eur"], config["real_discount_rate"]),
        )
        altered = cash.copy()
        altered["model_dispatch_cost_saving_eur"] += 1_000_000_000
        altered_summary = summarise_cashflow(
            altered, config["real_discount_rate"], design.technology
        )
        self.assertAlmostEqual(altered_summary["npv_eur"], summary["npv_eur"])
        self.assertGreater(
            altered_summary["lifetime_system_dispatch_cost_saving_eur"],
            summary["lifetime_system_dispatch_cost_saving_eur"],
        )


def _comparison_row(site_count=1, sites="A", gain=100.0, npv_eur=-10.0, capex=10.0):
    return {
        "technology": "bess",
        "site_count": site_count,
        "sites": sites,
        "wind_mw": 0.0,
        "battery_mw": 45.0,
        "battery_mwh": 220.0,
        "lifetime_net_renewable_gain_mwh": gain,
        "lifetime_project_revenue_eur": 20.0,
        "lifetime_energy_purchase_eur": 5.0,
        "lifetime_fixed_and_variable_om_eur": 2.0,
        "lifetime_maintenance_capex_eur": 3.0,
        "initial_capex_eur": capex,
        "lifetime_decommission_eur": 1.0,
        "lifetime_residual_eur": 0.5,
        "lifetime_system_dispatch_cost_saving_eur": 7.0,
        "npv_eur": npv_eur,
        "simple_payback_years": np.nan,
        "discounted_payback_years": np.nan,
    }


class CrossRunComparisonTests(unittest.TestCase):

    def test_dlr_hash_may_change_but_base_network_identity_must_match(self):
        common_network = {
            "origin": "gridkit.py:load",
            "network_meta": {"scenario": "WP2033", "scope": "north-west"},
            "buses": 15,
            "lines": 18,
            "transformers": 0,
            "links": 0,
            "snapshots": 168,
        }
        common_config = {
            "annualisation": {"mode": "repeat_block"},
            "prices": {"mode": "flat"},
            "costs": {"x": 1},
            "battery": {"x": 1},
            "real_discount_rate": 0.07,
            "currency_basis": "real EUR",
        }
        fixed = {
            "network": {**common_network, "network_hash": "fixed", "dlr_label": "none"},
            "config": common_config,
        }
        distributed = copy.deepcopy(fixed)
        dlr = {
            "network": {
                **common_network,
                "network_hash": "post-dlr",
                "dlr_label": "q3",
                "dlr_source_sha256": "abc",
            },
            "config": copy.deepcopy(common_config),
        }
        _same_basis(fixed, dlr, distributed)
        dlr["network"]["network_meta"] = {"scenario": "WP2033", "scope": "different"}
        with self.assertRaises(ValueError):
            _same_basis(fixed, dlr, distributed)

    def test_dlr_standalone_and_battery_increment_are_not_conflated(self):
        fixed = pd.DataFrame([_comparison_row(gain=100.0, npv_eur=-10.0)])
        dlr = pd.DataFrame([_comparison_row(gain=80.0, npv_eur=-9.0)])
        manifest = {
            "network": {"snapshots": 168},
            "config": {"annualisation": {"hours_per_year": 8760.0}, "years": 20},
        }
        result, audit = capacity_dlr_comparison(
            fixed, dlr, manifest,
            {"renewable_dispatch_down_mwh": 100.0},
            {"renewable_dispatch_down_mwh": 90.0},
        )
        expected_dlr = 10.0 * (8760.0 / 168.0) * 20
        self.assertAlmostEqual(audit["dlr_standalone_lifetime_gain_mwh"], expected_dlr)
        self.assertAlmostEqual(
            result.loc[0, "dlr_plus_bess_total_gain_vs_fixed_mwh"],
            expected_dlr + 80.0,
        )
        self.assertFalse(result.loc[0, "dlr_cost_included_in_conditional_bess_npv"])

    def test_distributed_delta_uses_same_size_single_site_reference(self):
        frame = pd.DataFrame([
            _comparison_row(site_count=1, sites="A", gain=100.0, npv_eur=-10.0, capex=10.0),
            _comparison_row(site_count=2, sites="A;B", gain=100.0, npv_eur=-12.0, capex=12.0),
        ])
        result = distributed_comparison(frame)
        two = result.loc[result["site_count"].eq(2)].iloc[0]
        self.assertAlmostEqual(two["technical_gain_delta_vs_single_mwh"], 0.0)
        self.assertAlmostEqual(two["undiscounted_cost_delta_vs_single_eur"], 2.0)
        self.assertAlmostEqual(two["npv_delta_vs_single_eur"], -2.0)


class DLRAndReportingTests(unittest.TestCase):
    def test_q3_multiplier_csv_is_timestamp_aligned_and_applied(self):
        network = SimpleNamespace()
        network.snapshots = pd.date_range("2033-01-01", periods=2, freq="h")
        network.lines = pd.DataFrame(
            {"s_nom": [100.0], "s_max_pu": [1.0]}, index=["L1"]
        )
        network.lines_t = SimpleNamespace(s_max_pu=pd.DataFrame())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dlr.csv"
            pd.DataFrame({
                "snapshot": network.snapshots[::-1],
                "L1": [1.2, 1.0],
            }).to_csv(path, index=False)
            multipliers, digest = apply_dlr_multiplier_csv(
                network, path, Path(directory)
            )
        np.testing.assert_allclose(multipliers["L1"], [1.0, 1.2])
        np.testing.assert_allclose(network.lines_t.s_max_pu["L1"], [1.0, 1.2])
        self.assertEqual(len(digest), 64)

    def test_no_build_wins_negative_npv(self):
        summary = pd.DataFrame([{
            "case_id": "bess_case",
            "technology": "bess",
            "bus": "A",
            "site_count": 1,
            "sites": "A",
            "allocations": "1.000000",
            "wind_mw": 0.0,
            "battery_mw": 45.0,
            "battery_mwh": 220.0,
            "npv_eur": -1.0,
            "lifetime_net_renewable_gain_mwh": 10.0,
        }])
        _, comparisons = build_comparisons(summary)
        economic = comparisons.loc[
            comparisons["selection"].eq("economic_max_conditional_NPV")
        ].iloc[0]
        self.assertEqual(economic["case_id"], "NO_BUILD")
        self.assertEqual(economic["npv_eur"], 0.0)


if __name__ == "__main__":
    unittest.main()
