from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "problem_3_1"))

from q6_priority_dispatch.core import (
    compare_policy_results,
    solve_lexicographic,
    status_assignment,
    target_line_hotspots,
    validate_assignment,
)


def toy_network(*, target_rating: float = 20.0, co_located: bool = False) -> pypsa.Network:
    """Triangle with target PTDFs 2/3 at A and 1/3 at B."""
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=1, freq="h"))
    for bus in ("A", "B", "C"):
        network.add("Bus", bus, v_nom=110.0)
    network.add("Line", "AC target", bus0="A", bus1="C", x=1.0, r=0.0, s_nom=target_rating)
    network.add("Line", "AB", bus0="A", bus1="B", x=1.0, r=0.0, s_nom=100.0)
    network.add("Line", "BC", bus0="B", bus1="C", x=1.0, r=0.0, s_nom=100.0)
    network.add(
        "Generator",
        "Wind A",
        bus="A",
        p_nom=10.0,
        carrier="wind",
        marginal_cost=0.0,
        p_max_pu=1.0,
    )
    network.add(
        "Generator",
        "Wind B",
        bus="A" if co_located else "B",
        p_nom=20.0,
        carrier="wind",
        marginal_cost=0.0,
        p_max_pu=1.0,
    )
    network.add(
        "Generator",
        "Backup",
        bus="C",
        p_nom=30.0,
        carrier="thermal",
        marginal_cost=100.0,
    )
    network.add(
        "Generator",
        "shed C",
        bus="C",
        p_nom=30.0,
        carrier="load shedding",
        marginal_cost=10000.0,
    )
    network.add("Load", "load C", bus="C", p_set=20.0)
    return network


class PriorityDispatchTests(unittest.TestCase):
    def assignment(self, network: pypsa.Network, priority=("Wind A",)) -> pd.DataFrame:
        return status_assignment(
            network,
            priority,
            source="unit_test_scenario",
            verified=False,
            assumption_note="Synthetic test assignment.",
        )

    def test_known_unequal_relief_case_has_exact_ten_mwh_inefficiency(self):
        network = toy_network(target_rating=20.0 / 3.0)
        assignment = self.assignment(network)
        neutral = solve_lexicographic(
            network,
            assignment,
            policy="neutral",
            label="neutral",
            lexicographic_tolerance_mwh=1e-7,
        )
        priority = solve_lexicographic(
            network,
            assignment,
            policy="priority",
            label="priority",
            lexicographic_tolerance_mwh=1e-7,
        )
        self.assertAlmostEqual(neutral.summary["wind_dispatch_down_mwh"], 10.0, places=5)
        self.assertAlmostEqual(priority.summary["wind_dispatch_down_mwh"], 20.0, places=5)
        self.assertAlmostEqual(
            priority.summary["wind_dispatch_down_mwh"]
            - neutral.summary["wind_dispatch_down_mwh"],
            10.0,
            places=5,
        )
        self.assertAlmostEqual(priority.summary["priority_dispatch_down_mwh"], 0.0, places=5)
        self.assertAlmostEqual(priority.summary["non_priority_dispatch_down_mwh"], 20.0, places=5)
        self.assertGreater(priority.summary["physical_generator_cost_eur"], neutral.summary["physical_generator_cost_eur"])

    def test_equal_ptdf_changes_allocation_not_total(self):
        network = toy_network(target_rating=20.0 / 3.0, co_located=True)
        assignment = self.assignment(network)
        neutral = solve_lexicographic(network, assignment, policy="neutral", label="neutral")
        priority = solve_lexicographic(network, assignment, policy="priority", label="priority")
        self.assertAlmostEqual(
            neutral.summary["wind_dispatch_down_mwh"],
            priority.summary["wind_dispatch_down_mwh"],
            places=4,
        )
        _, impacts = compare_policy_results(neutral, priority)
        self.assertAlmostEqual(float(impacts["additional_dispatch_down_mwh"].sum()), 0.0, places=4)
        # Equal PTDFs guarantee the same technical total. A solver may keep
        # the same allocation or redistribute it because both are degenerate.

    def test_uncongested_case_has_no_priority_technical_penalty(self):
        network = toy_network(target_rating=100.0)
        assignment = self.assignment(network)
        neutral = solve_lexicographic(network, assignment, policy="neutral", label="neutral")
        priority = solve_lexicographic(network, assignment, policy="priority", label="priority")
        self.assertAlmostEqual(
            neutral.summary["wind_dispatch_down_mwh"],
            priority.summary["wind_dispatch_down_mwh"],
            places=5,
        )
        self.assertAlmostEqual(neutral.summary["unserved_mwh"], 0.0, places=7)
        self.assertAlmostEqual(priority.summary["unserved_mwh"], 0.0, places=7)

    def test_one_status_class_matches_neutral_total(self):
        network = toy_network(target_rating=20.0 / 3.0)
        assignment = self.assignment(network, priority=("Wind A", "Wind B"))
        neutral = solve_lexicographic(network, assignment, policy="neutral", label="neutral")
        priority = solve_lexicographic(network, assignment, policy="priority", label="priority")
        self.assertAlmostEqual(
            neutral.summary["wind_dispatch_down_mwh"],
            priority.summary["wind_dispatch_down_mwh"],
            places=5,
        )

    def test_observed_mode_rejects_unverified_assignment(self):
        network = toy_network()
        assignment = self.assignment(network)
        with self.assertRaisesRegex(ValueError, "verified source"):
            validate_assignment(network, assignment, require_verified=True)

    def test_assignment_requires_exact_complete_generator_ids(self):
        network = toy_network()
        assignment = self.assignment(network).iloc[:1].copy()
        with self.assertRaisesRegex(ValueError, "cover every wind generator"):
            validate_assignment(network, assignment)

    def test_solution_reconciles_and_status_cost_is_not_booked(self):
        network = toy_network(target_rating=20.0 / 3.0)
        assignment = self.assignment(network)
        result = solve_lexicographic(network, assignment, policy="priority", label="priority")
        self.assertLess(result.summary["maximum_component_balance_error_mw"], 2e-4)
        self.assertLessEqual(result.summary["maximum_passive_loading_pu"], 1.00002)
        self.assertFalse(result.summary["status_penalty_in_physical_cost"])
        grouped = result.generator_hourly["dispatch_down_mwh"].sum()
        self.assertAlmostEqual(grouped, result.summary["wind_dispatch_down_mwh"], places=7)

    def test_binding_hotspot_uses_signed_directional_relief(self):
        network = toy_network(target_rating=20.0 / 3.0)
        assignment = self.assignment(network)
        result = solve_lexicographic(network, assignment, policy="priority", label="priority")
        hotspots = target_line_hotspots(result, assignment, minimum_loading_pu=0.999)
        target = hotspots.loc[hotspots["branch"].eq("AC target")]
        self.assertFalse(target.empty)
        self.assertEqual(int(target.iloc[0]["flow_direction_sign"]), 1)
        self.assertGreater(float(target.iloc[0]["priority_relief_mw_per_mw"]), 0.0)


if __name__ == "__main__":
    unittest.main()
