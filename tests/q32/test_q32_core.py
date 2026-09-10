"""Regression tests for Problem 3.2 constraint groups."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from problem_3_2.q32_constraint_groups.core import (
    constraint_memberships,
    directional_relief,
    haversine_km,
    merge_similar_modes,
    nonlocal_similar_pairs,
    normalized_positive_relief,
    reference_sensitivity,
    response_zones,
)
from problem_3_2.q32_constraint_groups.pipeline import _q6_crosswalk, load_config, validate_config


class ConstraintGroupTests(unittest.TestCase):
    def setUp(self):
        self.modes = pd.DataFrame(
            [
                {"mode_id": "L1::+1", "branch": "L1", "flow_sign": 1, "ac_component": 0,
                 "event_weighted_severity_hours": 2.0},
                {"mode_id": "L2::-1", "branch": "L2", "flow_sign": -1, "ac_component": 0,
                 "event_weighted_severity_hours": 1.0},
            ]
        )
        self.relief = pd.DataFrame(
            [[1.0, 0.8, -0.4, -0.5], [0.9, 0.7, -0.5, -0.6]],
            index=["L1::+1", "L2::-1"], columns=["A", "B", "C", "D"]
        )
        self.components = pd.Series(0, index=["A", "B", "C", "D"], dtype=int)

    def test_positive_relief_normalization(self):
        result = normalized_positive_relief(self.relief)
        self.assertAlmostEqual(result.at["L1::+1", "A"], 1.0)
        self.assertAlmostEqual(result.at["L1::+1", "B"], 0.8)
        self.assertEqual(result.at["L1::+1", "C"], 0.0)

    def test_constraint_groups_are_overlapping(self):
        members = constraint_memberships(self.modes, self.relief, 0.5, "test")
        counts = members.groupby("bus")["mode_id"].nunique()
        self.assertEqual(int(counts.loc["A"]), 2)
        self.assertEqual(int(counts.loc["B"]), 2)
        self.assertNotIn("C", counts.index)

    def test_mode_merge_requires_both_tests(self):
        members = constraint_memberships(self.modes, self.relief, 0.5, "test")
        mapping, audit = merge_similar_modes(
            self.modes, self.relief, members,
            jaccard_threshold=0.8, cosine_threshold=0.98
        )
        self.assertEqual(mapping["merged_constraint_group_id"].nunique(), 1)
        self.assertTrue(bool(audit.iloc[0]["merged"]))

    def test_directional_relief_respects_flow_sign(self):
        adjusted = pd.DataFrame(
            [[0.4, -0.2], [0.1, -0.7]], index=["L1", "L2"], columns=["A", "B"]
        )
        result = directional_relief(adjusted, self.modes)
        self.assertAlmostEqual(result.at["L1::+1", "A"], 0.4)
        self.assertAlmostEqual(result.at["L2::-1", "B"], 0.7)

    def test_reference_centering_is_component_local_and_invariant(self):
        shifted = self.relief - 0.25
        audit = reference_sensitivity(
            self.modes, self.relief, shifted, self.components, 0.5, "a", "b"
        )
        self.assertAlmostEqual(
            float(audit["maximum_absolute_centered_response_change"].max()), 0.0, places=12
        )
        self.assertAlmostEqual(
            float(audit["maximum_absolute_raw_relief_change"].max()), 0.25, places=12
        )

    def test_response_zones_are_mutually_exclusive(self):
        weights = pd.DataFrame(
            [[1.0, 0.0], [1.0, 1.0]],
            index=pd.date_range("2030-01-01", periods=2, freq="h"),
            columns=self.relief.index,
        )
        zones, features, summary = response_zones(
            self.components, self.modes, self.relief, weights,
            maximum_zones_per_component=3
        )
        self.assertEqual(len(zones), 4)
        self.assertEqual(zones["bus"].nunique(), 4)
        self.assertTrue(zones["response_zone_id"].notna().all())
        self.assertEqual(set(features.index), {"A", "B", "C", "D"})
        self.assertGreaterEqual(len(summary), 1)

    def test_nonlocal_pair_uses_geographic_and_electrical_filters(self):
        buses = pd.DataFrame(
            {"x": [-10.0, -6.0, -9.9], "y": [51.0, 55.0, 51.1]}, index=["A", "B", "C"]
        )
        components = pd.Series(0, index=buses.index, dtype=int)
        zones = pd.DataFrame(
            {"bus": buses.index, "response_zone_id": ["Z1", "Z1", "Z2"]}
        )
        features = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], index=buses.index)
        pairs = nonlocal_similar_pairs(
            buses, components, zones, features,
            minimum_distance_km=100.0, minimum_cosine_similarity=0.99, limit=10
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual({pairs.iloc[0]["bus_a"], pairs.iloc[0]["bus_b"]}, {"A", "B"})

    def test_haversine_and_config_validation(self):
        self.assertGreater(haversine_km(-10.0, 51.0, -6.0, 55.0), 400.0)
        config = load_config(None)
        validate_config(config)
        config["scope"] = "north-west"
        with self.assertRaises(ValueError):
            validate_config(config)


    def test_q6_crosswalk_uses_actual_status_impact_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q6 = root / "q6"
            q6.mkdir()
            pd.DataFrame(
                [
                    {"scenario": "network_neutral_static", "wind_dispatch_down_mwh": 20.0},
                    {"scenario": "network_priority_static", "wind_dispatch_down_mwh": 20.0},
                ]
            ).to_csv(q6 / "02_scenario_summary.csv", index=False)
            pd.DataFrame(
                {"additional_dispatch_down_mwh": [-5.0, 5.0]}
            ).to_csv(q6 / "06_generator_status_impacts.csv", index=False)
            result, audit = _q6_crosswalk(root, {"q6_result_dir": str(q6)})
            self.assertTrue(audit["exists"])
            self.assertEqual(len(result), 3)
            values = result.set_index("metric")["value"]
            self.assertAlmostEqual(values["priority_farms_protected_energy_mwh"], 5.0)


if __name__ == "__main__":
    unittest.main()
