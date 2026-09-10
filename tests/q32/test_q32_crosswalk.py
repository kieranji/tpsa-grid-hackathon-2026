"""Regression tests for Q3.2 renewable and similarity outputs."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from problem_3_2.q32_constraint_groups.pipeline import (
    _electrical_similarity_matrix,
    _renewable_crosswalk,
    load_config,
    validate_config,
)


class RenewableCrosswalkTests(unittest.TestCase):
    def test_all_configured_res_are_retained_including_zero_group(self):
        network = SimpleNamespace(
            generators=pd.DataFrame(
                {
                    "carrier": ["wind", "solar", "gas"],
                    "p_nom": [10.0, 20.0, 30.0],
                    "bus": ["A", "B", "A"],
                },
                index=["Wind A", "Solar B", "Gas A"],
            )
        )
        memberships = pd.DataFrame(
            {
                "bus": ["A"],
                "mode_id": ["L1::+1"],
                "branch": ["L1"],
                "flow_sign": [1],
                "normalized_positive_relief": [1.0],
                "relief_mw_per_mw_curtailment_or_charge": [0.5],
            }
        )
        zones = pd.DataFrame(
            {"bus": ["A", "B"], "response_zone_id": ["Z1", "Z2"]}
        )
        result = _renewable_crosswalk(
            network,
            {"renewable_carriers": ["wind", "solar"]},
            memberships,
            zones,
        )
        self.assertEqual(set(result["renewable_resource"]), {"Wind A", "Solar B"})
        solar = result.loc[result["renewable_resource"].eq("Solar B")].iloc[0]
        self.assertEqual(int(solar["constraint_group_count"]), 0)
        self.assertTrue(pd.isna(solar["mode_id"]))

    def test_similarity_is_component_local_and_featureless_is_undefined(self):
        features = pd.DataFrame(
            [[1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 0.0]],
            index=["A", "B", "C", "D"],
        )
        components = pd.Series([0, 0, 1, 0], index=features.index)
        result = _electrical_similarity_matrix(features, components)
        self.assertAlmostEqual(float(result.at["A", "B"]), 1.0)
        self.assertTrue(np.isnan(result.at["A", "C"]))
        self.assertTrue(np.isnan(result.at["D", "D"]))

    def test_empty_renewable_carrier_list_is_rejected(self):
        config = load_config(None)
        config["renewable_carriers"] = []
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
