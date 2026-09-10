"""Regression test preventing transitive chaining of dissimilar modes."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from problem_3_2.q32_constraint_groups.core import (
    constraint_memberships,
    merge_similar_modes,
)


class CompleteLinkMergeTests(unittest.TestCase):
    def test_intermediate_mode_cannot_chain_dissimilar_endpoints(self):
        angles = np.radians([0.0, 10.0, 20.0])
        relief = pd.DataFrame(
            [[np.cos(angle), np.sin(angle)] for angle in angles],
            index=["A", "B", "C"], columns=["X", "Y"]
        )
        modes = pd.DataFrame(
            [
                {"mode_id": mode, "branch": mode, "flow_sign": 1, "ac_component": 0}
                for mode in relief.index
            ]
        )
        memberships = constraint_memberships(modes, relief, 0.0, "test")
        mapping, pairs = merge_similar_modes(
            modes,
            relief,
            memberships,
            jaccard_threshold=0.4,
            cosine_threshold=0.98,
        )
        self.assertEqual(mapping["merged_constraint_group_id"].nunique(), 2)
        group = mapping.set_index("mode_id")["merged_constraint_group_id"]
        self.assertEqual(group["A"], group["B"])
        self.assertNotEqual(group["A"], group["C"])
        bc = pairs.loc[
            pairs["left_mode_id"].eq("B") & pairs["right_mode_id"].eq("C")
        ].iloc[0]
        self.assertTrue(bool(bc["pair_passes_thresholds"]))
        self.assertFalse(bool(bc["merged"]))


if __name__ == "__main__":
    unittest.main()
