"""Regression tests for deterministic Q4 technical tie-breaking."""
from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from problem_3_1.q4_techno_economic.reporting import build_comparisons


class Q4ReportingTieBreakTests(unittest.TestCase):
    @staticmethod
    def _summary(delta: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "case_id": "single", "technology": "bess", "bus": "A",
                    "site_count": 1, "sites": "A", "allocations": "1.0",
                    "wind_mw": 0.0, "battery_mw": 45.0, "battery_mwh": 220.0,
                    "lifetime_net_renewable_gain_mwh": 1000.0,
                    "npv_eur": -10.0, "initial_capex_eur": 50.0,
                },
                {
                    "case_id": "three", "technology": "bess", "bus": "A;B;C",
                    "site_count": 3, "sites": "A;B;C", "allocations": "0.33;0.33;0.34",
                    "wind_mw": 0.0, "battery_mw": 45.0, "battery_mwh": 220.0,
                    "lifetime_net_renewable_gain_mwh": 1000.0 + delta,
                    "npv_eur": -20.0, "initial_capex_eur": 55.0,
                },
            ]
        )

    def test_numerical_tie_selects_fewer_sites(self):
        _, comparisons = build_comparisons(self._summary(1e-7))
        technical = comparisons.loc[
            comparisons["selection"].eq("technical_max_net_RE_gain")
        ].iloc[0]
        self.assertEqual(technical["case_id"], "single")

    def test_material_technical_gain_is_not_treated_as_tie(self):
        _, comparisons = build_comparisons(self._summary(0.01))
        technical = comparisons.loc[
            comparisons["selection"].eq("technical_max_net_RE_gain")
        ].iloc[0]
        self.assertEqual(technical["case_id"], "three")


if __name__ == "__main__":
    unittest.main()
