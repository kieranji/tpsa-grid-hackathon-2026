from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/problem_3_1"))
from network_validation import (network_input_fingerprint, scale_thermal_limits,
                                source_bus_extract, validate_replay)
import question_2_battery_siting_sizing as q2


def network():
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Bus", "h", v_nom=220)
    n.add("Bus", "a", v_nom=110)
    n.add("Bus", "b", v_nom=110)
    n.add("Transformer", "ha", bus0="h", bus1="a", x=.16, r=.0025, s_nom=250)
    n.add("Transformer", "hb", bus0="h", bus1="b", x=.2, r=.003, s_nom=300)
    n.add("Line", "ab", bus0="a", bus1="b", x=10, r=1, s_nom=200)
    n.add("Generator", "external", bus="h", p_nom=300, marginal_cost=50)
    n.add("Generator", "wind", bus="a", p_nom=100, marginal_cost=0,
          p_max_pu=pd.Series([.9, .2], index=n.snapshots))
    n.add("Load", "demand", bus="b", p_set=120)
    return n


class NetworkValidationTests(unittest.TestCase):
    def test_thermal_scaling_preserves_transformer_impedance_and_flow(self):
        n = network()
        n.generators_t.p_set = pd.DataFrame({"external": [30., 100.], "wind": [90., 20.]}, index=n.snapshots)
        n.transformers_t.s_max_pu = pd.DataFrame({"ha": [.8, .9]}, index=n.snapshots)
        n.lpf()
        flow = n.lines_t.p0.copy()
        reactance = n.transformers.x_pu_eff.copy()
        scale_thermal_limits(n, 2)
        n.lpf()
        np.testing.assert_allclose(n.transformers.x_pu_eff, reactance)
        np.testing.assert_allclose(n.lines_t.p0, flow, atol=1e-9)
        np.testing.assert_allclose(n.transformers_t.s_max_pu.ha, [1.6, 1.8])
        self.assertEqual(n.transformers.at["ha", "s_nom"], 250)

    def test_fingerprint_covers_profiles_weights_and_boundary_bounds(self):
        n = network()
        identity = network_input_fingerprint(n)
        n.generators_t.p_max_pu.loc[n.snapshots[0], "wind"] = .7
        self.assertNotEqual(identity, network_input_fingerprint(n))
        identity = network_input_fingerprint(n)
        n.snapshot_weightings.iloc[0, 0] = 2
        self.assertNotEqual(identity, network_input_fingerprint(n))
        identity = network_input_fingerprint(n)
        n.generators.at["external", "p_min_pu"] = .1
        self.assertNotEqual(identity, network_input_fingerprint(n))

    def test_hourly_crossing_transformers_replay_without_hidden_slack(self):
        n = network()
        self.assertEqual(n.optimize(solver_name="highs", progress=False,
                         solver_options={"output_flag": False}), ("ok", "optimal"))
        reduced, crosswalk, profiles = source_bus_extract(n, {"a", "b"})
        self.assertEqual(len(crosswalk), 2)
        self.assertTrue((profiles.diff().iloc[1].abs() > 0).any())
        summary, _ = validate_replay(n, reduced, tolerance_mw=1e-6)
        self.assertTrue(summary["passed"], summary)
        self.assertFalse(summary["site_connection_and_merchant_dispatch_validated"])
        # Removing a boundary contribution must fail, even if LPF supplies it.
        reduced.generators_t.p_set.loc[:, "REPLAY_BOUNDARY_000"] += 1
        summary, _ = validate_replay(n, reduced, tolerance_mw=1e-6)
        self.assertFalse(summary["passed"])

    def test_q2_cache_key_changes_with_input_identity(self):
        n = network()
        ctx = SimpleNamespace(scenario="WP2033", scope="north-west", target_mode="line",
                              target_lines=["ab"], baseline_network=n,
                              input_identity=network_input_fingerprint(n))
        model = q2.BatteryModel("test", .1, .9, .9)
        key = q2.build_trial_key(ctx, ["a"], [1.], 25, 100, model, .99, "benchmark")
        n.lines.at["ab", "x"] *= 2
        ctx.input_identity = network_input_fingerprint(n)
        self.assertNotEqual(key, q2.build_trial_key(ctx, ["a"], [1.], 25, 100, model, .99, "benchmark"))

    def test_same_bess_counterfactual_detects_remaining_congestion(self):
        n = pypsa.Network()
        n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
        n.add("Bus", "a", v_nom=110)
        n.add("Bus", "b", v_nom=110)
        n.add("Line", "ab", bus0="a", bus1="b", x=10, r=1, s_nom=20)
        n.add("Generator", "wind", bus="a", p_nom=100, carrier="wind", marginal_cost=-1,
              p_max_pu=pd.Series(1., index=n.snapshots))
        n.add("Generator", "thermal", bus="b", p_nom=100, marginal_cost=50)
        n.add("Load", "demand", bus="b", p_set=100)
        ctx = SimpleNamespace(scenario="test", scope="test", original_ratings_mva={"ab": 20},
                              input_identity=network_input_fingerprint(n))
        normal = n.copy()
        model = q2.BatteryModel("test", .1, .9, .9)
        q2.add_battery_fleet(normal, ["b"], [1.], 25, 100, model)
        q2.solve_checked(normal, "test")
        with patch.object(q2.gridkit, "load", side_effect=lambda *_: n.copy()):
            result = q2.same_bess_target_residual(ctx, normal, ["b"], [1.], 25, 100, model)
        self.assertGreater(result["same_bess_residual_dispatch_down_mwh"], 100)
        self.assertFalse(result["same_bess_target_residual_pass"])


if __name__ == "__main__":
    unittest.main()
