from __future__ import annotations

from pathlib import Path
import re
import sys
import unittest

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "problem_3_1"))

from q5_shift_factors.core import (
    adjust_reference,
    branches,
    connected_component_count,
    factor_tables,
    lpf_transfer_delta,
    ptdf,
    rating_invariance,
    reference_weights,
    validate_with_lpf,
    wind_farms,
)
from q5_shift_factors.pipeline import load_config, validate_config


def toy_network() -> pypsa.Network:
    network = pypsa.Network()
    network.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    for bus in ("A", "B", "C"):
        network.add("Bus", bus, v_nom=110.0, x=-8.0, y=54.0)
    network.add("Line", "AB", bus0="A", bus1="B", x=1.0, r=0.0, s_nom=100.0)
    network.add("Line", "BC", bus0="B", bus1="C", x=1.0, r=0.0, s_nom=100.0)
    network.add(
        "Generator",
        "system slack",
        bus="C",
        control="Slack",
        p_nom=100.0,
        p_set=0.0,
        carrier="boundary",
    )
    network.add(
        "Generator",
        "Farm A",
        bus="A",
        control="PQ",
        p_nom=20.0,
        p_set=0.0,
        carrier="wind",
    )
    network.add(
        "Generator",
        "Farm B",
        bus="B",
        control="PQ",
        p_nom=30.0,
        p_set=0.0,
        carrier="wind",
    )
    network.add("Load", "load A", bus="A", p_set=1.0)
    network.add("Load", "load C", bus="C", p_set=3.0)
    return network


class PTDFCoreTests(unittest.TestCase):
    def test_radial_single_slack_has_known_factors(self):
        network = toy_network()
        base = ptdf(network)
        at_c = adjust_reference(base, reference_weights(network, "C"))
        self.assertAlmostEqual(float(at_c.at["AB", "A"]), 1.0, places=10)
        self.assertAlmostEqual(float(at_c.at["AB", "B"]), 0.0, places=10)
        self.assertAlmostEqual(float(at_c.at["AB", "C"]), 0.0, places=10)
        self.assertAlmostEqual(float(at_c.at["BC", "A"]), 1.0, places=10)
        self.assertAlmostEqual(float(at_c.at["BC", "B"]), 1.0, places=10)

    def test_load_weighted_reference_has_zero_weighted_mean(self):
        network = toy_network()
        weights = reference_weights(network, "load")
        self.assertAlmostEqual(float(weights.loc["A"]), 0.25)
        self.assertAlmostEqual(float(weights.loc["C"]), 0.75)
        adjusted = adjust_reference(ptdf(network), weights)
        residual = adjusted.to_numpy() @ weights.reindex(adjusted.columns).to_numpy()
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    def test_wind_mapping_uses_exact_model_bus(self):
        farms = wind_farms(toy_network())
        self.assertEqual(list(farms.index), ["Farm A", "Farm B"])
        self.assertEqual(farms.at["Farm A", "nearest_substation"], "A")
        self.assertEqual(
            farms.at["Farm A", "mapping_method"],
            "official_model_generator_bus_assignment",
        )

    def test_lpf_finite_difference_is_independent_validation(self):
        network = toy_network()
        farms = wind_farms(network)
        _, matrices, _ = factor_tables(
            network,
            ["AB", "BC"],
            farms,
            {"single_slack": "C"},
        )
        result = validate_with_lpf(
            network,
            ["AB", "BC"],
            farms,
            matrices["single_slack"],
            "C",
            delta_mw=1.0,
            tolerance=1e-8,
        )
        self.assertEqual(len(result), 4)
        self.assertTrue(bool(result["passed"].all()))
        self.assertLess(float(result["absolute_error"].max()), 1e-8)
        measured = lpf_transfer_delta(
            network,
            source_bus="A",
            weights=reference_weights(network, "C"),
        )
        self.assertAlmostEqual(float(measured.loc["AB"]), 1.0, places=8)

    def test_rating_only_dlr_does_not_change_ptdf(self):
        network = toy_network()
        farms = wind_farms(network)
        result = rating_invariance(
            network,
            ["AB"],
            farms,
            reference="load",
            multiplier=1.25,
        )
        self.assertTrue(bool(result.at[0, "ptdf_unchanged"]))
        self.assertLessEqual(
            float(result.at[0, "maximum_absolute_shift_factor_change"]), 1e-12
        )

    def test_connectivity_and_branch_orientation(self):
        network = toy_network()
        frame = branches(network)
        self.assertEqual(connected_component_count(network, frame), 1)
        self.assertEqual(frame.at["AB", "bus0"], "A")
        self.assertEqual(frame.at["AB", "bus1"], "B")


    def test_disconnected_farm_has_zero_factor_on_other_component(self):
        network = toy_network()
        network.add("Bus", "D", v_nom=110.0, x=-9.0, y=54.5)
        network.add(
            "Generator",
            "Farm D",
            bus="D",
            control="PQ",
            p_nom=10.0,
            p_set=0.0,
            carrier="wind",
        )
        network.add("Load", "load D", bus="D", p_set=2.0)
        farms = wind_farms(network)
        self.assertEqual(connected_component_count(network), 2)
        _, matrices, _ = factor_tables(
            network,
            ["AB"],
            farms,
            {"load_weighted": "load", "single_slack": "C"},
        )
        self.assertAlmostEqual(
            float(matrices["load_weighted"].at["Farm D", "AB"]), 0.0, places=12
        )
        self.assertTrue(np.isnan(matrices["single_slack"].at["Farm D", "AB"]))


class ConfigurationAndLanguageTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        config = load_config(None)
        validate_config(config)
        config["monitored_lines"] = []
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_owned_q4_q5_source_and_config_are_english_only(self):
        pattern = re.compile(r"[\u4e00-\u9fff]")
        paths = [
            ROOT / "src" / "problem_3_1" / "q4_techno_economic",
            ROOT / "src" / "problem_3_1" / "q5_shift_factors",
            ROOT / "src" / "problem_3_1" / "question_4_compare_results.py",
            ROOT / "src" / "problem_3_1" / "question_4_techno_economics.py",
            ROOT / "src" / "problem_3_1" / "question_5_shift_factors.py",
            ROOT / "configs" / "problem_3_1",
            ROOT / "tests" / "q4",
            ROOT / "tests" / "q5",
        ]
        offenders: list[str] = []
        for path in paths:
            files = [path] if path.is_file() else [
                candidate
                for candidate in path.rglob("*")
                if candidate.suffix in {".py", ".json", ".md"}
                and "__pycache__" not in candidate.parts
            ]
            for file in files:
                match = pattern.search(file.read_text(encoding="utf-8"))
                if match:
                    offenders.append(str(file.relative_to(ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
