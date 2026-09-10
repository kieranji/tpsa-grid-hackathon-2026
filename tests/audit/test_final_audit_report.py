"""Regression tests for the final audit-report entry point."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import final_audit_report as audit


class FinalAuditReportTests(unittest.TestCase):
    def test_default_root_is_parent_of_src_directory(self) -> None:
        expected = Path(audit.__file__).resolve().parents[1]
        self.assertEqual(audit.default_root(), expected)

    def test_read_rc_accepts_plain_and_prefixed_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plain.rc").write_text("0\n", encoding="utf-8")
            (root / "prefixed.rc").write_text("RC=0\n", encoding="utf-8")
            self.assertEqual(audit.read_rc(root, "plain.rc"), ("PASS", 0))
            self.assertEqual(audit.read_rc(root, "prefixed.rc"), ("PASS", 0))

    def test_q32_scope_gap_filename_matches_final_writer(self) -> None:
        self.assertEqual(audit.Q32_SCOPE_GAPS, Path("16_model_scope_and_gaps.csv"))

    def test_q4_dlr_crosswalk_filename_matches_final_artifact(self) -> None:
        expected = Path(
            "results/problem_3_1/question_4_final_comparison_20260910_v3/"
            "01_capacity_fixed_vs_dlr.csv"
        )
        self.assertEqual(audit.Q4_DLR_COMPARISON_DIR / expected.name, expected)

    def test_money_places_negative_sign_before_currency_symbol(self) -> None:
        self.assertEqual(audit.money(-1234.5), "-€1,234.50")


if __name__ == "__main__":
    unittest.main()
