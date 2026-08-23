import unittest

import pandas as pd

from utils import verify_category_consistency as verify


def _frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    """rows: (Kode Barang, Kategori Barang)."""
    return pd.DataFrame(
        {
            "Kode Barang": [r[0] for r in rows],
            "Kategori Barang": [r[1] for r in rows],
        }
    )


class TestFindMultiCategorySkus(unittest.TestCase):
    def test_reports_sku_recorded_under_two_categories(self):
        df = _frame(
            [
                ("FGS-00001", "Barang Semi FG (WIP-2)"),
                ("FGS-00001", "Barang Jadi (FG)"),
            ]
        )
        self.assertEqual(
            verify.find_multi_category_skus(df),
            {"FGS-00001": ["Barang Jadi (FG)", "Barang Semi FG (WIP-2)"]},
        )

    def test_ignores_sku_with_a_single_category(self):
        df = _frame([("PCG-00001", "Packaging"), ("PCG-00001", "Packaging")])
        self.assertEqual(verify.find_multi_category_skus(df), {})

    def test_naming_synonyms_are_not_a_category_difference(self):
        # Minuman / Minuman - FG is one category under two names; the pipeline
        # merges them, so reporting them would be a false alarm.
        df = _frame([("FGS-00006", "Minuman"), ("FGS-00006", "Minuman - FG")])
        self.assertEqual(verify.find_multi_category_skus(df, apply_synonyms=True), {})


class TestBuildReport(unittest.TestCase):
    def test_wip2_rows_left_in_the_source_do_not_fail_the_gate(self):
        # The 2026-08-22 relabel is handled in the normalization layer, not by
        # re-exporting the source, so WIP-2 rows are expected to stay in
        # dataset.csv forever. Failing on them would make the gate cry wolf.
        raw = _frame(
            [
                ("FGS-00001", "Barang Semi FG (WIP-2)"),
                ("FGS-00001", "Barang Jadi (FG)"),
            ]
        )
        normalized = _frame(
            [("FGS-00001", "Barang Jadi (FG)"), ("FGS-00001", "Barang Jadi (FG)")]
        )
        report = verify.build_report(raw, normalized)

        self.assertTrue(report["passed"])
        self.assertEqual(report["source_wip2_rows"], 1)
        self.assertEqual(report["normalized_multi_category"], {})
        # Still reported, just not fatal.
        self.assertIn("FGS-00001", report["source_multi_category"])

    def test_multi_category_after_normalization_fails_the_gate(self):
        raw = _frame([("FGS-00099", "Barang Semi FG (WIP-2)")])
        normalized = _frame(
            [
                ("FGS-00099", "Barang Semi FG (WIP-2)"),
                ("FGS-00099", "Barang Jadi (FG)"),
            ]
        )
        report = verify.build_report(raw, normalized)

        self.assertFalse(report["passed"])
        self.assertIn("FGS-00099", report["normalized_multi_category"])

    def test_passes_when_normalization_leaves_every_sku_single_category(self):
        raw = _frame([("PCG-00001", "Packaging")])
        normalized = _frame([("PCG-00001", "Packaging")])
        report = verify.build_report(raw, normalized)

        self.assertTrue(report["passed"])
        self.assertEqual(report["source_wip2_rows"], 0)
        self.assertEqual(report["categories"], ["Packaging"])

    def test_lists_the_categories_that_survive_normalization(self):
        raw = _frame([("PCG-00001", "Packaging")])
        normalized = _frame(
            [
                ("PCG-00001", "Packaging"),
                ("FGS-00001", "Barang Jadi (FG)"),
                ("FGS-00006", "Minuman - FG"),
            ]
        )
        report = verify.build_report(raw, normalized)
        self.assertEqual(
            report["categories"],
            ["Barang Jadi (FG)", "Minuman - FG", "Packaging"],
        )


if __name__ == "__main__":
    unittest.main()
