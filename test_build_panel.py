import unittest

import pandas as pd

import build_panel


class TestBuildDensePanel(unittest.TestCase):
    def test_single_day_pair_yields_one_row(self):
        df = pd.DataFrame({
            "Kode Barang": ["A"], "Nama Cabang": ["X"],
            "Tanggal": pd.to_datetime(["2024-01-01"]),
            "Kuantitas": [5],
            "Kategori Barang": ["Barang Jadi (FG)"], "Nama Barang": ["Widget"],
        })
        result = build_panel.build_dense_panel(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Kuantitas"].iloc[0], 5)

    def test_zero_fills_gap_days(self):
        df = pd.DataFrame({
            "Kode Barang": ["A", "A"], "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-05"]),
            "Kuantitas": [5, 9],
            "Kategori Barang": ["Barang Jadi (FG)"] * 2, "Nama Barang": ["Widget"] * 2,
        })
        result = build_panel.build_dense_panel(df).sort_values("Tanggal")
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result["Kuantitas"]), [5, 0, 0, 0, 9])

    def test_spans_leap_day_correctly(self):
        df = pd.DataFrame({
            "Kode Barang": ["A", "A"], "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2024-02-27", "2024-03-02"]),
            "Kuantitas": [1, 2],
            "Kategori Barang": ["Barang Jadi (FG)"] * 2, "Nama Barang": ["Widget"] * 2,
        })
        result = build_panel.build_dense_panel(df)
        self.assertEqual(len(result), 5)  # Feb 27, 28, 29, Mar 1, Mar 2
        self.assertIn(pd.Timestamp("2024-02-29"), list(result["Tanggal"]))

    def test_forward_fills_category_and_name_across_gap(self):
        df = pd.DataFrame({
            "Kode Barang": ["A", "A"], "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "Kuantitas": [5, 9],
            "Kategori Barang": ["Barang Jadi (FG)"] * 2, "Nama Barang": ["Widget"] * 2,
        })
        result = build_panel.build_dense_panel(df).sort_values("Tanggal")
        self.assertEqual(list(result["Nama Barang"]), ["Widget", "Widget", "Widget"])

    def test_keeps_separate_pairs_independent(self):
        df = pd.DataFrame({
            "Kode Barang": ["A", "B"], "Nama Cabang": ["X", "Y"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "Kuantitas": [5, 9],
            "Kategori Barang": ["Barang Jadi (FG)"] * 2, "Nama Barang": ["Widget", "Gadget"],
        })
        result = build_panel.build_dense_panel(df)
        self.assertEqual(len(result), 2)  # each pair has only 1 day of its own history


if __name__ == "__main__":
    unittest.main()
