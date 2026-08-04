import unittest

import pandas as pd

from utils import build_panel


def _daily_rows(pair, start, n_days, qty=1):
    return pd.DataFrame({
        "Kode Barang": [pair[0]] * n_days, "Nama Cabang": [pair[1]] * n_days,
        "Tanggal": pd.date_range(start, periods=n_days, freq="D"),
        "Kuantitas": [qty] * n_days,
    })


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


class TestFilterMinHistory(unittest.TestCase):
    def test_drops_pair_with_fewer_than_min_days_pre_cutoff(self):
        df = _daily_rows(("A", "X"), "2025-10-01", 59)  # 59 < 60
        result = build_panel.filter_min_history(df, cutoff=pd.Timestamp("2025-12-01"), min_days=60)
        self.assertEqual(len(result), 0)

    def test_keeps_pair_with_exactly_min_days_pre_cutoff(self):
        df = _daily_rows(("A", "X"), "2025-10-02", 60)  # exactly 60 days ending 2025-12-01 (exclusive)
        result = build_panel.filter_min_history(df, cutoff=pd.Timestamp("2025-12-01"), min_days=60)
        self.assertEqual(len(result), 60)

    def test_post_cutoff_rows_do_not_count_toward_threshold(self):
        pre = _daily_rows(("A", "X"), "2025-11-01", 30)      # 30 pre-cutoff days
        post = _daily_rows(("A", "X"), "2025-12-01", 31)      # 31 post-cutoff days (doesn't help)
        df = pd.concat([pre, post], ignore_index=True)
        result = build_panel.filter_min_history(df, cutoff=pd.Timestamp("2025-12-01"), min_days=60)
        self.assertEqual(len(result), 0)  # only 30 pre-cutoff days, below the 60-day threshold

    def test_kept_pair_retains_all_its_rows_including_post_cutoff(self):
        pre = _daily_rows(("A", "X"), "2025-09-01", 90)
        result = build_panel.filter_min_history(pre, cutoff=pd.Timestamp("2025-12-01"), min_days=60)
        self.assertEqual(len(result), 90)  # all rows kept, not just the pre-cutoff ones

    def test_pair_inactive_before_test_window_is_kept_if_history_sufficient(self):
        # Activity stopped in October 2025 (before the Dec test window) — the
        # spec says this is correct: the pair simply won't appear in test
        # rows once split, not something this filter should remove.
        df = _daily_rows(("A", "X"), "2025-08-01", 70)
        result = build_panel.filter_min_history(df, cutoff=pd.Timestamp("2025-12-01"), min_days=60)
        self.assertEqual(len(result), 70)


if __name__ == "__main__":
    unittest.main()
