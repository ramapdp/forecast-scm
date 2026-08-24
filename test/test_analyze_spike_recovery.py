import unittest

import numpy as np
import pandas as pd

from utils import analyze_spike_recovery as asr


def _series(qtys, start="2025-01-01", pair=("A", "X"), segment=1, capped=None):
    n = len(qtys)
    capped = [False] * n if capped is None else capped
    return pd.DataFrame({
        "Kode Barang": [pair[0]] * n,
        "Nama Cabang": [pair[1]] * n,
        "segment_id": [segment] * n,
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "Kuantitas": [float(q) for q in qtys],
        "is_capped": capped,
    })


class TestAddWindowMeans(unittest.TestCase):
    def test_windows_exclude_the_spike_day_itself(self):
        # 7 days of 1, spike day of 100, 7 days of 3.
        df = _series([1] * 7 + [100] + [3] * 7)
        out = asr.add_window_means(df, window=7)
        row = out.iloc[7]
        self.assertAlmostEqual(row["pre_mean"], 1.0)
        self.assertAlmostEqual(row["post_mean"], 3.0)

    def test_incomplete_window_at_segment_edge_is_null(self):
        df = _series([1] * 6 + [100] + [3] * 7)  # only 6 days before
        out = asr.add_window_means(df, window=7)
        self.assertTrue(pd.isna(out.iloc[6]["pre_mean"]))
        self.assertAlmostEqual(out.iloc[6]["post_mean"], 3.0)

    def test_windows_never_bridge_a_segment_boundary(self):
        # Same pair, two segments (a closure between them). The 7 days before
        # the spike live in segment 1 and must not feed segment 2's window.
        seg1 = _series([1] * 7, start="2025-01-01", segment=1)
        seg2 = _series([100] + [3] * 7, start="2025-03-01", segment=2)
        df = pd.concat([seg1, seg2], ignore_index=True)
        out = asr.add_window_means(df, window=7)
        spike = out[out["Kuantitas"] == 100].iloc[0]
        self.assertTrue(pd.isna(spike["pre_mean"]))

    def test_windows_are_computed_per_pair_not_across_pairs(self):
        a = _series([1] * 7 + [100] + [3] * 7, pair=("A", "X"))
        b = _series([50] * 15, pair=("B", "X"))
        df = pd.concat([a, b], ignore_index=True)
        out = asr.add_window_means(df, window=7)
        spike = out[out["Kuantitas"] == 100].iloc[0]
        self.assertAlmostEqual(spike["pre_mean"], 1.0)
        self.assertAlmostEqual(spike["post_mean"], 3.0)

    def test_excluding_capped_days_drops_them_from_both_windows(self):
        # A second capped day sits inside the post window; excluding it must
        # leave the mean of the six remaining days, not of all seven.
        qtys = [1] * 7 + [100] + [3, 3, 90, 3, 3, 3, 3]
        capped = [False] * 7 + [True] + [False, False, True, False, False, False, False]
        df = _series(qtys, capped=capped)
        out = asr.add_window_means(df, window=7, exclude_capped=True)
        row = out.iloc[7]
        self.assertAlmostEqual(row["post_mean"], 3.0)
        self.assertEqual(row["post_days"], 6)
        self.assertEqual(row["pre_days"], 7)

    def test_row_order_of_the_input_does_not_change_the_result(self):
        df = _series([1] * 7 + [100] + [3] * 7)
        shuffled = df.sample(frac=1.0, random_state=0)
        expected = asr.add_window_means(df, window=7)
        got = asr.add_window_means(shuffled, window=7)
        spike_expected = expected[expected["Kuantitas"] == 100].iloc[0]
        spike_got = got[got["Kuantitas"] == 100].iloc[0]
        self.assertAlmostEqual(spike_got["pre_mean"], spike_expected["pre_mean"])
        self.assertAlmostEqual(spike_got["post_mean"], spike_expected["post_mean"])


class TestCountCappedPerBranchDay(unittest.TestCase):
    def test_counts_all_categories_on_the_same_branch_day(self):
        df = pd.DataFrame({
            "Nama Cabang": ["X", "X", "X", "Y"],
            "Tanggal": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-01"]),
            "is_capped": [True, True, False, True],
        })
        out = asr.count_capped_per_branch_day(df)
        self.assertEqual(list(out["n_capped_same_day"]), [2, 2, 2, 1])

    def test_branch_day_without_any_capped_row_gets_zero(self):
        df = pd.DataFrame({
            "Nama Cabang": ["X"],
            "Tanggal": pd.to_datetime(["2025-01-01"]),
            "is_capped": [False],
        })
        out = asr.count_capped_per_branch_day(df)
        self.assertEqual(out.iloc[0]["n_capped_same_day"], 0)


class TestCompareWindows(unittest.TestCase):
    def _rows(self, pre, post):
        return pd.DataFrame({
            "pre_mean": pre, "post_mean": post,
            "Kuantitas": [100.0] * len(pre),
        })

    def test_drops_rows_with_an_incomplete_window(self):
        rows = self._rows([1.0, np.nan, 2.0], [1.0, 1.0, np.nan])
        stats = asr.compare_windows(rows)
        self.assertEqual(stats["n_total"], 3)
        self.assertEqual(stats["n_compared"], 1)

    def test_pull_forward_pattern_shows_post_below_pre(self):
        rows = self._rows([10.0] * 20, [2.0] * 20)
        stats = asr.compare_windows(rows)
        self.assertLess(stats["ratio_median"], 1.0)
        self.assertEqual(stats["share_post_below_pre"], 1.0)
        self.assertLess(stats["delta_pct"], 0)

    def test_flat_pattern_shows_no_change(self):
        rows = self._rows([10.0] * 20, [10.0] * 20)
        stats = asr.compare_windows(rows)
        self.assertAlmostEqual(stats["ratio_median"], 1.0)
        self.assertAlmostEqual(stats["delta_pct"], 0.0)

    def test_ratio_ignores_rows_whose_pre_window_is_zero(self):
        rows = self._rows([0.0, 4.0, 4.0], [5.0, 2.0, 2.0])
        stats = asr.compare_windows(rows)
        self.assertEqual(stats["n_ratio"], 2)
        self.assertAlmostEqual(stats["ratio_median"], 0.5)


if __name__ == "__main__":
    unittest.main()
