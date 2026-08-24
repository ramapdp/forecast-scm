import unittest

import pandas as pd

from utils.data_preprocessing import build_panel


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
            "Satuan": ["Porsi"],
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
            "Satuan": ["Porsi"] * 2,
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
            "Satuan": ["Porsi"] * 2,
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
            "Satuan": ["Porsi"] * 2,
        })
        result = build_panel.build_dense_panel(df).sort_values("Tanggal")
        self.assertEqual(list(result["Nama Barang"]), ["Widget", "Widget", "Widget"])

    def test_forward_fills_satuan_across_gap(self):
        # Satuan is a per-SKU constant the model needs downstream (a Porsi 3 and
        # a Kg 3 are not the same demand), so zero-filled days must carry it
        # like the other descriptive columns rather than leaving NaN.
        df = pd.DataFrame({
            "Kode Barang": ["A", "A"], "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "Kuantitas": [5, 9],
            "Kategori Barang": ["Barang Jadi (FG)"] * 2, "Nama Barang": ["Widget"] * 2,
            "Satuan": ["Porsi"] * 2,
        })
        result = build_panel.build_dense_panel(df).sort_values("Tanggal")
        self.assertEqual(list(result["Satuan"]), ["Porsi", "Porsi", "Porsi"])

    def test_keeps_separate_pairs_independent(self):
        df = pd.DataFrame({
            "Kode Barang": ["A", "B"], "Nama Cabang": ["X", "Y"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-10"]),
            "Kuantitas": [5, 9],
            "Kategori Barang": ["Barang Jadi (FG)"] * 2, "Nama Barang": ["Widget", "Gadget"],
            "Satuan": ["Porsi"] * 2,
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


class TestDensePanelSegments(unittest.TestCase):
    def _pair_frame(self):
        # Two active blocks either side of a closure: Jan 1-3 and Mar 1-3.
        dates = ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-03-01", "2024-03-02", "2024-03-03"]
        return pd.DataFrame({
            "Kode Barang": ["A"] * 6, "Nama Cabang": ["X"] * 6,
            "Tanggal": pd.to_datetime(dates), "Kuantitas": [1] * 6,
            "Kategori Barang": ["Barang Jadi (FG)"] * 6, "Nama Barang": ["Widget"] * 6,
            "Satuan": ["Porsi"] * 6,
        })

    def test_without_closures_output_matches_legacy_behaviour(self):
        df = self._pair_frame()
        result = build_panel.build_dense_panel(df)
        # Jan 1 -> Mar 3 inclusive is 63 days, all gap-filled as before.
        self.assertEqual(len(result), 63)
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_closure_removes_rows_and_starts_a_second_segment(self):
        df = self._pair_frame()
        closures = {"X": [(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-03-01"))]}
        result = build_panel.build_dense_panel(df, closures=closures).sort_values("Tanggal")
        self.assertEqual(len(result), 6)
        self.assertEqual(list(result[build_panel.SEGMENT_COL]), [1, 1, 1, 2, 2, 2])
        self.assertTrue(
            result[(result["Tanggal"] >= "2024-01-04") & (result["Tanggal"] < "2024-03-01")].empty
        )

    def test_open_ended_closure_truncates_the_tail(self):
        df = self._pair_frame()
        closures = {"X": [(pd.Timestamp("2024-01-04"), None)]}
        result = build_panel.build_dense_panel(df, closures=closures)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["Tanggal"].max(), pd.Timestamp("2024-01-03"))
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_closure_for_another_branch_has_no_effect(self):
        df = self._pair_frame()
        closures = {"Y": [(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-03-01"))]}
        result = build_panel.build_dense_panel(df, closures=closures)
        self.assertEqual(len(result), 63)

    def test_pair_entirely_inside_a_closure_is_dropped(self):
        df = self._pair_frame()
        other = df.copy()
        other["Nama Cabang"] = "Z"
        combined = pd.concat([df, other], ignore_index=True)
        closures = {"Z": [(pd.Timestamp("2023-12-01"), pd.Timestamp("2024-06-01"))]}
        result = build_panel.build_dense_panel(combined, closures=closures)
        self.assertEqual(set(result["Nama Cabang"]), {"X"})

    def test_every_segment_is_internally_dense(self):
        df = self._pair_frame()
        closures = {"X": [(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-03-01"))]}
        result = build_panel.build_dense_panel(df, closures=closures)
        for _, group in result.groupby(["Kode Barang", "Nama Cabang", build_panel.SEGMENT_COL]):
            spans = group["Tanggal"].sort_values().diff().dropna().dt.days
            self.assertTrue((spans == 1).all())


class TestDensePanelBreakpoints(unittest.TestCase):
    """A relocation splits a continuous run of dates into two segments.

    Unlike a closure it removes no rows -- the outlet kept trading, just at a
    different address serving a different market -- so the panel stays dense
    while every shift-based feature stops reaching across the move.
    """

    def _continuous_frame(self, branch="X", n_days=10):
        return pd.DataFrame({
            "Kode Barang": ["A"] * n_days, "Nama Cabang": [branch] * n_days,
            "Tanggal": pd.date_range("2024-01-01", periods=n_days, freq="D"),
            "Kuantitas": [1] * n_days,
            "Kategori Barang": ["Barang Jadi (FG)"] * n_days,
            "Nama Barang": ["Widget"] * n_days,
        "Satuan": ["Porsi"] * n_days,
        })

    def test_breakpoint_starts_a_new_segment_without_dropping_rows(self):
        df = self._continuous_frame()
        breakpoints = {"X": [pd.Timestamp("2024-01-05")]}
        result = build_panel.build_dense_panel(
            df, breakpoints=breakpoints
        ).sort_values("Tanggal")
        self.assertEqual(len(result), 10)
        self.assertEqual(
            list(result[build_panel.SEGMENT_COL]), [1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        )

    def test_breakpoint_date_itself_begins_the_new_segment(self):
        df = self._continuous_frame()
        breakpoints = {"X": [pd.Timestamp("2024-01-05")]}
        result = build_panel.build_dense_panel(df, breakpoints=breakpoints)
        moved_day = result[result["Tanggal"] == pd.Timestamp("2024-01-05")]
        self.assertEqual(moved_day[build_panel.SEGMENT_COL].iloc[0], 2)

    def test_breakpoint_for_another_branch_has_no_effect(self):
        df = self._continuous_frame()
        breakpoints = {"Y": [pd.Timestamp("2024-01-05")]}
        result = build_panel.build_dense_panel(df, breakpoints=breakpoints)
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_breakpoint_outside_the_pair_date_range_is_ignored(self):
        df = self._continuous_frame()
        breakpoints = {"X": [pd.Timestamp("2025-06-01")]}
        result = build_panel.build_dense_panel(df, breakpoints=breakpoints)
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_breakpoint_on_the_first_date_leaves_one_segment(self):
        df = self._continuous_frame()
        breakpoints = {"X": [pd.Timestamp("2024-01-01")]}
        result = build_panel.build_dense_panel(df, breakpoints=breakpoints)
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_closure_and_breakpoint_combine_into_three_segments(self):
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D")) + \
                list(pd.date_range("2024-03-01", periods=5, freq="D"))
        df = pd.DataFrame({
            "Kode Barang": ["A"] * 10, "Nama Cabang": ["X"] * 10,
            "Tanggal": pd.to_datetime(dates), "Kuantitas": [1] * 10,
            "Kategori Barang": ["Barang Jadi (FG)"] * 10, "Nama Barang": ["Widget"] * 10,
            "Satuan": ["Porsi"] * 10,
        })
        closures = {"X": [(pd.Timestamp("2024-01-06"), pd.Timestamp("2024-03-01"))]}
        breakpoints = {"X": [pd.Timestamp("2024-03-03")]}
        result = build_panel.build_dense_panel(
            df, closures=closures, breakpoints=breakpoints
        ).sort_values("Tanggal")
        self.assertEqual(
            list(result[build_panel.SEGMENT_COL]), [1, 1, 1, 1, 1, 2, 2, 3, 3, 3]
        )

    def test_segments_stay_internally_dense_after_a_breakpoint(self):
        df = self._continuous_frame()
        breakpoints = {"X": [pd.Timestamp("2024-01-05")]}
        result = build_panel.build_dense_panel(df, breakpoints=breakpoints)
        for _, group in result.groupby(["Kode Barang", "Nama Cabang", build_panel.SEGMENT_COL]):
            spans = group["Tanggal"].sort_values().diff().dropna().dt.days
            self.assertTrue((spans == 1).all())

    def test_breakpoints_none_reproduces_legacy_behaviour(self):
        df = self._continuous_frame()
        self.assertTrue(
            build_panel.build_dense_panel(df, breakpoints=None).equals(
                build_panel.build_dense_panel(df)
            )
        )


if __name__ == "__main__":
    unittest.main()
