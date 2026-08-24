import unittest

import numpy as np
import pandas as pd

from utils.eda import analyze_spike_comovement as asc


def _rows(records):
    """records: (kode, kategori, cabang, tanggal, capped, spike, ratio)"""
    df = pd.DataFrame(records, columns=[
        "Kode Barang", "Kategori Barang", "Nama Cabang", "Tanggal",
        "is_capped", "is_spike", "baseline_ratio"])
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df["Nama Barang"] = df["Kode Barang"]
    df["is_weekend"] = False
    return df


class TestStandalonePackagingSpikeDays(unittest.TestCase):
    def test_keeps_a_day_where_only_one_packaging_item_is_capped(self):
        df = _rows([
            ("PCG-1", "Packaging", "X", "2025-01-01", True, True, 6.0),
            ("FGS-1", "Barang Jadi (FG)", "X", "2025-01-01", False, False, 1.0),
        ])
        out = asc.standalone_packaging_spike_days(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["packaging_code"], "PCG-1")

    def test_drops_a_day_where_a_second_item_is_also_capped(self):
        df = _rows([
            ("PCG-1", "Packaging", "X", "2025-01-01", True, True, 6.0),
            ("FGS-1", "Barang Jadi (FG)", "X", "2025-01-01", True, True, 7.0),
        ])
        self.assertTrue(asc.standalone_packaging_spike_days(df).empty)

    def test_drops_a_solo_spike_that_is_not_packaging(self):
        df = _rows([("FGS-1", "Barang Jadi (FG)", "X", "2025-01-01", True, True, 6.0)])
        self.assertTrue(asc.standalone_packaging_spike_days(df).empty)

    def test_counts_are_per_branch_not_pooled_across_branches(self):
        # One capped item at X and one at Y on the same date: each branch-day
        # is standalone, so both must survive.
        df = _rows([
            ("PCG-1", "Packaging", "X", "2025-01-01", True, True, 6.0),
            ("PCG-2", "Packaging", "Y", "2025-01-01", True, True, 6.0),
        ])
        self.assertEqual(len(asc.standalone_packaging_spike_days(df)), 2)


class TestOrdinaryBranchDays(unittest.TestCase):
    def test_excludes_a_day_with_an_uncapped_event_window_spike(self):
        # is_spike True but not capped (event window) — still not an ordinary day.
        df = _rows([
            ("FGS-1", "Barang Jadi (FG)", "X", "2025-01-01", False, True, 6.0),
            ("FGS-1", "Barang Jadi (FG)", "X", "2025-01-02", False, False, 1.0),
        ])
        out = asc.ordinary_branch_days(df)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["Tanggal"], pd.Timestamp("2025-01-02"))


class TestPercentileWithin(unittest.TestCase):
    def test_value_above_all_reference_values_is_near_one(self):
        pct = asc.percentile_within(pd.Series([99.0]), pd.Series(np.arange(100.0)))
        self.assertGreater(pct.iloc[0], 0.98)

    def test_median_value_sits_near_one_half(self):
        pct = asc.percentile_within(pd.Series([50.0]), pd.Series(np.arange(101.0)))
        self.assertAlmostEqual(pct.iloc[0], 0.5, places=2)

    def test_ties_use_midrank_instead_of_counting_as_above(self):
        # Reference is all 1.0; a spike-day value of 1.0 is neither high nor low.
        pct = asc.percentile_within(pd.Series([1.0]), pd.Series([1.0] * 10))
        self.assertAlmostEqual(pct.iloc[0], 0.5)

    def test_empty_reference_yields_nan(self):
        pct = asc.percentile_within(pd.Series([1.0]), pd.Series(dtype=float))
        self.assertTrue(pd.isna(pct.iloc[0]))


class TestCompareItemRatios(unittest.TestCase):
    def _item_frame(self, branch, dates, ratios):
        df = pd.DataFrame({
            "Nama Cabang": branch, "Tanggal": pd.to_datetime(dates),
            "baseline_ratio": ratios})
        return df

    def test_reports_zero_when_no_spike_day_carries_the_item(self):
        item = self._item_frame("X", ["2025-02-01"], [1.0])
        spikes = pd.DataFrame({"Nama Cabang": ["Y"],
                               "Tanggal": pd.to_datetime(["2025-01-01"])})
        ordinary = pd.DataFrame({"Nama Cabang": ["X"],
                                 "Tanggal": pd.to_datetime(["2025-02-01"])})
        self.assertEqual(asc.compare_item_ratios(item, spikes, ordinary)["n_spike_days"], 0)

    def test_branch_with_too_few_ordinary_days_is_left_out_of_percentiles(self):
        dates = pd.date_range("2025-01-01", periods=11)
        item = self._item_frame("X", dates, [1.0] * 11)
        spikes = pd.DataFrame({"Nama Cabang": ["X"], "Tanggal": [dates[0]]})
        ordinary = pd.DataFrame({"Nama Cabang": ["X"] * 10, "Tanggal": dates[1:]})
        out = asc.compare_item_ratios(item, spikes, ordinary)
        self.assertEqual(out["n_spike_days"], 1)
        self.assertEqual(out["n_with_reference"], 0)

    def test_elevated_spike_days_push_the_mean_percentile_above_one_half(self):
        dates = pd.date_range("2025-01-01", periods=130)
        ratios = [5.0] * 30 + [1.0] * 100  # first 30 days elevated
        item = self._item_frame("X", dates, ratios)
        spikes = pd.DataFrame({"Nama Cabang": ["X"] * 30, "Tanggal": dates[:30]})
        ordinary = pd.DataFrame({"Nama Cabang": ["X"] * 100, "Tanggal": dates[30:]})
        out = asc.compare_item_ratios(item, spikes, ordinary)
        self.assertGreater(out["pct_mean"], 0.9)
        self.assertEqual(out["spike_ratio_median"], 5.0)
        self.assertEqual(out["base_ratio_median"], 1.0)

    def test_unrelated_spike_days_leave_the_mean_percentile_near_one_half(self):
        rng = np.random.default_rng(0)
        dates = pd.date_range("2025-01-01", periods=200)
        item = self._item_frame("X", dates, rng.normal(1.0, 0.3, 200))
        spikes = pd.DataFrame({"Nama Cabang": ["X"] * 50, "Tanggal": dates[:50]})
        ordinary = pd.DataFrame({"Nama Cabang": ["X"] * 150, "Tanggal": dates[50:]})
        out = asc.compare_item_ratios(item, spikes, ordinary)
        self.assertAlmostEqual(out["pct_mean"], 0.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()


class TestDayOfWeekMatching(unittest.TestCase):
    def test_matching_on_day_of_week_removes_a_pure_weekend_effect(self):
        # Demand is high every Sunday and normal otherwise; every spike day
        # is a Sunday. Unmatched, the spike days look extreme; matched
        # against other Sundays only, they are perfectly ordinary.
        dates = pd.date_range("2025-01-05", periods=420)  # starts on a Sunday
        dow = dates.dayofweek
        ratios = np.where(dow == 6, 3.0, 1.0)
        item = pd.DataFrame({
            "Nama Cabang": "X", "Tanggal": dates,
            "baseline_ratio": ratios, "day_of_week": dow})
        sundays = dates[dow == 6]
        spikes = pd.DataFrame({
            "Nama Cabang": "X", "Tanggal": sundays[:25],
            "day_of_week": 6})
        ordinary = pd.DataFrame({
            "Nama Cabang": "X", "Tanggal": dates.drop(sundays[:25]),
            "day_of_week": dates.drop(sundays[:25]).dayofweek})
        naive = asc.compare_item_ratios(item, spikes, ordinary)
        matched = asc.compare_item_ratios(
            item, spikes, ordinary, match_cols=["day_of_week"])
        self.assertGreater(naive["pct_mean"], 0.85)
        self.assertAlmostEqual(matched["pct_mean"], 0.5, delta=0.05)
