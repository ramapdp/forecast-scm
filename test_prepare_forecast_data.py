import unittest

import pandas as pd

import prepare_forecast_data


def _pair_series(qtys, start="2025-01-01", pair=("A", "X")):
    n = len(qtys)
    return pd.DataFrame({
        "Kode Barang": [pair[0]] * n, "Nama Cabang": [pair[1]] * n,
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "Kuantitas": qtys,
    })


class TestAddTargets(unittest.TestCase):
    def test_all_horizons_populated_with_correct_future_values(self):
        df = _pair_series(list(range(1, 11)))  # 10 days: 1..10
        result = prepare_forecast_data.add_targets(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        for h in range(1, 8):
            self.assertEqual(day1[f"target_h{h}"], h + 1)  # day1 + h -> value h+1

    def test_target_h1_is_next_day_not_current_day(self):
        df = _pair_series([10, 20, 30])
        result = prepare_forecast_data.add_targets(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        self.assertEqual(day1["target_h1"], 20)
        self.assertNotEqual(day1["target_h1"], 10)

    def test_targets_beyond_available_data_are_nan(self):
        df = _pair_series([1, 2, 3])  # only 3 days
        result = prepare_forecast_data.add_targets(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        self.assertEqual(day1["target_h1"], 2)
        self.assertEqual(day1["target_h2"], 3)
        self.assertTrue(pd.isna(day1["target_h3"]))
        self.assertTrue(pd.isna(day1["target_h7"]))


class TestAddLagFeatures(unittest.TestCase):
    def test_all_lags_populated_with_correct_past_values(self):
        df = _pair_series(list(range(1, 30)))  # 29 days: 1..29
        result = prepare_forecast_data.add_lag_features(df)
        day29 = result[result["Tanggal"] == pd.Timestamp("2025-01-29")].iloc[0]
        self.assertEqual(day29["lag_1"], 28)
        self.assertEqual(day29["lag_7"], 22)
        self.assertEqual(day29["lag_28"], 1)

    def test_higher_lags_are_nan_when_insufficient_history(self):
        df = _pair_series(list(range(1, 6)))  # only 5 days
        result = prepare_forecast_data.add_lag_features(df)
        day5 = result[result["Tanggal"] == pd.Timestamp("2025-01-05")].iloc[0]
        self.assertEqual(day5["lag_1"], 4)  # 1 day back is available
        self.assertTrue(pd.isna(day5["lag_7"]))
        self.assertTrue(pd.isna(day5["lag_28"]))


class TestAddRollingFeatures(unittest.TestCase):
    def test_rolling_mean_and_std_exclude_current_day(self):
        # Days 1..10 with Kuantitas 1..10. On day 8, the trailing 7-day
        # window (days 1-7, i.e. values 1..7) must be used — NOT days 2-8.
        df = _pair_series(list(range(1, 11)))
        result = prepare_forecast_data.add_rolling_features(df)
        day8 = result[result["Tanggal"] == pd.Timestamp("2025-01-08")].iloc[0]
        self.assertAlmostEqual(day8["roll_mean_7"], 4.0)  # mean(1..7)
        self.assertAlmostEqual(day8["roll_std_7"], pd.Series(range(1, 8)).std())

    def test_early_rows_are_nan_for_windows_larger_than_available_history(self):
        df = _pair_series(list(range(1, 6)))  # only 5 days
        result = prepare_forecast_data.add_rolling_features(df)
        day5 = result[result["Tanggal"] == pd.Timestamp("2025-01-05")].iloc[0]
        self.assertTrue(pd.isna(day5["roll_mean_7"]))
        self.assertTrue(pd.isna(day5["roll_mean_14"]))
        self.assertTrue(pd.isna(day5["roll_mean_28"]))


class TestComputeBranchStats(unittest.TestCase):
    def test_stats_computed_only_from_pre_cutoff_rows(self):
        train = pd.DataFrame({
            "Nama Cabang": ["X"] * 10,
            "Tanggal": pd.date_range("2025-11-01", periods=10, freq="D"),
            "Kuantitas": [10] * 10,  # steady 10/day
        })
        test_period = pd.DataFrame({
            "Nama Cabang": ["X"] * 5,
            "Tanggal": pd.date_range("2025-12-01", periods=5, freq="D"),
            "Kuantitas": [99999] * 5,  # extreme test-period values must not leak in
        })
        df = pd.concat([train, test_period], ignore_index=True)
        result = prepare_forecast_data.compute_branch_stats(df, cutoff=pd.Timestamp("2025-12-01"))
        branch_x = result[result["Nama Cabang"] == "X"].iloc[0]
        self.assertAlmostEqual(branch_x["branch_avg_daily_qty"], 10.0)

    def test_changing_test_period_values_does_not_change_output(self):
        train = pd.DataFrame({
            "Nama Cabang": ["X"] * 10,
            "Tanggal": pd.date_range("2025-11-01", periods=10, freq="D"),
            "Kuantitas": [10] * 10,
        })
        cutoff = pd.Timestamp("2025-12-01")
        test_a = pd.concat([train, pd.DataFrame({
            "Nama Cabang": ["X"], "Tanggal": [cutoff], "Kuantitas": [1],
        })], ignore_index=True)
        test_b = pd.concat([train, pd.DataFrame({
            "Nama Cabang": ["X"], "Tanggal": [cutoff], "Kuantitas": [999999],
        })], ignore_index=True)
        result_a = prepare_forecast_data.compute_branch_stats(test_a, cutoff=cutoff)
        result_b = prepare_forecast_data.compute_branch_stats(test_b, cutoff=cutoff)
        pd.testing.assert_frame_equal(result_a, result_b)

    def test_branch_volume_tier_ranks_distinct_branches(self):
        rows = []
        for branch, daily_qty in [("Small", 5), ("Medium", 50), ("Large", 500), ("Flagship", 5000)]:
            rows.append(pd.DataFrame({
                "Nama Cabang": [branch] * 10,
                "Tanggal": pd.date_range("2025-09-01", periods=10, freq="D"),
                "Kuantitas": [daily_qty] * 10,
            }))
        df = pd.concat(rows, ignore_index=True)
        result = prepare_forecast_data.compute_branch_stats(df, cutoff=pd.Timestamp("2025-12-01"))
        tiers = result.set_index("Nama Cabang")["branch_volume_tier"]
        self.assertNotEqual(tiers["Small"], tiers["Flagship"])


if __name__ == "__main__":
    unittest.main()
