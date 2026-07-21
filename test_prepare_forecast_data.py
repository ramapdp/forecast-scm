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


if __name__ == "__main__":
    unittest.main()
