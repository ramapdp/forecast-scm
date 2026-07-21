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


if __name__ == "__main__":
    unittest.main()
