import unittest

import pandas as pd

from utils import outlier_handling


def _pair_rows(pair, qtys, start="2025-01-01"):
    n = len(qtys)
    return pd.DataFrame({
        "Kode Barang": [pair[0]] * n, "Nama Cabang": [pair[1]] * n,
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "Kuantitas": qtys,
    })


class TestComputePairBaseline(unittest.TestCase):
    def test_eligible_pair_gets_correct_median(self):
        # 35 real-transaction days, mostly 10 with one high value — median
        # stays robust to the single outlier.
        qtys = [10] * 34 + [500]
        df = _pair_rows(("A", "X"), qtys, start="2025-01-01")
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertEqual(row["pair_median"], 10.0)
        self.assertTrue(row["pair_eligible"])

    def test_pair_below_min_history_is_ineligible(self):
        qtys = [10] * 29  # 29 < MIN_PAIR_HISTORY (30)
        df = _pair_rows(("A", "X"), qtys, start="2025-01-01")
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertFalse(row["pair_eligible"])

    def test_zero_fill_gap_days_do_not_count_toward_history(self):
        # 20 real transactions + 15 zero-quantity gap-fill rows = 35 panel
        # rows, but only 20 are real — below the 30-day minimum.
        real = _pair_rows(("A", "X"), [10] * 20, start="2025-01-01")
        gaps = _pair_rows(("A", "X"), [0] * 15, start="2025-01-21")
        df = pd.concat([real, gaps], ignore_index=True)
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertFalse(row["pair_eligible"])

    def test_test_period_rows_excluded_from_baseline(self):
        train = _pair_rows(("A", "X"), [10] * 30, start="2025-10-01")
        test_period = _pair_rows(("A", "X"), [99999] * 5, start="2025-12-01")
        df = pd.concat([train, test_period], ignore_index=True)
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        row = result[(result["Kode Barang"] == "A") & (result["Nama Cabang"] == "X")].iloc[0]
        self.assertEqual(row["pair_median"], 10.0)


if __name__ == "__main__":
    unittest.main()
