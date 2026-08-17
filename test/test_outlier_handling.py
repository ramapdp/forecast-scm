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

    def test_pair_that_only_ever_moves_in_whole_units_is_flagged(self):
        df = _pair_rows(("A", "X"), [10] * 30)
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertTrue(result.iloc[0]["pair_integer_only"])

    def test_pair_with_a_fractional_quantity_is_not_flagged(self):
        df = _pair_rows(("A", "X"), [10] * 29 + [10.5])
        result = outlier_handling.compute_pair_baseline(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertFalse(result.iloc[0]["pair_integer_only"])


def _with_event_flags(df, **flags):
    result = df.copy()
    for col in outlier_handling.EVENT_FLAG_COLS:
        result[col] = flags.get(col, False)
    return result


class TestApplyOutlierCapping(unittest.TestCase):
    def _baseline(self, pair=("A", "X"), median=10.0, eligible=True, integer_only=None):
        frame = pd.DataFrame({
            "Kode Barang": [pair[0]], "Nama Cabang": [pair[1]],
            "pair_median": [median], "pair_eligible": [eligible],
        })
        # integer_only=None reproduces a baseline frame built before the
        # whole-unit rounding existed, which must still be accepted.
        if integer_only is not None:
            frame["pair_integer_only"] = [integer_only]
        return frame

    def test_caps_spike_above_threshold_outside_event_window(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 50.0)  # 10 * 5.0
        self.assertTrue(result["is_spike"].iloc[0])
        self.assertEqual(result["baseline_ratio"].iloc[0], 100.0)

    def test_does_not_cap_value_below_threshold(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [40]))  # ratio 4.0 < 5.0
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 40.0)
        self.assertFalse(result["is_spike"].iloc[0])

    def test_exempts_spike_inside_event_window(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]), is_ramadan=True)
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 1000.0)  # not capped
        self.assertTrue(result["is_spike"].iloc[0])  # still flagged as detected

    def test_ineligible_pair_is_never_capped(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        baseline = self._baseline(median=10.0, eligible=False)
        result = outlier_handling.apply_outlier_capping(df, baseline)
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 1000.0)
        self.assertFalse(result["is_spike"].iloc[0])
        self.assertTrue(pd.isna(result["baseline_ratio"].iloc[0]))

    def test_gap_fill_zero_quantity_row_is_not_a_spike(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [0]))
        result = outlier_handling.apply_outlier_capping(df, self._baseline())
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 0.0)
        self.assertFalse(result["is_spike"].iloc[0])

    def test_whole_unit_pair_gets_a_whole_number_cap(self):
        # median 12.5 puts the raw cap at 62.5, which is meaningless for an
        # item counted in PCS. Round up, never down: the business criterion is
        # that the outlet does not run out.
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        result = outlier_handling.apply_outlier_capping(
            df, self._baseline(median=12.5, integer_only=True)
        )
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 63.0)

    def test_rounded_up_cap_never_exceeds_the_raw_quantity(self):
        # 62.6 / 12.5 = 5.008, so the row is capped -- but ceil(62.5) = 63
        # would push the capped value above the quantity actually issued and
        # break the capped <= raw invariant asserted in run_qa_checks.
        df = _with_event_flags(_pair_rows(("A", "X"), [62.6]))
        result = outlier_handling.apply_outlier_capping(
            df, self._baseline(median=12.5, integer_only=True)
        )
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 62.6)

    def test_pair_that_trades_in_fractions_keeps_the_exact_cap(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        result = outlier_handling.apply_outlier_capping(
            df, self._baseline(median=12.5, integer_only=False)
        )
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 62.5)

    def test_baseline_without_the_integer_flag_keeps_the_exact_cap(self):
        df = _with_event_flags(_pair_rows(("A", "X"), [1000]))
        result = outlier_handling.apply_outlier_capping(df, self._baseline(median=12.5))
        self.assertEqual(result["Kuantitas_capped"].iloc[0], 62.5)


if __name__ == "__main__":
    unittest.main()
