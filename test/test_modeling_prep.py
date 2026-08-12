import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep


def _event_items(rows):
    return pd.DataFrame(rows, columns=["Kode Barang", "is_event_driven"])


class TestAddEventFlag(unittest.TestCase):
    def test_marks_true_for_listed_event_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_marks_false_for_ordinary_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertFalse(bool(result.iloc[0]["is_event_driven"]))

    def test_is_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "  TRUE "]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_raises_when_a_sku_is_missing_from_the_list(self):
        df = pd.DataFrame({"Kode Barang": ["FGS-99999"]})
        items = _event_items([["PCG-00002", "true"]])
        with self.assertRaisesRegex(ValueError, "FGS-99999"):
            modeling_prep.add_event_flag(df, items)

    def test_does_not_mutate_the_input_frame(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00001", "false"]])
        modeling_prep.add_event_flag(df, items)
        self.assertNotIn("is_event_driven", df.columns)


def _series_frame(quantities, start="2024-01-01", item="I1", branch="B1"):
    return pd.DataFrame({
        "Kode Barang": [item] * len(quantities),
        "Nama Cabang": [branch] * len(quantities),
        "Tanggal": pd.date_range(start, periods=len(quantities), freq="D"),
        "Kuantitas": [float(q) for q in quantities],
    })


class TestClassifyPairs(unittest.TestCase):
    def test_daily_stable_demand_is_smooth(self):
        df = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "smooth")

    def test_daily_but_wildly_varying_demand_is_erratic(self):
        df = _series_frame([1, 50, 2, 80, 3, 90, 1, 70])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "erratic")

    def test_rare_but_consistent_demand_is_intermittent(self):
        df = _series_frame([10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "intermittent")

    def test_rare_and_bulky_demand_is_lumpy(self):
        df = _series_frame([5, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 90])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "lumpy")

    def test_pair_that_never_moved_is_lumpy(self):
        df = _series_frame([0, 0, 0, 0])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "lumpy")

    def test_segment_ignores_rows_at_or_after_the_cutoff(self):
        """The whole point of computing segments on train only: post-cutoff
        behaviour must not change the label."""
        train_only = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        with_future = pd.concat([
            train_only,
            _series_frame([0] * 60, start="2025-12-01"),
        ], ignore_index=True)
        cutoff = pd.Timestamp("2025-12-01")
        a = modeling_prep.classify_pairs(train_only, cutoff=cutoff).iloc[0]["demand_segment"]
        b = modeling_prep.classify_pairs(with_future, cutoff=cutoff).iloc[0]["demand_segment"]
        self.assertEqual(a, b)

    def test_every_row_of_a_pair_gets_the_same_label(self):
        df = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result["demand_segment"].nunique(), 1)


if __name__ == "__main__":
    unittest.main()
