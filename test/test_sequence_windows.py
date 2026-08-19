import unittest

import numpy as np
import pandas as pd

from utils import sequence_windows


def _panel(n_days=60, n_pairs=2, start="2025-01-01", seed=5):
    """A dense daily panel for a few pairs, shaped like model_input.parquet.

    Only the columns the windowing code reads are present; feature_cols is
    passed explicitly everywhere so the fixtures stay small.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for pair in range(n_pairs):
        parts.append(pd.DataFrame({
            "Tanggal": pd.date_range(start, periods=n_days, freq="D"),
            "Kode Barang": f"FGS-0000{pair}",
            "Nama Cabang": "KY001",
            "segment_id": 1,
            "feat_a": rng.normal(size=n_days).astype("float64"),
            "feat_b": rng.normal(size=n_days),
            "cat_idx": rng.integers(0, 3, size=n_days),
            "target_lead_time_cumulative": np.abs(rng.normal(size=n_days)) * 10,
        }))
    return pd.concat(parts, ignore_index=True)


FEATURES = ["feat_a", "feat_b", "cat_idx"]


class TestBuildIndex(unittest.TestCase):
    def test_dynamic_and_categorical_columns_are_separated(self):
        index = sequence_windows.build_index(_panel(), feature_cols=FEATURES,
                                             lookback=7)
        self.assertEqual(index["dynamic_cols"], ["feat_a", "feat_b"])
        self.assertEqual(index["idx_cols"], ["cat_idx"])

    def test_values_are_contiguous_float32_and_cats_are_int16(self):
        index = sequence_windows.build_index(_panel(), feature_cols=FEATURES,
                                             lookback=7)
        self.assertEqual(index["values"].dtype, np.dtype("float32"))
        self.assertTrue(index["values"].flags["C_CONTIGUOUS"])
        self.assertEqual(index["cats"].dtype, np.dtype("int16"))
        self.assertEqual(index["values"].shape, (120, 2))
        self.assertEqual(index["cats"].shape, (120, 1))

    def test_rows_are_sorted_by_segment_then_date(self):
        panel = _panel().sample(frac=1.0, random_state=0).reset_index(drop=True)
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        # positions restart at zero for each segment and never decrease within one
        self.assertEqual(index["positions"][0], 0)
        self.assertEqual(index["positions"].max(), 59)
        self.assertEqual(int((index["positions"] == 0).sum()), 2)

    def test_a_target_column_among_the_dynamic_columns_raises(self):
        """G2. The target must never become a window channel."""
        with self.assertRaises(ValueError) as caught:
            sequence_windows.build_index(
                _panel(),
                feature_cols=FEATURES + ["target_lead_time_cumulative"],
                lookback=7,
            )
        self.assertIn("target_lead_time_cumulative", str(caught.exception))

    def test_a_date_gap_inside_a_segment_raises(self):
        """G1. Window positions are only date arithmetic if the panel is dense."""
        panel = _panel()
        panel = panel.drop(index=30).reset_index(drop=True)
        with self.assertRaises(ValueError) as caught:
            sequence_windows.build_index(panel, feature_cols=FEATURES, lookback=7)
        self.assertIn("celah tanggal", str(caught.exception))

    def test_a_window_may_not_cross_a_segment_boundary(self):
        """G1. Two segments of the same pair are separated by a closure."""
        panel = _panel(n_days=40, n_pairs=1)
        second = panel.copy()
        second["segment_id"] = 2
        second["Tanggal"] = second["Tanggal"] + pd.Timedelta(days=100)
        both = pd.concat([panel, second], ignore_index=True)
        index = sequence_windows.build_index(both, feature_cols=FEATURES,
                                             lookback=7)
        # positions restart at 0 for the second segment, so no window built
        # from position >= lookback can reach back across the gap.
        self.assertEqual(int((index["positions"] == 0).sum()), 2)

    def test_the_lookup_maps_a_key_back_to_its_row_position(self):
        panel = _panel()
        index = sequence_windows.build_index(panel, feature_cols=FEATURES,
                                             lookback=7)
        key = ("FGS-00001", "KY001", 1, pd.Timestamp("2025-01-10"))
        position = index["lookup"].loc[key]
        self.assertEqual(index["dates"][position],
                         np.datetime64("2025-01-10", "D"))


if __name__ == "__main__":
    unittest.main()
