import unittest

import numpy as np
import pandas as pd

from utils import model_xgboost as xgb
from utils import modeling_prep, purging, walk_forward


FEATURES = ["feat_a", "feat_b", "cat_idx"]


def _dated_frame(n=400, seed=3, lead_time=3.0, start="2025-01-01"):
    """One pair's daily series, long enough to survive the 28-day warm-up cut."""
    rng = np.random.default_rng(seed)
    feat_a = rng.normal(size=n)
    return pd.DataFrame({
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "feat_a": feat_a,
        "feat_b": rng.normal(size=n),
        "cat_idx": rng.integers(0, 3, size=n),
        "target_lead_time_cumulative": np.abs(feat_a * 10 + 20),
        "lead_time_days": lead_time,
        "Kode Barang": "FGS-00001",
        "Nama Cabang": "KY001",
        "segment_id": 1,
    })


class TestSplitEarlyStopping(unittest.TestCase):
    def test_the_tail_is_the_last_thirty_days(self):
        train = _dated_frame(200)
        _, es_rows = xgb.split_early_stopping(train, tail_days=30)
        self.assertEqual(len(es_rows), 30)
        self.assertEqual(es_rows["Tanggal"].max(), train["Tanggal"].max())

    def test_no_es_date_precedes_a_fit_date(self):
        fit_rows, es_rows = xgb.split_early_stopping(_dated_frame(200), tail_days=30)
        self.assertLess(fit_rows["Tanggal"].max(), es_rows["Tanggal"].min())

    def test_the_boundary_is_purged(self):
        """lead_time_days is 3, so the three fit rows nearest the tail carry a
        label built partly out of the early-stopping window."""
        train = _dated_frame(200, lead_time=3.0)
        fit_rows, es_rows = xgb.split_early_stopping(train, tail_days=30)
        es_start = es_rows["Tanggal"].min()
        self.assertTrue(
            (fit_rows["Tanggal"] + pd.Timedelta(days=3) < es_start).all()
        )

    def test_a_longer_lead_time_purges_more(self):
        short = xgb.split_early_stopping(_dated_frame(200, lead_time=1.0))[0]
        long = xgb.split_early_stopping(_dated_frame(200, lead_time=4.0))[0]
        self.assertLess(len(long), len(short))

    def test_the_two_parts_do_not_overlap(self):
        fit_rows, es_rows = xgb.split_early_stopping(_dated_frame(200))
        self.assertEqual(set(fit_rows.index) & set(es_rows.index), set())

    def test_a_training_window_too_short_to_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "terlalu pendek"):
            xgb.split_early_stopping(_dated_frame(20), tail_days=30)

    def test_an_empty_frame_is_refused(self):
        with self.assertRaisesRegex(ValueError, "kosong"):
            xgb.split_early_stopping(_dated_frame(0), tail_days=30)


class TestSearchSpace(unittest.TestCase):
    def test_n_estimators_is_not_searched(self):
        """Early stopping decides the round count; searching it wastes budget."""
        self.assertNotIn("n_estimators", xgb.SEARCH_SPACE)

    def test_encoding_offers_all_three_modes(self):
        self.assertEqual(set(xgb.SEARCH_SPACE["encoding"]),
                         {"ordinal", "native", "one_hot"})

    def test_defaults_cover_every_searched_key(self):
        for key in xgb.SEARCH_SPACE:
            self.assertIn(key, xgb.DEFAULT_PARAMS)


if __name__ == "__main__":
    unittest.main()
