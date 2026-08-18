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


class TestEncode(unittest.TestCase):
    def _pair(self):
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "cat_idx": [0, 1, 2]})
        valid = pd.DataFrame({"feat_a": [4.0, 5.0], "cat_idx": [1, 0]})
        return train, valid

    def test_ordinal_passes_the_index_through(self):
        train, valid = self._pair()
        train_out, valid_out, enable = xgb.encode(train, valid, "ordinal",
                                                  idx_cols=["cat_idx"])
        self.assertFalse(enable)
        self.assertEqual(list(train_out.columns), ["feat_a", "cat_idx"])
        self.assertEqual(list(train_out["cat_idx"]), [0, 1, 2])

    def test_native_makes_the_index_categorical(self):
        train, valid = self._pair()
        train_out, valid_out, enable = xgb.encode(train, valid, "native",
                                                  idx_cols=["cat_idx"])
        self.assertTrue(enable)
        self.assertEqual(str(train_out["cat_idx"].dtype), "category")
        self.assertEqual(str(valid_out["cat_idx"].dtype), "category")

    def test_native_gives_validation_the_training_categories(self):
        train, valid = self._pair()
        train_out, valid_out, _ = xgb.encode(train, valid, "native",
                                             idx_cols=["cat_idx"])
        self.assertEqual(list(train_out["cat_idx"].cat.categories),
                         list(valid_out["cat_idx"].cat.categories))

    def test_native_turns_an_unseen_category_into_a_null(self):
        """XGBoost consumes NaN natively; an unseen level must not become a
        different level's code."""
        train, _ = self._pair()
        valid = pd.DataFrame({"feat_a": [4.0], "cat_idx": [99]})
        _, valid_out, _ = xgb.encode(train, valid, "native", idx_cols=["cat_idx"])
        self.assertTrue(valid_out["cat_idx"].isna().all())

    def test_one_hot_expands_and_drops_the_index(self):
        train, valid = self._pair()
        train_out, valid_out, enable = xgb.encode(train, valid, "one_hot",
                                                  idx_cols=["cat_idx"])
        self.assertFalse(enable)
        self.assertNotIn("cat_idx", train_out.columns)
        self.assertEqual(list(train_out.columns), list(valid_out.columns))

    def test_every_mode_preserves_row_count_and_order(self):
        train, valid = self._pair()
        for encoding in ("ordinal", "native", "one_hot"):
            train_out, valid_out, _ = xgb.encode(train, valid, encoding,
                                                 idx_cols=["cat_idx"])
            self.assertEqual(len(train_out), 3, encoding)
            self.assertEqual(len(valid_out), 2, encoding)
            self.assertEqual(list(valid_out["feat_a"]), [4.0, 5.0], encoding)

    def test_a_validation_only_category_never_shifts_columns(self):
        train, _ = self._pair()
        valid = pd.DataFrame({"feat_a": [4.0, 5.0], "cat_idx": [1, 7]})
        for encoding in ("ordinal", "native", "one_hot"):
            train_out, valid_out, _ = xgb.encode(train, valid, encoding,
                                                 idx_cols=["cat_idx"])
            self.assertEqual(list(train_out.columns), list(valid_out.columns),
                             encoding)

    def test_an_unknown_encoding_is_refused(self):
        train, valid = self._pair()
        with self.assertRaisesRegex(ValueError, "encoding"):
            xgb.encode(train, valid, "embedding", idx_cols=["cat_idx"])


class TestApplyEncoding(unittest.TestCase):
    def _fit_layout(self, encoding):
        train = pd.DataFrame({"feat_a": [1.0, 2.0, 3.0], "cat_idx": [0, 1, 2]})
        train_out, _, _ = xgb.encode(train, train, encoding, idx_cols=["cat_idx"])
        return (list(train_out.columns),
                xgb.training_categories(train, idx_cols=["cat_idx"]))

    def test_it_reproduces_the_training_columns_in_every_mode(self):
        for encoding in ("ordinal", "native", "one_hot"):
            columns, categories = self._fit_layout(encoding)
            frame = pd.DataFrame({"cat_idx": [2, 0], "feat_a": [9.0, 8.0]})
            out, _ = xgb.apply_encoding(frame, encoding, columns, categories,
                                        idx_cols=["cat_idx"])
            self.assertEqual(list(out.columns), columns, encoding)

    def test_a_shuffled_input_column_order_does_not_change_the_output(self):
        columns, categories = self._fit_layout("one_hot")
        frame = pd.DataFrame({"cat_idx": [2, 0], "feat_a": [9.0, 8.0]})
        out, _ = xgb.apply_encoding(frame, "one_hot", columns, categories,
                                    idx_cols=["cat_idx"])
        self.assertEqual(list(out.columns), columns)

    def test_native_restores_the_recorded_categories(self):
        columns, categories = self._fit_layout("native")
        frame = pd.DataFrame({"feat_a": [9.0], "cat_idx": [1]})
        out, enable = xgb.apply_encoding(frame, "native", columns, categories,
                                         idx_cols=["cat_idx"])
        self.assertTrue(enable)
        self.assertEqual(list(out["cat_idx"].cat.categories), categories["cat_idx"])


if __name__ == "__main__":
    unittest.main()
