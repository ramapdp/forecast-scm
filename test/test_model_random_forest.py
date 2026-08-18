import unittest

import numpy as np
import pandas as pd

from utils import model_random_forest as rf


FEATURES = ["feat_a", "feat_b", "cat_idx"]


def _frame(n=200, seed=0, target_scale=1.0):
    rng = np.random.default_rng(seed)
    feat_a = rng.normal(size=n)
    return pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": rng.normal(size=n),
        "cat_idx": rng.integers(0, 3, size=n),
        "target_lead_time_cumulative": np.abs(feat_a * 10 + 20) * target_scale,
    })


class TestAssertNoNan(unittest.TestCase):
    def test_passes_on_a_clean_frame(self):
        rf.assert_no_nan(_frame(), FEATURES)

    def test_names_the_offending_column(self):
        frame = _frame()
        frame.loc[0, "feat_b"] = np.nan
        with self.assertRaisesRegex(ValueError, "feat_b"):
            rf.assert_no_nan(frame, FEATURES)


class TestEstimateLeafMemoryBytes(unittest.TestCase):
    def test_scales_linearly_with_tree_count(self):
        base = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 50,
                "max_samples_leaf": 20, "max_samples": None}
        doubled = {**base, "n_estimators": 200}
        self.assertEqual(
            rf.estimate_leaf_memory_bytes(doubled, 1_000_000),
            2 * rf.estimate_leaf_memory_bytes(base, 1_000_000),
        )

    def test_a_bigger_min_samples_leaf_costs_less(self):
        small = {"n_estimators": 200, "max_depth": 30, "min_samples_leaf": 20,
                 "max_samples_leaf": 20, "max_samples": None}
        large = {**small, "min_samples_leaf": 200}
        self.assertLess(
            rf.estimate_leaf_memory_bytes(large, 1_000_000),
            rf.estimate_leaf_memory_bytes(small, 1_000_000),
        )

    def test_depth_caps_the_node_count(self):
        """A depth-12 tree cannot exceed 2**13 nodes however many rows it sees."""
        params = {"n_estimators": 1, "max_depth": 12, "min_samples_leaf": 1,
                  "max_samples_leaf": 1, "max_samples": None}
        self.assertLessEqual(
            rf.estimate_leaf_memory_bytes(params, 10_000_000),
            2 ** 13 * 8,
        )

    def test_bootstrap_fraction_reduces_the_estimate(self):
        full = {"n_estimators": 200, "max_depth": 30, "min_samples_leaf": 50,
                "max_samples_leaf": 20, "max_samples": None}
        half = {**full, "max_samples": 0.5}
        self.assertLess(
            rf.estimate_leaf_memory_bytes(half, 1_000_000),
            rf.estimate_leaf_memory_bytes(full, 1_000_000),
        )

    def test_the_configuration_the_spec_rejects_blows_the_budget(self):
        params = {"n_estimators": 200, "max_depth": 40, "min_samples_leaf": 1,
                  "max_samples_leaf": 1, "max_samples": None}
        self.assertGreater(
            rf.estimate_leaf_memory_bytes(params, 1_280_000),
            rf.MEMORY_BUDGET_BYTES,
        )

    def test_the_default_configuration_fits_the_budget(self):
        self.assertLess(
            rf.estimate_leaf_memory_bytes(rf.DEFAULT_PARAMS, 1_280_000),
            rf.MEMORY_BUDGET_BYTES,
        )


class TestExpandOneHot(unittest.TestCase):
    def test_row_count_and_order_are_unchanged(self):
        train, valid = _frame(60, seed=1), _frame(40, seed=2)
        train_X, valid_X = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertEqual(len(train_X), 60)
        self.assertEqual(len(valid_X), 40)
        self.assertTrue((train_X["feat_a"].to_numpy() == train[FEATURES]["feat_a"].to_numpy()).all())

    def test_columns_match_between_train_and_valid(self):
        train, valid = _frame(60, seed=1), _frame(40, seed=2)
        train_X, valid_X = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertEqual(list(train_X.columns), list(valid_X.columns))

    def test_a_category_absent_from_training_does_not_shift_columns(self):
        train = _frame(60, seed=1)
        train["cat_idx"] = 0
        valid = _frame(40, seed=2)
        valid["cat_idx"] = 7
        train_X, valid_X = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertEqual(list(train_X.columns), list(valid_X.columns))
        self.assertEqual(int(valid_X["cat_idx_0"].sum()), 0)

    def test_the_index_column_itself_is_gone(self):
        train, valid = _frame(60, seed=1), _frame(40, seed=2)
        train_X, _ = rf.expand_one_hot(train[FEATURES], valid[FEATURES], ["cat_idx"])
        self.assertNotIn("cat_idx", train_X.columns)


class TestMakeFitPredict(unittest.TestCase):
    def _params(self, **overrides):
        return {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 5,
                "max_samples_leaf": 20, "random_state": 0, **overrides}

    def test_returns_one_prediction_per_validation_row(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        self.assertEqual(predict(train, valid).shape, (80,))

    def test_predictions_are_never_negative(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        self.assertTrue((predict(train, valid) >= 0).all())

    def test_the_high_quantile_sits_above_the_low_one(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        low = rf.make_fit_predict(self._params(), feature_cols=FEATURES, quantile=0.1)
        high = rf.make_fit_predict(self._params(), feature_cols=FEATURES, quantile=0.9)
        self.assertTrue((high(train, valid) >= low(train, valid)).all())

    def test_log_target_returns_predictions_on_the_original_scale(self):
        """Quantiles are equivariant under log1p, so inverting must land back
        in the target's own range — not in log space.
        """
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(log_target=True), feature_cols=FEATURES)
        prediction = predict(train, valid)
        self.assertGreater(prediction.mean(), 5.0)

    def test_one_hot_runs_end_to_end(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(one_hot=True), feature_cols=FEATURES)
        self.assertEqual(predict(train, valid).shape, (80,))

    def test_the_same_seed_gives_the_same_predictions(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        np.testing.assert_array_equal(predict(train, valid), predict(train, valid))

    def test_a_nan_feature_is_rejected_rather_than_imputed(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        train.loc[0, "feat_a"] = np.nan
        predict = rf.make_fit_predict(self._params(), feature_cols=FEATURES)
        with self.assertRaisesRegex(ValueError, "feat_a"):
            predict(train, valid)

    def test_an_over_budget_configuration_is_refused_before_fitting(self):
        train, valid = _frame(300, seed=1), _frame(80, seed=2)
        predict = rf.make_fit_predict(
            self._params(n_estimators=200, max_depth=40, min_samples_leaf=1,
                         max_samples_leaf=50),
            feature_cols=FEATURES,
            memory_budget=1000,
        )
        with self.assertRaisesRegex(MemoryError, "budget"):
            predict(train, valid)


if __name__ == "__main__":
    unittest.main()
