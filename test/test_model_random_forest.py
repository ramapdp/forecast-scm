import unittest

import numpy as np
import pandas as pd

from utils import model_random_forest as rf
from utils import modeling_prep, purging, walk_forward


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


class TestSampleSearchSpace(unittest.TestCase):
    def test_returns_the_requested_number_of_candidates(self):
        self.assertEqual(len(rf.sample_search_space(18, n_train=1_280_000, seed=1)), 18)

    def test_candidates_are_distinct(self):
        candidates = rf.sample_search_space(18, n_train=1_280_000, seed=1)
        signatures = {tuple(sorted(c.items(), key=lambda kv: kv[0])) for c in candidates}
        self.assertEqual(len(signatures), 18)

    def test_the_same_seed_reproduces_the_same_list(self):
        first = rf.sample_search_space(10, n_train=1_280_000, seed=7)
        second = rf.sample_search_space(10, n_train=1_280_000, seed=7)
        self.assertEqual(first, second)

    def test_different_seeds_give_different_lists(self):
        first = rf.sample_search_space(10, n_train=1_280_000, seed=7)
        second = rf.sample_search_space(10, n_train=1_280_000, seed=8)
        self.assertNotEqual(first, second)

    def test_every_candidate_fits_the_memory_budget(self):
        for candidate in rf.sample_search_space(18, n_train=1_280_000, seed=1):
            self.assertLessEqual(
                rf.estimate_leaf_memory_bytes(candidate, 1_280_000),
                rf.MEMORY_BUDGET_BYTES,
                candidate,
            )

    def test_every_candidate_carries_a_full_parameter_set(self):
        for candidate in rf.sample_search_space(5, n_train=1_280_000, seed=1):
            for key in rf.DEFAULT_PARAMS:
                self.assertIn(key, candidate)

    def test_a_tiny_budget_that_admits_nothing_raises(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            rf.sample_search_space(18, n_train=1_280_000, seed=1, memory_budget=10)

    def test_only_searched_parameters_vary(self):
        candidates = rf.sample_search_space(18, n_train=1_280_000, seed=1)
        self.assertEqual({c["n_estimators"] for c in candidates},
                         {rf.DEFAULT_PARAMS["n_estimators"]})
        self.assertEqual({c["random_state"] for c in candidates},
                         {rf.DEFAULT_PARAMS["random_state"]})


class TestSelectBest(unittest.TestCase):
    def test_picks_the_lowest_pinball(self):
        candidates = [{"max_depth": 12}, {"max_depth": 16}, {"max_depth": 20}]
        results = pd.DataFrame({
            "candidate_id": [0, 1, 2],
            "pinball": [5.0, 3.0, 4.0],
        })
        self.assertEqual(rf.select_best(results, candidates), {"max_depth": 16})

    def test_ignores_a_candidate_that_failed(self):
        candidates = [{"max_depth": 12}, {"max_depth": 16}]
        results = pd.DataFrame({
            "candidate_id": [0, 1],
            "pinball": [np.nan, 4.0],
        })
        self.assertEqual(rf.select_best(results, candidates), {"max_depth": 16})

    def test_raises_when_every_candidate_failed(self):
        candidates = [{"max_depth": 12}]
        results = pd.DataFrame({"candidate_id": [0], "pinball": [np.nan]})
        with self.assertRaisesRegex(ValueError, "tidak ada kandidat"):
            rf.select_best(results, candidates)


import tempfile
from pathlib import Path


def _dated_frame(n=400, seed=3):
    """One pair's series, long enough that the 28-day warm-up cut leaves rows."""
    frame = _frame(n, seed=seed)
    frame["Tanggal"] = pd.date_range("2025-01-01", periods=n, freq="D")
    frame["lead_time_days"] = 3.0
    frame["Kode Barang"] = "FGS-00001"
    frame["Nama Cabang"] = "KY001"
    return frame


class TestFitFinal(unittest.TestCase):
    def _params(self):
        return {"n_estimators": 20, "max_depth": 6, "min_samples_leaf": 5,
                "max_samples_leaf": 20, "random_state": 0}

    def test_bundle_records_what_prediction_needs(self):
        bundle = rf.fit_final(_dated_frame(), self._params(),
                              feature_cols=FEATURES, n_estimators=20)
        for key in ("model", "params", "feature_cols", "columns", "quantile", "n_train"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["feature_cols"], FEATURES)

    def test_training_stops_before_december(self):
        frame = _dated_frame(n=400)
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        eligible = frame[frame["Tanggal"] < pd.Timestamp("2025-12-01")]
        self.assertLessEqual(bundle["n_train"], len(eligible))
        self.assertGreater(bundle["n_train"], 0)

    def test_the_boundary_is_purged(self):
        """lead_time_days is 3, so 2025-11-29 onward is contaminated."""
        frame = _dated_frame(n=400)
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        safe = frame[frame["Tanggal"] <= pd.Timestamp("2025-11-27")]
        self.assertLessEqual(bundle["n_train"], len(safe))

    def test_rows_without_a_target_are_dropped(self):
        """The last days of a segment have no label; they cannot be trained on."""
        frame = _dated_frame(n=400)
        blank = frame["Tanggal"].between("2025-06-01", "2025-06-05")
        frame.loc[blank, "target_lead_time_cumulative"] = np.nan
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES,
                              n_estimators=20)
        clean = _dated_frame(n=400)
        reference = rf.fit_final(clean, self._params(), feature_cols=FEATURES,
                                 n_estimators=20)
        self.assertEqual(bundle["n_train"], reference["n_train"] - int(blank.sum()))

    def test_the_warmup_window_is_excluded(self):
        """Training rows are the eligible rows, not every row before December."""
        frame = _dated_frame(n=400)
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES,
                              n_estimators=20)
        expected = walk_forward.eligible_rows(frame)
        expected = expected[purging.lookahead_safe_mask(
            expected, modeling_prep.TEST_START)]
        self.assertEqual(bundle["n_train"], len(expected))

    def test_final_tree_count_overrides_the_searched_one(self):
        bundle = rf.fit_final(_dated_frame(), self._params(),
                              feature_cols=FEATURES, n_estimators=33)
        self.assertEqual(bundle["params"]["n_estimators"], 33)

    def test_predict_bundle_returns_one_value_per_row(self):
        frame = _dated_frame()
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        self.assertEqual(rf.predict_bundle(bundle, frame.head(25)).shape, (25,))

    def test_predict_bundle_is_non_negative(self):
        frame = _dated_frame()
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        self.assertTrue((rf.predict_bundle(bundle, frame.head(25)) >= 0).all())

    def test_one_hot_bundle_predicts_when_a_column_order_differs(self):
        frame = _dated_frame()
        params = {**self._params(), "one_hot": True}
        bundle = rf.fit_final(frame, params, feature_cols=FEATURES, n_estimators=20)
        shuffled = frame.head(25)[list(reversed(FEATURES)) + ["Tanggal", "lead_time_days"]]
        self.assertEqual(rf.predict_bundle(bundle, shuffled).shape, (25,))

    def test_a_saved_bundle_predicts_identically_after_loading(self):
        frame = _dated_frame()
        bundle = rf.fit_final(frame, self._params(), feature_cols=FEATURES, n_estimators=20)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "bundle.joblib")
            rf.save_bundle(bundle, path)
            reloaded = rf.load_bundle(path)
        np.testing.assert_array_equal(
            rf.predict_bundle(bundle, frame.head(25)),
            rf.predict_bundle(reloaded, frame.head(25)),
        )


class TestRunSearchCheckpoint(unittest.TestCase):
    """A six-hour search must not lose fourteen finished candidates because the
    fifteenth was killed by the OS."""

    def _panel(self):
        rows = []
        for i, date in enumerate(pd.date_range("2025-05-01", periods=245, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        from utils import modeling_prep
        return modeling_prep.assign_folds(pd.DataFrame(rows))

    def _candidates(self):
        return [
            {**rf.DEFAULT_PARAMS, "n_estimators": 5, "max_depth": 4,
             "min_samples_leaf": 5, "max_samples_leaf": 5, "max_depth_label": d}
            for d in (1, 2)
        ]

    def test_writes_a_row_per_candidate_as_it_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            rf.run_search(self._panel(), self._candidates(), folds=(1,),
                          feature_cols=FEATURES, verbose=False,
                          checkpoint_path=path)
            self.assertEqual(len(pd.read_csv(path)), 2)

    def test_checkpoint_matches_the_returned_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            returned = rf.run_search(self._panel(), self._candidates(), folds=(1,),
                                     feature_cols=FEATURES, verbose=False,
                                     checkpoint_path=path)
            saved = pd.read_csv(path)
            self.assertEqual(len(saved), len(returned))
            self.assertEqual(list(saved["candidate_id"]), list(returned["candidate_id"]))

    def test_runs_without_a_checkpoint_path(self):
        returned = rf.run_search(self._panel(), self._candidates(), folds=(1,),
                                 feature_cols=FEATURES, verbose=False)
        self.assertEqual(len(returned), 2)


class TestRunSearchResume(unittest.TestCase):
    """What just happened in practice: a six-hour search died at candidate 12
    and the checkpoint held. Restarting from zero would have thrown away the
    twelve finished fits, so a resume has to be the default, not an option."""

    def _panel(self):
        rows = []
        for i, date in enumerate(pd.date_range("2025-05-01", periods=245, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        from utils import modeling_prep
        return modeling_prep.assign_folds(pd.DataFrame(rows))

    def _candidates(self, n=3):
        depths = [4, 5, 6]
        return [
            {**rf.DEFAULT_PARAMS, "n_estimators": 5, "max_depth": depths[i],
             "min_samples_leaf": 5, "max_samples_leaf": 5}
            for i in range(n)
        ]

    def _run(self, path, candidates, **kwargs):
        return rf.run_search(self._panel(), candidates, folds=(1,),
                             feature_cols=FEATURES, verbose=False,
                             checkpoint_path=path, **kwargs)

    def test_a_finished_candidate_is_not_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            candidates = self._candidates()
            self._run(path, candidates)

            # Forge an unmistakable value; a resume must return it untouched.
            saved = pd.read_csv(path)
            saved.loc[saved["candidate_id"] == 0, "pinball"] = -999.0
            saved = saved[saved["candidate_id"] < 2]
            saved.to_csv(path, index=False)

            resumed = self._run(path, candidates)
            self.assertEqual(len(resumed), 3)
            row = resumed[resumed["candidate_id"] == 0].iloc[0]
            self.assertEqual(float(row["pinball"]), -999.0)

    def test_resume_false_recomputes_everything(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            candidates = self._candidates()
            self._run(path, candidates)
            saved = pd.read_csv(path)
            saved.loc[saved["candidate_id"] == 0, "pinball"] = -999.0
            saved.to_csv(path, index=False)

            fresh = self._run(path, candidates, resume=False)
            row = fresh[fresh["candidate_id"] == 0].iloc[0]
            self.assertNotEqual(float(row["pinball"]), -999.0)

    def test_results_come_back_in_candidate_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            candidates = self._candidates()
            self._run(path, candidates)
            saved = pd.read_csv(path)
            saved[saved["candidate_id"] == 1].to_csv(path, index=False)

            resumed = self._run(path, candidates)
            self.assertEqual(list(resumed["candidate_id"]), [0, 1, 2])

    def test_a_checkpoint_from_a_different_search_space_is_refused(self):
        """Silently mixing candidates from two different spaces would produce a
        winner that never existed."""
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path, self._candidates())
            different = self._candidates()
            different[0]["max_depth"] = 11
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                self._run(path, different)

    def test_a_missing_checkpoint_simply_runs_everything(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "absent.csv")
            result = self._run(path, self._candidates())
            self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
