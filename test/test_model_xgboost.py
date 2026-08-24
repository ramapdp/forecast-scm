import unittest

import numpy as np
import pandas as pd

from utils.modelling import model_xgboost as xgb
from utils.modelling import evaluation, modeling_prep, purging, walk_forward


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
        "target_lead_time_cumulative_capped": np.abs(feat_a * 10 + 20),
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


class TestMakeFitPredict(unittest.TestCase):
    def _params(self, **overrides):
        return {**xgb.DEFAULT_PARAMS, "max_depth": 3, "learning_rate": 0.3,
                "min_child_weight": 1, "random_state": 0, **overrides}

    def _split(self, n=300):
        frame = _dated_frame(n)
        return frame.iloc[:250], frame.iloc[250:]

    def test_returns_one_column_per_quantile_point(self):
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          max_rounds=40)(train, valid)
        self.assertEqual(prediction.shape,
                         (len(valid), len(evaluation.QUANTILE_SET_A)))

    def test_the_default_grid_is_the_evaluation_grid(self):
        self.assertEqual(tuple(xgb.QUANTILES), evaluation.QUANTILE_SET_A)

    def test_the_service_level_constant_survives_as_a_scalar(self):
        """B-9 ships at one quantile; the grid is how the model is *scored*."""
        self.assertEqual(xgb.QUANTILE, 0.9)

    def test_a_grid_of_any_size_flows_through(self):
        """Nothing downstream may assume nineteen: Tahap B hands out a grid
        whose length comes from the critical-ratio spread."""
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          quantiles=(0.5, 0.8, 0.95),
                                          max_rounds=40)(train, valid)
        self.assertEqual(prediction.shape, (len(valid), 3))

    def test_predictions_are_never_negative(self):
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          max_rounds=40)(train, valid)
        self.assertTrue((prediction >= 0).all())

    def test_the_columns_follow_the_requested_order(self):
        """One fit, two columns, in the order asked for. Compared on the mean
        rather than row by row on purpose: a composite quantile head has no
        structural monotonicity guarantee, and crossing is something this
        project measures (`crossing_rate`) rather than asserts away."""
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          quantiles=(0.1, 0.9),
                                          max_rounds=60)(train, valid)
        self.assertGreater(prediction[:, 1].mean(), prediction[:, 0].mean())

    def test_predictions_are_left_unsorted(self):
        """A post-hoc sort would drive `crossing_rate` to zero by hiding the
        miscalibration it exists to expose. What comes back is the head's own
        output — crossing and all."""
        train, valid = self._split()
        prediction = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                          quantiles=(0.9, 0.1),
                                          max_rounds=60)(train, valid)
        self.assertGreater(prediction[:, 0].mean(), prediction[:, 1].mean())

    def test_every_encoding_runs_end_to_end(self):
        train, valid = self._split()
        for encoding in ("ordinal", "native", "one_hot"):
            prediction = xgb.make_fit_predict(
                self._params(encoding=encoding), feature_cols=FEATURES,
                max_rounds=40, idx_cols=["cat_idx"],
            )(train, valid)
            self.assertEqual(prediction.shape,
                             (len(valid), len(evaluation.QUANTILE_SET_A)),
                             encoding)

    def test_one_hot_really_expands_on_this_frame(self):
        """Guards the test suite itself: without idx_cols the synthetic frame
        has no column matching the real IDX_COLS, so every encoding would
        quietly become a no-op and these tests would prove nothing."""
        train, _ = self._split()
        expanded, _, _ = xgb.encode(train[FEATURES], train[FEATURES], "one_hot",
                                    idx_cols=["cat_idx"])
        self.assertGreater(len(expanded.columns), len(FEATURES))

    def test_log_target_returns_predictions_on_the_original_scale(self):
        train, valid = self._split()
        logged = xgb.make_fit_predict(self._params(log_target=True),
                                      feature_cols=FEATURES, max_rounds=60)(train, valid)
        self.assertGreater(logged.mean(), 5.0)

    def test_the_same_seed_gives_the_same_predictions(self):
        train, valid = self._split()
        first = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                     max_rounds=40)(train, valid)
        second = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                      max_rounds=40)(train, valid)
        np.testing.assert_allclose(first, second)

    def test_a_nan_feature_is_rejected_rather_than_imputed(self):
        train, valid = self._split()
        train = train.copy()
        train.loc[train.index[0], "feat_a"] = np.nan
        with self.assertRaisesRegex(ValueError, "feat_a"):
            xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                 max_rounds=40)(train, valid)

    def test_the_round_count_is_recorded_per_call(self):
        train, valid = self._split()
        fit_predict = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                           max_rounds=40)
        fit_predict(train, valid)
        fit_predict(train, valid)
        self.assertEqual(len(fit_predict.best_iterations), 2)
        self.assertTrue(all(count >= 1 for count in fit_predict.best_iterations))

    def test_the_second_fit_sees_every_training_row(self):
        """The refit is the whole point: XGBoost must end up trained on the
        same population the Random Forest saw, tail included."""
        train, valid = self._split()
        seen = []
        original = xgb.build_estimator

        def spy(params, n_estimators, enable_categorical=False,
                early_stopping_rounds=None, quantiles=xgb.QUANTILES,
                device=xgb.DEFAULT_DEVICE):
            model = original(params, n_estimators,
                             enable_categorical=enable_categorical,
                             early_stopping_rounds=early_stopping_rounds,
                             quantiles=quantiles, device=device)
            real_fit = model.fit

            def fit(X, y, **kwargs):
                seen.append(len(X))
                return real_fit(X, y, **kwargs)

            model.fit = fit
            return model

        xgb.build_estimator = spy
        try:
            xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                 max_rounds=40)(train, valid)
        finally:
            xgb.build_estimator = original

        fit_rows, es_rows = xgb.split_early_stopping(train)
        self.assertEqual(seen[0], len(fit_rows))
        self.assertEqual(seen[1], len(train))

    def test_the_second_fit_uses_the_round_count_the_first_chose(self):
        train, valid = self._split()
        rounds = []
        original = xgb.build_estimator

        def spy(params, n_estimators, enable_categorical=False,
                early_stopping_rounds=None, quantiles=xgb.QUANTILES,
                device=xgb.DEFAULT_DEVICE):
            rounds.append(n_estimators)
            return original(params, n_estimators,
                            enable_categorical=enable_categorical,
                            early_stopping_rounds=early_stopping_rounds,
                            quantiles=quantiles, device=device)

        xgb.build_estimator = spy
        try:
            fit_predict = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                               max_rounds=40)
            fit_predict(train, valid)
        finally:
            xgb.build_estimator = original

        self.assertEqual(rounds[0], 40)
        self.assertEqual(rounds[1], fit_predict.best_iterations[0])


class TestWalkForwardIntegration(unittest.TestCase):
    def _panel(self, periods=245):
        rows = []
        for i, date in enumerate(pd.date_range("2025-05-01", periods=periods, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "target_lead_time_cumulative_capped": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        return modeling_prep.assign_folds(pd.DataFrame(rows))

    def test_it_plugs_into_run_fold_unchanged(self):
        results = walk_forward.run_fold(
            self._panel(), 1,
            xgb.make_fit_predict({**xgb.DEFAULT_PARAMS, "max_depth": 3,
                                  "min_child_weight": 1},
                                 feature_cols=FEATURES, max_rounds=30,
                                 tail_days=14),
            model_name="xgboost",
        )
        self.assertIn("xgboost", set(results["model"]))
        self.assertTrue(results["pinball"].notna().all())

    def test_no_training_row_reaches_december(self):
        frame = self._panel(periods=300)
        seen_max = []
        fit_predict = xgb.make_fit_predict(
            {**xgb.DEFAULT_PARAMS, "max_depth": 3, "min_child_weight": 1},
            feature_cols=FEATURES, max_rounds=20, tail_days=14,
        )

        def spy(train, valid):
            seen_max.append(train["Tanggal"].max())
            return fit_predict(train, valid)

        walk_forward.run_walk_forward(frame, spy, model_name="xgboost")
        for stamp in seen_max:
            self.assertLess(stamp, pd.Timestamp("2025-12-01"))


import tempfile
from pathlib import Path


class TestFitFinal(unittest.TestCase):
    def _params(self, **overrides):
        return {**xgb.DEFAULT_PARAMS, "max_depth": 3, "learning_rate": 0.3,
                "min_child_weight": 1, "random_state": 0, **overrides}

    def _bundle(self, frame=None, **overrides):
        return xgb.fit_final(frame if frame is not None else _dated_frame(400),
                             self._params(**overrides), feature_cols=FEATURES,
                             max_rounds=40, tail_days=14, idx_cols=["cat_idx"])

    def test_bundle_records_what_prediction_needs(self):
        bundle = self._bundle()
        for key in ("model", "params", "feature_cols", "columns", "categories",
                    "idx_cols", "encoding", "log_target", "best_iteration",
                    "quantiles", "n_train"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["feature_cols"], FEATURES)
        self.assertEqual(tuple(bundle["quantiles"]), xgb.QUANTILES)

    def test_training_stops_before_december(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        eligible = frame[frame["Tanggal"] < pd.Timestamp("2025-12-01")]
        self.assertLessEqual(bundle["n_train"], len(eligible))
        self.assertGreater(bundle["n_train"], 0)

    def test_the_december_boundary_is_purged(self):
        """lead_time_days is 3, so 2025-11-29 onward is contaminated."""
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        safe = frame[frame["Tanggal"] <= pd.Timestamp("2025-11-27")]
        self.assertLessEqual(bundle["n_train"], len(safe))

    def test_rows_without_a_target_are_dropped(self):
        frame = _dated_frame(400)
        blank = frame["Tanggal"].between("2025-06-01", "2025-06-05")
        frame.loc[blank, "target_lead_time_cumulative"] = np.nan
        frame.loc[blank, "target_lead_time_cumulative_capped"] = np.nan
        bundle = self._bundle(frame)
        reference = self._bundle(_dated_frame(400))
        self.assertEqual(bundle["n_train"], reference["n_train"] - int(blank.sum()))

    def test_the_warmup_window_is_excluded(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        expected = walk_forward.eligible_rows(frame)
        expected = expected[purging.lookahead_safe_mask(
            expected, pd.Timestamp("2025-12-01"))]
        self.assertEqual(bundle["n_train"], len(expected))

    def test_the_final_model_is_trained_on_every_eligible_row(self):
        """Tail included: the tail chooses the round count, then rejoins."""
        bundle = self._bundle()
        self.assertEqual(bundle["model"].n_estimators, bundle["best_iteration"])

    def test_predict_bundle_returns_one_column_per_quantile_point(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        self.assertEqual(xgb.predict_bundle(bundle, frame).shape,
                         (len(frame), len(evaluation.QUANTILE_SET_A)))

    def test_predict_bundle_is_non_negative(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        self.assertTrue((xgb.predict_bundle(bundle, frame) >= 0).all())

    def test_predict_bundle_ignores_the_input_column_order(self):
        for encoding in ("ordinal", "native", "one_hot"):
            frame = _dated_frame(400)
            bundle = self._bundle(frame, encoding=encoding)
            shuffled = frame[list(reversed(frame.columns))]
            np.testing.assert_allclose(
                xgb.predict_bundle(bundle, frame),
                xgb.predict_bundle(bundle, shuffled),
                err_msg=encoding,
            )

    def test_a_saved_bundle_predicts_identically_after_loading(self):
        frame = _dated_frame(400)
        bundle = self._bundle(frame)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "bundle.joblib")
            xgb.save_bundle(bundle, path)
            reloaded = xgb.load_bundle(path)
        np.testing.assert_allclose(xgb.predict_bundle(bundle, frame),
                                   xgb.predict_bundle(reloaded, frame))

    def test_log_target_bundles_predict_on_the_original_scale(self):
        frame = _dated_frame(400)
        raw = self._bundle(frame, log_target=False)
        logged = self._bundle(frame, log_target=True)
        self.assertLess(
            abs(xgb.predict_bundle(logged, frame).mean()
                - xgb.predict_bundle(raw, frame).mean()),
            20.0,
        )


class TestSearchWrappers(unittest.TestCase):
    def test_the_default_budget_is_thirty_candidates(self):
        self.assertEqual(len(xgb.sample_search_space()), xgb.N_CANDIDATES)
        self.assertEqual(xgb.N_CANDIDATES, 30)

    def test_the_same_seed_reproduces_the_same_list(self):
        self.assertEqual(xgb.sample_search_space(6, seed=1),
                         xgb.sample_search_space(6, seed=1))

    def test_candidates_are_distinct(self):
        keys = sorted(xgb.SEARCH_SPACE)
        drawn = xgb.sample_search_space(20, seed=1)
        signatures = {tuple(c[k] for k in keys) for c in drawn}
        self.assertEqual(len(signatures), 20)

    def test_every_candidate_carries_a_full_parameter_set(self):
        for candidate in xgb.sample_search_space(5, seed=1):
            for key in xgb.DEFAULT_PARAMS:
                self.assertIn(key, candidate)

    def test_only_searched_parameters_vary(self):
        candidates = xgb.sample_search_space(20, seed=1)
        self.assertEqual({c["random_state"] for c in candidates},
                         {xgb.DEFAULT_PARAMS["random_state"]})

    def test_the_search_folds_are_three_and_five(self):
        self.assertEqual(xgb.SEARCH_FOLDS, (3, 5))

    def test_run_search_scores_every_candidate(self):
        # Starts in 2024 on purpose: run_search cannot pass tail_days through to
        # make_fit_predict, so the fold's training window has to be long enough
        # for the real 30-day tail plus the 28-day warm-up cut.
        rows = []
        for i, date in enumerate(pd.date_range("2024-10-01", periods=460, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "target_lead_time_cumulative_capped": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        panel = modeling_prep.assign_folds(pd.DataFrame(rows))
        candidates = [{**xgb.DEFAULT_PARAMS, "max_depth": d, "min_child_weight": 1}
                      for d in (3, 4)]
        results = xgb.run_search(panel, candidates, folds=(1,),
                                 feature_cols=FEATURES, verbose=False)
        self.assertEqual(list(results["candidate_id"]), [0, 1])
        self.assertTrue(results["pinball"].notna().all())
        for key in xgb.SEARCH_SPACE:
            self.assertIn(key, results.columns)


class TestDevice(unittest.TestCase):
    """Where the booster runs is a parameter, not a hardcoded constant.

    The 19-point grid makes XGBoost the most expensive of the three models
    (T-14: `one_output_per_tree` builds 19 trees per boosting round), so the
    search may have to run on a rented GPU. What matters for the methodology
    is that the device is *recorded* — a candidate chosen on one device and
    refitted on another was not selected under the arithmetic that produced it.
    """

    def _params(self, **overrides):
        return {**xgb.DEFAULT_PARAMS, "max_depth": 3, "learning_rate": 0.3,
                "min_child_weight": 1, "random_state": 0, **overrides}

    def test_the_booster_runs_on_cpu_unless_told_otherwise(self):
        estimator = xgb.build_estimator(self._params(), n_estimators=5)
        self.assertEqual(estimator.get_params()["device"], "cpu")

    def test_the_requested_device_reaches_the_estimator(self):
        estimator = xgb.build_estimator(self._params(), n_estimators=5,
                                        device="cuda")
        self.assertEqual(estimator.get_params()["device"], "cuda")

    def test_an_unknown_device_reaches_the_booster_through_fit_predict(self):
        """Forwarding is proved by the library rejecting it, not by a mock."""
        fit_predict = xgb.make_fit_predict(self._params(), feature_cols=FEATURES,
                                           quantiles=(0.5,), tail_days=14,
                                           max_rounds=5, device="bukan-device")
        frame = _dated_frame(200)
        with self.assertRaises(Exception) as caught:
            fit_predict(frame.iloc[:150], frame.iloc[150:])
        self.assertIn("bukan-device", str(caught.exception))

    def test_run_search_forwards_the_device_to_every_candidate(self):
        """run_search records a rejected candidate rather than raising, so a
        device it cannot honour shows up as an error on every row."""
        rows = []
        for i, date in enumerate(pd.date_range("2024-10-01", periods=460, freq="D")):
            rows.append({
                "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "target_lead_time_cumulative_capped": float(i % 7),
                "lead_time_days": 3.0, "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4), "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
            })
        panel = modeling_prep.assign_folds(pd.DataFrame(rows))
        candidates = [{**xgb.DEFAULT_PARAMS, "max_depth": 3, "min_child_weight": 1}]
        results = xgb.run_search(panel, candidates, folds=(1,),
                                 feature_cols=FEATURES, verbose=False,
                                 device="bukan-device")
        self.assertTrue(results["pinball"].isna().all())
        self.assertIn("bukan-device", str(results["error"].iloc[0]))

    def test_the_bundle_records_the_device_it_was_fitted_on(self):
        bundle = xgb.fit_final(_dated_frame(400), self._params(),
                               feature_cols=FEATURES, max_rounds=40,
                               tail_days=14, idx_cols=["cat_idx"])
        self.assertEqual(bundle["device"], "cpu")
        self.assertEqual(bundle["model"].get_params()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()


class TestRunSearchForwarding(unittest.TestCase):
    """Notebook memanggil pembungkus per model, bukan model_common. Kalau
    `only` berhenti di sini, pemecahan shard tidak pernah sampai ke mesinnya."""

    def test_only_and_provenance_reach_model_common(self):
        from unittest import mock
        with mock.patch.object(xgb.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            xgb.run_search(pd.DataFrame(), [{"a": 1}], only=[3, 4],
                           provenance={"device": "cuda:0"})
        self.assertEqual(spy.call_args.kwargs["only"], [3, 4])
        self.assertEqual(spy.call_args.kwargs["provenance"],
                         {"device": "cuda:0"})

    def test_the_defaults_stay_none(self):
        from unittest import mock
        with mock.patch.object(xgb.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            xgb.run_search(pd.DataFrame(), [{"a": 1}])
        self.assertIsNone(spy.call_args.kwargs["only"])
        self.assertIsNone(spy.call_args.kwargs["provenance"])
