import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep, walk_forward


def _panel(n_days=245, pairs=(("I1", "B1"), ("I2", "B1")), start="2025-05-01"):
    """Two pairs spanning 2025-05-01..2025-12-31, so every fold and the locked
    December window are represented.
    """
    rows = []
    for item, branch in pairs:
        for i, date in enumerate(pd.date_range(start, periods=n_days, freq="D")):
            rows.append({
                "Kode Barang": item,
                "Nama Cabang": branch,
                "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "target_lead_time_cumulative_capped": float(i % 7),
                "lead_time_days": 3.0,
                "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4),
                "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i),
                "feat_b": float(i % 3),
            })
    return modeling_prep.assign_folds(pd.DataFrame(rows))


FEATURES = ["feat_a", "feat_b"]


class TestEligibleRows(unittest.TestCase):
    def test_drops_the_first_28_days_of_each_pair(self):
        result = walk_forward.eligible_rows(_panel())
        first = result[result["Kode Barang"] == "I1"]["Tanggal"].min()
        self.assertEqual(first, pd.Timestamp("2025-05-29"))

    def test_drops_every_december_row(self):
        result = walk_forward.eligible_rows(_panel())
        self.assertEqual(len(result[result["Tanggal"] >= modeling_prep.TEST_START]), 0)

    def test_drops_rows_with_a_null_target(self):
        panel = _panel()
        blank = panel["Tanggal"] == pd.Timestamp("2025-07-15")
        panel.loc[blank, "target_lead_time_cumulative"] = np.nan
        panel.loc[blank, "target_lead_time_cumulative_capped"] = np.nan
        result = walk_forward.eligible_rows(panel)
        self.assertEqual(len(result[result["Tanggal"] == pd.Timestamp("2025-07-15")]), 0)

    def test_keeps_every_original_column(self):
        panel = _panel()
        result = walk_forward.eligible_rows(panel)
        self.assertEqual(set(panel.columns), set(result.columns))

    def test_matches_to_tabular_row_for_row(self):
        """The contract: the tabular adapter and the runner must agree on the
        row set exactly, or a cross-model comparison compares different data.
        """
        panel = _panel()
        pre_december = panel[panel["Tanggal"] < modeling_prep.TEST_START]
        expected = modeling_prep.to_tabular(pre_december, FEATURES)["keys"]
        result = walk_forward.eligible_rows(panel)
        key_cols = ["Kode Barang", "Nama Cabang", "segment_id", "Tanggal"]
        self.assertEqual(
            set(map(tuple, expected[key_cols].to_numpy())),
            set(map(tuple, result[key_cols].to_numpy())),
        )

    def test_does_not_mutate_the_input_frame(self):
        panel = _panel()
        before = len(panel)
        walk_forward.eligible_rows(panel)
        self.assertEqual(len(panel), before)


class TestPrepareFold(unittest.TestCase):
    def test_validation_is_exactly_the_folds_month(self):
        prepared = walk_forward.prepare_fold(_panel(), 1)
        valid = prepared["valid"]
        self.assertEqual(valid["Tanggal"].min(), pd.Timestamp("2025-07-01"))
        self.assertEqual(valid["Tanggal"].max(), pd.Timestamp("2025-07-31"))

    def test_no_training_row_reaches_the_fold_boundary(self):
        for fold_id in walk_forward.FOLDS:
            prepared = walk_forward.prepare_fold(_panel(), fold_id)
            boundary = modeling_prep.FOLD_STARTS[fold_id - 1]
            self.assertLess(prepared["train"]["Tanggal"].max(), boundary)

    def test_purging_removes_rows_whose_target_window_crosses_the_boundary(self):
        """lead_time_days is 3, so 2025-06-30 sums demand through 2025-07-03 —
        three days of the validation month.
        """
        prepared = walk_forward.prepare_fold(_panel(), 1)
        last = prepared["train"]["Tanggal"].max()
        self.assertLessEqual(last, pd.Timestamp("2025-06-27"))

    def test_no_fold_ever_sees_december(self):
        for fold_id in walk_forward.FOLDS:
            prepared = walk_forward.prepare_fold(_panel(), fold_id)
            for part in ("train", "valid"):
                late = prepared[part]["Tanggal"] >= modeling_prep.TEST_START
                self.assertEqual(int(late.sum()), 0, f"fold {fold_id} {part}")

    def test_training_window_expands_with_the_fold_number(self):
        sizes = [len(walk_forward.prepare_fold(_panel(), f)["train"]) for f in walk_forward.FOLDS]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_train_and_valid_never_share_a_row(self):
        prepared = walk_forward.prepare_fold(_panel(), 3)
        overlap = set(prepared["train"].index) & set(prepared["valid"].index)
        self.assertEqual(overlap, set())

    def test_prepared_flag_skips_the_second_cut(self):
        panel = _panel()
        once = walk_forward.eligible_rows(panel)
        from_raw = walk_forward.prepare_fold(panel, 2)["valid"]
        from_prepared = walk_forward.prepare_fold(once, 2, prepared=True)["valid"]
        pd.testing.assert_frame_equal(from_raw, from_prepared)

    def test_rejects_an_out_of_range_fold(self):
        with self.assertRaises(ValueError):
            walk_forward.prepare_fold(_panel(), 9)


QUANTILES = (0.1, 0.5, 0.9)


def _matrix(values, quantiles=QUANTILES):
    """A point forecast widened into the (n, len(quantiles)) matrix the runner
    now requires. Deliberately identical across quantiles: these fixtures test
    the plumbing, and a flat matrix makes a mis-indexed column visible."""
    return np.repeat(np.asarray(values, dtype=float)[:, None], len(quantiles), axis=1)


def _perfect(train, valid):
    """A model that cheats. Used to assert the plumbing, not the modeling:
    a perfect prediction must score MAE 0, so any non-zero MAE means the
    runner mis-aligned predictions with labels.
    """
    return _matrix(valid["target_lead_time_cumulative"].to_numpy(dtype=float))


def _constant(value):
    def fit_predict(train, valid):
        return _matrix(np.full(len(valid), float(value)))
    return fit_predict


def _spread(train, valid):
    """A crude but ordered quantile model: low, middle and high. Its coverage
    must climb with tau, which a flat fixture cannot show."""
    actual = valid["target_lead_time_cumulative"].to_numpy(dtype=float)
    return np.column_stack([actual - 2.0, actual, actual + 2.0])


def _overall(results, model="rf"):
    return results[(results["model"] == model) & results["group_col"].isna()]


class TestRunFold(unittest.TestCase):
    def test_a_perfect_model_scores_zero_error_at_every_quantile(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        overall = _overall(results)
        self.assertEqual(len(overall), len(QUANTILES))
        self.assertAlmostEqual(float(overall["mae"].max()), 0.0)
        self.assertAlmostEqual(float(overall["pinball"].max()), 0.0)

    def test_one_row_per_quantile_carrying_its_own_tau(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        self.assertEqual(sorted(_overall(results)["quantile"]), list(QUANTILES))

    def test_predictions_are_aligned_row_by_row_not_just_in_count(self):
        """Reversing the prediction vector must change the score. If it does
        not, the runner is comparing sorted or re-indexed values.
        """
        def reversed_model(train, valid):
            return _perfect(train, valid)[::-1]

        straight = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                         quantiles=QUANTILES)
        flipped = walk_forward.run_fold(_panel(), 1, reversed_model,
                                        model_name="rf", quantiles=QUANTILES)
        self.assertAlmostEqual(float(_overall(straight)["mae"].max()), 0.0)
        self.assertGreater(float(_overall(flipped)["mae"].max()), 0.0)

    def test_columns_are_not_transposed(self):
        """The one failure a flat fixture cannot catch: reading the quantile
        columns in the wrong order still yields the right shape."""
        def descending(train, valid):
            return _spread(train, valid)[:, ::-1]

        results = walk_forward.run_fold(_panel(), 1, descending, model_name="rf",
                                        quantiles=QUANTILES)
        overall = _overall(results).set_index("quantile")
        self.assertGreater(overall.loc[0.1, "coverage"],
                           overall.loc[0.9, "coverage"])

    def test_coverage_climbs_with_the_quantile(self):
        results = walk_forward.run_fold(_panel(), 1, _spread, model_name="rf",
                                        quantiles=QUANTILES)
        coverage = _overall(results).sort_values("quantile")["coverage"].tolist()
        self.assertEqual(coverage, sorted(coverage))
        self.assertLess(coverage[0], coverage[-1])

    def test_every_naive_baseline_is_scored_at_every_quantile(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        self.assertEqual(
            set(results["model"].unique()),
            {"rf", "naive_zero", "naive_lag_1", "naive_roll_mean_7"},
        )
        for model in results["model"].unique():
            self.assertEqual(len(_overall(results, model)), len(QUANTILES), model)

    def test_model_and_baselines_are_scored_on_identical_row_counts(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        overall = results[results["group_col"].isna()]
        self.assertEqual(overall["n"].nunique(), 1)

    def test_reports_each_group_column(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        self.assertEqual(
            set(results["group_col"].dropna().unique()),
            set(walk_forward.GROUP_COLS),
        )

    def test_group_row_counts_sum_to_the_overall_count(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        rf = results[(results["model"] == "rf") & (results["quantile"] == 0.9)]
        overall = int(rf[rf["group_col"].isna()].iloc[0]["n"])
        for group_col in walk_forward.GROUP_COLS:
            grouped = rf[rf["group_col"] == group_col]
            self.assertEqual(int(grouped["n"].sum()), overall, group_col)

    def test_carries_every_metric_column(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf",
                                        quantiles=QUANTILES)
        for column in ["n", "quantile", "mae", "pinball", "coverage", "fill_rate",
                       "shortfall_units", "overstock_units", "crossing_rate"]:
            self.assertIn(column, results.columns)

    def test_rejects_a_prediction_of_the_wrong_length(self):
        def short(train, valid):
            return _matrix(np.zeros(len(valid) - 1))

        with self.assertRaisesRegex(ValueError, "bentuk"):
            walk_forward.run_fold(_panel(), 1, short, quantiles=QUANTILES)

    def test_rejects_a_one_dimensional_prediction(self):
        """G7. A point forecast broadcast across 19 quantiles would score
        badly but not visibly wrongly — the shapes still line up."""
        def flat(train, valid):
            return np.zeros(len(valid))

        with self.assertRaisesRegex(ValueError, "bentuk"):
            walk_forward.run_fold(_panel(), 1, flat, quantiles=QUANTILES)

    def test_rejects_a_prediction_with_the_wrong_quantile_count(self):
        def two_columns(train, valid):
            return np.zeros((len(valid), 2))

        with self.assertRaisesRegex(ValueError, "bentuk"):
            walk_forward.run_fold(_panel(), 1, two_columns, quantiles=QUANTILES)


class TestCrossingRate(unittest.TestCase):
    def test_monotone_predictions_report_no_crossing(self):
        results = walk_forward.run_fold(_panel(), 1, _spread, model_name="rf",
                                        quantiles=QUANTILES)
        self.assertEqual(float(_overall(results)["crossing_rate"].max()), 0.0)

    def test_inverted_predictions_are_reported_not_raised(self):
        """A composite pinball head has no monotonicity guarantee, so crossing
        is a measurement to write down — never a crash mid-search."""
        def crossed(train, valid):
            return _spread(train, valid)[:, ::-1]

        results = walk_forward.run_fold(_panel(), 1, crossed, model_name="rf",
                                        quantiles=QUANTILES)
        self.assertEqual(float(_overall(results)["crossing_rate"].max()), 1.0)

    def test_baselines_cannot_cross(self):
        results = walk_forward.run_fold(_panel(), 1, _spread, model_name="rf",
                                        quantiles=QUANTILES)
        baseline = _overall(results, "naive_zero")
        self.assertEqual(float(baseline["crossing_rate"].max()), 0.0)


class TestRunWalkForward(unittest.TestCase):
    def test_covers_every_fold(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf",
                                                quantiles=QUANTILES)
        self.assertEqual(sorted(results["fold_id"].unique()), list(walk_forward.FOLDS))

    def test_a_huge_constant_overshoots_and_a_zero_undershoots(self):
        high = walk_forward.run_walk_forward(_panel(), _constant(1000),
                                             model_name="rf", quantiles=QUANTILES)
        low = walk_forward.run_walk_forward(_panel(), _constant(0),
                                            model_name="rf", quantiles=QUANTILES)
        self.assertAlmostEqual(
            walk_forward.pooled_metric(high, "rf", "coverage", quantile=0.9), 1.0)
        self.assertLess(
            walk_forward.pooled_metric(low, "rf", "coverage", quantile=0.9), 1.0)

    def test_pooled_metric_weights_folds_by_row_count(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf",
                                                quantiles=QUANTILES)
        self.assertAlmostEqual(
            walk_forward.pooled_metric(results, "rf", "pinball", quantile=0.9), 0.0)

    def test_pooled_metric_can_be_restricted_to_the_search_folds(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf",
                                                quantiles=QUANTILES)
        value = walk_forward.pooled_metric(results, "rf", "pinball", folds=(3, 5),
                                           quantile=0.9)
        self.assertAlmostEqual(value, 0.0)

    def test_pooled_metric_isolates_one_quantile(self):
        """Without the filter this would silently average three quantiles'
        coverage into a number that describes none of them."""
        results = walk_forward.run_walk_forward(_panel(), _spread, model_name="rf",
                                                quantiles=QUANTILES)
        low = walk_forward.pooled_metric(results, "rf", "coverage", quantile=0.1)
        high = walk_forward.pooled_metric(results, "rf", "coverage", quantile=0.9)
        self.assertLess(low, high)

    def test_pooled_metric_refuses_a_meaningless_cross_quantile_average(self):
        """Coverage averaged over 0.1 and 0.9 describes neither, and would look
        perfectly reasonable in a results table."""
        results = walk_forward.run_walk_forward(_panel(), _spread, model_name="rf",
                                                quantiles=QUANTILES)
        with self.assertRaisesRegex(ValueError, "lintas kuantil"):
            walk_forward.pooled_metric(results, "rf", "coverage")

    def test_pinball_and_crossing_rate_may_be_averaged_across_the_grid(self):
        results = walk_forward.run_walk_forward(_panel(), _spread, model_name="rf",
                                                quantiles=QUANTILES)
        self.assertFalse(np.isnan(walk_forward.pooled_metric(results, "rf", "pinball")))
        self.assertFalse(
            np.isnan(walk_forward.pooled_metric(results, "rf", "crossing_rate")))

    def test_is_deterministic_for_a_deterministic_model(self):
        first = walk_forward.run_walk_forward(_panel(), _constant(5),
                                              model_name="rf", quantiles=QUANTILES)
        second = walk_forward.run_walk_forward(_panel(), _constant(5),
                                               model_name="rf", quantiles=QUANTILES)
        pd.testing.assert_frame_equal(first, second)


class TestPooledK1(unittest.TestCase):
    def test_k1_is_the_mean_of_the_per_quantile_pooled_pinball(self):
        results = walk_forward.run_walk_forward(_panel(), _spread, model_name="rf",
                                                quantiles=QUANTILES)
        per_quantile = [
            walk_forward.pooled_metric(results, "rf", "pinball", quantile=tau)
            for tau in QUANTILES
        ]
        self.assertAlmostEqual(walk_forward.pooled_k1(results, "rf"),
                               sum(per_quantile) / len(per_quantile))

    def test_a_perfect_model_has_k1_zero(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf",
                                                quantiles=QUANTILES)
        self.assertAlmostEqual(walk_forward.pooled_k1(results, "rf"), 0.0)

    def test_k1_can_be_restricted_to_a_subset_of_folds(self):
        """A model that only misbehaves on fold 5 must look fine when fold 5
        is excluded — otherwise the fold filter is not filtering."""
        def only_bad_on_fold_5(train, valid):
            inflated = 100.0 if int(valid["fold_id"].iloc[0]) == 5 else 0.0
            return _matrix(np.full(len(valid), inflated))

        results = walk_forward.run_walk_forward(_panel(), only_bad_on_fold_5,
                                                model_name="rf", quantiles=QUANTILES)
        self.assertGreater(walk_forward.pooled_k1(results, "rf", folds=(3, 5)),
                           walk_forward.pooled_k1(results, "rf", folds=(3,)))

    def test_the_baselines_get_a_k1_too(self):
        """K1 is only meaningful against a floor measured the same way."""
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf",
                                                quantiles=QUANTILES)
        self.assertGreater(walk_forward.pooled_k1(results, "naive_zero"), 0.0)


class TestCoverageByQuantile(unittest.TestCase):
    def test_one_row_per_quantile_with_its_target(self):
        results = walk_forward.run_walk_forward(_panel(), _spread, model_name="rf",
                                                quantiles=QUANTILES)
        table = walk_forward.coverage_by_quantile(results, "rf")
        self.assertEqual(list(table["quantile"]), list(QUANTILES))
        self.assertEqual(list(table["target"]), list(QUANTILES))

    def test_gap_is_signed_so_the_direction_of_the_miss_survives(self):
        """K2 asks whether a model misses the same way at every point. An
        absolute gap would erase exactly that."""
        results = walk_forward.run_walk_forward(_panel(), _constant(1000),
                                                model_name="rf", quantiles=QUANTILES)
        table = walk_forward.coverage_by_quantile(results, "rf")
        self.assertTrue((table["gap"] > 0).all())


if __name__ == "__main__":
    unittest.main()


class TestTargetSeparation(unittest.TestCase):
    """Latih di `..._capped`, nilai di target mentah (keputusan 2026-08-24).

    Kedua target hidup berdampingan di panel yang sama, jadi satu-satunya yang
    menjaga keduanya tidak tertukar adalah tes ini: sebuah model yang meramal
    target capped dengan sempurna harus tetap mendapat error, karena yang
    dinilai adalah permintaan mentah.
    """

    def _panel_with_both_targets(self):
        panel = _panel()
        panel["target_lead_time_cumulative_capped"] = (
            panel["target_lead_time_cumulative"] * 0.5
        )
        return panel

    def test_scores_against_the_raw_target_not_the_capped_one(self):
        panel = self._panel_with_both_targets()
        frame = walk_forward.eligible_rows(panel)
        valid = frame[frame["fold_id"] == 1]

        def fit_predict(train, valid_rows):
            capped = valid_rows["target_lead_time_cumulative_capped"].to_numpy(float)
            return np.repeat(capped[:, None], 3, axis=1)

        result = walk_forward.run_fold(frame, 1, fit_predict, model_name="m",
                                       quantiles=(0.1, 0.5, 0.9), prepared=True)
        row = result[(result["model"] == "m") & result["group_col"].isna()
                     & np.isclose(result["quantile"], 0.5)]
        expected = float((valid["target_lead_time_cumulative"]
                          - valid["target_lead_time_cumulative_capped"]).abs().mean())
        self.assertAlmostEqual(float(row["mae"].iloc[0]), expected, places=9)
        self.assertGreater(expected, 0.0)

    def test_requires_the_capped_target_column(self):
        panel = self._panel_with_both_targets().drop(
            columns=["target_lead_time_cumulative_capped"])
        with self.assertRaises(KeyError) as ctx:
            walk_forward.eligible_rows(panel)
        self.assertIn("target_lead_time_cumulative_capped", str(ctx.exception))

    def test_refuses_when_the_two_targets_disagree_about_missing_rows(self):
        panel = self._panel_with_both_targets()
        mask = panel["Tanggal"] == pd.Timestamp("2025-07-15")
        panel.loc[mask, "target_lead_time_cumulative_capped"] = np.nan
        with self.assertRaises(ValueError) as ctx:
            walk_forward.eligible_rows(panel)
        self.assertIn("pola nilai kosong", str(ctx.exception))
