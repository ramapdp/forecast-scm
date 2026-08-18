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
        panel.loc[panel["Tanggal"] == pd.Timestamp("2025-07-15"), "target_lead_time_cumulative"] = np.nan
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


def _perfect(train, valid):
    """A model that cheats. Used to assert the plumbing, not the modeling:
    a perfect prediction must score MAE 0, so any non-zero MAE means the
    runner mis-aligned predictions with labels.
    """
    return valid["target_lead_time_cumulative"].to_numpy(dtype=float)


def _constant(value):
    def fit_predict(train, valid):
        return np.full(len(valid), float(value))
    return fit_predict


class TestRunFold(unittest.TestCase):
    def test_a_perfect_model_scores_zero_error(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        overall = results[(results["model"] == "rf") & results["group_col"].isna()]
        self.assertEqual(len(overall), 1)
        self.assertAlmostEqual(float(overall.iloc[0]["mae"]), 0.0)
        self.assertAlmostEqual(float(overall.iloc[0]["pinball"]), 0.0)

    def test_predictions_are_aligned_row_by_row_not_just_in_count(self):
        """Reversing the prediction vector must change the score. If it does
        not, the runner is comparing sorted or re-indexed values.
        """
        def reversed_model(train, valid):
            return _perfect(train, valid)[::-1]

        straight = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        flipped = walk_forward.run_fold(_panel(), 1, reversed_model, model_name="rf")
        straight_mae = float(straight[straight["group_col"].isna() & (straight["model"] == "rf")].iloc[0]["mae"])
        flipped_mae = float(flipped[flipped["group_col"].isna() & (flipped["model"] == "rf")].iloc[0]["mae"])
        self.assertAlmostEqual(straight_mae, 0.0)
        self.assertGreater(flipped_mae, 0.0)

    def test_every_naive_baseline_is_scored_too(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        self.assertEqual(
            set(results["model"].unique()),
            {"rf", "naive_zero", "naive_lag_1", "naive_roll_mean_7"},
        )

    def test_model_and_baselines_are_scored_on_identical_row_counts(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        overall = results[results["group_col"].isna()]
        self.assertEqual(overall["n"].nunique(), 1)

    def test_reports_each_group_column(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        self.assertEqual(
            set(results["group_col"].dropna().unique()),
            set(walk_forward.GROUP_COLS),
        )

    def test_group_row_counts_sum_to_the_overall_count(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        rf = results[results["model"] == "rf"]
        overall = int(rf[rf["group_col"].isna()].iloc[0]["n"])
        for group_col in walk_forward.GROUP_COLS:
            grouped = rf[rf["group_col"] == group_col]
            self.assertEqual(int(grouped["n"].sum()), overall, group_col)

    def test_carries_every_metric_column(self):
        results = walk_forward.run_fold(_panel(), 1, _perfect, model_name="rf")
        for column in ["n", "mae", "pinball", "coverage", "fill_rate",
                       "shortfall_units", "overstock_units"]:
            self.assertIn(column, results.columns)

    def test_rejects_a_prediction_of_the_wrong_length(self):
        def short(train, valid):
            return np.zeros(len(valid) - 1)

        with self.assertRaisesRegex(ValueError, "panjang"):
            walk_forward.run_fold(_panel(), 1, short)


class TestRunWalkForward(unittest.TestCase):
    def test_covers_every_fold(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf")
        self.assertEqual(sorted(results["fold_id"].unique()), list(walk_forward.FOLDS))

    def test_a_huge_constant_overshoots_and_a_zero_undershoots(self):
        high = walk_forward.run_walk_forward(_panel(), _constant(1000), model_name="rf")
        low = walk_forward.run_walk_forward(_panel(), _constant(0), model_name="rf")
        self.assertAlmostEqual(walk_forward.pooled_metric(high, "rf", "coverage"), 1.0)
        self.assertLess(walk_forward.pooled_metric(low, "rf", "coverage"), 1.0)

    def test_pooled_metric_weights_folds_by_row_count(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf")
        self.assertAlmostEqual(walk_forward.pooled_metric(results, "rf", "pinball"), 0.0)

    def test_pooled_metric_can_be_restricted_to_the_search_folds(self):
        results = walk_forward.run_walk_forward(_panel(), _perfect, model_name="rf")
        value = walk_forward.pooled_metric(results, "rf", "pinball", folds=(3, 5))
        self.assertAlmostEqual(value, 0.0)

    def test_is_deterministic_for_a_deterministic_model(self):
        first = walk_forward.run_walk_forward(_panel(), _constant(5), model_name="rf")
        second = walk_forward.run_walk_forward(_panel(), _constant(5), model_name="rf")
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
