import unittest

import numpy as np
import pandas as pd

from utils import evaluation


class TestPinballLoss(unittest.TestCase):
    def test_alpha_half_is_half_the_absolute_error(self):
        y = pd.Series([10.0, 20.0])
        pred = pd.Series([12.0, 16.0])
        self.assertAlmostEqual(
            evaluation.pinball_loss(y, pred, alpha=0.5),
            evaluation.mae(y, pred) / 2,
        )

    def test_under_forecast_is_penalised_by_alpha(self):
        """Predicting 90 when demand is 100 is a stockout; at alpha 0.9 the
        shortfall costs 0.9 per unit."""
        y = pd.Series([100.0])
        self.assertAlmostEqual(
            evaluation.pinball_loss(y, pd.Series([90.0]), alpha=0.9), 9.0
        )

    def test_over_forecast_is_penalised_by_one_minus_alpha(self):
        y = pd.Series([100.0])
        self.assertAlmostEqual(
            evaluation.pinball_loss(y, pd.Series([110.0]), alpha=0.9), 1.0
        )

    def test_a_stockout_costs_more_than_the_same_overstock(self):
        y = pd.Series([100.0])
        under = evaluation.pinball_loss(y, pd.Series([90.0]), alpha=0.9)
        over = evaluation.pinball_loss(y, pd.Series([110.0]), alpha=0.9)
        self.assertGreater(under, over)

    def test_perfect_forecast_costs_nothing(self):
        y = pd.Series([5.0, 0.0, 91.0])
        self.assertEqual(evaluation.pinball_loss(y, y.copy(), alpha=0.9), 0.0)

    def test_null_targets_are_excluded_not_counted_as_zero(self):
        y = pd.Series([100.0, np.nan])
        pred = pd.Series([90.0, 0.0])
        self.assertAlmostEqual(evaluation.pinball_loss(y, pred, alpha=0.9), 9.0)


class TestQuantileCoverage(unittest.TestCase):
    """The honest test for a quantile model: train at 0.9 and roughly 90% of
    actuals should land at or below the forecast."""

    def test_coverage_counts_actuals_at_or_below_the_forecast(self):
        y = pd.Series([1.0, 2.0, 3.0, 100.0])
        pred = pd.Series([5.0, 5.0, 5.0, 5.0])
        self.assertAlmostEqual(evaluation.quantile_coverage(y, pred), 0.75)

    def test_full_coverage_when_every_forecast_is_high_enough(self):
        y = pd.Series([1.0, 2.0])
        self.assertEqual(evaluation.quantile_coverage(y, pd.Series([9.0, 9.0])), 1.0)

    def test_null_targets_are_excluded(self):
        y = pd.Series([1.0, np.nan])
        self.assertEqual(evaluation.quantile_coverage(y, pd.Series([9.0, 0.0])), 1.0)


def _frame():
    return pd.DataFrame({
        "target_lead_time_cumulative": [10.0, 20.0, np.nan, 8.0],
        "target_lead_time_cumulative_capped": [10.0, 20.0, np.nan, 8.0],
        "lag_1": [3.0, 5.0, 1.0, 2.0],
        "roll_mean_7": [4.0, 6.0, 1.0, 2.5],
        "lead_time_days": [2, 3, 1, 4],
    })


class TestNaiveBaselines(unittest.TestCase):
    def test_zero_baseline_predicts_nothing(self):
        preds = evaluation.naive_predictions(_frame())
        self.assertTrue((preds["naive_zero"] == 0).all())

    def test_lag_baseline_scales_yesterday_by_the_lead_time(self):
        preds = evaluation.naive_predictions(_frame())
        self.assertEqual(list(preds["naive_lag_1"]), [6.0, 15.0, 1.0, 8.0])

    def test_rolling_baseline_scales_the_weekly_mean_by_the_lead_time(self):
        preds = evaluation.naive_predictions(_frame())
        self.assertEqual(list(preds["naive_roll_mean_7"]), [8.0, 18.0, 1.0, 10.0])

    def test_negative_forecasts_are_clipped_to_zero(self):
        """Demand cannot be negative, and a negative forecast would flatter
        the pinball score by over-penalising nothing."""
        df = _frame()
        df["roll_mean_7"] = -5.0
        preds = evaluation.naive_predictions(df)
        self.assertTrue((preds["naive_roll_mean_7"] >= 0).all())

    def test_null_features_become_zero_forecasts(self):
        df = _frame()
        df.loc[0, "roll_mean_7"] = np.nan
        preds = evaluation.naive_predictions(df)
        self.assertEqual(preds["naive_roll_mean_7"].iloc[0], 0.0)

    def test_all_three_baselines_are_produced(self):
        self.assertEqual(
            sorted(evaluation.naive_predictions(_frame())),
            ["naive_lag_1", "naive_roll_mean_7", "naive_zero"],
        )


class TestFillRate(unittest.TestCase):
    """The data owner's success criterion is 'the outlet does not run out',
    which is a fill rate, not an error magnitude."""

    def test_fill_rate_is_one_when_every_forecast_covers_demand(self):
        y = pd.Series([10.0, 20.0])
        self.assertEqual(evaluation.fill_rate(y, pd.Series([10.0, 25.0])), 1.0)

    def test_fill_rate_is_the_share_of_demand_actually_met(self):
        y = pd.Series([10.0, 10.0])
        self.assertAlmostEqual(evaluation.fill_rate(y, pd.Series([10.0, 5.0])), 0.75)

    def test_overstock_does_not_compensate_for_a_shortfall(self):
        # Surplus at one outlet-day cannot be shipped back in time to cover a
        # stockout at another, so fill rate must not net the two off.
        y = pd.Series([10.0, 10.0])
        self.assertAlmostEqual(evaluation.fill_rate(y, pd.Series([100.0, 5.0])), 0.75)

    def test_fill_rate_excludes_null_targets(self):
        y = pd.Series([10.0, np.nan])
        self.assertEqual(evaluation.fill_rate(y, pd.Series([10.0, 0.0])), 1.0)

    def test_fill_rate_of_a_window_with_no_demand_is_one(self):
        # Nothing was demanded, so nothing went unserved -- a division by zero
        # here would poison every per-segment table with NaN.
        y = pd.Series([0.0, 0.0])
        self.assertEqual(evaluation.fill_rate(y, pd.Series([0.0, 0.0])), 1.0)


class TestShortfallAndOverstockUnits(unittest.TestCase):
    def test_shortfall_counts_only_under_forecast_units(self):
        y = pd.Series([10.0, 10.0])
        self.assertEqual(evaluation.shortfall_units(y, pd.Series([4.0, 50.0])), 6.0)

    def test_overstock_counts_only_over_forecast_units(self):
        y = pd.Series([10.0, 10.0])
        self.assertEqual(evaluation.overstock_units(y, pd.Series([4.0, 50.0])), 40.0)

    def test_a_perfect_forecast_has_neither(self):
        y = pd.Series([10.0, 10.0])
        self.assertEqual(evaluation.shortfall_units(y, y), 0.0)
        self.assertEqual(evaluation.overstock_units(y, y), 0.0)

    def test_null_targets_are_excluded(self):
        y = pd.Series([10.0, np.nan])
        self.assertEqual(evaluation.shortfall_units(y, pd.Series([0.0, 0.0])), 10.0)


class TestQuantileSet(unittest.TestCase):
    """Tahap A grid: 19 evenly spaced points, which is what makes the mean
    pinball approach CRPS rather than sample one service level."""

    def test_stage_a_has_nineteen_points_from_5_to_95_percent(self):
        self.assertEqual(len(evaluation.QUANTILE_SET_A), 19)
        self.assertAlmostEqual(evaluation.QUANTILE_SET_A[0], 0.05)
        self.assertAlmostEqual(evaluation.QUANTILE_SET_A[-1], 0.95)

    def test_the_business_service_level_is_one_of_the_points(self):
        """B-9 commits to 0.9. A grid that misses it would report K1 without
        ever scoring the quantile the business actually ships at."""
        self.assertIn(evaluation.DEFAULT_ALPHA, evaluation.QUANTILE_SET_A)

    def test_points_are_strictly_increasing(self):
        points = list(evaluation.QUANTILE_SET_A)
        self.assertEqual(points, sorted(set(points)))


class TestCostCoverageShare(unittest.TestCase):
    def _table(self):
        return pd.DataFrame({
            "Kode Barang": ["A", "B", "C"],
            "cost_confidence": ["tinggi", "rendah", None],
        })

    def test_share_is_volume_weighted_not_sku_counted(self):
        """B-10 closes on 80% of *volume*, not 80% of SKUs — one high-volume
        SKU can carry the threshold that a dozen rare ones cannot."""
        volume = pd.Series({"A": 90.0, "B": 5.0, "C": 5.0})
        self.assertAlmostEqual(
            evaluation.cost_coverage_share(self._table(), volume), 0.9
        )

    def test_blank_confidence_is_not_precise(self):
        volume = pd.Series({"A": 0.0, "B": 0.0, "C": 10.0})
        self.assertEqual(evaluation.cost_coverage_share(self._table(), volume), 0.0)

    def test_sku_absent_from_the_cost_table_counts_as_imprecise(self):
        volume = pd.Series({"A": 10.0, "D": 90.0})
        self.assertAlmostEqual(
            evaluation.cost_coverage_share(self._table(), volume), 0.1
        )


class TestResolveQuantileSet(unittest.TestCase):
    def _table(self, confidence):
        return pd.DataFrame({"Kode Barang": ["A"], "cost_confidence": [confidence]})

    def test_below_the_threshold_is_stage_a(self):
        result = evaluation.resolve_quantile_set(
            cost_table=self._table("rendah"), volume_by_sku=pd.Series({"A": 1.0})
        )
        self.assertEqual(result, evaluation.QUANTILE_SET_A)

    def test_at_the_threshold_without_critical_ratios_refuses_to_guess(self):
        """Falling back to Tahap A once B-10 has closed would silently keep
        evaluating on a grid the design says is superseded."""
        with self.assertRaises(NotImplementedError):
            evaluation.resolve_quantile_set(
                cost_table=self._table("tinggi"),
                volume_by_sku=pd.Series({"A": 1.0}),
            )

    def test_at_the_threshold_with_critical_ratios_is_stage_b(self):
        ratios = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        result = evaluation.resolve_quantile_set(
            cost_table=self._table("tinggi"),
            volume_by_sku=pd.Series({"A": 1.0}),
            critical_ratios=ratios,
        )
        self.assertEqual(len(result), len(evaluation.QUANTILE_SET_B_PERCENTILES))
        self.assertEqual(list(result), sorted(result))


class TestQuantileSetB(unittest.TestCase):
    def test_percentiles_of_the_critical_ratio_spread(self):
        ratios = pd.Series(np.linspace(0.0, 1.0, 101))
        result = evaluation.quantile_set_b(ratios)
        self.assertEqual([round(value, 2) for value in result],
                         [0.10, 0.25, 0.50, 0.75, 0.90])

    def test_duplicate_percentiles_collapse(self):
        """A cost table where every segment lands on the same critical ratio
        must not produce a grid that scores the same point five times."""
        result = evaluation.quantile_set_b(pd.Series([0.9] * 20))
        self.assertEqual(len(result), 1)


class TestAsQuantileFrame(unittest.TestCase):
    QUANTILES = (0.1, 0.5, 0.9)

    def test_a_matrix_keeps_its_columns_in_quantile_order(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        frame = evaluation.as_quantile_frame(matrix, self.QUANTILES)
        self.assertEqual(list(frame.columns), list(self.QUANTILES))
        self.assertEqual(frame.iloc[1, 2], 6.0)

    def test_a_point_forecast_is_broadcast_across_every_quantile(self):
        """The naive baselines produce one number per row. Scoring them at
        every tau is what keeps the floor comparable to K1."""
        frame = evaluation.as_quantile_frame(pd.Series([7.0, 8.0]), self.QUANTILES)
        self.assertEqual(frame.shape, (2, 3))
        self.assertTrue((frame[0.1] == frame[0.9]).all())

    def test_wrong_column_count_is_refused(self):
        with self.assertRaises(ValueError):
            evaluation.as_quantile_frame(np.zeros((2, 2)), self.QUANTILES)


class TestScoreQuantiles(unittest.TestCase):
    QUANTILES = (0.1, 0.5, 0.9)

    def _scored(self):
        y = pd.Series([10.0, 20.0])
        predictions = np.array([[5.0, 9.0, 12.0], [8.0, 18.0, 25.0]])
        return evaluation.score_quantiles(y, predictions, self.QUANTILES), y, predictions

    def test_one_row_per_quantile(self):
        scored, _, _ = self._scored()
        self.assertEqual(list(scored["quantile"]), list(self.QUANTILES))

    def test_each_row_is_scored_at_its_own_tau(self):
        scored, y, predictions = self._scored()
        for position, tau in enumerate(self.QUANTILES):
            expected = evaluation.pinball_loss(
                y, pd.Series(predictions[:, position]), alpha=tau
            )
            self.assertAlmostEqual(
                scored.loc[scored["quantile"] == tau, "pinball"].iloc[0], expected
            )

    def test_coverage_is_reported_per_quantile(self):
        scored, _, _ = self._scored()
        self.assertEqual(scored.loc[scored["quantile"] == 0.1, "coverage"].iloc[0], 0.0)
        self.assertEqual(scored.loc[scored["quantile"] == 0.9, "coverage"].iloc[0], 1.0)


class TestK1Score(unittest.TestCase):
    def test_k1_is_the_unweighted_mean_of_the_per_quantile_pinball(self):
        scored = pd.DataFrame({"quantile": [0.1, 0.5, 0.9],
                               "pinball": [1.0, 2.0, 6.0],
                               "n": [10, 10, 10]})
        self.assertAlmostEqual(evaluation.k1_score(scored), 3.0)

    def test_row_counts_do_not_reweight_the_quantile_average(self):
        """Every quantile is scored on the identical rows, so an n-weighted
        average would only reintroduce rounding noise. Unweighted is the
        design (Bagian 2 of the multi-quantile spec)."""
        scored = pd.DataFrame({"quantile": [0.1, 0.9],
                               "pinball": [1.0, 3.0],
                               "n": [1, 1_000_000]})
        self.assertAlmostEqual(evaluation.k1_score(scored), 2.0)


class TestCrossingRate(unittest.TestCase):
    QUANTILES = (0.1, 0.5, 0.9)

    def test_monotone_predictions_do_not_cross(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0]])
        self.assertEqual(evaluation.crossing_rate(matrix, self.QUANTILES), 0.0)

    def test_rate_is_the_share_of_rows_with_at_least_one_inversion(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 3.0, 5.0]])
        self.assertEqual(evaluation.crossing_rate(matrix, self.QUANTILES), 0.5)

    def test_one_row_with_two_inversions_still_counts_once(self):
        matrix = np.array([[3.0, 2.0, 1.0]])
        self.assertEqual(evaluation.crossing_rate(matrix, self.QUANTILES), 1.0)


class TestEvaluateBaselines(unittest.TestCase):
    def test_returns_one_row_per_baseline_per_quantile(self):
        result = evaluation.evaluate_baselines(_frame(), quantiles=(0.5, 0.9))
        self.assertEqual(len(result), 6)
        self.assertEqual(
            sorted(result["baseline"].unique()),
            ["naive_lag_1", "naive_roll_mean_7", "naive_zero"],
        )

    def test_the_quantile_is_carried_through(self):
        result = evaluation.evaluate_baselines(_frame(), quantiles=(0.5, 0.9))
        self.assertEqual(sorted(result["quantile"].unique()), [0.5, 0.9])

    def test_reports_mae_pinball_and_coverage(self):
        result = evaluation.evaluate_baselines(_frame(), quantiles=(0.9,))
        for col in ["mae", "pinball", "coverage", "n"]:
            self.assertIn(col, result.columns)

    def test_reports_the_service_level_metrics(self):
        result = evaluation.evaluate_baselines(_frame(), quantiles=(0.9,))
        for col in ["fill_rate", "shortfall_units", "overstock_units"]:
            self.assertIn(col, result.columns)

    def test_zero_baseline_leaves_every_unit_unserved(self):
        result = evaluation.evaluate_baselines(
            _frame(), quantiles=(0.9,)).set_index("baseline")
        self.assertEqual(result.loc["naive_zero", "fill_rate"], 0.0)
        self.assertEqual(result.loc["naive_zero", "shortfall_units"], 10 + 20 + 8)
        self.assertEqual(result.loc["naive_zero", "overstock_units"], 0.0)

    def test_row_count_excludes_null_targets(self):
        result = evaluation.evaluate_baselines(_frame(), quantiles=(0.9,))
        self.assertTrue((result["n"] == 3).all())

    def test_zero_baseline_mae_equals_mean_demand(self):
        result = evaluation.evaluate_baselines(
            _frame(), quantiles=(0.9,)).set_index("baseline")
        self.assertAlmostEqual(result.loc["naive_zero", "mae"], (10 + 20 + 8) / 3)


class TestEvaluateByGroup(unittest.TestCase):
    """Metrics must be reportable per demand segment: a global MAE is
    dominated by the smooth pairs, where predicting anything is easy."""

    def _grouped(self):
        df = _frame()
        df["demand_segment"] = ["smooth", "smooth", "lumpy", "lumpy"]
        return df

    def test_one_row_per_group_per_baseline(self):
        result = evaluation.evaluate_baselines(
            self._grouped(), group_col="demand_segment", quantiles=(0.9,))
        self.assertEqual(len(result), 6)

    def test_group_column_is_carried_through(self):
        result = evaluation.evaluate_baselines(
            self._grouped(), group_col="demand_segment", quantiles=(0.9,))
        self.assertEqual(set(result["demand_segment"]), {"smooth", "lumpy"})

    def test_groups_partition_the_evaluated_rows(self):
        result = evaluation.evaluate_baselines(
            self._grouped(), group_col="demand_segment", quantiles=(0.9,))
        per_baseline = result[result["baseline"] == "naive_zero"]
        self.assertEqual(per_baseline["n"].sum(), 3)


if __name__ == "__main__":
    unittest.main()
