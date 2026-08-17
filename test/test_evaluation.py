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


class TestEvaluateBaselines(unittest.TestCase):
    def test_returns_one_row_per_baseline(self):
        result = evaluation.evaluate_baselines(_frame())
        self.assertEqual(len(result), 3)
        self.assertEqual(
            sorted(result["baseline"]),
            ["naive_lag_1", "naive_roll_mean_7", "naive_zero"],
        )

    def test_reports_mae_pinball_and_coverage(self):
        result = evaluation.evaluate_baselines(_frame())
        for col in ["mae", "pinball", "coverage", "n"]:
            self.assertIn(col, result.columns)

    def test_reports_the_service_level_metrics(self):
        result = evaluation.evaluate_baselines(_frame())
        for col in ["fill_rate", "shortfall_units", "overstock_units"]:
            self.assertIn(col, result.columns)

    def test_zero_baseline_leaves_every_unit_unserved(self):
        result = evaluation.evaluate_baselines(_frame()).set_index("baseline")
        self.assertEqual(result.loc["naive_zero", "fill_rate"], 0.0)
        self.assertEqual(result.loc["naive_zero", "shortfall_units"], 10 + 20 + 8)
        self.assertEqual(result.loc["naive_zero", "overstock_units"], 0.0)

    def test_row_count_excludes_null_targets(self):
        result = evaluation.evaluate_baselines(_frame())
        self.assertTrue((result["n"] == 3).all())

    def test_zero_baseline_mae_equals_mean_demand(self):
        result = evaluation.evaluate_baselines(_frame()).set_index("baseline")
        self.assertAlmostEqual(result.loc["naive_zero", "mae"], (10 + 20 + 8) / 3)


class TestEvaluateByGroup(unittest.TestCase):
    """Metrics must be reportable per demand segment: a global MAE is
    dominated by the smooth pairs, where predicting anything is easy."""

    def _grouped(self):
        df = _frame()
        df["demand_segment"] = ["smooth", "smooth", "lumpy", "lumpy"]
        return df

    def test_one_row_per_group_per_baseline(self):
        result = evaluation.evaluate_baselines(self._grouped(), group_col="demand_segment")
        self.assertEqual(len(result), 6)

    def test_group_column_is_carried_through(self):
        result = evaluation.evaluate_baselines(self._grouped(), group_col="demand_segment")
        self.assertEqual(set(result["demand_segment"]), {"smooth", "lumpy"})

    def test_groups_partition_the_evaluated_rows(self):
        result = evaluation.evaluate_baselines(self._grouped(), group_col="demand_segment")
        per_baseline = result[result["baseline"] == "naive_zero"]
        self.assertEqual(per_baseline["n"].sum(), 3)


if __name__ == "__main__":
    unittest.main()
