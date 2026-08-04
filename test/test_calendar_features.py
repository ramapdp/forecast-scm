import unittest

import pandas as pd

from utils import calendar_features


class TestBasicCalendarFeatures(unittest.TestCase):
    def test_day_of_week_known_dates(self):
        # 2024-01-01 is a Monday (dayofweek 0); 2024-01-06 is a Saturday (5)
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-06"]))
        result = calendar_features.day_of_week(dates)
        self.assertEqual(list(result), [0, 5])

    def test_day_of_month(self):
        dates = pd.Series(pd.to_datetime(["2024-01-15", "2024-02-28"]))
        result = calendar_features.day_of_month(dates)
        self.assertEqual(list(result), [15, 28])

    def test_month(self):
        dates = pd.Series(pd.to_datetime(["2024-03-01", "2024-12-31"]))
        result = calendar_features.month(dates)
        self.assertEqual(list(result), [3, 12])

    def test_is_weekend(self):
        # Saturday and Sunday True, Monday False
        dates = pd.Series(pd.to_datetime(["2024-01-06", "2024-01-07", "2024-01-08"]))
        result = calendar_features.is_weekend(dates)
        self.assertEqual(list(result), [True, True, False])


class TestIsNationalHoliday(unittest.TestCase):
    def test_new_years_day_both_years(self):
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2025-01-01"]))
        result = calendar_features.is_national_holiday(dates)
        self.assertEqual(list(result), [True, True])

    def test_independence_day_both_years(self):
        dates = pd.Series(pd.to_datetime(["2024-08-17", "2025-08-17"]))
        result = calendar_features.is_national_holiday(dates)
        self.assertEqual(list(result), [True, True])

    def test_ordinary_date_is_not_a_holiday(self):
        dates = pd.Series(pd.to_datetime(["2024-05-15"]))
        result = calendar_features.is_national_holiday(dates)
        self.assertEqual(list(result), [False])


class TestRamadanFeatures(unittest.TestCase):
    def test_is_ramadan_on_start_end_and_mid_dates(self):
        dates = pd.Series(pd.to_datetime(["2024-03-11", "2024-03-25", "2024-04-09", "2024-04-10"]))
        result = calendar_features.is_ramadan(dates)
        self.assertEqual(list(result), [True, True, True, False])

    def test_days_into_ramadan(self):
        dates = pd.Series(pd.to_datetime(["2024-03-11", "2024-03-12", "2024-04-09"]))
        result = calendar_features.days_into_ramadan(dates)
        self.assertEqual(list(result), [0, 1, 29])

    def test_days_into_ramadan_nan_outside_ramadan(self):
        dates = pd.Series(pd.to_datetime(["2024-04-10"]))
        result = calendar_features.days_into_ramadan(dates)
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_days_until_ramadan(self):
        dates = pd.Series(pd.to_datetime(["2024-03-10", "2024-03-01"]))
        result = calendar_features.days_until_ramadan(dates)
        self.assertEqual(list(result), [1, 10])

    def test_days_until_ramadan_nan_during_or_after_ramadan(self):
        dates = pd.Series(pd.to_datetime(["2024-03-11", "2024-04-09"]))
        result = calendar_features.days_until_ramadan(dates)
        self.assertTrue(result.isna().all())


class TestEidAlFitrFeatures(unittest.TestCase):
    def test_is_eid_al_fitr_on_and_off_the_date(self):
        dates = pd.Series(pd.to_datetime(["2024-04-10", "2024-04-11", "2025-03-31"]))
        result = calendar_features.is_eid_al_fitr(dates)
        self.assertEqual(list(result), [True, False, True])

    def test_days_since_within_window(self):
        dates = pd.Series(pd.to_datetime(["2024-04-10", "2024-04-11", "2024-04-17"]))  # 0, 1, 7 days after
        result = calendar_features.days_since_eid_al_fitr(dates)
        self.assertEqual(list(result), [0, 1, 7])

    def test_days_since_nan_before_the_holiday(self):
        dates = pd.Series(pd.to_datetime(["2024-04-09"]))
        result = calendar_features.days_since_eid_al_fitr(dates)
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_days_since_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2024-05-01"]))  # far beyond the 14-day window
        result = calendar_features.days_since_eid_al_fitr(dates)
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_days_until_within_window(self):
        dates = pd.Series(pd.to_datetime(["2024-04-09", "2024-04-03"]))  # 1, 7 days before
        result = calendar_features.days_until_eid_al_fitr(dates)
        self.assertEqual(list(result), [1, 7])

    def test_days_until_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2024-03-01"]))  # far beyond the 14-day window
        result = calendar_features.days_until_eid_al_fitr(dates)
        self.assertTrue(pd.isna(result.iloc[0]))


class TestEidAlAdhaFeatures(unittest.TestCase):
    def test_is_eid_al_adha_on_and_off_the_date(self):
        dates = pd.Series(pd.to_datetime(["2024-06-17", "2024-06-18", "2025-06-06"]))
        result = calendar_features.is_eid_al_adha(dates)
        self.assertEqual(list(result), [True, False, True])

    def test_days_since_within_window(self):
        dates = pd.Series(pd.to_datetime(["2024-06-17", "2024-06-24"]))  # 0, 7 days after
        result = calendar_features.days_since_eid_al_adha(dates)
        self.assertEqual(list(result), [0, 7])

    def test_days_until_within_window(self):
        dates = pd.Series(pd.to_datetime(["2024-06-16", "2024-06-10"]))  # 1, 7 days before
        result = calendar_features.days_until_eid_al_adha(dates)
        self.assertEqual(list(result), [1, 7])

    def test_days_until_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2024-05-01"]))
        result = calendar_features.days_until_eid_al_adha(dates)
        self.assertTrue(pd.isna(result.iloc[0]))


class TestIndependenceDayFeatures(unittest.TestCase):
    def test_is_independence_day_on_and_off_the_date(self):
        dates = pd.Series(pd.to_datetime(["2024-08-17", "2024-08-18", "2025-08-17"]))
        result = calendar_features.is_independence_day(dates)
        self.assertEqual(list(result), [True, False, True])

    def test_days_since_within_window(self):
        dates = pd.Series(pd.to_datetime(["2024-08-17", "2024-08-24"]))  # 0, 7 days after
        result = calendar_features.days_since_independence_day(dates)
        self.assertEqual(list(result), [0, 7])

    def test_days_since_nan_before_the_holiday(self):
        dates = pd.Series(pd.to_datetime(["2024-08-16"]))
        result = calendar_features.days_since_independence_day(dates)
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_days_since_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2024-09-15"]))
        result = calendar_features.days_since_independence_day(dates)
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_days_until_within_window(self):
        dates = pd.Series(pd.to_datetime(["2024-08-16", "2024-08-10"]))  # 1, 7 days before
        result = calendar_features.days_until_independence_day(dates)
        self.assertEqual(list(result), [1, 7])

    def test_days_until_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2024-07-01"]))
        result = calendar_features.days_until_independence_day(dates)
        self.assertTrue(pd.isna(result.iloc[0]))


class TestNewYearFeatures(unittest.TestCase):
    def test_is_new_year_on_and_off_the_date(self):
        dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2025-01-01"]))
        result = calendar_features.is_new_year(dates)
        self.assertEqual(list(result), [True, False, True])

    def test_days_since_within_window(self):
        dates = pd.Series(pd.to_datetime(["2025-01-01", "2025-01-10"]))  # 0, 9 days after
        result = calendar_features.days_since_new_year(dates)
        self.assertEqual(list(result), [0, 9])

    def test_days_since_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2025-01-20"]))
        result = calendar_features.days_since_new_year(dates)
        self.assertTrue(pd.isna(result.iloc[0]))

    def test_days_until_within_window_same_year(self):
        # 2025-01-01 itself: 0 days until
        dates = pd.Series(pd.to_datetime(["2025-01-01"]))
        result = calendar_features.days_until_new_year(dates)
        self.assertEqual(list(result), [0])

    def test_days_until_wraps_to_next_year_in_december(self):
        dates = pd.Series(pd.to_datetime(["2024-12-20", "2024-12-31"]))  # 12, 1 days until 2025-01-01
        result = calendar_features.days_until_new_year(dates)
        self.assertEqual(list(result), [12, 1])

    def test_days_until_nan_beyond_proximity_window(self):
        dates = pd.Series(pd.to_datetime(["2024-12-01"]))
        result = calendar_features.days_until_new_year(dates)
        self.assertTrue(pd.isna(result.iloc[0]))


class TestCheckYearCoverage(unittest.TestCase):
    def test_raises_on_uncovered_year(self):
        df = pd.DataFrame({"Tanggal": pd.to_datetime(["2026-01-01"])})
        with self.assertRaises(ValueError):
            calendar_features.add_calendar_features(df)

    def test_no_error_for_covered_years(self):
        df = pd.DataFrame({"Tanggal": pd.to_datetime(["2024-01-01", "2025-12-31"])})
        result = calendar_features.add_calendar_features(df)
        self.assertEqual(len(result), 2)


class TestAddCalendarFeatures(unittest.TestCase):
    def test_wires_all_expected_columns(self):
        df = pd.DataFrame({"Tanggal": pd.to_datetime(["2024-03-11", "2024-06-17", "2024-07-01"])})
        result = calendar_features.add_calendar_features(df)
        expected_cols = {
            "day_of_week", "day_of_month", "month", "is_weekend", "is_national_holiday",
            "is_ramadan", "days_into_ramadan", "days_until_ramadan",
            "is_eid_al_fitr", "days_since_eid_al_fitr", "days_until_eid_al_fitr",
            "is_eid_al_adha", "days_since_eid_al_adha", "days_until_eid_al_adha",
            "is_independence_day", "days_since_independence_day", "days_until_independence_day",
            "is_new_year", "days_since_new_year", "days_until_new_year",
        }
        self.assertTrue(expected_cols.issubset(set(result.columns)))

    def test_ramadan_start_and_eid_al_adha_rows_have_correct_flags(self):
        df = pd.DataFrame({"Tanggal": pd.to_datetime(["2024-03-11", "2024-06-17"])})
        result = calendar_features.add_calendar_features(df)
        self.assertTrue(result.iloc[0]["is_ramadan"])
        self.assertTrue(result.iloc[1]["is_eid_al_adha"])


if __name__ == "__main__":
    unittest.main()
