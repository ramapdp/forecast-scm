import unittest

import pandas as pd

import calendar_features


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


if __name__ == "__main__":
    unittest.main()
