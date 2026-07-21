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


if __name__ == "__main__":
    unittest.main()
