import datetime
import unittest

import merge_dataset


class TestParseTanggal(unittest.TestCase):
    def test_parses_day_month_year(self):
        result = merge_dataset.parse_tanggal("01 Jan 2024")
        self.assertEqual(result, datetime.date(2024, 1, 1))

    def test_parses_end_of_year_date(self):
        result = merge_dataset.parse_tanggal("31 Dec 2025")
        self.assertEqual(result, datetime.date(2025, 12, 31))


if __name__ == "__main__":
    unittest.main()
