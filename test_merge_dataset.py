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


class TestNormalizeRow(unittest.TestCase):
    def test_row_with_exact_field_count_is_unchanged(self):
        row = [
            "01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
            "KY038 - Kebuli Yaman Talaga Bestari", "Porsi", "2",
        ]
        self.assertEqual(merge_dataset.normalize_row(row), row)

    def test_row_with_empty_trailing_fields_is_trimmed(self):
        row = [
            "01 Jan 2025", "Minuman - FG", "FGS-00014", "Club Mineral 600 ml",
            "KY003 - Kebuli Yaman Serang", "Botol", "4", "", "",
        ]
        self.assertEqual(merge_dataset.normalize_row(row), row[:7])

    def test_row_with_non_empty_trailing_field_raises(self):
        row = [
            "01 Jan 2025", "Minuman - FG", "FGS-00014", "Club Mineral 600 ml",
            "KY003 - Kebuli Yaman Serang", "Botol", "4", "unexpected", "",
        ]
        with self.assertRaises(ValueError):
            merge_dataset.normalize_row(row)

    def test_row_with_too_few_fields_raises(self):
        row = ["01 Jan 2025", "Minuman - FG", "FGS-00014"]
        with self.assertRaises(ValueError):
            merge_dataset.normalize_row(row)


if __name__ == "__main__":
    unittest.main()
