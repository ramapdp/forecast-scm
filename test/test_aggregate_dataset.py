import tempfile
import unittest
from pathlib import Path

from utils import aggregate_dataset
from utils import merge_dataset


class TestParseKuantitas(unittest.TestCase):
    def test_parses_plain_integer_string(self):
        self.assertEqual(aggregate_dataset.parse_kuantitas("220"), 220)

    def test_parses_comma_decimal_whole_number(self):
        self.assertEqual(aggregate_dataset.parse_kuantitas("103,0"), 103)

    def test_parses_comma_decimal_fractional_value(self):
        self.assertEqual(aggregate_dataset.parse_kuantitas("103,5"), 103.5)


class TestAggregateRows(unittest.TestCase):
    def test_sums_kuantitas_for_rows_with_matching_key(self):
        rows = [
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "220,0"],
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "1,0"],
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "2,0"],
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "2,0"],
        ]
        self.assertEqual(aggregate_dataset.aggregate_rows(rows), [
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "225.0"],
        ])

    def test_keeps_rows_with_different_keys_separate(self):
        rows = [
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Porsi", "2,0"],
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Botol", "3,0"],
        ]
        self.assertEqual(aggregate_dataset.aggregate_rows(rows), [
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Porsi", "2.0"],
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Botol", "3.0"],
        ])

    def test_sums_rows_with_fractional_kuantitas(self):
        rows = [
            ["01 Feb 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "103,0"],
            ["01 Feb 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "4,5"],
        ]
        self.assertEqual(aggregate_dataset.aggregate_rows(rows), [
            ["01 Feb 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "107.5"],
        ])

    def test_output_order_follows_first_occurrence_across_dates(self):
        rows = [
            ["01 Jan 2024", "Barang Jadi (FG)", "A1", "Item A1",
             "KY001 - Branch", "Porsi", "1,0"],
            ["02 Jan 2024", "Barang Jadi (FG)", "B1", "Item B1",
             "KY002 - Branch", "Porsi", "3,0"],
            ["01 Jan 2024", "Barang Jadi (FG)", "A1", "Item A1",
             "KY001 - Branch", "Porsi", "4,0"],
            ["03 Jan 2024", "Barang Jadi (FG)", "C1", "Item C1",
             "KY003 - Branch", "Porsi", "5,0"],
        ]
        result = aggregate_dataset.aggregate_rows(rows)
        self.assertEqual([row[2] for row in result], ["A1", "B1", "C1"])
        self.assertEqual(result[0][6], "5.0")


class TestMain(unittest.TestCase):
    def test_main_aggregates_file_in_place(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);A1;Item A1;KY001 - Branch;Porsi;1,0\n"
            "01 Jan 2024;Barang Jadi (FG);A1;Item A1;KY001 - Branch;Porsi;4,5\n"
            "02 Jan 2024;Barang Jadi (FG);B1;Item B1;KY002 - Branch;Porsi;3,0\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

            aggregate_dataset.main(path=path)

            rows = merge_dataset.read_rows(path)

        self.assertEqual(rows, [
            ["01 Jan 2024", "Barang Jadi (FG)", "A1", "Item A1",
             "KY001 - Branch", "Porsi", "5.5"],
            ["02 Jan 2024", "Barang Jadi (FG)", "B1", "Item B1",
             "KY002 - Branch", "Porsi", "3.0"],
        ])


if __name__ == "__main__":
    unittest.main()
