import tempfile
import unittest
from pathlib import Path

import aggregate_dataset
import merge_dataset


class TestAggregateRows(unittest.TestCase):
    def test_sums_kuantitas_for_rows_with_matching_key(self):
        rows = [
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "220"],
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "1"],
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "2"],
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "2"],
        ]
        self.assertEqual(aggregate_dataset.aggregate_rows(rows), [
            ["01 Jan 2024", "Barang Semi FG (WIP-2)", "FGS-00001", "Ayam Kebuli (0.9)",
             "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Potong", "225"],
        ])

    def test_keeps_rows_with_different_keys_separate(self):
        rows = [
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Porsi", "2"],
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Botol", "3"],
        ]
        self.assertEqual(aggregate_dataset.aggregate_rows(rows), rows)

    def test_output_order_follows_first_occurrence_across_dates(self):
        rows = [
            ["01 Jan 2024", "Barang Jadi (FG)", "A1", "Item A1",
             "KY001 - Branch", "Porsi", "1"],
            ["02 Jan 2024", "Barang Jadi (FG)", "B1", "Item B1",
             "KY002 - Branch", "Porsi", "3"],
            ["01 Jan 2024", "Barang Jadi (FG)", "A1", "Item A1",
             "KY001 - Branch", "Porsi", "4"],
            ["03 Jan 2024", "Barang Jadi (FG)", "C1", "Item C1",
             "KY003 - Branch", "Porsi", "5"],
        ]
        result = aggregate_dataset.aggregate_rows(rows)
        self.assertEqual([row[2] for row in result], ["A1", "B1", "C1"])
        self.assertEqual(result[0][6], "5")


class TestMain(unittest.TestCase):
    def test_main_aggregates_file_in_place(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);A1;Item A1;KY001 - Branch;Porsi;1\n"
            "01 Jan 2024;Barang Jadi (FG);A1;Item A1;KY001 - Branch;Porsi;4\n"
            "02 Jan 2024;Barang Jadi (FG);B1;Item B1;KY002 - Branch;Porsi;3\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))

            aggregate_dataset.main(path=path)

            rows = merge_dataset.read_rows(path)

        self.assertEqual(rows, [
            ["01 Jan 2024", "Barang Jadi (FG)", "A1", "Item A1",
             "KY001 - Branch", "Porsi", "5"],
            ["02 Jan 2024", "Barang Jadi (FG)", "B1", "Item B1",
             "KY002 - Branch", "Porsi", "3"],
        ])


if __name__ == "__main__":
    unittest.main()
