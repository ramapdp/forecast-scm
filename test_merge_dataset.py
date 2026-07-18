import datetime
import tempfile
import unittest
from pathlib import Path

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


class TestReadRows(unittest.TestCase):
    def test_reads_and_normalizes_rows(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas;;\n"
            "01 Jan 2025;Minuman - FG;FGS-00014;Club Mineral 600 ml;"
            "KY003 - Kebuli Yaman Serang;Botol;4;;\n"
            "02 Jan 2025;Barang Jadi (FG);FGS-00005;Sambal - FG;"
            "KY038 - Kebuli Yaman Talaga Bestari;Porsi;2;;\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            rows = merge_dataset.read_rows(path)
        self.assertEqual(rows, [
            ["01 Jan 2025", "Minuman - FG", "FGS-00014", "Club Mineral 600 ml",
             "KY003 - Kebuli Yaman Serang", "Botol", "4"],
            ["02 Jan 2025", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Porsi", "2"],
        ])

    def test_skips_fully_blank_trailing_row(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2025;Minuman - FG;FGS-00014;Club Mineral 600 ml;"
            "KY003 - Kebuli Yaman Serang;Botol;4\n"
            ";;;;;;\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            rows = merge_dataset.read_rows(path)
        self.assertEqual(rows, [
            ["01 Jan 2025", "Minuman - FG", "FGS-00014", "Club Mineral 600 ml",
             "KY003 - Kebuli Yaman Serang", "Botol", "4"],
        ])


class TestMergeAndSort(unittest.TestCase):
    def test_merges_and_sorts_chronologically_with_stable_ties(self):
        file_a = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "03 Jan 2024;Barang Jadi (FG);A1;Item A1;KY001 - Branch;Porsi;1\n"
            "01 Jan 2024;Barang Jadi (FG);A2;Item A2;KY001 - Branch;Porsi;2\n"
        )
        file_b = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);B1;Item B1;KY002 - Branch;Porsi;3\n"
            "02 Jan 2024;Barang Jadi (FG);B2;Item B2;KY002 - Branch;Porsi;4\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            path_a.write_bytes(b"\xef\xbb\xbf" + file_a.encode("utf-8"))
            path_b.write_bytes(b"\xef\xbb\xbf" + file_b.encode("utf-8"))
            rows = merge_dataset.merge_and_sort([path_a, path_b])
        # Both A2 and B1 are dated 01 Jan 2024; A2 must come first because
        # file_a is listed (and therefore read) before file_b.
        self.assertEqual([row[2] for row in rows], ["A2", "B1", "B2", "A1"])


class TestWriteRows(unittest.TestCase):
    def test_writes_bom_semicolon_header_and_rows(self):
        rows = [
            ["01 Jan 2024", "Barang Jadi (FG)", "FGS-00005", "Sambal - FG",
             "KY038 - Kebuli Yaman Talaga Bestari", "Porsi", "2"],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.csv"
            merge_dataset.write_rows(rows, path)
            raw = path.read_bytes()
            round_tripped = merge_dataset.read_rows(path)
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertEqual(round_tripped, rows)
        text = raw.decode("utf-8-sig")
        self.assertEqual(
            text.splitlines()[0],
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas",
        )


class TestMain(unittest.TestCase):
    def test_main_writes_merged_sorted_output(self):
        file_a = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "02 Jan 2024;Barang Jadi (FG);A1;Item A1;KY001 - Branch;Porsi;1\n"
        )
        file_b = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);B1;Item B1;KY002 - Branch;Porsi;3\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.csv"
            path_b = Path(tmpdir) / "b.csv"
            out_path = Path(tmpdir) / "merged.csv"
            path_a.write_bytes(b"\xef\xbb\xbf" + file_a.encode("utf-8"))
            path_b.write_bytes(b"\xef\xbb\xbf" + file_b.encode("utf-8"))
            merge_dataset.main(source_paths=[path_a, path_b], output_path=out_path)
            rows = merge_dataset.read_rows(out_path)
        self.assertEqual([row[2] for row in rows], ["B1", "A1"])


if __name__ == "__main__":
    unittest.main()
