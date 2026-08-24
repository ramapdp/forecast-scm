import csv
import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATE_FORMAT = "%d %b %Y"


# ── PARSING & VALIDASI BARIS ────────────────────────────────────────────────

def parse_tanggal(value: str) -> datetime.date:
    """Ubah string tanggal format '01 Jan 2024' menjadi objek datetime.date."""
    return datetime.datetime.strptime(value.strip(), DATE_FORMAT).date()


EXPECTED_FIELD_COUNT = 7


def normalize_row(row: list[str]) -> list[str]:
    """Pastikan baris punya tepat EXPECTED_FIELD_COUNT kolom.

    Kolom ekstra yang tidak kosong dianggap corrupt dan akan raise ValueError,
    karena itu berarti delimiter salah atau ada field yang tidak diharapkan.
    """
    if len(row) < EXPECTED_FIELD_COUNT:
        raise ValueError(f"Row has fewer than {EXPECTED_FIELD_COUNT} fields: {row!r}")
    kept, extra = row[:EXPECTED_FIELD_COUNT], row[EXPECTED_FIELD_COUNT:]
    if any(field.strip() for field in extra):
        raise ValueError(f"Unexpected non-empty trailing field(s) in row: {row!r}")
    return kept


# ── I/O ─────────────────────────────────────────────────────────────────────

def read_rows(path) -> list[list[str]]:
    """Baca semua baris data dari CSV (delimiter titik koma), skip header.

    Baris kosong dibuang. Setiap baris divalidasi oleh normalize_row sebelum
    dikembalikan, sehingga caller bisa langsung menggunakan indeks kolom.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        return [
            normalize_row(row)
            for row in reader
            if any(field.strip() for field in row)
        ]


def write_rows(rows, path) -> None:
    """Tulis baris data ke CSV dengan header standar FIELDNAMES."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)


# ── MERGE & SORTING ──────────────────────────────────────────────────────────

def merge_and_sort(paths) -> list[list[str]]:
    """Gabungkan semua file CSV sumber lalu urutkan kronologis berdasarkan Tanggal.

    Pengurutan dilakukan pada kolom pertama (Tanggal) setelah di-parse ke
    datetime.date, bukan sebagai string, sehingga '01 Feb 2024' < '01 Mar 2024'
    secara benar meski secara alfabet '01 Feb...' > '01 Jan...'.
    """
    all_rows: list[list[str]] = []
    for path in paths:
        all_rows.extend(read_rows(path))
    all_rows.sort(key=lambda row: parse_tanggal(row[0]))
    return all_rows


FIELDNAMES = [
    "Tanggal", "Kategori Barang", "Kode Barang", "Nama Barang",
    "Nama Cabang", "Satuan", "Kuantitas",
]


SOURCE_FILES = [
    str(BASE_DIR / "dataset/csv/jan-24.csv"),
    str(BASE_DIR / "dataset/csv/feb-24.csv"),
    str(BASE_DIR / "dataset/csv/mar-24.csv"),
    str(BASE_DIR / "dataset/csv/apr-des-24.csv"),
    str(BASE_DIR / "dataset/csv/jan-des-25.csv"),
]

OUTPUT_FILE = str(BASE_DIR / "dataset/csv/dataset.csv")


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

def main(source_paths=SOURCE_FILES, output_path=OUTPUT_FILE) -> None:
    """Merge semua file sumber, urutkan, lalu tulis ke OUTPUT_FILE."""
    rows = merge_and_sort(source_paths)
    write_rows(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
