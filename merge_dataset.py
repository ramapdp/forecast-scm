import csv
import datetime

DATE_FORMAT = "%d %b %Y"


def parse_tanggal(value: str) -> datetime.date:
    return datetime.datetime.strptime(value.strip(), DATE_FORMAT).date()


EXPECTED_FIELD_COUNT = 7


def normalize_row(row: list[str]) -> list[str]:
    if len(row) < EXPECTED_FIELD_COUNT:
        raise ValueError(f"Row has fewer than {EXPECTED_FIELD_COUNT} fields: {row!r}")
    kept, extra = row[:EXPECTED_FIELD_COUNT], row[EXPECTED_FIELD_COUNT:]
    if any(field.strip() for field in extra):
        raise ValueError(f"Unexpected non-empty trailing field(s) in row: {row!r}")
    return kept


def read_rows(path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        return [normalize_row(row) for row in reader]


def merge_and_sort(paths) -> list[list[str]]:
    all_rows: list[list[str]] = []
    for path in paths:
        all_rows.extend(read_rows(path))
    all_rows.sort(key=lambda row: parse_tanggal(row[0]))
    return all_rows
