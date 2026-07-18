# Merge Dataset CSVs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the five `dataset/*.csv` goods-out logs into one chronologically sorted `dataset/dataset.csv`.

**Architecture:** A single stdlib-only Python script, `merge_dataset.py`, built bottom-up as small, independently testable functions (parse a date → normalize a row → read a file → merge+sort many files → write the result → wire it together as `main()`).

**Tech Stack:** Python 3 standard library only (`csv`, `datetime`, `unittest`, `tempfile`, `pathlib`). No external dependencies.

## Global Constraints

- No external dependencies — stdlib only (per design doc).
- Output: `dataset/dataset.csv`, `;`-delimited, UTF-8 with BOM (`utf-8-sig`), single canonical 7-column header: `Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas`.
- `Tanggal` is parsed (format `%d %b %Y`, e.g. `01 Jan 2024`) only for sorting; the original text is written back out unchanged.
- Source files are read in this exact chronological order, always: `dataset/jan-24.csv`, `dataset/feb-24.csv`, `dataset/mar-24.csv`, `dataset/apr-des-24.csv`, `dataset/jan-des-25.csv`. This, combined with a stable sort, is what makes same-date rows keep source order.
- Rows are normalized to exactly 7 fields. If a dropped trailing field (present in `jan-des-25.csv`) is ever non-empty, raise `ValueError` — never silently discard data.
- Discovered running against the real files (not anticipated when this plan was written): `dataset/jan-24.csv`, `dataset/feb-24.csv`, and `dataset/apr-des-24.csv` each have 3 fully-blank trailing rows (`;;;;;;`, 7 empty fields, export artifact). `read_rows` skips any row where every field is empty after stripping, before it reaches `normalize_row` — this is not the same as the non-empty-trailing-field error case above and must not weaken it.
- This project now has a local-only git repository (initialized after this plan was written, on branch `master`, with an initial baseline commit). Each task should be committed on completion; nothing gets pushed anywhere.

---

### Task 1: Date parsing

**Files:**
- Create: `merge_dataset.py`
- Create: `test_merge_dataset.py`

**Interfaces:**
- Produces: `merge_dataset.parse_tanggal(value: str) -> datetime.date`

- [ ] **Step 1: Write the failing test (and module scaffolding)**

Create `merge_dataset.py` with just the imports and constant `parse_tanggal` will need:

```python
import csv
import datetime

DATE_FORMAT = "%d %b %Y"
```

Create `test_merge_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `FAIL` / `AttributeError: module 'merge_dataset' has no attribute 'parse_tanggal'`

- [ ] **Step 3: Implement `parse_tanggal`**

Append to `merge_dataset.py`:

```python
def parse_tanggal(value: str) -> datetime.date:
    return datetime.datetime.strptime(value.strip(), DATE_FORMAT).date()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (2 tests)

---

### Task 2: Row normalization

**Files:**
- Modify: `merge_dataset.py`
- Modify: `test_merge_dataset.py`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces: `merge_dataset.normalize_row(row: list[str]) -> list[str]` — raises `ValueError` if the row has fewer than 7 fields, or if any field beyond the 7th is non-empty after stripping whitespace.

- [ ] **Step 1: Write the failing tests**

Append to `test_merge_dataset.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `FAIL` / `AttributeError: module 'merge_dataset' has no attribute 'normalize_row'`

- [ ] **Step 3: Implement `normalize_row`**

Append to `merge_dataset.py`:

```python
EXPECTED_FIELD_COUNT = 7


def normalize_row(row: list[str]) -> list[str]:
    if len(row) < EXPECTED_FIELD_COUNT:
        raise ValueError(f"Row has fewer than {EXPECTED_FIELD_COUNT} fields: {row!r}")
    kept, extra = row[:EXPECTED_FIELD_COUNT], row[EXPECTED_FIELD_COUNT:]
    if any(field.strip() for field in extra):
        raise ValueError(f"Unexpected non-empty trailing field(s) in row: {row!r}")
    return kept
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (6 tests)

---

### Task 3: Reading a single CSV file

**Files:**
- Modify: `merge_dataset.py`
- Modify: `test_merge_dataset.py`

**Interfaces:**
- Consumes: `merge_dataset.normalize_row` (Task 2)
- Produces: `merge_dataset.read_rows(path) -> list[list[str]]` — opens `path` as `utf-8-sig`, `;`-delimited, skips the header row, returns normalized data rows.

- [ ] **Step 1: Write the failing test**

Append to `test_merge_dataset.py` (add `import tempfile` and `from pathlib import Path` to the top of the file alongside the existing imports):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `FAIL` / `AttributeError: module 'merge_dataset' has no attribute 'read_rows'`

- [ ] **Step 3: Implement `read_rows`**

Append to `merge_dataset.py`:

```python
def read_rows(path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        return [normalize_row(row) for row in reader]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (7 tests)

---

### Task 4: Merging and sorting multiple files

**Files:**
- Modify: `merge_dataset.py`
- Modify: `test_merge_dataset.py`

**Interfaces:**
- Consumes: `merge_dataset.read_rows` (Task 3), `merge_dataset.parse_tanggal` (Task 1)
- Produces: `merge_dataset.merge_and_sort(paths) -> list[list[str]]` — reads all `paths` in the given order, returns all rows stably sorted ascending by parsed `Tanggal`.

- [ ] **Step 1: Write the failing test**

Append to `test_merge_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `FAIL` / `AttributeError: module 'merge_dataset' has no attribute 'merge_and_sort'`

- [ ] **Step 3: Implement `merge_and_sort`**

Append to `merge_dataset.py`:

```python
def merge_and_sort(paths) -> list[list[str]]:
    all_rows: list[list[str]] = []
    for path in paths:
        all_rows.extend(read_rows(path))
    all_rows.sort(key=lambda row: parse_tanggal(row[0]))
    return all_rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (8 tests)

---

### Task 5: Writing the merged CSV

**Files:**
- Modify: `merge_dataset.py`
- Modify: `test_merge_dataset.py`

**Interfaces:**
- Consumes: `merge_dataset.read_rows` (Task 3, used by the test to round-trip)
- Produces: `merge_dataset.FIELDNAMES: list[str]` (the 7 canonical column names), `merge_dataset.write_rows(rows: list[list[str]], path) -> None` — writes `path` as `utf-8-sig`, `;`-delimited, with `FIELDNAMES` as the header row.

- [ ] **Step 1: Write the failing test**

Append to `test_merge_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `FAIL` / `AttributeError: module 'merge_dataset' has no attribute 'write_rows'`

- [ ] **Step 3: Implement `FIELDNAMES` and `write_rows`**

Append to `merge_dataset.py`:

```python
FIELDNAMES = [
    "Tanggal", "Kategori Barang", "Kode Barang", "Nama Barang",
    "Nama Cabang", "Satuan", "Kuantitas",
]


def write_rows(rows, path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(FIELDNAMES)
        writer.writerows(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (9 tests)

---

### Task 6: Wiring it together (`main`)

**Files:**
- Modify: `merge_dataset.py`
- Modify: `test_merge_dataset.py`

**Interfaces:**
- Consumes: `merge_dataset.merge_and_sort` (Task 4), `merge_dataset.write_rows` (Task 5), `merge_dataset.read_rows` (Task 3, used by the test)
- Produces: `merge_dataset.SOURCE_FILES: list[str]`, `merge_dataset.OUTPUT_FILE: str`, `merge_dataset.main(source_paths=SOURCE_FILES, output_path=OUTPUT_FILE) -> None`

- [ ] **Step 1: Write the failing test**

Append to `test_merge_dataset.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `FAIL` / `AttributeError: module 'merge_dataset' has no attribute 'main'`

- [ ] **Step 3: Implement `SOURCE_FILES`, `OUTPUT_FILE`, and `main`**

Append to `merge_dataset.py`:

```python
SOURCE_FILES = [
    "dataset/jan-24.csv",
    "dataset/feb-24.csv",
    "dataset/mar-24.csv",
    "dataset/apr-des-24.csv",
    "dataset/jan-des-25.csv",
]

OUTPUT_FILE = "dataset/dataset.csv"


def main(source_paths=SOURCE_FILES, output_path=OUTPUT_FILE) -> None:
    rows = merge_and_sort(source_paths)
    write_rows(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (10 tests)

---

### Task 7: Run against the real dataset and verify

**Files:**
- None created/modified — this task executes `merge_dataset.py` against the real files in `dataset/`.

**Interfaces:**
- Consumes: `merge_dataset.main` (Task 6) via `python3 merge_dataset.py`

- [ ] **Step 1: Run the full test suite one more time**

Run: `python3 -m unittest test_merge_dataset -v`
Expected: `OK` (10 tests)

- [ ] **Step 2: Run the script against the real dataset**

Run (from repo root): `python3 merge_dataset.py`
Expected: `Wrote 1548288 rows to dataset/dataset.csv`

- [ ] **Step 3: Verify the output row count**

Run: `wc -l dataset/dataset.csv`
Expected: `1548289` (1,548,288 data rows + 1 header — this is `(48236-1) + (45599-1) + (61693-1) + (509496-1) + (883269-1) + 1`, i.e. every source data row plus one merged header)

- [ ] **Step 4: Spot-check chronological ordering and format**

Run: `head -3 dataset/dataset.csv`
Expected: header row, then data rows starting `01 Jan 2024;...`

Run: `tail -3 dataset/dataset.csv`
Expected: data rows ending `31 Dec 2025;...`

Run: `python3 -c "import pathlib; print(pathlib.Path('dataset/dataset.csv').read_bytes()[:3])"`
Expected: `b'\xef\xbb\xbf'` (BOM present)
