# Merge dataset CSVs — design

## Purpose

Combine the five monthly/period `dataset/*.csv` goods-out logs (`jan-24.csv`, `feb-24.csv`, `mar-24.csv`, `apr-des-24.csv`, `jan-des-25.csv`) into a single `dataset/dataset.csv`, sorted chronologically by the `Tanggal` column, as a foundation for later forecasting work in `data-processing.ipynb`.

## Approach

Implement as a standalone Python script, `merge_dataset.py`, at the repo root, using only the Python standard library (`csv`, `datetime`) — no external dependencies, since the repo has no dependency manifest yet. Pandas can be introduced later, alongside a real `requirements.txt`, when actual forecasting work begins.

## Behavior

- **Input order**: read the 5 source files in chronological order — `jan-24.csv`, `feb-24.csv`, `mar-24.csv`, `apr-des-24.csv`, `jan-des-25.csv`.
- **Schema normalization**: keep only the first 7 fields per row (`Tanggal`, `Kategori Barang`, `Kode Barang`, `Nama Barang`, `Nama Cabang`, `Satuan`, `Kuantitas`). This drops the 2 trailing empty columns present only in `jan-des-25.csv`. If a dropped trailing field is ever non-empty for any row, the script raises an error instead of silently discarding it.
- **Date parsing**: parse `Tanggal` (format `"01 Jan 2024"`, i.e. `%d %b %Y`) into a `datetime` for sorting purposes only. The original text format is preserved unchanged in the output.
- **Sorting**: perform a stable sort of all rows by parsed date, ascending. Because rows are read in chronological file order first, rows sharing the same date retain their original relative order (source file order, then in-file order) after the stable sort.
- **Error handling**: any row whose `Tanggal` fails to parse raises an error immediately, aborting the run — bad/malformed data should be surfaced, not silently dropped.
- **Output**: write `dataset/dataset.csv` with a single header row, `;` delimiter, and UTF-8 BOM (`utf-8-sig` encoding) — matching the source file format exactly.

## Out of scope

- Deduplication or cleanup of `xxx.`-prefixed SKU/name values (see `CLAUDE.md` note) — rows are passed through unchanged.
- Any conversion to comma-delimited or ISO-date format.
- Excel (`dataset/excel/*.xlsx`) files — CSV sources only.
