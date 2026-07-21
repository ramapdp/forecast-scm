# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is a git repository with a Python 3.9.6 virtual environment (`.venv`) and `requirements.txt` (pandas, pyarrow, holidays). The data-prep pipeline currently includes `merge_dataset.py` and `aggregate_dataset.py` (which clean/combine raw CSVs into `dataset/dataset.csv`); the full pipeline will also include `normalize_items.py`, `build_panel.py`, `calendar_features.py`, and `prepare_forecast_data.py` (which transform the cleaned data into model-ready train/test parquet files), orchestrated by `data-processing.ipynb`. Refer to `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md` for the full design.

- `data-processing.ipynb` — Jupyter notebook with 15 cells of exploratory and QA code, intended as the entry point for data processing/forecasting work ("forecast-scm" = supply-chain forecasting).
- `dataset/` — raw transactional data, described below.

## Dataset

`dataset/*.csv` are goods-issued ("Barang Keluar" = goods out) transaction logs, one row per line item, exported from the `dataset/excel/*.xlsx` originals (each CSV corresponds 1:1 to an `.xlsx` file of the same period).

Columns (semicolon-delimited, UTF-8 with BOM, header row in Indonesian):

| Column | Meaning |
|---|---|
| `Tanggal` | Transaction date, `DD Mon YYYY` (e.g. `01 Jan 2024`) |
| `Kategori Barang` | Item category — e.g. `Bahan Baku (RM)` (raw material), `Barang Dalam Process (WIP-1)`, `Barang Jadi (FG)` (finished goods), `Minuman - FG`, `Packaging`, `Snack`, `Tambahan`, `Barang Umum` |
| `Kode Barang` | SKU code, e.g. `FGS-00001` |
| `Nama Barang` | Item name |
| `Nama Cabang` | Branch, formatted `KY0NN - Kebuli Yaman <location>` (67 branches as of the current data) |
| `Satuan` | Unit of measure — `Kg`, `Potong`, `Porsi`, `Botol`, `PCS`, `Pack`, `Ekor`, `Galon` |
| `Kuantitas` | Quantity issued |

Notes / quirks to be aware of when writing any ingestion code:

- **Files partition time, not category** — together the five CSVs cover Jan 2024–Dec 2025 with no overlap: `jan-24.csv` (Jan 2024), `feb-24.csv` (Feb 2024), `mar-24.csv` (Mar 2024), `apr-des-24.csv` (Apr–Dec 2024), `jan-des-25.csv` (Jan–Dec 2025).
- `jan-des-25.csv` has two extra trailing empty columns (9 fields vs. 7 in the other files) — strip these rather than assuming a uniform schema across files.
- Some `Kode Barang`/`Nama Barang` values are prefixed with `xxx.` (e.g. `xxx.FGS-00003` / `xxx.Iga Sapi Kebuli`) in `apr-des-24.csv` — this looks like an in-source marker (possibly discontinued/excluded items) rather than a distinct SKU; confirm intent with the data owner before treating it as a normal product code.
- Encoding: files start with a UTF-8 BOM (`﻿`) — strip it when parsing (e.g. pandas: `encoding="utf-8-sig"`), and use `;` as the separator.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Commands

- Run the full data-prep pipeline: `.venv/bin/python3 prepare_forecast_data.py`
- Run all tests: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v`
- Run one module's tests: `.venv/bin/python3 -m unittest test_normalize_items -v`
