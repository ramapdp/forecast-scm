# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This repository currently contains **only raw data and an empty analysis notebook** — there is no application code, no dependency manifest (no `requirements.txt`/`pyproject.toml`/etc.), no README, and it is not yet a git repository. There are no build, lint, or test commands to run because no code exists yet.

- `data-processing.ipynb` — empty Jupyter notebook (0 bytes), presumably the intended entry point for data processing/forecasting work ("forecast-scm" = supply-chain forecasting).
- `dataset/` — raw transactional data, described below.

When adding the first real code to this repo, also set up the corresponding tooling (dependency manifest, README) and update this file with real commands rather than guessing at conventions.

## Dataset

`dataset/*.csv` are goods-issued ("Barang Keluar" = goods out) transaction logs, one row per line item, exported from the `dataset/excel/*.xlsx` originals (each CSV corresponds 1:1 to an `.xlsx` file of the same period).

Columns (semicolon-delimited, UTF-8 with BOM, header row in Indonesian):

| Column | Meaning |
|---|---|
| `Tanggal` | Transaction date, `DD Mon YYYY` (e.g. `01 Jan 2024`) |
| `Kategori Barang` | Item category — e.g. `Bahan Baku (RM)` (raw material), `Barang Dalam Process (WIP-1)`, `Barang Jadi (FG)` (finished goods), `Minuman - FG`, `Packaging`, `Snack`, `Tambahan`, `Barang Umum` |
| `Kode Barang` | SKU code, e.g. `FGS-00001` |
| `Nama Barang` | Item name |
| `Nama Cabang` | Branch, formatted `KY0NN - Kebuli Yaman <location>` (49 branches as of the current data) |
| `Satuan` | Unit of measure — `Kg`, `Potong`, `Porsi`, `Botol`, `PCS`, `Pack`, `Ekor`, `Galon` |
| `Kuantitas` | Quantity issued |

Notes / quirks to be aware of when writing any ingestion code:

- **Files partition time, not category** — together the five CSVs cover Jan 2024–Dec 2025 with no overlap: `jan-24.csv` (Jan 2024), `feb-24.csv` (Feb 2024), `mar-24.csv` (Mar 2024), `apr-des-24.csv` (Apr–Dec 2024), `jan-des-25.csv` (Jan–Dec 2025).
- `jan-des-25.csv` has two extra trailing empty columns (9 fields vs. 7 in the other files) — strip these rather than assuming a uniform schema across files.
- Some `Kode Barang`/`Nama Barang` values are prefixed with `xxx.` (e.g. `xxx.FGS-00003` / `xxx.Iga Sapi Kebuli`) in `apr-des-24.csv` — this looks like an in-source marker (possibly discontinued/excluded items) rather than a distinct SKU; confirm intent with the data owner before treating it as a normal product code.
- Encoding: files start with a UTF-8 BOM (`﻿`) — strip it when parsing (e.g. pandas: `encoding="utf-8-sig"`), and use `;` as the separator.
