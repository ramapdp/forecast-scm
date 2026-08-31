# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is a git repository with a Python 3.9.6 virtual environment (`.venv`) and `requirements.txt` (pandas, pyarrow, holidays). The data-prep pipeline (`utils/data_preprocessing/normalize_items.py`, `build_panel.py`, `calendar_features.py`, `outlier_handling.py`, `outlet_features.py`, `prepare_forecast_data.py`) cleans/combines raw CSVs (via `merge_dataset.py`/`aggregate_dataset.py`) into `dataset/csv/dataset.csv`, then transforms it into a cleaned + feature-engineered `dataset/model_ready/featured.parquet`, and finally splits that into model-ready `dataset/model_ready/{train,test}.parquet`. Refer to `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`, `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md`, and `docs/superpowers/specs/2026-08-15-outlet-lifecycle-handling-design.md` for the full design, and to `docs/pipeline-overview.md` for a plain-language walkthrough of the whole flow.

Modeling sits on top of that. `utils/modelling/modeling_prep.py` turns `featured.parquet` into `dataset/model_ready/model_input.parquet` (already imputed — do not call `impute_features()` on it again), `utils/modelling/walk_forward.py` runs the shared five-fold walk-forward evaluation every model is scored through, and `utils/modelling/model_random_forest.py` supplies the first model: a 0.9-quantile Random Forest built on `quantile-forest`. December 2025 is a locked test set that no fold may touch. See `docs/superpowers/specs/2026-08-12-modeling-preprocessing-design.md` and `docs/superpowers/specs/2026-08-18-random-forest-modeling-design.md` for the design, and `docs/hasil-modeling-rf.md` for the measured results.

`utils/modelling/model_common.py` holds the parts of that machinery no single model owns — the random search with its checkpoint/resume, one-hot expansion, and the bundle format — and `utils/modelling/model_xgboost.py` supplies the second model: a 0.9-quantile XGBoost (`reg:quantileerror`) whose boosting rounds come from early stopping on a purged 30-day tail of each fold's training window, then a refit on the full training rows so it is trained on the same population the forest saw. See `docs/superpowers/specs/2026-08-19-xgboost-modeling-design.md`. Its results document has been archived to `docs/bak/hasil-modeling-xgb.single-quantile.bak.md` — those numbers predate the multi-quantile criterion, and no current one exists until Phase 3 runs XGBoost. XGBoost needs the OpenMP runtime, which pip cannot supply: `brew install libomp` once on macOS.

Hand-maintained config lives alongside the raw data in `dataset/` (all of which is gitignored): `outlets.csv`, `outlet_name_overrides.csv`, `outlet_mapping.csv`, `event_driven_items.csv`, and `outlet_closures.csv` (intervals when a branch was not operating — days inside one produce no panel rows, and `segment_id` numbers the contiguous active blocks so no lag, rolling window, or target bridges a closure).

- `notebook/data_processing.ipynb` — Jupyter notebook of exploratory, feature-engineering and QA code that produces `featured.parquet`; the entry point for data processing/forecasting work ("forecast-scm" = supply-chain forecasting). `notebook/merge_and_aggregate.ipynb` is a companion notebook for the merge/aggregate stage; `notebook/train_test_split.ipynb` reads `featured.parquet` and produces the final train/test split. All of these except the `modeling_*.ipynb` notebooks are self-contained (see "Notebook convention" below); the modeling notebooks still add the repo root to `sys.path` in an early cell so they can import the shared modeling machinery from the root-level `utils` package.
- `utils/` — the code, split into four subpackages (namespace package, no root `__init__.py`; every module is run as `python3 -m utils.<subpackage>.<module>` from the repo root):
  - `merge_split_data/` — `merge_dataset.py`, `aggregate_dataset.py`, `sync_outlets.py`
  - `data_preprocessing/` — `normalize_items.py`, `build_panel.py`, `calendar_features.py`, `outlet_features.py`, `outlier_handling.py`, `prepare_forecast_data.py`
  - `modelling/` — `modeling_prep.py`, `walk_forward.py`, `purging.py`, `sequence_windows.py`, `evaluation.py`, `model_common.py`, `model_random_forest.py`, `model_xgboost.py`, `model_lstm.py`
  - `eda/` — `verify_category_consistency.py`, `analyze_spike_recovery.py`, `analyze_spike_comovement.py`, `generate_item_cost_margin_template.py`

  Imports inside a subpackage are relative (`from . import build_panel`); imports across subpackages are absolute (`from utils.data_preprocessing import build_panel`). Each module's `BASE_DIR` is `Path(__file__).resolve().parents[2]` — the repo root, three levels up from `utils/<subpackage>/<module>.py`. A module moved to a different depth must have that constant adjusted or every `dataset/` path silently resolves inside `utils/`.
- `test/` — unittest suites (`test_*.py`), one per pipeline module.
- `dataset/` — raw transactional data, described below.

## Notebook convention

Since 2026-08-25 the notebooks are being refactored to stand on their own: every function a
notebook runs is defined **inline in the notebook**, with no `sys.path.append("..")` and no
`from utils... import`. Applied to every notebook in `notebook/` as of 2026-08-26. The `modeling_*.ipynb` notebooks
get a deliberate exception (see below).

`utils/` and `test/` do not go away — their role changes. They are where code is written and
tested; the notebook is where it is run. The workflow for changing any logic:

1. Edit the function in `utils/<subpackage>/<module>.py`.
2. Run its tests: `.venv/bin/python3 -m unittest test.test_<module> -v`.
3. Copy the function **verbatim** into the matching notebook cell.

Keep the notebook copy byte-identical to the `utils` original. Each refactored notebook ends
with an optional "Cek sinkron dengan `utils/`" cell that compares `inspect.getsource()` of every
pair and names any function that has drifted, which only works while the two copies match
character for character. Where the notebook must differ — e.g. `split_train_test` calls a local
`lookahead_safe_mask` instead of `purging.lookahead_safe_mask` — that cell normalizes the one
known difference explicitly rather than loosening the comparison.

Exception for the modeling notebooks (decided 2026-08-26): `modeling_rf.ipynb`,
`modeling_xgb.ipynb`, and `modeling_lstm.ipynb` inline only their own model code
(`model_random_forest.py` / `model_xgboost.py` / `model_lstm.py`, plus `sequence_windows.py` in the
LSTM notebook since nothing else uses it) and keep importing the shared machinery — `modeling_prep`,
`walk_forward`, `evaluation`, `model_common`, `run_config`, `purging`. Inlining that too would put
the same 2,129 lines into three notebooks (four, counting `modeling_prep.ipynb`), which costs more
than the separation it buys. Because those imports stay, the model code is copied **verbatim**, with
no qualifier rewriting — `model_common.x` inside it still resolves.

Their artefact paths (`SEARCH_FILE`, `RESULTS_FILE`) come from `run_config`, not from the model
module's constants, so `FORECAST_CHECKPOINT_DIR` applies to every cell including the ones that read
the results table back. The sync cell skips those two names for that reason.

These three cannot be verified by running them — a full pass takes hours. Verify a change by
executing the definition cells plus a small fit (see the harness pattern used on 2026-08-26: run
cells up to the run-config cell, then one `make_fit_predict` on a few thousand rows), and rely on
the sync cell for the rest.

`eda.ipynb` follows the style rules below, so its narrative — business context, per-section
findings, findings table, open questions, pre-modeling checklist — lives in
`docs/detail-tahap-eda.md` (Bagian 1), not in the notebook. Re-run the notebook whenever
`dataset.csv` is regenerated and update that document if the numbers move.

Style rules for the notebooks (set by the data owner on `merge_and_aggregate.ipynb`):

- Markdown cells contain **a heading and nothing else**: `# Title` for the opening cell, then
  numbered `## N. Step name` per stage. No prose paragraphs, tables, or bullet lists in markdown.
- All explanation lives in the code: every function carries a docstring, and constants that need
  context get an inline comment.
- Longer rationale that does not fit a docstring goes in a comment block **above** the `def` —
  outside what `inspect.getsource()` returns, so it does not break the sync check — or here in
  `CLAUDE.md`.
- Each notebook resolves the repo root with a `find_base_dir()` helper that walks up from the
  working directory looking for `dataset/csv/`, so the notebook runs from `notebook/` (Jupyter)
  and from the repo root (`nbconvert --execute`) alike. Never hardcode `..`.

## Dataset

`dataset/*.csv` are goods-issued ("Barang Keluar" = goods out) transaction logs, one row per line item, exported from the `dataset/excel/*.xlsx` originals (each CSV corresponds 1:1 to an `.xlsx` file of the same period).

Columns (semicolon-delimited, UTF-8 with BOM, header row in Indonesian):

| Column | Meaning |
|---|---|
| `Tanggal` | **Pickup date**, `DD Mon YYYY` (e.g. `01 Jan 2024`) — the day goods were handed to the customer, *not* the day the order was placed (see the note below) |
| `Kategori Barang` | Item category — e.g. `Bahan Baku (RM)` (raw material), `Barang Dalam Process (WIP-1)`, `Barang Jadi (FG)` (finished goods), `Minuman - FG`, `Packaging`, `Snack`, `Tambahan`, `Barang Umum` |
| `Kode Barang` | SKU code, e.g. `FGS-00001` |
| `Nama Barang` | Item name |
| `Nama Cabang` | Branch, formatted `KY0NN - Kebuli Yaman <location>` (67 branches as of the current data) |
| `Satuan` | Unit of measure — `Kg`, `Potong`, `Porsi`, `Botol`, `PCS`, `Pack`, `Ekor`, `Galon`, `Gr`, `Cup`, `Roll`, `Bungkus` (12 distinct values total; `Bungkus` occurs only once in the whole dataset) |
| `Kuantitas` | Quantity issued |

Notes / quirks to be aware of when writing any ingestion code:

- **`Tanggal` is the pickup date, not the order date** (confirmed by the data owner 2026-08-15). A customer who orders on Monday for Thursday pickup produces a row dated *Thursday*; the outlet manager relays that order to head office on Monday, but nothing is written until the goods actually leave. The POS deliberately does not store the order date, because orders can be cancelled and only realized transactions are recorded. Every series in this project therefore runs on the pickup-time axis — features and target are consistent on that axis, but they measure *realized* demand, not when demand arose. This is a hard limitation on forecast accuracy (head office already knows part of the coming days' demand from relayed orders, and that information exists in no dataset here), and the current business logic is that the model serves demand *outside* pre-orders — which the team handles separately. See `docs/batasan-penelitian.md` (B-1, B-2, B-3) before designing any feature or metric that assumes otherwise.
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

- Run the full data-prep pipeline (cleansing + split, end to end): `.venv/bin/python3 -m utils.data_preprocessing.prepare_forecast_data` (must be run as a module from the repo root — the packages use relative imports)
- Run the pipeline via the notebooks (cleansing, then split): `jupyter nbconvert --to notebook --execute --inplace --allow-errors notebook/data_processing.ipynb notebook/train_test_split.ipynb`
- Run the Random Forest modeling notebook (benchmark, search, final walk-forward; takes hours): `.venv/bin/python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook/modeling_rf.ipynb`
- Run the XGBoost modeling notebook (benchmark, search, final walk-forward; takes hours): `.venv/bin/python3 -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebook/modeling_xgb.ipynb`
- Run **part** of a modeling notebook, one kernel, saving after every cell: `.venv/bin/python3 run_cells.py notebook/modeling_xgb.ipynb 2-10,14` (`--list` prints the cell indices). Exists because `nbconvert --execute` only knows "the whole notebook", while the search runs for days and the walk-forward has no checkpoint at all — one process means one kill at hour 120 erases both. `run_xgb_all.sh` chains the three XGBoost stages through it.
- Phase 3 is split across two machines since 2026-08-26: hyperparameter search for XGBoost and LSTM on a Windows PC with an RTX 4060 Ti 8 GB, walk-forward and final fit on this Mac. Which cells go where, and what breaks if they are swapped, is in `docs/runbook-pencarian-gpu-windows.md`.
- Run all tests: `.venv/bin/python3 -m unittest discover -p "test_*.py" -v`
- Run one module's tests: `.venv/bin/python3 -m unittest test.test_normalize_items -v`

Note: `notebook/data_processing.ipynb`'s QA section asserts `(featured["Kuantitas"] >= 0).all()`, added for a known raw-data anomaly (negative Kuantitas at branch KY011, 2024-02-29). As of 2026-07-28 there are no negative-Kuantitas rows in `dataset/dataset.csv` or `dataset/feb-24.csv`, and the presence of `dataset/feb-24_No_Minus.csv`/`.xlsx` suggests the data owner already supplied a corrected February 2024 export — so the assertion should now pass cleanly. This has not been reconfirmed with the data owner; until it is, keep `--allow-errors` on the nbconvert run, or use `.venv/bin/python3 -m utils.data_preprocessing.prepare_forecast_data` directly (unaffected by this notebook-only assertion).
