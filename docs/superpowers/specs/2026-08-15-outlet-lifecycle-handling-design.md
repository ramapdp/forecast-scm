# Outlet lifecycle handling (closure, reopening, relocation) — design

## Purpose

`build_panel.build_dense_panel()` (`utils/build_panel.py:26`) reindexes every
`(Kode Barang, Nama Cabang)` pair to a dense daily range spanning that pair's own
first-to-last transaction date, filling gap days with `Kuantitas = 0`. That is
correct for ordinary quiet days, but wrong when a branch stops operating
entirely for a stretch: those days are not zero demand, they are *no outlet*.
Because `canonicalize_branch_names()` merges an old branch code into its
successor, a close-then-reopen cycle collapses into one continuous pair range
and the entire closed stretch is fabricated as zeros.

This is not a hypothetical. Two branches in the current dataset are affected,
and a third case is in progress right now.

## Key finding: 19,304 fabricated rows in the current dataset

Measured against `dataset/model_ready/featured.parquet` (1,522,868 rows):

| Branch | Closed window | Fabricated rows | Real transactions in window |
|---|---|---|---|
| `KY011 - Kebuli Yaman Bekasi Galaxy` | 2024-03-01 → 2025-07-17 (505 days) | **17,640** (68.6% of that branch's rows) | 0 |
| `KY056 - Kebuli Yaman Tigaraksa` | 2024-10-01 → 2024-11-21 (53 days) | **1,664** | 0 |

All 19,304 rows land in the training split and contaminate `lag_*`, `roll_*`,
`target_h*`, `target_lead_time_cumulative`, and the branch-level statistics.

The distortion of `compute_branch_stats()` output is the most visible symptom.
`branch_avg_daily_qty` is the mean of daily branch totals; 505 zero days drag it
down by a factor of 3.6:

| `KY011 - Kebuli Yaman Bekasi Galaxy` | Current | Correct |
|---|---|---|
| Days entering the mean | 700 | 196 |
| `branch_avg_daily_qty` | 104.0 | **371.3** |
| `branch_demand_cv` | 1.863 | **0.502** |
| Volume rank among 59 branches | **#59 (smallest)** | #46 |

The model is currently told that the branch is the chain's smallest and most
erratic. It is neither.

## Owner-confirmed facts (2026-08-15)

1. **`KY011 - Kebuli Yaman Bekasi Galaxy`** did not operate at all between
   2024-03-01 and 2025-07-17, then reopened 2025-07-18 under code `KY069`
   (mapped back to `KY011` by `dataset/outlet_name_overrides.csv:10`).
2. **`KY056 - Kebuli Yaman Tigaraksa`** was temporarily closed 2024-10-01 →
   2024-11-21 and resumed afterwards. Its pre-2024-03-01 history belongs to it
   (relocated from `KY035 - Kebuli Yaman Antapani`) — **already handled**:
   `outlet_name_overrides.csv:13` merges the two, and the canonical Tigaraksa
   series already starts 2024-01-01 with 777 non-zero rows in Jan–Feb 2024.
   No change needed for that part.
3. **`Kebuli Yaman Cikarang Pusat`**: `KY047 - Kebuli Yaman Ciomas` closed
   2025-11-30 and the replacement outlet had not opened as of 2025-12-31.
   The December absence is a genuine relocation-in-progress, not missing data.
4. **`KY073 - Kebuli Yaman Cilebut`** opened 2025-12-19 and is still operating.
5. **`Kebab Saudagar - Kutabumi`** is permanently out of business and its data is
   not needed. Already excluded via `normalize_items.EXCLUDED_BRANCHES`
   (`utils/normalize_items.py:130`).

## Approach: segmented dense panel

Three representations were considered.

**A. Segmented panel (chosen).** `build_dense_panel` emits no rows at all for
closed days and numbers each contiguous active block with `segment_id`. All
shift-based functions group by `(pair, segment)`.

**B. Keep the rows, flag `is_closed`, drop them late.** Rejected: the density
invariant survives, but `lag_*` and `roll_mean_*` for days *after* reopening
still read the closed period's zeros, so `roll_mean_28` immediately after
reopening is dragged toward zero. Closing that hole requires per-function
masking, which ends up more invasive than A.

**C. Break the identity — let the reopened branch be a separate branch.**
Rejected: it discards the deliberate relocation-continuity decision documented
in `2026-08-08-lead-time-integration-design.md`, turns every reopened branch
into a cold-start case, and does not generalise to the four relocations whose
dates are still lower bounds.

A is cheap because the seven functions that depend on panel density **already
accept a `pair_cols` parameter**: `add_targets`, `add_lag_features`,
`add_rolling_features`, `add_lead_time_target` (`prepare_forecast_data.py`), and
`drop_warmup_rows`, `to_tabular`, `to_sequences` (`modeling_prep.py`). No
function body is rewritten; the composite functions pass a longer key.

## New data file: `dataset/outlet_closures.csv`

Semicolon-delimited, matching `outlet_mapping.csv` and
`outlet_name_overrides.csv`.

```
Nama Outlet;tanggal_tutup;tanggal_buka;alasan
KY011 - Kebuli Yaman Bekasi Galaxy;2024-03-01;2025-07-18;tutup total, buka kembali dengan kode KY069
KY056 - Kebuli Yaman Tigaraksa;2024-10-01;2024-11-22;tutup sementara
Kebuli Yaman Cikarang Pusat;2025-12-01;;relokasi dari KY047 Ciomas, belum buka per akhir data
```

- Keyed on the **canonical `Nama Outlet`**, like `RELOCATION_DATES`, because the
  file is consumed after `canonicalize_branch_names()`.
- The interval is **`[tanggal_tutup, tanggal_buka)`** — closed from
  `tanggal_tutup` inclusive through the day before `tanggal_buka`. An empty
  `tanggal_buka` means still closed through the end of the data.
- Applies **per branch**, not per pair: a closed outlet issues no SKU.
- `alasan` is free text for humans; the pipeline never reads it.

## Module changes

### `utils/outlet_features.py`

```python
CLOSURES_FILE = str(BASE_DIR / "dataset/outlet_closures.csv")
MIN_GAP_WARN_DAYS = 14

def load_closures(
    path: str = CLOSURES_FILE,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]]
```

Returns intervals keyed by canonical branch name, so `build_panel` never needs
to know this file's column names. Raises `ValueError` — matching
`parse_delivery_days`'s fail-loud convention — when a date fails to parse, when
`tanggal_buka <= tanggal_tutup`, or when two intervals for one branch overlap. A
missing file yields `{}` (closures are optional; the pipeline must still run on
a fresh checkout).

```python
def detect_unrecorded_gaps(
    df: pd.DataFrame,
    closures: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]],
    branch_col: str = "Nama Cabang",
    date_col: str = "Tanggal",
    min_gap_days: int = MIN_GAP_WARN_DAYS,
) -> list[dict]
```

Scans per-branch transaction-date gaps and returns those `>= min_gap_days` that
are not already covered by a recorded interval. It **returns** findings; the
caller prints them. Detection never segments anything on its own — the config
file stays the single authority.

The threshold counts **missing days** (dates with no transaction between two
consecutive transaction dates). `min_gap_days = 14` is calibrated against the
real data: the longest clearly benign gap is 7 missing days
(`Kebuli Yaman Citayam`, relocation handover), then 4 (`KY003 - Kebuli Yaman
Serang`, `Kebuli Yaman Cadas`). At 14 the two confirmed closures fire
(`KY011` 504 missing days, `KY056` 52) and nothing else does.
`KY068 - Kebuli Yaman Kramatwatu` sits just below at 13 missing days
(2025-06-28 → 2025-07-10) — worth asking the data owner about, but not worth
lowering the threshold for.

This also closes a documentation gap: `docs/dokumentasi-preprocessing-id.md` §4
claims dropped branches are printed every run, but `filter_matched_branches`
never printed anything. This warning becomes the promised visibility mechanism.

### `utils/build_panel.py`

```python
SEGMENT_COL = "segment_id"

def build_dense_panel(
    df, pair_cols=PAIR_COLS, date_col="Tanggal", qty_col="Kuantitas",
    carry_cols=CARRY_COLS,
    closures: dict[str, list[tuple[pd.Timestamp, pd.Timestamp | None]]] | None = None,
) -> pd.DataFrame
```

Per pair: build the full daily range as today, **remove dates falling inside any
closure interval for that pair's branch**, then number each remaining contiguous
block `1, 2, …` chronologically into `segment_id`. A pair whose entire range sits
inside a closure produces no rows and disappears.

`closures=None` reproduces today's behaviour exactly, with `segment_id = 1`
everywhere — this is the backward-compatibility guarantee the tests pin.

### `utils/prepare_forecast_data.py`

```python
SEGMENT_COLS = PAIR_COLS + [build_panel.SEGMENT_COL]
```

`build_featured_dataset()` loads closures alongside the other master data and
passes them to `build_dense_panel`, then prints any `detect_unrecorded_gaps`
findings. `engineer_features()` passes `pair_cols=SEGMENT_COLS` to `add_targets`,
`add_lag_features`, `add_rolling_features`, and `add_lead_time_target`.

### `utils/modeling_prep.py`

`build_model_input()` passes `pair_cols=SEGMENT_COLS` to `drop_warmup_rows`,
`to_tabular`, and `to_sequences`, so no lookback window bridges a closure.

## What deliberately stays coarser than segment level

| Function | Level | Rationale |
|---|---|---|
| `filter_min_history` | **pair** | History across segments is still real history. Short segments are removed automatically by the now-segment-aware `drop_warmup_rows` at `L=28`; a second rule would be redundant. |
| `compute_pair_baseline` | **pair** | A median across segments is more stable, and closed days no longer exist to drag it down. |
| `compute_branch_stats` | **branch** | This is precisely what gets fixed — closed days stop entering the mean (the KY011 correction above). |
| `add_branch_age_days` | **branch**, from first-ever date | Branch age is an identity property, not a per-segment one. |
| `classify_pairs` (ADI/CV²) | **pair** | ADI measures demand sparsity; closed days are not quiet days and must not count. Removing the rows achieves this without touching the function. |

## Treatment of the three open cases

**Cikarang Pusat.** One row in `outlet_closures.csv` from 2025-12-01 with an
empty `tanggal_buka`. Zero effect on the current dataset — the pair already ends
2025-11-30 — but the moment 2026 data arrives, the closed stretch is not
fabricated. Once `tanggal_buka` is known it also replaces today's
`RELOCATION_DATES["Kebuli Yaman Cikarang Pusat"] = 2025-11-30  # lower bound`.

That update stays a **manual checklist item, not an automatic derivation**: the
two tables do not correspond in general. `KY056 - Kebuli Yaman Tigaraksa`
appears in both with unrelated dates — relocation 2024-03-01, temporary closure
2024-10-01 → 2024-11-22 — so a blanket "reopening date is the relocation date"
rule would corrupt it. Only relocations that *are* a closure qualify.

**KY073 Cilebut.** No mechanical change, and `MIN_HISTORY_DAYS` stays **60**.
Lowering the threshold cannot help: `filter_min_history` counts days *before the
cutoff*, and Cilebut has zero pre-cutoff days, so any threshold ≥ 1 excludes it.
It is a pure cold-start case, recorded as such in
`docs/todolist-data-preprocessing.md` and §15 of the Indonesian documentation.
It enters the dataset by itself once it has ≥ 60 days before the next cutoff, with
no code change. A new QA assertion prints branches that pass the name filter but
vanish after `filter_min_history`, so future cases surface deliberately rather
than by accident.

**Kebab Saudagar - Kutabumi.** Already in `EXCLUDED_BRANCHES`. Add a comment
recording the owner confirmation and date, matching the comment style of
`RELOCATION_DATES`.

## Schema and volume impact

- `FEATURED_COLUMNS` gains `segment_id` → **64 columns** for
  `featured`/`train`/`test`; `model_input.parquet` → 76.
- `featured.parquet` drops the 19,304 fabricated rows to **≈1,503,564**. The
  exact figure must be re-measured during implementation: removing closed days
  can push a pair below `MIN_HISTORY_DAYS`, cascading a few more removals.
- The December test split is unchanged (55,046 rows) — no closure interval
  overlaps December 2025.

## QA assertions (added to `run_qa_checks()`)

1. No row's `Tanggal` falls inside a recorded closure interval for its branch.
2. `segment_id` per pair starts at 1 and is contiguous.
3. No date gap exists *within* a segment — the density invariant that
   `add_targets`'s positional `shift(-h)` depends on.

## Testing (TDD, failing test first)

- `build_dense_panel` with one closure → no rows inside the window;
  `segment_id` is 1 before and 2 after.
- `build_dense_panel` with `closures=None` → output identical to today's
  behaviour except for the constant `segment_id` column.
- **`add_targets` with `pair_cols=SEGMENT_COLS` → the last row of segment 1 has
  `target_h1 = NaN`, not segment 2's first value.** The most important test:
  without segmentation the target silently jumps the closure.
- `add_rolling_features` → the first `roll_mean_28` in segment 2 is NaN, not
  contaminated by the closure's zeros.
- `add_lead_time_target` → the forward window never sums across a closure.
- `to_sequences` → no 28-day window bridges two segments.
- `load_closures` → `ValueError` on unparseable dates, on
  `tanggal_buka <= tanggal_tutup`, and on overlapping intervals for one branch;
  `{}` on a missing file.
- `detect_unrecorded_gaps` → a recorded gap raises no warning, an unrecorded one
  does.
- Regression on real data: `KY011`'s `branch_avg_daily_qty` moves from ≈104 to
  ≈371 and `branch_demand_cv` from ≈1.86 to ≈0.50.

## Out of scope

- **Cold-start fallback.** Cilebut and the 842 pairs dropped by
  `MIN_HISTORY_DAYS` still receive no forecast. Deferred to the modelling phase;
  already recorded as limitation #5.
- Changing `MIN_HISTORY_DAYS`, the outlier algorithm, or the fold scheme.
- Automatic closure inference. Detection warns; humans decide.

## Open questions

1. The four relocations still on lower-bound dates (`Mayor Oking`,
   `Teluk Pucung`, `Bukit Gading Balaraja`, `Grand Wisata Bekasi`) will likely
   produce the same close-then-reopen pattern in the next refresh.
   `detect_unrecorded_gaps` will catch them, but only after the data arrives.
2. The loss of 19,304 training rows must be reported as a data-quality
   correction, not as an unexplained reduction in sample size.

## Resolved questions

- **Bekasi Galaxy reopened at the same site** (owner-confirmed 2026-08-15). It
  therefore gets no `RELOCATION_DATES` entry, `days_since_relocation` stays null
  for it, and its existing classification as a duplicate branch code rather than
  a relocation is correct. The closure interval alone describes its lifecycle.

## References

- `docs/superpowers/specs/2026-08-08-outlier-handling-design.md` — the
  "`Kuantitas` is never 0 in raw data" finding this design builds on.
- `docs/superpowers/specs/2026-08-08-lead-time-integration-design.md` — the
  relocation-continuity decision.
- `docs/outlet_relocation_notes.md` — old → new outlet mappings.
- `docs/dokumentasi-preprocessing-id.md` — Indonesian pipeline documentation to
  update once implemented.
