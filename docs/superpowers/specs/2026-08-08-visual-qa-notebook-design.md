# Visual QA section for `data-processing.ipynb` — design

## Purpose

`notebook/eda.ipynb` explores `dataset/dataset.csv` (raw, pre-pipeline) with charts covering
`Kuantitas` distribution/outliers (section 3) and time-series/seasonality (section 5). `notebook/data-processing.ipynb`
runs the full cleansing + feature-engineering pipeline (branch drop, dense-panel building,
min-history filtering, category canonicalization, outlier capping, calendar features) and already
has numeric QA assertions (bagian 8: non-negative `Kuantitas`, no duplicate `(pair, Tanggal)`,
outlier-capping invariants, lag leakage spot-check, outlet-join sanity) — but no visualizations.
This spec adds a new section that reproduces analogous distribution/outlier and seasonality charts
computed on `featured` (the post-pipeline output), so the pipeline's transformations can be
sanity-checked visually, not just via row counts and asserts.

## Pre-check: does `data-processing.ipynb` already match `docs/todolist-data-preprocessing.md`?

Verified before starting this design — yes, no gaps found for pipeline-code items:

- 🔴 region/lead-time integration (marked done 2026-08-08): confirmed present in notebook bagian 7
  (`apply_region_features`, `add_lead_time_target`, QA cells `49b2b61b`/`5f4860c9`).
- 🟡 outlier-handling wired into the notebook (marked done 2026-08-08): confirmed present in section 5
  (`compute_pair_baseline`/`apply_outlier_capping`) and section 8 QA cell `80494edc`, in the documented
  order (after calendar features, before lag/rolling).
- All remaining unchecked items in that file are either data-owner confirmations (no code
  change needed) or separately-scoped gap-engineering items (QA-in-script, cold-start fallback for
  filtered pairs) — none block this task.

## Placement

New section appended after section 8 (QA checks) and before the final concluding markdown cell
(`4cb8c25d`), operating on the `featured` DataFrame already built by earlier cells. Self-contained:
adds its own `matplotlib`/palette setup (data-processing.ipynb doesn't otherwise plot), rather than
touching earlier cells.

## Style

Reuse eda.ipynb's palette/style constants verbatim (`notebook/eda.ipynb` cell 5) so charts look
consistent between the two notebooks:

```python
BLUE = "#2a78d6"
BLUE_DARK = "#173f73"
plt.rcParams.update({
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "font.size": 10,
})
```

## Chart 1: `Kuantitas` distribution, raw vs capped

2×2 subplot grid — rows = `Kuantitas` (raw) / `Kuantitas_capped`, columns = linear / `log1p` scale.
Same histogram style as eda.ipynb cell 20 (`bins=60`, `color=BLUE`), extended to a second row so the
effect of capping on the distribution's tail is visible (capping touches 8,507 of 1,340,034 rows —
a difference only visible on the log1p scale, not linear).

## Chart 2: `Kuantitas` by `Kategori Barang`, raw vs capped

1×2 subplot, boxplot per `Kategori Barang` sorted by descending median, log y-scale,
`showfliers=False` — same style as eda.ipynb cell 21, computed once for raw `Kuantitas` (left) and
once for `Kuantitas_capped` (right) so category-level shifts from capping are visible side by side.
Both subplots share the same y-axis limits for direct comparison.

## Chart 3: Daily total `Kuantitas`, Ramadan/Eid overlay

Line chart of daily total `Kuantitas` (raw) with 7-day rolling mean, Ramadan periods shaded,
Eid al-Fitr/Eid al-Adha dates marked with vertical lines — same style as eda.ipynb cell 33, reusing
`calendar_features.RAMADAN_PERIODS`/`EID_AL_FITR_DATES`/`EID_AL_ADHA_DATES` (module already imported
in `data-processing.ipynb` cell `db5a0791`). Confirms the seasonal pattern EDA found in the raw data
survives branch-dropping (section 2) and min-history filtering (section 3).

## Chart 4: Mean daily total by day-of-week (overall)

Bar chart, mean total `Kuantitas` per day-of-week, all branches/items combined — same style as
eda.ipynb cell 34. Reuses the `day_of_week` column already added by `calendar_features.add_calendar_features`
(section 4) instead of recomputing `_dow`.

## Chart 5: Mean daily total by day-of-week, per `Kategori Barang`

3×4 grid of small bar charts, one per `Kategori Barang`, each on its own y-scale — same style as
eda.ipynb cell 35. Uses the canonicalized `Kategori Barang` column (post `canonicalize_item_categories`),
so this also serves as an indirect visual check that category canonicalization didn't introduce
category-level seasonality artifacts.

## Markdown framing

A short intro markdown cell before the charts states: this section is a post-pipeline visual
sanity-check, not a re-run of `eda.ipynb`; some distributional/seasonal differences from EDA's raw
charts are expected and not bugs (dropped branches lacking an `outlets.csv` match, pairs dropped
by `MIN_HISTORY_DAYS`, category canonicalization, outlier capping) — output should still resemble
EDA's raw-data shape overall, not diverge structurally.

## Out of scope

- Reproducing eda.ipynb's other chart groups (data health bagian 2, temporal coverage bagian 4, item×outlet
  structure section 6, lead-time proxy section 7, month-seasonality bar, branch×day-of-week deviation heatmap,
  year-over-year consistency) — not requested for this pass.
- Moving any of these charts or the existing section 8 numeric QA into `utils/prepare_forecast_data.py`'s
  `main()` — charts are notebook-only by nature; the numeric-QA-in-script gap is already tracked
  separately in `docs/todolist-data-preprocessing.md` (🟡 "7 QA assertion cuma ada di notebook").
- Any change to pipeline logic (`utils/*.py`) — this is a notebook-only, visualization-only addition.
