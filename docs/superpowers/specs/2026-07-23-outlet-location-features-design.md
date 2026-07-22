# Outlet location & delivery-channel features — design

## Purpose

`dataset/outlets.csv` (an outlet master file: `Nama Outlet`, `Alamat`, `Kecamatan`, `Kota`, `has_shopee`, `has_gofood`, `has_grabfood`) has become available. This fills a gap explicitly flagged in `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`'s "Outlet (branch) characteristic features" section, which noted no external outlet master data was available and derived branch features purely from transaction history. This spec adds `kota`, `has_shopee`, `has_gofood`, and `has_grabfood` as new static, per-branch features in that same pipeline stage.

## Data quality found

Comparing `outlets.csv` (62 rows) against `dataset.csv`'s 67 distinct `Nama Cabang` values surfaced two join problems:

1. **Name mismatch**: `dataset.csv` uses `KY0NN - Kebuli Yaman <Name>` (sometimes with a `(Pusat)` suffix); `outlets.csv` uses bare `Kebuli Yaman <Name>`, and several names differ beyond that prefix (e.g. `Kuta Bumi (PUSAT)` vs `Kutabumi (Pusat)`, `Rawalumbu` vs `Rawalumbu (Bekasi)`, `Cimanggu` vs `Cimanggu Bogor`, `TOD M1` vs `TOD M1 Bandara`, `Tangcity` vs `TangCity Mall`). `outlets.csv` also has outlets with no transactions at all (newer/unopened sites), and `dataset.csv` has branches (`Kebab Saudagar - Kutabumi`, `TOD M1 Bandara`) with no obvious `outlets.csv` counterpart by name. `KY011` and `KY069` both render as "Bekasi Galaxy" but only one outlet row exists for it.
2. **Bad `Kota` values**: 4 outlets (`Bogor Ciampea`, `Kuta Bumi (PUSAT)`, `Jatimakmur`, `Kota Harapan Indah`) have `Kota` filled with a province name (`Jawa Barat`, `Banten`) instead of an actual city — a source data-entry gap.

## Module: `outlet_features.py`

New module, same shape as `calendar_features.py`: pure functions, its own `test_outlet_features.py`, wired into `prepare_forecast_data.py`'s `main()` as an added feature-engineering step. Not implemented as notebook-only logic — consistent with every other pipeline stage (`normalize_items.py`, `build_panel.py`, `calendar_features.py`), each of which has a dedicated tested module even though the design narrative for the overall pipeline describes it as notebook-orchestrated.

Output: one row per `Nama Cabang` carrying `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, joined onto the dense daily panel by branch (static per branch, no date dependency, joined identically onto both train and test).

## `dataset/outlet_name_overrides.csv`

New file, columns: `Nama Cabang`, `Nama Outlet`, `Kota Override` (optional, blank except the 4 known bad rows and any ambiguous `Kota` values — see below).

Matching logic, in order:

1. **Override lookup first** — if `Nama Cabang` appears in the override file, use its paired `Nama Outlet` to pull that row from `outlets.csv`. If `Kota Override` is non-blank, it wins over `outlets.csv`'s own `Kota` value for that row.
2. **Automatic fallback** — for any `Nama Cabang` not in the override file, strip the `KY0NN - ` prefix and any trailing `(Pusat)`/parenthetical, then substring/case-insensitive-match against `outlets.csv`'s `Nama Outlet`.
3. **Unmatched** — if neither step finds a row, the branch gets `kota = "Unknown"` and all three channel flags `NaN`. Branches are never dropped for lacking an outlet match — consistent with keeping every branch that passes the existing minimum-history filter.

The override file is where the specific fixes get encoded: `Kebab Saudagar - Kutabumi`, `TOD M1 Bandara`, `Rawalumbu (Bekasi)` → `Rawalumbu`, `Cimanggu Bogor` → `Cimanggu`, `TangCity Mall` → `Tangcity`, `Kuta Bumi (PUSAT)`, the `KY011`/`KY069` Bekasi Galaxy duplicate, and the 4 province-only `Kota` corrections. It is populated by the data owner (not derived automatically), since only they can confirm these correspondences.

## Kota normalization

After any `Kota Override` is applied, remaining `Kota` values get light string cleanup only — no semantic merging:

- Strip leading `Kota `/`Kabupaten ` and trim whitespace, but **keep the Kota/Kabupaten distinction** (e.g. `Kota Bogor` and `Kabupaten Bogor` remain two different values — city vs. regency areas can have genuinely different demand patterns, not just a naming quirk).
- Jakarta's `Jakarta Timur`/`Utara`/`Selatan`/`Barat` pass through unchanged (already city-level).
- Ambiguous bare values (`Bogor`, `Tangerang`, `Bekasi` — unclear whether these mean the city or the regency) are **not** guessed algorithmically; they're resolved through `Kota Override`, same as the 4 known-bad province rows.

`kota` is kept as a plain categorical column, not encoded — same treatment as `Kode Barang`/`Nama Cabang`/`Kategori Barang` in the existing design. Actual encoding (one-hot / label / embedding) stays a modeling-stage concern.

`Kecamatan` is not used as a feature: 56 distinct values across 62 outlets is near-1:1 with outlet identity, too sparse to let a global model pooling series across branches (per the existing design's modeling strategy) generalize across it.

## Channel features

`has_shopee`, `has_gofood`, `has_grabfood` are kept as three separate boolean features (not collapsed into a channel count), preserving platform-specific signal — e.g. GrabFood-only demand may behave differently from Shopee-only demand.

## Leakage treatment

`kota`, `has_shopee`, `has_gofood`, `has_grabfood` are static exogenous master data — not derived from `Kuantitas` history. Unlike `branch_avg_daily_qty`/`branch_volume_tier`/`branch_demand_cv` (which require a train-only-freeze rule per the existing design), these features join directly onto both train and test rows with no leakage risk.

## QA checks (added to the existing in-notebook QA section)

- Count of branches still `kota == "Unknown"` after overrides are applied — expect 0; investigate any that remain.
- Every `Nama Cabang` in the panel maps to exactly one outlet row (no fan-out from the substring match producing duplicate joins).

## Out of scope

- Any outlet attribute beyond `Kota`/`Kecamatan`/channel flags (e.g. size/format, opening date) — not present in `outlets.csv` today.
- Encoding strategy for `kota` — deferred to the modeling stage, same as other categorical identifiers.
- Retroactively resolving `outlets.csv` rows with no matching `Nama Cabang` at all (unopened/no-transaction outlets) — irrelevant to this pipeline since they contribute no rows to the panel.
