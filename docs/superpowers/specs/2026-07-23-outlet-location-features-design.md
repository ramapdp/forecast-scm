# Outlet location & delivery-channel features — design

## Purpose

`dataset/outlets.csv` (an outlet master file: `Nama Outlet`, `Alamat`, `Kecamatan`, `Kota`, `has_shopee`, `has_gofood`, `has_grabfood`) has become available. This fills a gap explicitly flagged in `docs/superpowers/specs/2026-07-21-forecast-data-prep-design.md`'s "Outlet (branch) characteristic features" section, which noted no external outlet master data was available and derived branch features purely from transaction history. This spec adds `kota`, `has_shopee`, `has_gofood`, and `has_grabfood` as new static, per-branch features in that same pipeline stage.

## Data quality found

Comparing `outlets.csv` (62 rows) against `dataset.csv`'s 67 distinct `Nama Cabang` values originally surfaced two join problems. The first is now mostly resolved at the source; the second remains open:

1. **Name mismatch (mostly resolved)**: `outlets.csv` originally used bare `Kebuli Yaman <Name>` while `dataset.csv` uses `KY0NN - Kebuli Yaman <Name>` (sometimes with a `(Pusat)` suffix), and several names differed beyond that prefix (e.g. `Kuta Bumi (PUSAT)` vs `Kutabumi (Pusat)`, `Rawalumbu` vs `Rawalumbu (Bekasi)`, `Cimanggu` vs `Cimanggu Bogor`, `TOD M1` vs `TOD M1 Bandara`, `Tangcity` vs `TangCity Mall`). The data owner subsequently updated the source `outlets.json` (regenerated into `dataset/outlets.csv` via `sync_outlets.py`) to prefix most `Nama Outlet` values with their `KY0NN - ` code, so `Nama Outlet` now matches `Nama Cabang` byte-for-byte for 52 of the 67 branches. Two known duplicates remain and are handled via `outlet_name_overrides.csv` rather than relying on fuzzy matching: `KY011`/`KY069` both correspond to the single "Bekasi Galaxy" outlet row, and `TOD M1 Bandara` (a no-prefix `Nama Cabang` row, distinct from `KY051 - kebuli Yaman TOD M1 Bandara`) is the same physical outlet recorded under two different branch strings — `TOD M1 Bandara` is the outlet's old/legacy name, confirmed by the data owner (2026-07-26). `outlets.csv` also still has outlets with no transactions at all (newer/unopened sites), and 10 `dataset.csv` branches (`Tambun`, `Condet`, `Antapani`, `Aryana Karawaci`, `Ciomas`, `Bantarjati Bogor`, `Ciputat Timur`, `Dukuh Zamrud`, `Citayam`, `Bintara`) plus the already-excluded `Kebab Saudagar - Kutabumi` have no `outlets.csv` counterpart at all. These branches are treated as no longer existing and their rows are dropped from the dataset entirely before the panel is built (see "Branch existence filtering" below) — an update from this spec's original decision to keep them with `kota="Unknown"`.
2. **Bad/ambiguous `Kota` values (open)**: 8 outlets still have a `Kota` value that's either a province (`Jawa Barat`, `Banten` — 4 outlets: `Kutabumi (Pusat)`, `Bogor Ciampea`, `Jatimakmur`, `Kota Harapan Indah`) or a bare regency/city name that's ambiguous between the `Kota` and `Kabupaten` variant (`Bogor`, `Tangerang`, `Bekasi` — 4 outlets: `Cibinong`, `Cikeas`, `Sepatan`, `Rawalumbu (Bekasi)`) — a source data-entry gap the `Nama Outlet` fix didn't touch. `outlet_name_overrides.csv` carries best-guess `Kota Override` values for all eight, derived from each outlet's `Kecamatan` (e.g. Kecamatan Jatiuwung → Kota Tangerang, Kecamatan Cibinong → Kabupaten Bogor), pending data-owner confirmation. Two clean-cut bare values, `Cilegon` and `Pandeglang`, are left as-is: neither has a `Kota`/`Kabupaten` counterpart to be ambiguous with.

## Module: `outlet_features.py`

New module, same shape as `calendar_features.py`: pure functions, its own `test_outlet_features.py`, wired into `prepare_forecast_data.py`'s `main()` as an added feature-engineering step. Not implemented as notebook-only logic — consistent with every other pipeline stage (`normalize_items.py`, `build_panel.py`, `calendar_features.py`), each of which has a dedicated tested module even though the design narrative for the overall pipeline describes it as notebook-orchestrated.

Output: one row per `Nama Cabang` carrying `kota`, `has_shopee`, `has_gofood`, `has_grabfood`, joined onto the dense daily panel by branch (static per branch, no date dependency, joined identically onto both train and test).

## `dataset/outlet_name_overrides.csv`

New file, columns: `Nama Cabang`, `Nama Outlet`, `Kota Override` (optional, blank except the 8 bad/ambiguous `Kota` rows and the 2 known duplicates — see below).

Matching logic, in order:

1. **Override lookup first** — if `Nama Cabang` appears in the override file, use its paired `Nama Outlet` to pull that row from `outlets.csv`. If `Kota Override` is non-blank, it wins over `outlets.csv`'s own `Kota` value for that row.
2. **Automatic fallback** — for any `Nama Cabang` not in the override file, strip the `KY0NN - ` prefix and any trailing `(Pusat)`/parenthetical, then substring/case-insensitive-match against `outlets.csv`'s `Nama Outlet`.
3. **Unmatched** — if neither step finds a row, `build_outlet_features` returns `kota = "Unknown"` and all three channel flags `NaN` for that branch. This remains the function's contract for direct/unit-test calls, but it's no longer a reachable path in the full pipeline: unmatched branches are dropped from the dataset entirely before `build_outlet_features` ever sees them (see "Branch existence filtering" below).

The override file (`outlet_name_overrides.csv`) currently has 10 rows: the 8 `Kota Override` corrections from "Data quality found" above, plus 2 rows resolving the known duplicates (`KY069 - Kebuli Yaman Bekasi Galaxy` → `KY011 - Kebuli Yaman Bekasi Galaxy`; `TOD M1 Bandara` → `KY051 - kebuli Yaman TOD M1 Bandara`). It was drafted with best-guess mappings (derived from `Kecamatan` for the `Kota` corrections) rather than confirmed live with the data owner. The `TOD M1 Bandara` → `KY051` row is now confirmed by the data owner (2026-07-26); the `KY069`/`KY011` Bekasi Galaxy duplicate and the 8 `Kota Override` corrections remain best-guess and are still flagged for their review before being treated as final, per the matching logic's reliance on this file being authoritative.

## Branch existence filtering

`outlets.csv` (via `outlets.json`/`sync_outlets.py`) is treated as the source of truth for which branches still exist. `outlet_features.filter_matched_branches(df, outlets_df, overrides_df)` drops every row whose `Nama Cabang` can't be resolved by either matching step above (override or automatic fallback) — reusing the same `match_branch_to_outlet` logic, so "matched" means exactly the same thing here as it does for the `kota`/channel join.

It's called in `prepare_forecast_data.main()` immediately after `normalize_items.load_and_normalize()`, before `build_panel.build_dense_panel()` — so dropped branches never reach the dense panel, lag/rolling features, branch stats, or the model-ready parquet files at all. `outlets_df`/`overrides_df` are loaded once at the top of `main()` and reused for both this filter and the later `apply_outlet_features` join, so the two stages can never disagree about what counts as a match.

**Residual risk**: this can't distinguish "branch closed" from "branch is new and just hasn't been added to `outlets.csv` yet" — both look identical (no match). The QA check below prints the dropped-branch list on every run specifically so an unexpected new name is easy to spot before it silently disappears from the data.

## Kota normalization

After any `Kota Override` is applied, remaining `Kota` values get light string cleanup only — trim whitespace, no semantic merging:

- The `Kota `/`Kabupaten ` prefix, where already present in the source data (e.g. `Kota Bekasi`, `Kabupaten Bogor`), is **kept, not stripped** — city vs. regency areas can have genuinely different demand patterns, so collapsing `Kota Bogor` and `Kabupaten Bogor` down to a bare `Bogor` would destroy real signal, not just a naming quirk.
- Jakarta's `Jakarta Timur`/`Utara`/`Selatan`/`Barat` pass through unchanged (already city-level, no `Kota`/`Kabupaten` prefix in the source).
- Ambiguous bare values (`Bogor`, `Tangerang`, `Bekasi` — unclear whether these mean the city or the regency) are **not** guessed algorithmically; they're resolved through `Kota Override`, same as the 4 outright-bad province rows. Bare values with no `Kota`/`Kabupaten` counterpart to be ambiguous with (`Cilegon`, `Pandeglang`) pass through unchanged.

`kota` is kept as a plain categorical column, not encoded — same treatment as `Kode Barang`/`Nama Cabang`/`Kategori Barang` in the existing design. Actual encoding (one-hot / label / embedding) stays a modeling-stage concern.

`Kecamatan` is not used as a feature: 56 distinct values across 62 outlets is near-1:1 with outlet identity, too sparse to let a global model pooling series across branches (per the existing design's modeling strategy) generalize across it.

## Channel features

`has_shopee`, `has_gofood`, `has_grabfood` are kept as three separate boolean features (not collapsed into a channel count), preserving platform-specific signal — e.g. GrabFood-only demand may behave differently from Shopee-only demand.

A 5th derived feature, `can_order_online`, is added alongside these three (not instead of them): `True` if any of `has_shopee`/`has_gofood`/`has_grabfood` is `"Yes"`, `False` if all three are `"No"`, and unknown (`NaN`) if the branch has no outlet match at all — mirroring the `kota="Unknown"` treatment for unmatched branches rather than guessing. This gives a simple yes/no view of online-order availability without discarding the per-platform detail the three flags carry. As with `kota="Unknown"`, this `NaN` path is a defensive fallback in `build_outlet_features` that the full pipeline no longer reaches, since unmatched branches are filtered out beforehand.

## Leakage treatment

`kota`, `has_shopee`, `has_gofood`, `has_grabfood` are static exogenous master data — not derived from `Kuantitas` history. Unlike `branch_avg_daily_qty`/`branch_volume_tier`/`branch_demand_cv` (which require a train-only-freeze rule per the existing design), these features join directly onto both train and test rows with no leakage risk.

## Bias / column-count considerations

Adding these features brings the model-ready panel from 44 to 49 columns. This isn't a meaningful overfitting risk on its own: no high-cardinality identifier (`Kode Barang`, `Nama Cabang`, `Kategori Barang`, and now `kota`) is one-hot encoded — all stay as single plain categorical columns with encoding deferred to the modeling stage — so column count isn't inflated by dummy variables, and the row:column ratio is large enough that tree-based models (the pipeline's target) aren't at real risk of overfitting purely from feature count. The one thing actually worth watching is multicollinearity among `lag_*`/`roll_*` features, which are all derived from the same `Kuantitas` history and correlate heavily with each other — harmless for tree ensembles, relevant only if a linear/regularized model is ever fit. The ~15% of branches with no `outlets.csv` match no longer contribute a "no outlet master data" signal to worry about, since those branches (and their ~10% of rows) are now dropped from the dataset entirely rather than kept as `kota="Unknown"` — see "Branch existence filtering" above; the trade-off is a modest reduction in training data volume in exchange for not modeling branches that no longer exist.

## QA checks (added to the existing in-notebook QA section)

- Count and list of branches dropped for having no `outlets.csv` match, printed on every run — expect exactly the 10 known branches listed in "Data quality found" above. Investigate immediately if the list ever changes, since that could mean either a genuinely new/active branch was dropped (see the residual risk noted in "Branch existence filtering") or a matching regression.
- No branch remains with `kota == "Unknown"` in `featured` — since unmatched branches are now dropped upstream, this should always hold; a failure here signals a bug in the filtering step, not a normal data-quality finding.
- Every `Nama Cabang` in the panel maps to exactly one outlet row (no fan-out from the substring match producing duplicate joins).
- Distribution of `can_order_online` (True/False counts; no `Unknown` expected anymore) as a sanity check.

## Out of scope

- Any outlet attribute beyond `Kota`/`Kecamatan`/channel flags (e.g. size/format, opening date) — not present in `outlets.csv` today.
- Encoding strategy for `kota` — deferred to the modeling stage, same as other categorical identifiers.
- Retroactively resolving `outlets.csv` rows with no matching `Nama Cabang` at all (unopened/no-transaction outlets) — irrelevant to this pipeline since they contribute no rows to the panel.
