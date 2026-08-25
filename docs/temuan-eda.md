# Temuan EDA — `dataset/csv/dataset.csv`

Dokumen ini memuat narasi dan kesimpulan dari `notebook/eda.ipynb`. Notebook itu sendiri
hanya berisi kode dan grafik: sejak refactor 2026-08-26 sel markdown di notebook dibatasi
pada judul bagian saja (lihat "Notebook convention" di `CLAUDE.md`), sehingga narasi
tingkat-dokumen — konteks bisnis, open questions, dan checklist pra-modeling — hidup di sini.

Angka-angka di bawah berasal dari run EDA atas `dataset/csv/dataset.csv` per 2026-08-06.
Jalankan ulang `notebook/eda.ipynb` setiap kali `dataset.csv` diregenerasi, lalu perbarui
tabel ini kalau ada yang berubah.

This notebook is a **fully independent** EDA pass over the cleaned, aggregated transaction
log produced by `notebook/merge_and_aggregate.ipynb` (`merge_dataset.py` + `aggregate_dataset.py`
&rarr; `dataset/dataset.csv`). It does not modify, replace, or depend on
`notebook/data-processing.ipynb` — some duplication of basic checks (missing values, unique
counts, etc.) between the two notebooks is expected and accepted.

## Business context

The goal is demand forecasting for every item at every outlet, horizon &le; 4 days. The central
SCM team currently ships to **Region 1** outlets every **Monday & Thursday**, and to **Region 2**
outlets every **Tuesday & Friday** — so the eventual model needs to predict *cumulative* demand
over the lead-time window until the next shipment (a 3-day or 4-day window, depending on region
and shipment day).

**Important constraint:** as of this notebook, there is no Region 1 / Region 2 (or any
delivery-schedule) mapping anywhere in the data (`dataset/outlets.csv` only has
`Kecamatan`/`Kota`). This notebook does **not** guess at or invent that mapping — the
day-of-week and lead-time-window analyses below are deliberately generic/non-region-segmented
proxies. The gap itself is called out explicitly in the final "Open Questions / Data Gaps"
section.

## 9. Conclusion & Pre-Modeling Verification Checklist

One row per finding: the conclusion from this EDA pass, and — if anything — what still needs
a data-owner/SCM decision before this dataset feeds `normalize_items.py` / `build_panel.py` /
`prepare_forecast_data.py`.

| # | Finding / Conclusion | Verify before modeling |
|---|---|---|
| 1 | Data quality is clean: 693,563 rows, 2024-01-01–2025-12-31, 0 missing values, 0 duplicates. | None — ready as-is. |
| 2 | `Kuantitas` is heavily right-skewed (median 5, mean ~30.4, max 5,250) and, as of the 2026-08-06 source refresh, genuinely fractional (dtype float64, e.g. `5.8`, `92.5`) rather than integer-only — confirmed intentional by the data owner, not an export artifact. | Eyeball the top-20 outlier rows (§3) for unit-multiplier / data-entry errors before treating them as genuine bulk orders. Make sure downstream preprocessing/model code treats Kuantitas as continuous, not integer-count. |
| 3 | Branch date-completeness is strong: 66/67 branches ≥95%; only KY056 (Kebuli Yaman Tigaraksa) is below, at 92.3%. | Ask data owner whether KY056's gap is a reporting gap, a temporary closure, or a newer outlet. |
| 4 | Demand is highly intermittent at the item-branch-pair level (median 64% zero-demand days; 57.1% of pairs >50% zero). | None — this is a modeling-method decision (avoid pure ARIMA-style continuous-demand assumptions), not a data question. |
| 5 | Volume is concentrated: the top 6.6% of item-branch pairs drive 80% of total volume. | None — informs evaluation weighting, not a data question. |
| 6 | 78.3% of item-branch pairs (3,040/3,882) clear the existing 60-day `MIN_HISTORY_DAYS` threshold; ~22% would be dropped under current rules. | Data owner to review the failing pairs — new SKU, new outlet, or reporting gap? — before deciding to drop them or relax the threshold. |
| 7 | `xxx.`-prefixed items can't always be auto-merged into their non-prefixed counterpart — e.g. `xxx.FGS.00069` "Cendol Pandan" vs `FGS`-coded "Cendol" have different names, so normalization can't assume they're the same SKU. | Data owner decision required: drop, keep as a separate SKU, or manually confirm the merge. |
| 8 | No negative `Kuantitas` rows found in the current `dataset/dataset.csv` — the known KY011 2024-02-29 anomaly is resolved in this run, using the current `dataset/csv/feb-24.csv` (already comma-decimal formatted, consistent with the other four source months). | Not yet permanently confirmed by the data owner — rerun this check every time `dataset.csv` is regenerated. |
| 9 | **(New check)** Every `Kode Barang` maps to exactly one `Satuan` — 0/109 SKUs have more than one unit of measure. | None — confirmed consistent, safe to aggregate `Kuantitas` per SKU. |
| 10 | **(New check)** 27/109 SKUs (24.8%) appear under more than one `Kategori Barang` over time — e.g. `Minuman` → `Minuman - FG`, `Barang Semi FG (WIP-2)` → `Barang Jadi (FG)`, `Snack` → `Snack (FG)`. The consistent pairing pattern looks like a mid-2024 category-taxonomy rename rather than random data-entry noise. | Confirm with the data owner whether this is an intentional taxonomy rename, and if so decide whether `normalize_items.py` should use each SKU's most-recent category (current pipeline assumes one fixed category per SKU) or treat category as time-varying. |
| 11 | No Region 1 (Mon/Thu) / Region 2 (Tue/Fri) delivery-schedule mapping exists in `dataset/outlets.csv` — it only has `Kecamatan`/`Kota`. This blocks region-segmented versions of the day-of-week (§5) and lead-time-window (§7) analyses, which are generic proxies only. | Ask the SCM team for a branch → region (or branch → delivery-day) column on `dataset/outlets.csv`, joined the same way `outlet_features.py` already joins `has_shopee`/`has_gofood`/`has_grabfood`. |
| 12 | Lead-time proxy: a 4-day rolling window is mildly more predictable than a 3-day window (median CV ~1.12 vs ~1.25). | This is a generic, non-region-segmented proxy — don't finalize until #11's region mapping lands. |
| 13 | Clear weekly and Ramadan/Eid-linked seasonality exist overall and per category (§5). | Confirm `calendar_features.py`'s `RAMADAN_PERIODS`/`EID_AL_FITR_DATES`/`EID_AL_ADHA_DATES` fully cover 2024-2025 via `check_year_coverage`. |

**Priority order:** rows 7, 10, and 11 are the biggest blockers — each can silently distort
`normalize_items.py` or `build_panel.py` output if skipped, and none can be resolved from the
data alone.

## Open Questions / Data Gaps

Bagian §14 di notebook (`## 14. Open questions — cek data`) hanya menjalankan dua
pemeriksaan ringan: kolom apa saja yang tersedia di `dataset/outlets.csv` (untuk menunjukkan
mapping region belum ada — baris 11 di tabel), dan apakah masih ada `Kuantitas` negatif
(baris 8). Daftar pertanyaan terbukanya sendiri adalah baris 3, 6, 7, 10, dan 11 pada tabel
di atas.
