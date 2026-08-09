# Visual QA Section for `data-processing.ipynb` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new visual-QA section to `notebook/data-processing.ipynb` that reproduces `eda.ipynb`'s distribution/outlier (§3) and seasonality (§5) charts, computed on the post-pipeline `featured` DataFrame instead of raw `dataset.csv`.

**Architecture:** Pure notebook addition — two groups of new cells inserted between the existing §8 QA cells and the `export_featured` call, using `matplotlib` with `eda.ipynb`'s own palette/style constants (redefined locally since `data-processing.ipynb` doesn't currently import `matplotlib`). No changes to any `utils/*.py` module.

**Tech Stack:** Jupyter notebook, `matplotlib`, `numpy`, `pandas` (all already in `requirements.txt`); reuses `calendar_features` (already imported in the notebook).

## Global Constraints

- Notebook-only change — no edits to `utils/*.py` (per spec's "Out of scope").
- New cells go after cell id `dda7ce32` (`featured.info()`) and before cell id `ac0b1f5a` (`export_featured(featured)`), so charts run on fully-QA'd data before export.
- Reuse `eda.ipynb` cell 5's exact palette/style constants: `BLUE = "#2a78d6"`, `BLUE_DARK = "#173f73"`, and the `plt.rcParams.update({...})` block (spines off, light grid, `figure.dpi=100`, `font.size=10`).
- Reuse existing notebook state — do not recompute `_dow`; use the `day_of_week` column already added by `calendar_features.add_calendar_features`.
- Verify with `jupyter nbconvert --to notebook --execute --inplace --allow-errors notebook/data-processing.ipynb` (per `CLAUDE.md`'s documented command and the `--allow-errors` caveat for the not-yet-reconfirmed KY011 assertion), then confirm programmatically that none of the *new* cells produced an error output and each chart cell produced an `image/png` output.
- Reference source: `docs/superpowers/specs/2026-08-08-visual-qa-notebook-design.md`.

---

### Task 1: Distribution/outlier visual QA cells (raw vs `Kuantitas_capped`)

**Files:**
- Modify: `notebook/data-processing.ipynb` — insert 4 new cells after cell id `dda7ce32`, before cell id `ac0b1f5a`.

**Interfaces:**
- Consumes: notebook-global `featured` DataFrame (columns `Kuantitas`, `Kuantitas_capped`, `Kategori Barang`) already built by earlier cells; no new functions.
- Produces: notebook-global `plt`, `np`, `BLUE`, `BLUE_DARK` names, in scope for Task 2's cells (inserted immediately after this task's cells, same kernel run).

- [ ] **Step 1: Insert the setup cell**

Insert a new **code** cell immediately after cell id `dda7ce32`:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

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

- [ ] **Step 2: Insert the section intro markdown cell**

Insert a new **markdown** cell right after the Step 1 cell:

```markdown
### 9. Visual QA — distribution/outlier & seasonality (vs `eda.ipynb`)
Section ini adalah pengecekan visual pasca-pipeline terhadap `featured`, bukan re-run `eda.ipynb`
(yang mengeksplorasi `dataset.csv` mentah sebelum drop cabang/filter min-history/capping/canonicalize
kategori). Beberapa perbedaan bentuk distribusi/musiman dari `eda.ipynb` **diperkirakan** dan bukan
bug: cabang yang di-drop karena tidak match `outlets.csv` (§2), pair yang gagal `MIN_HISTORY_DAYS`
(§3), kategori yang di-canonicalize ke versi terbaru, dan efek outlier-capping (§5) — tapi bentuk
keseluruhannya harus tetap menyerupai `eda.ipynb`, bukan menyimpang secara struktural.
```

- [ ] **Step 3: Insert the distribution histogram cell (raw vs capped, linear vs log1p)**

Insert a new **code** cell right after the Step 2 cell:

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(featured["Kuantitas"], bins=60, color=BLUE)
axes[0, 0].set_title("Kuantitas (raw) — linear scale")
axes[0, 0].set_xlabel("Kuantitas")
axes[0, 0].set_ylabel("Row count")

axes[0, 1].hist(np.log1p(featured["Kuantitas"]), bins=60, color=BLUE)
axes[0, 1].set_title("Kuantitas (raw) — log1p scale")
axes[0, 1].set_xlabel("log1p(Kuantitas)")
axes[0, 1].set_ylabel("Row count")

axes[1, 0].hist(featured["Kuantitas_capped"], bins=60, color=BLUE_DARK)
axes[1, 0].set_title("Kuantitas_capped — linear scale")
axes[1, 0].set_xlabel("Kuantitas_capped")
axes[1, 0].set_ylabel("Row count")

axes[1, 1].hist(np.log1p(featured["Kuantitas_capped"]), bins=60, color=BLUE_DARK)
axes[1, 1].set_title("Kuantitas_capped — log1p scale")
axes[1, 1].set_xlabel("log1p(Kuantitas_capped)")
axes[1, 1].set_ylabel("Row count")

fig.suptitle("Kuantitas distribution: raw vs capped")
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Insert the boxplot-by-category cell (raw vs capped, shared y-axis)**

Insert a new **code** cell right after the Step 3 cell:

```python
order = (
    featured.groupby("Kategori Barang", observed=True)["Kuantitas"]
    .median()
    .sort_values(ascending=False)
    .index
)
raw_by_cat = [featured.loc[featured["Kategori Barang"] == cat, "Kuantitas"] for cat in order]
capped_by_cat = [featured.loc[featured["Kategori Barang"] == cat, "Kuantitas_capped"] for cat in order]

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

box_raw = axes[0].boxplot(raw_by_cat, tick_labels=list(order), patch_artist=True, showfliers=False)
for patch in box_raw["boxes"]:
    patch.set_facecolor(BLUE)
    patch.set_alpha(0.6)
axes[0].set_yscale("log")
axes[0].set_ylabel("Kuantitas (log scale)")
axes[0].set_title("Raw Kuantitas by Kategori Barang")
for label in axes[0].get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

box_capped = axes[1].boxplot(capped_by_cat, tick_labels=list(order), patch_artist=True, showfliers=False)
for patch in box_capped["boxes"]:
    patch.set_facecolor(BLUE_DARK)
    patch.set_alpha(0.6)
axes[1].set_title("Kuantitas_capped by Kategori Barang")
for label in axes[1].get_xticklabels():
    label.set_rotation(45)
    label.set_ha("right")

fig.suptitle("Kuantitas by Kategori Barang: raw vs capped (sorted by raw median)")
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Execute the notebook and verify the new cells**

Run:
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --allow-errors notebook/data-processing.ipynb
```
Expected: command exits 0 (this may take several minutes — the pipeline reprocesses ~1.34M rows; allow up to 10 minutes).

Then verify no error output on the new cells and that both chart cells rendered an image:
```bash
.venv/bin/python3 - <<'EOF'
import json
nb = json.load(open("notebook/data-processing.ipynb"))
markers = [
    "Kuantitas distribution: raw vs capped",
    "Kuantitas by Kategori Barang: raw vs capped",
]
found = 0
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if any(m in src for m in markers):
        found += 1
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        images = [o for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
        assert not errors, f"cell errored: {errors}"
        assert images, "no image/png output produced"
print(f"OK — {found}/2 chart cells verified, no errors, images present")
EOF
```
Expected output: `OK — 2/2 chart cells verified, no errors, images present`

- [ ] **Step 6: Commit**

```bash
git add notebook/data-processing.ipynb
git commit -m "$(cat <<'EOF'
feat: add distribution/outlier visual QA to data-processing notebook

Reproduces eda.ipynb's raw-Kuantitas histogram/boxplot style, computed on
featured (raw vs Kuantitas_capped) to visually confirm outlier-capping
behavior alongside the existing numeric QA asserts.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Seasonality visual QA cells

**Files:**
- Modify: `notebook/data-processing.ipynb` — insert 3 new cells after Task 1's last inserted cell (the boxplot cell from Step 4), before cell id `ac0b1f5a`.

**Interfaces:**
- Consumes: notebook-global `featured`, `plt`, `np`, `BLUE`, `BLUE_DARK` from Task 1; notebook-global `calendar_features` module (already imported in cell `db5a0791`) and its `RAMADAN_PERIODS`, `EID_AL_FITR_DATES`, `EID_AL_ADHA_DATES` dict attributes; `featured["day_of_week"]` column (already added by `calendar_features.add_calendar_features` in §4).
- Produces: nothing consumed by later tasks (this is the last content task).

- [ ] **Step 1: Insert the daily-total time series cell (Ramadan/Eid overlay)**

Insert a new **code** cell right after Task 1's boxplot cell:

```python
daily_total = featured.groupby("Tanggal")["Kuantitas"].sum()
rolling_mean = daily_total.rolling(7, min_periods=1).mean()

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(daily_total.index, daily_total.values, color=BLUE, alpha=0.35, linewidth=0.8, label="Daily total")
ax.plot(rolling_mean.index, rolling_mean.values, color=BLUE_DARK, linewidth=1.5, label="7-day rolling mean")

for _, (start, end) in calendar_features.RAMADAN_PERIODS.items():
    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#eda100", alpha=0.15)

for _, eid_date in calendar_features.EID_AL_FITR_DATES.items():
    ax.axvline(pd.Timestamp(eid_date), color="#1baf7a", linestyle="--", linewidth=1)

for _, eid_date in calendar_features.EID_AL_ADHA_DATES.items():
    ax.axvline(pd.Timestamp(eid_date), color="#eb6834", linestyle="--", linewidth=1)

ax.set_title("Featured daily total Kuantitas (Ramadan shaded, Eid dates marked)")
ax.set_xlabel("Date")
ax.set_ylabel("Total Kuantitas")
ax.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Insert the day-of-week (overall) bar chart cell**

Insert a new **code** cell right after the Step 1 cell:

```python
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

dow_daily_total = featured.groupby(["Tanggal", "day_of_week"])["Kuantitas"].sum().reset_index()
dow_mean = dow_daily_total.groupby("day_of_week")["Kuantitas"].mean().reindex(range(7))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(dow_labels, dow_mean.values, color=BLUE)
ax.set_title("Mean daily total Kuantitas by day of week (featured, all branches/items)")
ax.set_ylabel("Mean total Kuantitas")
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Insert the day-of-week per-category grid cell**

Insert a new **code** cell right after the Step 2 cell:

```python
categories = sorted(str(c) for c in featured["Kategori Barang"].unique())

fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()

for i, cat in enumerate(categories):
    sub = featured[featured["Kategori Barang"] == cat]
    sub_daily = sub.groupby(["Tanggal", "day_of_week"])["Kuantitas"].sum().reset_index()
    sub_dow_mean = sub_daily.groupby("day_of_week")["Kuantitas"].mean().reindex(range(7)).fillna(0)
    axes[i].bar(dow_labels, sub_dow_mean.values, color=BLUE)
    axes[i].set_title(cat, fontsize=9)
    axes[i].tick_params(axis="x", labelsize=7)

for j in range(len(categories), len(axes)):
    axes[j].axis("off")

fig.suptitle("Day-of-week pattern by Kategori Barang, post-pipeline (each panel own scale)")
plt.tight_layout()
plt.show()
```

Note: `featured["Kategori Barang"]` currently has 8 distinct values (verified against `dataset/model_ready/featured.parquet`), so the 3×4 grid has 4 unused panels — Step 3's loop already turns those off via `axes[j].axis("off")`.

- [ ] **Step 4: Execute the notebook and verify the new cells**

Run:
```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --allow-errors notebook/data-processing.ipynb
```
Expected: exits 0 (allow up to 10 minutes).

Then verify:
```bash
.venv/bin/python3 - <<'EOF'
import json
nb = json.load(open("notebook/data-processing.ipynb"))
markers = [
    "Featured daily total Kuantitas (Ramadan shaded, Eid dates marked)",
    "Mean daily total Kuantitas by day of week (featured, all branches/items)",
    "Day-of-week pattern by Kategori Barang, post-pipeline",
]
found = 0
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if any(m in src for m in markers):
        found += 1
        errors = [o for o in cell.get("outputs", []) if o.get("output_type") == "error"]
        images = [o for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
        assert not errors, f"cell errored: {errors}"
        assert images, "no image/png output produced"
print(f"OK — {found}/3 chart cells verified, no errors, images present")
EOF
```
Expected output: `OK — 3/3 chart cells verified, no errors, images present`

Also confirm no cell anywhere in the notebook (old or new) produced an error, since this is the final full run before committing:
```bash
.venv/bin/python3 - <<'EOF'
import json
nb = json.load(open("notebook/data-processing.ipynb"))
error_cells = [i for i, c in enumerate(nb["cells"]) if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
assert not error_cells, f"cells with errors (0-indexed): {error_cells}"
print("OK — no error outputs anywhere in the notebook")
EOF
```
Expected output: `OK — no error outputs anywhere in the notebook`

- [ ] **Step 5: Commit**

```bash
git add notebook/data-processing.ipynb
git commit -m "$(cat <<'EOF'
feat: add seasonality visual QA to data-processing notebook

Reproduces eda.ipynb's daily-total (Ramadan/Eid overlay) and day-of-week
seasonality charts, computed on featured to confirm the seasonal pattern
EDA found in raw data survives branch-dropping and min-history filtering.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
