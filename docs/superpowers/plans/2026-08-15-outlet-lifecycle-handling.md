# Outlet Lifecycle Handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `build_dense_panel` from fabricating zero-demand rows for days a branch was closed, by segmenting each pair's daily panel on owner-confirmed closure intervals.

**Architecture:** A new `dataset/outlet_closures.csv` records `[tanggal_tutup, tanggal_buka)` intervals per canonical branch. `outlet_features.load_closures()` parses it into a dict of intervals; `build_panel.build_dense_panel()` removes those dates from each pair's reindex range and numbers the remaining contiguous blocks into a new `segment_id` column. The seven functions that depend on panel density already accept a `pair_cols` parameter, so they are made segment-aware by passing a longer key rather than by rewriting them. A separate detector warns about unrecorded gaps but never acts on them.

**Tech Stack:** Python 3.9.6, pandas, pyarrow, unittest. No new dependencies.

## Global Constraints

- Python **3.9.6** — no `X | Y` union syntax in annotations. Use `typing.Optional` / `typing.Union`. Builtin generics (`list[str]`, `dict[str, ...]`) are fine; the codebase already uses them.
- All project CSVs are **semicolon-delimited, `encoding="utf-8-sig"`**.
- Config errors **fail loud** (`raise ValueError`) rather than defaulting silently — matching `outlet_features.parse_delivery_days`.
- QA assertion messages are written in **Indonesian**, matching `run_qa_checks()`.
- Tests are **unittest**, one file per pipeline module, run from the repo root.
- `MIN_HISTORY_DAYS` stays **60**. It is out of scope for this plan.
- Pipeline modules use relative imports and must be run as modules: `.venv/bin/python3 -m utils.<name>`.
- Baseline before starting: **274 tests passing**.
- Spec: `docs/superpowers/specs/2026-08-15-outlet-lifecycle-handling-design.md`

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `dataset/outlet_closures.csv` | Owner-confirmed closure intervals | Create |
| `utils/outlet_features.py` | Branch master data: loading, matching, region/outlet features. Gains closure loading + gap detection because both are branch-lifecycle concerns. | Modify |
| `utils/build_panel.py` | Dense daily panel construction and history filtering. Gains segmentation; stays ignorant of the CSV's column names by receiving a pre-parsed dict. | Modify |
| `utils/prepare_forecast_data.py` | Single definition of feature-engineering step order + QA | Modify |
| `utils/modeling_prep.py` | Modelling-stage transforms and model adapters | Modify |
| `utils/normalize_items.py` | Item/branch normalisation and exclusions | Modify (comment only) |
| `test/test_outlet_features.py` | Tests for closures + gap detection | Modify |
| `test/test_build_panel.py` | Tests for segmentation | Modify |
| `test/test_prepare_forecast_data.py` | Tests for wiring + QA assertions | Modify |
| `test/test_modeling_prep.py` | Tests for segment-aware adapters | Modify |
| `docs/*.md` | Documentation | Modify |

---

### Task 1: Closure config file and loader

**Files:**
- Create: `dataset/outlet_closures.csv`
- Modify: `utils/outlet_features.py` (add after `load_region_mapping`, line 33)
- Test: `test/test_outlet_features.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `outlet_features.CLOSURES_FILE: str`; `outlet_features.load_closures(path: str = CLOSURES_FILE) -> dict[str, list[tuple[pd.Timestamp, Optional[pd.Timestamp]]]]`. The dict is keyed by canonical branch name; each tuple is `(start, end)` where `end is None` means "still closed". Tasks 2, 3, 4 and 5 all consume this exact shape.

- [ ] **Step 1: Create the closure data file**

Create `dataset/outlet_closures.csv` with exactly this content (semicolon-delimited, no BOM needed on write — pandas reads it with `utf-8-sig` either way):

```
Nama Outlet;tanggal_tutup;tanggal_buka;alasan
KY011 - Kebuli Yaman Bekasi Galaxy;2024-03-01;2025-07-18;tutup total, buka kembali di lokasi yang sama dengan kode KY069 (konfirmasi pemilik data 2026-08-15)
KY056 - Kebuli Yaman Tigaraksa;2024-10-01;2024-11-22;tutup sementara (konfirmasi pemilik data 2026-08-15)
Kebuli Yaman Cikarang Pusat;2025-12-01;;relokasi dari KY047 Ciomas, belum buka per akhir data (konfirmasi pemilik data 2026-08-15)
```

- [ ] **Step 2: Write the failing tests**

Add to `test/test_outlet_features.py`:

```python
import tempfile
from pathlib import Path


class TestLoadClosures(unittest.TestCase):
    def _write(self, body):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        )
        tmp.write("Nama Outlet;tanggal_tutup;tanggal_buka;alasan\n" + body)
        tmp.close()
        return tmp.name

    def test_parses_closed_and_reopened_interval(self):
        path = self._write("Cabang A;2024-03-01;2025-07-18;tutup\n")
        result = outlet_features.load_closures(path)
        self.assertEqual(list(result), ["Cabang A"])
        start, end = result["Cabang A"][0]
        self.assertEqual(start, pd.Timestamp("2024-03-01"))
        self.assertEqual(end, pd.Timestamp("2025-07-18"))

    def test_empty_tanggal_buka_means_still_closed(self):
        path = self._write("Cabang A;2025-12-01;;masih tutup\n")
        result = outlet_features.load_closures(path)
        self.assertIsNone(result["Cabang A"][0][1])

    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(outlet_features.load_closures("/tmp/tidak-ada-file.csv"), {})

    def test_unparseable_date_raises(self):
        path = self._write("Cabang A;01 Mar 2024;2025-07-18;salah format\n")
        with self.assertRaises(ValueError):
            outlet_features.load_closures(path)

    def test_reopen_before_close_raises(self):
        path = self._write("Cabang A;2025-07-18;2024-03-01;terbalik\n")
        with self.assertRaises(ValueError):
            outlet_features.load_closures(path)

    def test_overlapping_intervals_raise(self):
        path = self._write(
            "Cabang A;2024-01-01;2024-06-01;satu\nCabang A;2024-05-01;2024-08-01;dua\n"
        )
        with self.assertRaises(ValueError):
            outlet_features.load_closures(path)

    def test_real_file_has_the_three_confirmed_closures(self):
        result = outlet_features.load_closures()
        self.assertIn("KY011 - Kebuli Yaman Bekasi Galaxy", result)
        self.assertIn("KY056 - Kebuli Yaman Tigaraksa", result)
        self.assertIn("Kebuli Yaman Cikarang Pusat", result)
        self.assertIsNone(result["Kebuli Yaman Cikarang Pusat"][0][1])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_outlet_features.TestLoadClosures -v`
Expected: FAIL — `AttributeError: module 'utils.outlet_features' has no attribute 'load_closures'`

- [ ] **Step 4: Implement `load_closures`**

In `utils/outlet_features.py`, add the constant next to the other file paths (after line 11):

```python
CLOSURES_FILE = str(BASE_DIR / "dataset/outlet_closures.csv")
```

And add this function immediately after `load_region_mapping` (line 33):

```python
def load_closures(
    path: str = CLOSURES_FILE,
) -> dict[str, list[tuple[pd.Timestamp, Optional[pd.Timestamp]]]]:
    """Read recorded outlet closure intervals, keyed by canonical branch name.

    Each interval is [tanggal_tutup, tanggal_buka) — closed from tanggal_tutup
    inclusive through the day *before* tanggal_buka. An empty tanggal_buka
    means the outlet is still closed through the end of the data.

    Keyed on canonical `Nama Outlet` (like RELOCATION_DATES) because callers
    consume this after canonicalize_branch_names has merged old branch codes
    into their successors. Returns {} when the file is absent so the pipeline
    still runs on a checkout that has no closures recorded yet.
    """
    if not Path(path).exists():
        return {}

    raw = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    closures: dict[str, list[tuple[pd.Timestamp, Optional[pd.Timestamp]]]] = {}

    for _, row in raw.iterrows():
        branch = str(row["Nama Outlet"]).strip()
        start = pd.to_datetime(row["tanggal_tutup"], format="%Y-%m-%d", errors="coerce")
        if pd.isna(start):
            raise ValueError(
                f"tanggal_tutup tidak valid untuk {branch!r}: {row['tanggal_tutup']!r}"
            )

        raw_end = row["tanggal_buka"]
        if pd.isna(raw_end) or not str(raw_end).strip():
            end = None
        else:
            end = pd.to_datetime(raw_end, format="%Y-%m-%d", errors="coerce")
            if pd.isna(end):
                raise ValueError(f"tanggal_buka tidak valid untuk {branch!r}: {raw_end!r}")
            if end <= start:
                raise ValueError(
                    f"tanggal_buka <= tanggal_tutup untuk {branch!r}: "
                    f"{end.date()} <= {start.date()}"
                )

        closures.setdefault(branch, []).append((start, end))

    for branch, intervals in closures.items():
        intervals.sort(key=lambda interval: interval[0])
        for (earlier_start, earlier_end), (later_start, _) in zip(intervals, intervals[1:]):
            if earlier_end is None or later_start < earlier_end:
                raise ValueError(
                    f"Interval tutup tumpang tindih untuk {branch!r}: "
                    f"{earlier_start.date()} dan {later_start.date()}"
                )

    return closures
```

`Optional` and `Path` are already imported at the top of the module (lines 2–3).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_outlet_features -v`
Expected: PASS, all tests including the 7 new ones.

- [ ] **Step 6: Commit**

```bash
git add dataset/outlet_closures.csv utils/outlet_features.py test/test_outlet_features.py
git commit -m "feat: add the outlet closure interval config and loader"
```

---

### Task 2: Unrecorded-gap detector

**Files:**
- Modify: `utils/outlet_features.py` (add after `load_closures`)
- Test: `test/test_outlet_features.py`

**Interfaces:**
- Consumes: the closure dict shape from Task 1.
- Produces: `outlet_features.MIN_GAP_WARN_DAYS: int = 14`; `outlet_features.detect_unrecorded_gaps(df, closures, branch_col="Nama Cabang", date_col="Tanggal", min_gap_days=MIN_GAP_WARN_DAYS) -> list[dict]`. Each finding is `{"branch": str, "gap_start": pd.Timestamp, "gap_end": pd.Timestamp, "gap_days": int}` where `gap_days` counts **missing** days and `gap_start`/`gap_end` are the first and last missing day inclusive. Task 4 consumes this to print warnings.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_outlet_features.py`:

```python
class TestDetectUnrecordedGaps(unittest.TestCase):
    def _frame(self, branch, dates):
        return pd.DataFrame({
            "Nama Cabang": [branch] * len(dates),
            "Tanggal": pd.to_datetime(dates),
        })

    def test_no_gap_yields_no_findings(self):
        df = self._frame("A", pd.date_range("2024-01-01", periods=30, freq="D"))
        self.assertEqual(outlet_features.detect_unrecorded_gaps(df, {}), [])

    def test_gap_below_threshold_is_ignored(self):
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-01-16", periods=5, freq="D"))
        df = self._frame("A", dates)
        self.assertEqual(
            outlet_features.detect_unrecorded_gaps(df, {}, min_gap_days=14), []
        )

    def test_unrecorded_gap_is_reported_with_missing_day_bounds(self):
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-02-01", periods=5, freq="D"))
        df = self._frame("A", dates)
        findings = outlet_features.detect_unrecorded_gaps(df, {}, min_gap_days=14)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["branch"], "A")
        self.assertEqual(findings[0]["gap_start"], pd.Timestamp("2024-01-06"))
        self.assertEqual(findings[0]["gap_end"], pd.Timestamp("2024-01-31"))
        self.assertEqual(findings[0]["gap_days"], 26)

    def test_recorded_gap_is_not_reported(self):
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-02-01", periods=5, freq="D"))
        df = self._frame("A", dates)
        closures = {"A": [(pd.Timestamp("2024-01-06"), pd.Timestamp("2024-02-01"))]}
        self.assertEqual(
            outlet_features.detect_unrecorded_gaps(df, closures, min_gap_days=14), []
        )

    def test_open_ended_closure_covers_a_trailing_gap(self):
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-02-01", periods=5, freq="D"))
        df = self._frame("A", dates)
        closures = {"A": [(pd.Timestamp("2024-01-06"), None)]}
        self.assertEqual(
            outlet_features.detect_unrecorded_gaps(df, closures, min_gap_days=14), []
        )

    def test_partially_recorded_gap_is_still_reported(self):
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-02-01", periods=5, freq="D"))
        df = self._frame("A", dates)
        closures = {"A": [(pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-20"))]}
        findings = outlet_features.detect_unrecorded_gaps(df, closures, min_gap_days=14)
        self.assertEqual(len(findings), 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_outlet_features.TestDetectUnrecordedGaps -v`
Expected: FAIL — `AttributeError: module 'utils.outlet_features' has no attribute 'detect_unrecorded_gaps'`

- [ ] **Step 3: Implement the detector**

In `utils/outlet_features.py`, add the constant next to `CLOSURES_FILE`:

```python
# Warn about transaction gaps of at least this many MISSING days that are not
# recorded in outlet_closures.csv. Calibrated against the real data: the
# longest clearly benign gap is 7 missing days (Citayam's relocation handover),
# and at 14 exactly the two confirmed closures fire. KY068 Kramatwatu sits just
# under at 13 missing days — worth asking the data owner about, not worth
# lowering the threshold for.
MIN_GAP_WARN_DAYS = 14
```

And add these functions after `load_closures`:

```python
def _gap_is_recorded(
    branch: str,
    gap_start: pd.Timestamp,
    gap_end: pd.Timestamp,
    closures: dict,
) -> bool:
    for start, end in closures.get(branch, []):
        last_closed_day = gap_end if end is None else end - pd.Timedelta(days=1)
        if start <= gap_start and gap_end <= last_closed_day:
            return True
    return False


def detect_unrecorded_gaps(
    df: pd.DataFrame,
    closures: dict[str, list[tuple[pd.Timestamp, Optional[pd.Timestamp]]]],
    branch_col: str = "Nama Cabang",
    date_col: str = "Tanggal",
    min_gap_days: int = MIN_GAP_WARN_DAYS,
) -> list[dict]:
    """Find long transaction gaps that outlet_closures.csv does not explain.

    Returns findings rather than printing them, so the caller decides how to
    report. Detection never segments anything on its own — outlet_closures.csv
    stays the sole authority over what the pipeline treats as closed.
    """
    findings = []
    for branch, group in df.groupby(branch_col, observed=True):
        dates = pd.Series(sorted(pd.to_datetime(group[date_col]).unique()))
        if len(dates) < 2:
            continue
        for position in range(1, len(dates)):
            missing_days = (dates.iloc[position] - dates.iloc[position - 1]).days - 1
            if missing_days < min_gap_days:
                continue
            gap_start = dates.iloc[position - 1] + pd.Timedelta(days=1)
            gap_end = dates.iloc[position] - pd.Timedelta(days=1)
            if _gap_is_recorded(branch, gap_start, gap_end, closures):
                continue
            findings.append({
                "branch": branch,
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_days": missing_days,
            })
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_outlet_features -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add utils/outlet_features.py test/test_outlet_features.py
git commit -m "feat: warn about transaction gaps not recorded as closures"
```

---

### Task 3: Segment the dense panel

**Files:**
- Modify: `utils/build_panel.py:26-48`
- Test: `test/test_build_panel.py`

**Interfaces:**
- Consumes: the closure dict shape from Task 1.
- Produces: `build_panel.SEGMENT_COL: str = "segment_id"`; `build_dense_panel(df, pair_cols=PAIR_COLS, date_col="Tanggal", qty_col="Kuantitas", carry_cols=CARRY_COLS, closures=None, branch_col="Nama Cabang") -> pd.DataFrame` with a new integer `segment_id` column, 1-based and contiguous per pair. Tasks 4, 5 and 6 all rely on that column name.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_build_panel.py`:

```python
class TestDensePanelSegments(unittest.TestCase):
    def _pair_frame(self):
        # Two active blocks either side of a closure: Jan 1-3 and Mar 1-3.
        dates = ["2024-01-01", "2024-01-02", "2024-01-03",
                 "2024-03-01", "2024-03-02", "2024-03-03"]
        return pd.DataFrame({
            "Kode Barang": ["A"] * 6, "Nama Cabang": ["X"] * 6,
            "Tanggal": pd.to_datetime(dates), "Kuantitas": [1] * 6,
            "Kategori Barang": ["Barang Jadi (FG)"] * 6, "Nama Barang": ["Widget"] * 6,
        })

    def test_without_closures_output_matches_legacy_behaviour(self):
        df = self._pair_frame()
        result = build_panel.build_dense_panel(df)
        # Jan 1 -> Mar 3 inclusive is 63 days, all gap-filled as before.
        self.assertEqual(len(result), 63)
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_closure_removes_rows_and_starts_a_second_segment(self):
        df = self._pair_frame()
        closures = {"X": [(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-03-01"))]}
        result = build_panel.build_dense_panel(df, closures=closures).sort_values("Tanggal")
        self.assertEqual(len(result), 6)
        self.assertEqual(list(result[build_panel.SEGMENT_COL]), [1, 1, 1, 2, 2, 2])
        self.assertTrue(
            result[(result["Tanggal"] >= "2024-01-04") & (result["Tanggal"] < "2024-03-01")].empty
        )

    def test_open_ended_closure_truncates_the_tail(self):
        df = self._pair_frame()
        closures = {"X": [(pd.Timestamp("2024-01-04"), None)]}
        result = build_panel.build_dense_panel(df, closures=closures)
        self.assertEqual(len(result), 3)
        self.assertEqual(result["Tanggal"].max(), pd.Timestamp("2024-01-03"))
        self.assertEqual(set(result[build_panel.SEGMENT_COL]), {1})

    def test_closure_for_another_branch_has_no_effect(self):
        df = self._pair_frame()
        closures = {"Y": [(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-03-01"))]}
        result = build_panel.build_dense_panel(df, closures=closures)
        self.assertEqual(len(result), 63)

    def test_pair_entirely_inside_a_closure_is_dropped(self):
        df = self._pair_frame()
        other = df.copy()
        other["Nama Cabang"] = "Z"
        combined = pd.concat([df, other], ignore_index=True)
        closures = {"Z": [(pd.Timestamp("2023-12-01"), pd.Timestamp("2024-06-01"))]}
        result = build_panel.build_dense_panel(combined, closures=closures)
        self.assertEqual(set(result["Nama Cabang"]), {"X"})

    def test_every_segment_is_internally_dense(self):
        df = self._pair_frame()
        closures = {"X": [(pd.Timestamp("2024-01-04"), pd.Timestamp("2024-03-01"))]}
        result = build_panel.build_dense_panel(df, closures=closures)
        for _, group in result.groupby(["Kode Barang", "Nama Cabang", build_panel.SEGMENT_COL]):
            spans = group["Tanggal"].sort_values().diff().dropna().dt.days
            self.assertTrue((spans == 1).all())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_build_panel.TestDensePanelSegments -v`
Expected: FAIL — `AttributeError: module 'utils.build_panel' has no attribute 'SEGMENT_COL'`

- [ ] **Step 3: Implement segmentation**

In `utils/build_panel.py`, change the import block at the top from:

```python
import pandas as pd
```

to:

```python
from typing import Optional

import pandas as pd
```

Add the constant next to `CARRY_COLS` (line 6):

```python
SEGMENT_COL = "segment_id"
```

Replace `build_dense_panel` (lines 26–48) entirely with:

```python
def _drop_closed_dates(dates: pd.DatetimeIndex, intervals: list) -> pd.DatetimeIndex:
    values = pd.Series(dates)
    keep = pd.Series(True, index=values.index)
    for start, end in intervals:
        closed = values >= start if end is None else (values >= start) & (values < end)
        keep &= ~closed
    return pd.DatetimeIndex(values[keep])


def _segment_ids(dates: pd.Series) -> pd.Series:
    # A new segment begins wherever two kept dates are more than one day
    # apart — which is exactly where a closure interval was removed.
    starts_new_segment = dates.diff().dt.days.fillna(1) > 1
    return starts_new_segment.cumsum().astype(int) + 1


def build_dense_panel(
    df: pd.DataFrame,
    pair_cols: list[str] = PAIR_COLS,
    date_col: str = "Tanggal",
    qty_col: str = "Kuantitas",
    carry_cols: list[str] = CARRY_COLS,
    closures: Optional[dict] = None,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    """Reindex each pair to a dense daily panel over its own active range.

    Days inside a recorded closure interval for the pair's branch produce no
    rows at all: the outlet did not exist then, so zero-filling them would
    fabricate demand history. Each contiguous run of kept dates is numbered
    into SEGMENT_COL, and callers group by pair + segment so no lag, rolling
    window, target shift, or LSTM sequence ever bridges a closure.

    closures=None reproduces the pre-segmentation behaviour exactly, with
    segment_id == 1 everywhere.
    """
    closures = closures or {}
    pieces = []
    for keys, group in df.groupby(pair_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values(date_col)
        full_range = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")

        branch = dict(zip(pair_cols, keys)).get(branch_col)
        active_range = _drop_closed_dates(full_range, closures.get(branch, []))
        if len(active_range) == 0:
            continue

        dense = group.set_index(date_col).reindex(active_range)
        dense[qty_col] = dense[qty_col].fillna(0)
        for col in carry_cols:
            dense[col] = dense[col].ffill().bfill()
        for pair_col, key in zip(pair_cols, keys):
            dense[pair_col] = key
        dense = dense.reset_index().rename(columns={"index": date_col})
        dense[SEGMENT_COL] = _segment_ids(dense[date_col])
        pieces.append(dense[pair_cols + [date_col, qty_col] + carry_cols + [SEGMENT_COL]])

    result = pd.concat(pieces, ignore_index=True)
    return result.sort_values(pair_cols + [date_col]).reset_index(drop=True)
```

- [ ] **Step 4: Run the full build_panel suite**

Run: `.venv/bin/python3 -m unittest test.test_build_panel -v`
Expected: PASS — the 6 new tests plus every pre-existing one, since `closures=None` preserves the old behaviour.

- [ ] **Step 5: Commit**

```bash
git add utils/build_panel.py test/test_build_panel.py
git commit -m "feat: segment the dense panel on recorded closure intervals"
```

---

### Task 4: Wire segmentation through feature engineering

**Files:**
- Modify: `utils/prepare_forecast_data.py` (`FEATURED_COLUMNS`, `engineer_features`, `build_featured_dataset`)
- Test: `test/test_prepare_forecast_data.py`

**Interfaces:**
- Consumes: `outlet_features.load_closures`, `outlet_features.detect_unrecorded_gaps`, `build_panel.SEGMENT_COL`.
- Produces: `prepare_forecast_data.SEGMENT_COLS: list[str]` = `["Kode Barang", "Nama Cabang", "segment_id"]`, consumed by Task 5 and mirrored by Task 6. `build_featured_dataset` gains a `closures_path: str = outlet_features.CLOSURES_FILE` parameter; `engineer_features` gains `closures: Optional[dict] = None` (used only by QA in Task 5, accepted here so the signature is stable).

- [ ] **Step 1: Write the failing test**

Add to `test/test_prepare_forecast_data.py`:

```python
class TestSegmentAwareFeatures(unittest.TestCase):
    def _panel(self):
        # One pair, two segments: 2024-01-01..2024-01-05 and 2024-03-01..2024-03-05.
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-03-01", periods=5, freq="D"))
        return pd.DataFrame({
            "Kode Barang": ["A"] * 10, "Nama Cabang": ["X"] * 10,
            "Tanggal": dates, "Kuantitas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "segment_id": [1] * 5 + [2] * 5,
        })

    def test_targets_do_not_jump_across_a_segment_boundary(self):
        result = prepare_forecast_data.add_targets(
            self._panel(), pair_cols=prepare_forecast_data.SEGMENT_COLS
        )
        last_of_segment_1 = result[
            (result["segment_id"] == 1) & (result["Tanggal"] == pd.Timestamp("2024-01-05"))
        ]
        self.assertTrue(pd.isna(last_of_segment_1["target_h1"].iloc[0]))

    def test_targets_would_jump_without_segment_cols(self):
        # Guards the guard: with plain PAIR_COLS the target silently reaches
        # across the closure, which is the bug segmentation exists to prevent.
        result = prepare_forecast_data.add_targets(
            self._panel(), pair_cols=prepare_forecast_data.PAIR_COLS
        )
        last_of_segment_1 = result[result["Tanggal"] == pd.Timestamp("2024-01-05")]
        self.assertEqual(last_of_segment_1["target_h1"].iloc[0], 6)

    def test_lags_do_not_cross_a_segment_boundary(self):
        result = prepare_forecast_data.add_lag_features(
            self._panel(), pair_cols=prepare_forecast_data.SEGMENT_COLS
        )
        first_of_segment_2 = result[result["Tanggal"] == pd.Timestamp("2024-03-01")]
        self.assertTrue(pd.isna(first_of_segment_2["lag_1"].iloc[0]))

    def test_rolling_does_not_cross_a_segment_boundary(self):
        panel = self._panel()
        result = prepare_forecast_data.add_rolling_features(
            panel, pair_cols=prepare_forecast_data.SEGMENT_COLS, windows=[3]
        )
        first_of_segment_2 = result[result["Tanggal"] == pd.Timestamp("2024-03-01")]
        self.assertTrue(pd.isna(first_of_segment_2["roll_mean_3"].iloc[0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestSegmentAwareFeatures -v`
Expected: FAIL — `AttributeError: module 'utils.prepare_forecast_data' has no attribute 'SEGMENT_COLS'`

- [ ] **Step 3: Add the constant and the schema column**

In `utils/prepare_forecast_data.py`, after `PAIR_COLS = build_panel.PAIR_COLS` (line 13):

```python
SEGMENT_COL = build_panel.SEGMENT_COL
# The grouping key for every shift-based feature. Grouping by pair alone would
# let a lag, rolling window, or target reach across a closure gap and treat two
# sides of a months-long shutdown as consecutive days.
SEGMENT_COLS = PAIR_COLS + [SEGMENT_COL]
```

In `FEATURED_COLUMNS`, add `"segment_id"` immediately after `"Nama Barang"`:

```python
    "Kode Barang", "Nama Cabang", "Tanggal", "Kuantitas", "Kategori Barang",
    "Nama Barang", "segment_id", "day_of_week", "day_of_month", "month", "is_weekend",
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestSegmentAwareFeatures -v`
Expected: PASS

- [ ] **Step 5: Pass `SEGMENT_COLS` to the four shift-based functions**

In `engineer_features`, change the four calls (currently plain calls with no `pair_cols`):

```python
    df = add_targets(df, pair_cols=SEGMENT_COLS)
    df = outlet_features.apply_region_features(df, region_df)
    df = apply_outlet_features(df, outlets_df, overrides_df)
    df = outlet_features.add_relocation_feature(df)
    df = add_lead_time_target(df, pair_cols=SEGMENT_COLS)
    df = add_lag_features(df, pair_cols=SEGMENT_COLS, qty_col="Kuantitas_capped")
    df = add_rolling_features(df, pair_cols=SEGMENT_COLS, qty_col="Kuantitas_capped")
```

Leave `compute_branch_stats`, `apply_branch_stats`, and `add_branch_age_days` untouched — they are branch-level by design, and `compute_pair_baseline` / `apply_outlier_capping` stay pair-level (see the spec's "What deliberately stays coarser than segment level").

- [ ] **Step 6: Load closures in `build_featured_dataset` and warn about gaps**

Add a `closures_path` parameter to `build_featured_dataset` and use it. The function becomes:

```python
def build_featured_dataset(
    input_path: str = normalize_items.RAW_DATA_FILE,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
    region_path: str = outlet_features.REGION_MAPPING_FILE,
    closures_path: str = outlet_features.CLOSURES_FILE,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> pd.DataFrame:
    outlets_df = outlet_features.load_outlets(outlets_path)
    overrides_df = outlet_features.load_overrides(overrides_path)
    region_df = outlet_features.load_region_mapping(region_path)
    closures = outlet_features.load_closures(closures_path)
    df = normalize_items.load_and_normalize(input_path)
    df = outlet_features.filter_matched_branches(df, outlets_df, overrides_df)
    df = outlet_features.canonicalize_branch_names(df, outlets_df, overrides_df)
    df = normalize_items.reaggregate_daily(df)

    for finding in outlet_features.detect_unrecorded_gaps(df, closures):
        print(
            f"[WARN] Cabang {finding['branch']!r} punya gap {finding['gap_days']} hari "
            f"({finding['gap_start'].date()}..{finding['gap_end'].date()}) "
            f"belum tercatat di {closures_path} — hari-hari itu akan diisi nol."
        )

    df = build_panel.build_dense_panel(df, closures=closures)
    df = build_panel.filter_min_history(df, cutoff=cutoff, min_days=min_history_days)
    return engineer_features(
        df,
        outlets_df=outlets_df,
        overrides_df=overrides_df,
        region_df=region_df,
        cutoff=cutoff,
        min_pair_history=min_pair_history,
        spike_ratio_threshold=spike_ratio_threshold,
    )
```

- [ ] **Step 7: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py"`
Expected: `OK`. If a pre-existing end-to-end test fails on the new `segment_id` column, update that test's expected column list — do not remove the column.

- [ ] **Step 8: Commit**

```bash
git add utils/prepare_forecast_data.py test/test_prepare_forecast_data.py
git commit -m "feat: group shift-based features by pair and segment"
```

---

### Task 5: QA assertions for the segmentation invariants

**Files:**
- Modify: `utils/prepare_forecast_data.py` (`run_qa_checks`, `main`)
- Test: `test/test_prepare_forecast_data.py`

**Interfaces:**
- Consumes: `SEGMENT_COLS` from Task 4, closure dict from Task 1.
- Produces: `run_qa_checks(df, closures=None)` — the added `closures` parameter defaults to `None` so existing callers keep working.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_prepare_forecast_data.py`:

```python
class TestSegmentQaChecks(unittest.TestCase):
    def _minimal_featured(self):
        dates = list(pd.date_range("2024-01-01", periods=3, freq="D"))
        return pd.DataFrame({
            "Kode Barang": ["A"] * 3, "Nama Cabang": ["X"] * 3,
            "Tanggal": dates, "Kuantitas": [1.0, 2.0, 3.0],
            "Kuantitas_capped": [1.0, 2.0, 3.0], "segment_id": [1, 1, 1],
            "kota": ["Kota Tangerang"] * 3, "kawasan": [1, 1, 1],
        })

    def test_clean_frame_passes(self):
        prepare_forecast_data.run_qa_checks(self._minimal_featured(), closures={})

    def test_row_inside_a_closure_fails(self):
        df = self._minimal_featured()
        closures = {"X": [(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))]}
        with self.assertRaises(AssertionError):
            prepare_forecast_data.run_qa_checks(df, closures=closures)

    def test_segment_ids_not_starting_at_one_fail(self):
        df = self._minimal_featured()
        df["segment_id"] = [2, 2, 2]
        with self.assertRaises(AssertionError):
            prepare_forecast_data.run_qa_checks(df, closures={})

    def test_date_gap_inside_one_segment_fails(self):
        df = self._minimal_featured()
        df.loc[2, "Tanggal"] = pd.Timestamp("2024-01-10")
        with self.assertRaises(AssertionError):
            prepare_forecast_data.run_qa_checks(df, closures={})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_prepare_forecast_data.TestSegmentQaChecks -v`
Expected: FAIL — `TypeError: run_qa_checks() got an unexpected keyword argument 'closures'`

- [ ] **Step 3: Extend `run_qa_checks`**

Change the signature and append the three assertions before the end of the function:

```python
def run_qa_checks(df: pd.DataFrame, closures: Optional[dict] = None) -> None:
```

Add at the end of the function body:

```python
    for branch, intervals in (closures or {}).items():
        branch_rows = df[df["Nama Cabang"] == branch]
        for start, end in intervals:
            inside = (
                branch_rows["Tanggal"] >= start
                if end is None
                else (branch_rows["Tanggal"] >= start) & (branch_rows["Tanggal"] < end)
            )
            assert not inside.any(), (
                f"Ditemukan {int(inside.sum())} baris di dalam periode tutup "
                f"{branch!r} ({start.date()}..)"
            )

    segment_starts = df.groupby(PAIR_COLS, observed=True)["segment_id"].min()
    assert (segment_starts == 1).all(), "Ada pasangan yang segment_id-nya tidak mulai dari 1"

    segment_counts = df.groupby(PAIR_COLS, observed=True)["segment_id"].nunique()
    segment_maxima = df.groupby(PAIR_COLS, observed=True)["segment_id"].max()
    assert (segment_counts == segment_maxima).all(), "segment_id tidak kontinu per pasangan"

    spans = (
        df.sort_values(SEGMENT_COLS + ["Tanggal"])
        .groupby(SEGMENT_COLS, observed=True)["Tanggal"]
        .diff()
        .dt.days
        .dropna()
    )
    assert (spans == 1).all(), "Ada lubang tanggal di dalam satu segmen"
```

`Optional` must be imported — add `from typing import Optional` to the top of `utils/prepare_forecast_data.py` (it currently imports only `numpy`, `pandas`, `Path`, and the sibling modules).

- [ ] **Step 4: Pass closures from `main`**

In `main()`, add the `closures_path` parameter alongside the other paths and thread it through:

```python
def main(
    input_path: str = normalize_items.RAW_DATA_FILE,
    output_dir: str = MODEL_READY_DIR,
    min_history_days: int = build_panel.MIN_HISTORY_DAYS,
    cutoff: pd.Timestamp = TEST_START,
    outlets_path: str = outlet_features.OUTLETS_FILE,
    overrides_path: str = outlet_features.OVERRIDES_FILE,
    region_path: str = outlet_features.REGION_MAPPING_FILE,
    closures_path: str = outlet_features.CLOSURES_FILE,
    min_pair_history: int = outlier_handling.MIN_PAIR_HISTORY,
    spike_ratio_threshold: float = outlier_handling.SPIKE_RATIO_THRESHOLD,
) -> None:
    df = build_featured_dataset(
        input_path=input_path,
        min_history_days=min_history_days,
        cutoff=cutoff,
        outlets_path=outlets_path,
        overrides_path=overrides_path,
        region_path=region_path,
        closures_path=closures_path,
        min_pair_history=min_pair_history,
        spike_ratio_threshold=spike_ratio_threshold,
    )
    run_qa_checks(df, closures=outlet_features.load_closures(closures_path))
```

Leave the rest of `main()` unchanged.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add utils/prepare_forecast_data.py test/test_prepare_forecast_data.py
git commit -m "feat: assert the closure and segment invariants in QA"
```

---

### Task 6: Make the model adapters segment-aware

**Files:**
- Modify: `utils/modeling_prep.py` (`drop_warmup_rows:265`, `to_tabular:283`, `to_sequences:349`)
- Test: `test/test_modeling_prep.py`

**Interfaces:**
- Consumes: the `segment_id` column produced in Task 3.
- Produces: `modeling_prep.SEGMENT_COL`, `modeling_prep.SEGMENT_COLS`, and a private `_resolve_pair_cols(df, pair_cols)`. The `keys` frames returned by both adapters gain `segment_id`, which `validate_contract` compares like any other key column — no change needed there.

- [ ] **Step 1: Write the failing tests**

Add to `test/test_modeling_prep.py`:

```python
class TestSegmentAwareAdapters(unittest.TestCase):
    def _frame(self):
        dates = list(pd.date_range("2024-01-01", periods=6, freq="D"))
        dates += list(pd.date_range("2024-06-01", periods=6, freq="D"))
        return pd.DataFrame({
            "Kode Barang": ["A"] * 12, "Nama Cabang": ["X"] * 12,
            "Tanggal": dates, "segment_id": [1] * 6 + [2] * 6,
            "feat": list(range(12)), "target_lead_time_cumulative": list(range(12)),
            "fold_id": [float("nan")] * 12,
        })

    def test_warmup_is_cut_per_segment(self):
        result = modeling_prep.drop_warmup_rows(self._frame(), lookback=3)
        # 3 rows survive in each of the two segments, not 9 across one series.
        self.assertEqual(len(result), 6)
        self.assertEqual(sorted(result["segment_id"].unique()), [1, 2])
        self.assertEqual(result["Tanggal"].min(), pd.Timestamp("2024-01-04"))

    def test_frames_without_segment_id_still_group_by_pair(self):
        df = self._frame().drop(columns=["segment_id"])
        result = modeling_prep.drop_warmup_rows(df, lookback=3)
        self.assertEqual(len(result), 9)

    def test_sequences_never_bridge_two_segments(self):
        result = modeling_prep.to_sequences(
            self._frame(), feature_cols=["feat"], lookback=3
        )
        self.assertEqual(len(result["X"]), 6)
        for window in result["X"]:
            values = [int(v) for v in window[:, 0]]
            self.assertTrue(all(b - a == 1 for a, b in zip(values, values[1:])))

    def test_adapters_agree_on_segmented_input(self):
        df = self._frame()
        tabular = modeling_prep.to_tabular(df, feature_cols=["feat"], lookback=3)
        sequences = modeling_prep.to_sequences(df, feature_cols=["feat"], lookback=3)
        modeling_prep.validate_contract(tabular, sequences)
        self.assertIn("segment_id", tabular["keys"].columns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python3 -m unittest test.test_modeling_prep.TestSegmentAwareAdapters -v`
Expected: FAIL — `test_warmup_is_cut_per_segment` reports 9 rows instead of 6, because the warm-up cut currently groups by pair only.

- [ ] **Step 3: Add the segment constants and resolver**

In `utils/modeling_prep.py`, next to `PAIR_COLS` (line 20):

```python
SEGMENT_COL = "segment_id"
SEGMENT_COLS = PAIR_COLS + [SEGMENT_COL]
```

Add this helper just above `drop_warmup_rows` (line 265):

```python
def _resolve_pair_cols(df: pd.DataFrame, pair_cols: Optional[list]) -> list:
    """Group by segment when the frame carries one, otherwise by pair.

    An explicit pair_cols argument always wins. Falling back on the column's
    presence keeps fixtures and callers that predate segmentation working,
    while guaranteeing that any frame built from the segmented panel never
    lets a warm-up cut or an LSTM window bridge a closure.
    """
    if pair_cols is not None:
        return pair_cols
    return SEGMENT_COLS if SEGMENT_COL in df.columns else PAIR_COLS
```

`Optional` is **not** currently imported in `utils/modeling_prep.py` — add `from typing import Optional` above the `import json` line.

- [ ] **Step 4: Use the resolver in the three adapters**

In `drop_warmup_rows`, replace `pair_cols = pair_cols or PAIR_COLS` with:

```python
    pair_cols = _resolve_pair_cols(df, pair_cols)
```

In `to_tabular`, replace `pair_cols = pair_cols or PAIR_COLS` with:

```python
    pair_cols = _resolve_pair_cols(df, pair_cols)
```

In `to_sequences`, replace `pair_cols = pair_cols or PAIR_COLS` with:

```python
    pair_cols = _resolve_pair_cols(df, pair_cols)
```

Leave `compute_pair_demand_stats` and `classify_pairs` on `pair_cols or PAIR_COLS` — ADI and CV² are pair-level statistics by design, and closed days no longer exist to distort them.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add utils/modeling_prep.py test/test_modeling_prep.py
git commit -m "feat: cut warm-up and build sequences per segment"
```

---

### Task 7: Regenerate artifacts and verify the real-data correction

**Files:**
- Modify: `utils/normalize_items.py:130` (comment only)
- Regenerates: `dataset/model_ready/{featured,train,test}.parquet`, `model_input.parquet`, `category_mapping.json`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: regenerated parquet artifacts with 64 / 76 columns.

- [ ] **Step 1: Document the Kebab Saudagar exclusion**

In `utils/normalize_items.py`, replace line 130 with:

```python
# Kebab Saudagar is a different brand that briefly issued goods through this
# system (2025-12-20..2025-12-31, 137 rows). Data owner confirmed 2026-08-15
# that it is no longer operating and its data is not needed.
EXCLUDED_BRANCHES = {"Kebab Saudagar - Kutabumi"}
```

- [ ] **Step 2: Run the data-prep pipeline end to end**

Run: `.venv/bin/python3 -m utils.prepare_forecast_data`

Expected: no `[WARN]` lines (all three long gaps are recorded), no `AssertionError`, and a final line of the form `Wrote <n> train rows and 55046 test rows to .../model_ready`. The train count should be roughly 19,304 lower than the previous 1,467,822 — expect ≈1,448,518, possibly slightly lower if removing closed days pushed a pair under `MIN_HISTORY_DAYS`.

- [ ] **Step 3: Run the modelling-prep pipeline**

Run: `.venv/bin/python3 -m utils.modeling_prep`
Expected: `Wrote <n> rows x 76 columns to .../model_input.parquet`

- [ ] **Step 4: Verify the correction on real data**

Run:

```bash
.venv/bin/python3 -c "
import pandas as pd
f = pd.read_parquet('dataset/model_ready/featured.parquet')
print('baris   :', len(f))
print('kolom   :', len(f.columns))
print('segments:', sorted(f['segment_id'].unique()))
ky011 = f[f['Nama Cabang'] == 'KY011 - Kebuli Yaman Bekasi Galaxy']
closed = ky011[(ky011.Tanggal >= '2024-03-01') & (ky011.Tanggal < '2025-07-18')]
print('baris KY011 di jendela tutup:', len(closed), '(harus 0)')
print('branch_avg_daily_qty        :', round(ky011.branch_avg_daily_qty.iloc[0], 1), '(harus ~371)')
print('branch_demand_cv            :', round(ky011.branch_demand_cv.iloc[0], 3), '(harus ~0.50)')
tig = f[f['Nama Cabang'] == 'KY056 - Kebuli Yaman Tigaraksa']
print('baris Tigaraksa di jendela  :', len(tig[(tig.Tanggal >= '2024-10-01') & (tig.Tanggal < '2024-11-22')]), '(harus 0)')
print('Tigaraksa mulai             :', tig.Tanggal.min().date(), '(harus 2024-01-01, riwayat Antapani)')
"
```

Expected: 64 columns, zero rows in both closure windows, `branch_avg_daily_qty` ≈ 371, `branch_demand_cv` ≈ 0.50, Tigaraksa still starting 2024-01-01. If `branch_avg_daily_qty` is still ≈104, the closure rows were not removed — go back to Task 3.

- [ ] **Step 5: Commit**

```bash
git add utils/normalize_items.py dataset/model_ready
git commit -m "chore: regenerate artifacts without fabricated closure rows"
```

---

### Task 8: Update the documentation

**Files:**
- Modify: `docs/dokumentasi-preprocessing-id.md`, `docs/pipeline-overview.md`, `docs/todolist-data-preprocessing.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: the verified row/column counts from Task 7. Use the **measured** numbers, not this plan's estimates.

- [ ] **Step 1: Update `docs/dokumentasi-preprocessing-id.md`**

- section 2 "Sumber data": add a row for `dataset/outlet_closures.csv` — "Interval tutup per outlet (relokasi, tutup sementara); 3 baris terkonfirmasi pemilik data 2026-08-15".
- bagian 3 flow diagram: annotate stage 5 as `build_panel.py (panel harian padat per segmen + filter riwayat min.)`.
- section 3 size table and section 13 artifact table: replace 1.522.868 / 1.467.822 / 63 with the measured values and 64 columns; `model_input.parquet` becomes 76 columns.
- section 5: add a subsection explaining that days inside a recorded closure produce no rows, that `segment_id` numbers the contiguous active blocks, and that every shift-based feature groups by pair + segment. Include the KY011 before/after table (104,0 → 371,3; cv 1,863 → 0,502; peringkat #59 → #46) as the worked example.
- bagian 8.4 leakage table: add a row — "Fitur bergeser dikelompokkan per (pasangan, segmen) | Lag, rolling, dan target tidak pernah melintasi periode outlet tutup".
- section 9 QA list: change "7 asersi" to the new count and describe the three added checks.
- section 15: add Cilebut as a known cold-start case, and add `KY068 - Kebuli Yaman Kramatwatu` (gap 13 hari, 2025-06-28..2025-07-10) as a borderline gap awaiting owner confirmation.
- bagian 16 glossary: add "**Segmen**" — "Blok tanggal aktif kontinu milik satu pasangan; dipisahkan oleh periode outlet tutup".

- [ ] **Step 2: Update `docs/pipeline-overview.md`**

Add `segment_id` to the stage-5 description and note that stages producing `target_*`, `lag_*`, and `roll_*` group by pair + segment.

- [ ] **Step 3: Update `docs/todolist-data-preprocessing.md`**

Add a completed entry for this work, and add open items: confirm the Kramatwatu 13-day gap; fill `tanggal_buka` for Cikarang Pusat once the outlet opens and re-derive `RELOCATION_DATES` from it; watch for the same close-then-reopen pattern in the four lower-bound relocations.

- [ ] **Step 4: Update `CLAUDE.md`**

In the "Project state" paragraph, add `outlet_closures.csv` to the list of dataset config files and add this design spec to the "Refer to" list.

- [ ] **Step 5: Verify the suite still passes and commit**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py"`
Expected: `OK`

```bash
git add docs CLAUDE.md
git commit -m "docs: document closure segmentation and the segment_id column"
```

---

## Notes for the implementer

- **Do not build an automatic `RELOCATION_DATES` ← `tanggal_buka` derivation.** The spec describes deriving Cikarang Pusat's relocation date from its reopening date, but that rule does not generalise: `KY056 - Kebuli Yaman Tigaraksa` appears in both tables with *different, unrelated* dates (relocation 2024-03-01, temporary closure 2024-10-01 → 2024-11-22), so a blanket "these must agree" assertion would fire falsely. Keep it a manual step on the todolist (Task 8, Step 3).
- **The single most important test is `test_targets_would_jump_without_segment_cols` in Task 4.** It pins the exact bug this work exists to prevent: without segment grouping, `target_h1` on the last day before a closure silently becomes the first day after it. If you find yourself tempted to delete that test, stop.
- `filter_min_history` stays pair-level on purpose. History either side of a closure is still real history; short segments are removed automatically by the now-segment-aware `drop_warmup_rows` at `L=28`.
- The notebooks (`notebook/data-processing.ipynb`, `notebook/train_test_split.ipynb`, `notebook/modeling_prep.ipynb`) call the composite functions rather than re-listing steps, so they inherit these changes without edits. Re-running them is optional; if you do, use `--allow-errors` as `CLAUDE.md` describes.
