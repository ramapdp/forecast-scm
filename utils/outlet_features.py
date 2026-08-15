import re
from typing import Optional, Union
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

OUTLETS_FILE = str(BASE_DIR / "dataset/outlets.csv")
OVERRIDES_FILE = str(BASE_DIR / "dataset/outlet_name_overrides.csv")
REGION_MAPPING_FILE = str(BASE_DIR / "dataset/outlet_mapping.csv")
CLOSURES_FILE = str(BASE_DIR / "dataset/outlet_closures.csv")

# Warn about transaction gaps of at least this many MISSING days that are not
# recorded in outlet_closures.csv. Calibrated against the real data: the
# longest clearly benign gap is 7 missing days (Citayam's relocation handover),
# and at 14 exactly the two confirmed closures fire. KY068 Kramatwatu sits just
# under at 13 missing days — worth asking the data owner about, not worth
# lowering the threshold for.
MIN_GAP_WARN_DAYS = 14

PREFIX_RE = re.compile(r"^KY\d+\s*-\s*", re.IGNORECASE)
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")

CHANNEL_COLS = ["has_shopee", "has_gofood", "has_grabfood"]

INDONESIAN_WEEKDAYS = {
    "Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3,
    "Jumat": 4, "Sabtu": 5, "Minggu": 6,
}


def load_outlets(path: str = OUTLETS_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def load_overrides(path: str = OVERRIDES_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)


def load_region_mapping(path: str = REGION_MAPPING_FILE) -> pd.DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8-sig")


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


def parse_delivery_days(hari_pengiriman: str) -> set[int]:
    tokens = re.split(r"\bdan\b|,", hari_pengiriman)
    days = set()
    for token in tokens:
        name = token.strip()
        if not name:
            continue
        if name not in INDONESIAN_WEEKDAYS:
            raise ValueError(f"Unrecognized weekday token: {name!r} in {hari_pengiriman!r}")
        days.add(INDONESIAN_WEEKDAYS[name])
    return days


def compute_lead_time_days(day_of_week: int, delivery_days: set[int]) -> int:
    # Always strictly forward: if day_of_week is itself a delivery day, this
    # returns the days to the *next* occurrence, never 0.
    for d in range(1, 8):
        if (day_of_week + d) % 7 in delivery_days:
            return d
    raise ValueError(f"No delivery day found for day_of_week={day_of_week}")


def apply_region_features(
    df: pd.DataFrame,
    region_df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
    date_col: str = "Tanggal",
) -> pd.DataFrame:
    # region_df's `new_name` matches Nama Cabang once branches are canonicalized
    # (see canonicalize_branch_names) — join is a straight rename, no fuzzy matching needed.
    features = region_df[["new_name", "kawasan", "hari_pengiriman"]].rename(
        columns={"new_name": branch_col}
    )
    result = df.merge(features, on=branch_col, how="left")

    # lead_time_days depends only on (day_of_week, hari_pengiriman) — a handful
    # of distinct combinations even across millions of rows — so compute a
    # small lookup table once and merge back, instead of .apply() per row.
    # itertuples() can't expose a leading-underscore column as an attribute
    # (renamed positionally instead), so avoid that name for the temp column.
    result["day_of_week_tmp"] = result[date_col].dt.weekday
    combos = result[["day_of_week_tmp", "hari_pengiriman"]].drop_duplicates().dropna()
    # list comprehension over itertuples, not .apply(axis=1) — .apply() on an
    # empty frame (all branches unmatched) can't infer the output is a Series
    # and returns an empty DataFrame instead, breaking the assignment below.
    combos["lead_time_days"] = [
        compute_lead_time_days(int(row.day_of_week_tmp), parse_delivery_days(row.hari_pengiriman))
        for row in combos.itertuples()
    ]
    result = result.merge(combos, on=["day_of_week_tmp", "hari_pengiriman"], how="left")
    return result.drop(columns=["day_of_week_tmp"])


def _strip_for_matching(nama_cabang: str) -> str:
    stripped = PREFIX_RE.sub("", nama_cabang)
    stripped = TRAILING_PAREN_RE.sub("", stripped)
    return stripped.strip()


def match_branch_to_outlet(
    nama_cabang: str, outlets_df: pd.DataFrame, overrides_df: pd.DataFrame
) -> tuple[Optional[pd.Series], Optional[str]]:
    override_row = overrides_df[overrides_df["Nama Cabang"] == nama_cabang]
    if not override_row.empty:
        nama_outlet = override_row.iloc[0]["Nama Outlet"]
        kota_override = override_row.iloc[0]["Kota Override"]
        outlet_row = outlets_df[outlets_df["Nama Outlet"] == nama_outlet]
        if outlet_row.empty:
            return None, None
        return outlet_row.iloc[0], (kota_override if pd.notna(kota_override) and kota_override else None)

    stripped = _strip_for_matching(nama_cabang)
    candidates = outlets_df[
        outlets_df["Nama Outlet"].str.casefold().apply(lambda name: stripped.casefold() in name)
    ]
    if len(candidates) == 1:
        return candidates.iloc[0], None
    return None, None


def normalize_kota(kota: Optional[str], kota_override: Optional[str]) -> str:
    if kota_override:
        return kota_override.strip()
    if kota is None or (isinstance(kota, float) and pd.isna(kota)):
        return "Unknown"
    return kota.strip()


def _to_bool(value) -> Union[bool, float]:
    if pd.isna(value):
        return float("nan")
    return value.strip().lower() == "yes"


def filter_matched_branches(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    branches = df[branch_col].unique()
    matched = {
        b for b in branches
        if match_branch_to_outlet(b, outlets_df, overrides_df)[0] is not None
    }
    return df[df[branch_col].isin(matched)].reset_index(drop=True)


def canonicalize_branch_names(
    df: pd.DataFrame,
    outlets_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    branch_col: str = "Nama Cabang",
) -> pd.DataFrame:
    branches = df[branch_col].unique()
    canonical_names = {}
    for nama_cabang in branches:
        outlet_row, _ = match_branch_to_outlet(nama_cabang, outlets_df, overrides_df)
        canonical_names[nama_cabang] = (
            nama_cabang if outlet_row is None else outlet_row["Nama Outlet"]
        )
    result = df.copy()
    result[branch_col] = result[branch_col].map(canonical_names)
    return result


# Physical relocation dates, confirmed by data owner (2026-08-11) — see
# docs/outlet_relocation_notes.md. Keyed by canonical `Nama Outlet` (the name
# branches are renamed to by canonicalize_branch_names), since merged old/new
# branch codes share one canonical identity by the time this runs.
#
# Tigaraksa/Cadas/Citayam/Bintara: EXACT dates — the first day of native data
# under the new code, directly observable in the raw data (old code stops,
# new code starts within days).
#
# Mayor Oking/Cikarang Pusat/Teluk Pucung/Bukit Gading Balaraja/Grand Wisata
# Bekasi: the old code never stops appearing in the current data — data owner
# confirmed (2026-08-11) these relocations happened after the dataset's
# coverage ends, so no exact date is derivable. These use the LAST date the
# old code appears as a lower-bound proxy: every pre-relocation row correctly
# gets a negative days_since_relocation, but the magnitude under-estimates
# the true distance to relocation (which happened sometime after this date).
# Re-derive properly once a data refresh shows the old code stop / a new
# code start.
RELOCATION_DATES: dict[str, pd.Timestamp] = {
    "KY056 - Kebuli Yaman Tigaraksa": pd.Timestamp("2024-03-01"),
    "Kebuli Yaman Cadas": pd.Timestamp("2025-10-03"),
    "Kebuli Yaman Citayam": pd.Timestamp("2025-11-07"),
    "Kebuli Yaman Bintara": pd.Timestamp("2025-11-28"),
    "Kebuli Yaman Mayor Oking": pd.Timestamp("2025-12-31"),  # lower bound
    "Kebuli Yaman Cikarang Pusat": pd.Timestamp("2025-11-30"),  # lower bound
    "Kebuli Yaman Teluk Pucung": pd.Timestamp("2025-12-31"),  # lower bound
    "Kebuli Yaman Bukit Gading Balaraja": pd.Timestamp("2025-12-31"),  # lower bound
    "Kebuli Yaman Grand Wisata Bekasi": pd.Timestamp("2025-12-31"),  # lower bound
}


def add_relocation_feature(
    df: pd.DataFrame,
    relocation_dates: dict[str, pd.Timestamp] = RELOCATION_DATES,
    branch_col: str = "Nama Cabang",
    date_col: str = "Tanggal",
) -> pd.DataFrame:
    result = df.copy()
    relocation_date = result[branch_col].map(relocation_dates)
    result["days_since_relocation"] = (result[date_col] - relocation_date).dt.days
    return result


def build_outlet_features(
    branch_names: list[str], outlets_df: pd.DataFrame, overrides_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for nama_cabang in branch_names:
        outlet_row, kota_override = match_branch_to_outlet(nama_cabang, outlets_df, overrides_df)
        if outlet_row is None:
            rows.append(
                {
                    "Nama Cabang": nama_cabang,
                    "kota": "Unknown",
                    "has_shopee": float("nan"),
                    "has_gofood": float("nan"),
                    "has_grabfood": float("nan"),
                    "can_order_online": float("nan"),
                }
            )
            continue

        channels = {col: _to_bool(outlet_row[col]) for col in CHANNEL_COLS}
        can_order_online = (
            float("nan")
            if any(pd.isna(v) for v in channels.values())
            else any(channels.values())
        )
        rows.append(
            {
                "Nama Cabang": nama_cabang,
                "kota": normalize_kota(outlet_row["Kota"], kota_override),
                **channels,
                "can_order_online": can_order_online,
            }
        )
    return pd.DataFrame(rows)
