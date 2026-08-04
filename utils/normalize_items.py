import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_FILE = str(BASE_DIR / "dataset/dataset.csv")
DATE_FORMAT = "%d %b %Y"

XXX_PREFIX_RE = re.compile(r"^xxx\.\s*", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"(?<=[A-Za-z])\.(?=\d)")
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
# Also matches the digit->lowercase-letter boundary (zero-width, no \s there) so
# that "250ml" and "250 ml" normalize to the same value for comparison.
WHITESPACE_RE = re.compile(r"(?<=\d)(?=[a-z])|(\s+)")


def strip_xxx_prefix(value: str) -> str:
    return XXX_PREFIX_RE.sub("", value)


def unify_separator(code: str) -> str:
    return SEPARATOR_RE.sub("-", code)


def normalize_name_for_comparison(name: str) -> str:
    name = strip_xxx_prefix(name)
    name = TRAILING_PAREN_RE.sub("", name)
    name = WHITESPACE_RE.sub(" ", name).strip()
    return name


def resolve_conditional_normalization(
    df: pd.DataFrame, transform, code_col: str = "Kode Barang", name_col: str = "Nama Barang"
) -> dict[str, str]:
    names_by_raw = df.groupby(code_col)[name_col].apply(
        lambda s: {normalize_name_for_comparison(n) for n in s}
    )
    transformed = {raw: transform(raw) for raw in names_by_raw.index}
    groups: dict[str, list[str]] = {}
    for raw, t in transformed.items():
        groups.setdefault(t, []).append(raw)

    mapping: dict[str, str] = {}
    for t, raws in groups.items():
        all_names: set[str] = set()
        for raw in raws:
            all_names |= names_by_raw[raw]
        if len(all_names) == 1:
            for raw in raws:
                mapping[raw] = t
        else:
            for raw in raws:
                mapping[raw] = raw
    return mapping


def build_normalized_code_map(
    df: pd.DataFrame, code_col: str = "Kode Barang", name_col: str = "Nama Barang"
) -> dict[str, str]:
    pass1 = resolve_conditional_normalization(df, strip_xxx_prefix, code_col, name_col)

    df_pass1 = df.copy()
    df_pass1[code_col] = df_pass1[code_col].map(pass1)

    pass2 = resolve_conditional_normalization(df_pass1, unify_separator, code_col, name_col)

    return {raw: pass2[pass1[raw]] for raw in pass1}


def apply_item_normalization(df: pd.DataFrame) -> pd.DataFrame:
    code_map = build_normalized_code_map(df)
    result = df.copy()
    result["Kode Barang"] = result["Kode Barang"].map(code_map)
    return result


def canonicalize_item_names(
    df: pd.DataFrame, code_col: str = "Kode Barang", name_col: str = "Nama Barang"
) -> pd.DataFrame:
    result = df.copy()
    is_clean = result[name_col].map(strip_xxx_prefix) == result[name_col]
    canonical = result.loc[is_clean].groupby(code_col)[name_col].first()
    result[name_col] = result[code_col].map(canonical).fillna(result[name_col])
    return result


AGG_SPEC = {"Kuantitas": "sum", "Kategori Barang": "first", "Nama Barang": "first", "Satuan": "first"}

EXCLUDED_BRANCHES = {"Kebab Saudagar - Kutabumi"}


def exclude_branches(
    df: pd.DataFrame, branches: set[str] = EXCLUDED_BRANCHES, branch_col: str = "Nama Cabang"
) -> pd.DataFrame:
    return df[~df[branch_col].isin(branches)].reset_index(drop=True)


EXCLUDED_ITEMS = {"xxx.FGS.00066", "xxx.FGS.00069"}


def exclude_items(
    df: pd.DataFrame, items: set[str] = EXCLUDED_ITEMS, code_col: str = "Kode Barang"
) -> pd.DataFrame:
    return df[~df[code_col].isin(items)].reset_index(drop=True)


EXPLICIT_ITEM_RENAMES: dict[str, tuple[str, str]] = {
    "xxx.FGS.00067": ("FGS-00068", "Ayam Crispy Spicy - FG"),
}


def apply_item_renames(
    df: pd.DataFrame,
    renames: dict[str, tuple[str, str]] = EXPLICIT_ITEM_RENAMES,
    code_col: str = "Kode Barang",
    name_col: str = "Nama Barang",
) -> pd.DataFrame:
    result = df.copy()
    matched = result[code_col].isin(renames)
    raw_codes = result.loc[matched, code_col]
    result.loc[matched, code_col] = raw_codes.map(lambda c: renames[c][0])
    result.loc[matched, name_col] = raw_codes.map(lambda c: renames[c][1])
    return result


def reaggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Kode Barang", "Tanggal", "Nama Cabang"], as_index=False).agg(AGG_SPEC)


def load_and_normalize(path: str = RAW_DATA_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format=DATE_FORMAT)
    df = exclude_branches(df)
    df = apply_item_renames(df)
    df = exclude_items(df)
    df = apply_item_normalization(df)
    df = canonicalize_item_names(df)
    df = reaggregate_daily(df)
    return df
