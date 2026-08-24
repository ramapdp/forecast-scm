import re
from pathlib import Path

import pandas as pd

from utils.merge_split_data import merge_dataset

BASE_DIR = Path(__file__).resolve().parents[2]

# Single source of truth: the file merge_dataset writes is the file every
# downstream stage reads. Kept as a reference rather than a second literal so
# a move of the merged output cannot leave this pointing at a path that no
# longer exists.
RAW_DATA_FILE = merge_dataset.OUTPUT_FILE
DATE_FORMAT = "%d %b %Y"

XXX_PREFIX_RE = re.compile(r"^xxx\.\s*", re.IGNORECASE)
SEPARATOR_RE = re.compile(r"(?<=[A-Za-z])\.(?=\d)")
TRAILING_PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")
# Also matches the digit->lowercase-letter boundary (zero-width, no \s there) so
# that "250ml" and "250 ml" normalize to the same value for comparison.
WHITESPACE_RE = re.compile(r"(?<=\d)(?=[a-z])|(\s+)")


# ── TEXT NORMALIZATION HELPERS ───────────────────────────────────────────────

def strip_xxx_prefix(value: str) -> str:
    """Buang prefix 'xxx.' (case-insensitive) dari kode barang.

    Prefix ini muncul di data sumber lama sebagai penanda sementara dan tidak
    membawa makna bisnis — menghapusnya menyatukan kode yang sebenarnya sama.
    """
    return XXX_PREFIX_RE.sub("", value)


def unify_separator(code: str) -> str:
    """Ganti titik antara huruf dan angka dengan strip (mis. 'FGS.00001' → 'FGS-00001').

    Data sumber menggunakan campuran titik dan strip sebagai separator pada
    kode barang. Penyatuan ke strip memungkinkan kode yang sama dikenali sebagai
    kunci yang identik tanpa mengubah makna.
    """
    return SEPARATOR_RE.sub("-", code)


def normalize_name_for_comparison(name: str) -> str:
    name = strip_xxx_prefix(name)
    name = TRAILING_PAREN_RE.sub("", name)
    name = WHITESPACE_RE.sub(" ", name).strip()
    return name



# ── CODE MAP BUILDING ────────────────────────────────────────────────────────

def resolve_conditional_normalization(
    df: pd.DataFrame, transform, code_col: str = "Kode Barang", name_col: str = "Nama Barang"
) -> dict[str, str]:
    """Terapkan transform pada kode barang, tapi hanya jika semua kode yang tergabung
    setelah transform punya nama barang yang identik (setelah normalisasi nama).

    Ini mencegah dua item berbeda secara fisik (mis. dua SKU dengan nama berbeda)
    digabungkan hanya karena kode mereka mirip setelah transformasi. Jika ada
    konflik nama, kode asli dipertahankan tanpa perubahan.
    """
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
    """Bangun peta kode_asli → kode_normal dalam dua pass berurutan.

    Pass 1: strip prefix 'xxx.' (kondisional — lihat resolve_conditional_normalization).
    Pass 2: ganti titik-sebagai-separator dengan strip, diterapkan pada hasil pass 1.

    Dua pass terpisah diperlukan karena transformasi pertama bisa mengubah
    konteks untuk transformasi kedua: kode yang tadinya identik setelah strip-xxx
    mungkin masih berbeda titik/strip-nya.
    """
    pass1 = resolve_conditional_normalization(df, strip_xxx_prefix, code_col, name_col)

    df_pass1 = df.copy()
    df_pass1[code_col] = df_pass1[code_col].map(pass1)

    pass2 = resolve_conditional_normalization(df_pass1, unify_separator, code_col, name_col)

    return {raw: pass2[pass1[raw]] for raw in pass1}


def apply_item_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan peta normalisasi kode ke seluruh DataFrame, mengganti kolom Kode Barang."""
    code_map = build_normalized_code_map(df)
    result = df.copy()
    result["Kode Barang"] = result["Kode Barang"].map(code_map)
    return result



# ── CATEGORY CANONICALIZATION ─────────────────────────────────────────────────

def canonicalize_item_names(
    df: pd.DataFrame, code_col: str = "Kode Barang", name_col: str = "Nama Barang"
) -> pd.DataFrame:
    """Seragamkan Nama Barang tiap kode ke satu nama kanonik (nama tanpa prefix 'xxx.').

    Baris dengan prefix 'xxx.' dianggap nama sementara; baris bersih dijadikan
    referensi. Jika semua baris suatu kode masih pakai prefix, nama aslinya dipertahankan.
    """
    result = df.copy()
    is_clean = result[name_col].map(strip_xxx_prefix) == result[name_col]
    canonical = result.loc[is_clean].groupby(code_col)[name_col].first()
    result[name_col] = result[code_col].map(canonical).fillna(result[name_col])
    return result


CATEGORY_SYNONYMS: dict[str, str] = {
    "Minuman": "Minuman - FG",
    "Snack": "Snack (FG)",
}

# SKUs whose recorded category must be rewritten for their whole history.
# The synonym-collapse rule above cannot express these: it only merges labels
# that mean the same thing everywhere, while each entry here is a statement
# about one specific SKU.
#
# 2026-08-10 (data owner) — FGS-00014 (Club Mineral 600ml) was recorded as
# WIP-2 early on but is actually a drink, so it reads as Minuman - FG for its
# whole history. That confirmation also established the general rule that
# Barang Semi FG (WIP-2) and Barang Jadi (FG) are genuinely different
# categories, which still holds and is enforced in
# canonicalize_item_categories() below.
#
# 2026-08-22 (data owner) — supersedes the 2026-08-10 confirmation for the ten
# SKUs listed below, and only for those. WIP-2 turned out to be an old
# administrative label for them: how the goods are handled never changed, only
# the category name was updated. Their whole history therefore reads as Barang
# Jadi (FG) rather than being left time-varying. FGS-00014 is NOT part of this
# group — it moves to Minuman - FG, not FG, and its 2026-08-10 entry stands.
_ADMINISTRATIVE_WIP2_RELABEL = [
    "FGS-00001",  # Ayam Kebuli (0.9)
    "FGS-00002",  # Kambing Kebuli
    "FGS-00003",  # Iga Sapi Kebuli
    "FGS-00004",  # Nasi Kebuli
    "FGS-00005",  # Sambal - FG
    "FGS-00012",  # Samosa Beef Original (RM)
    "FGS-00013",  # Samosa Beef Spicy (RM)
    "FGS-00018",  # Kambing Kebuli Aqiqah Betina - FG
    "FGS-00049",  # Iga Dino - FG
    "FGS-00053",  # Ayam Kebuli (0.6)
]

EXPLICIT_CATEGORY_OVERRIDES: dict[str, str] = {
    "FGS-00014": "Minuman - FG",
    **{code: "Barang Jadi (FG)" for code in _ADMINISTRATIVE_WIP2_RELABEL},
}


def canonicalize_item_categories(
    df: pd.DataFrame,
    code_col: str = "Kode Barang",
    category_col: str = "Kategori Barang",
    date_col: str = "Tanggal",
) -> pd.DataFrame:
    """Seragamkan kategori per SKU menggunakan sinonim dan override eksplisit.

    Dua langkah:
    1. Sinonim (CATEGORY_SYNONYMS): satukan varian penamaan yang identik secara
       bisnis (mis. 'Minuman' → 'Minuman - FG') hanya jika SKU itu memang
       hanya punya satu kategori setelah penyatuan.
    2. Override eksplisit (EXPLICIT_CATEGORY_OVERRIDES): tulis ulang kategori
       untuk SKU tertentu yang sudah dikonfirmasi pemilik data, terlepas dari
       apa yang tertulis di sumber.
    """
    result = df.copy()
    normalized = result[category_col].replace(CATEGORY_SYNONYMS)

    # Only collapse a SKU's history to one category when every recorded
    # category is the same after synonym normalization (e.g. Minuman /
    # Minuman - FG). Genuinely different categories (e.g. WIP-2 vs Barang
    # Jadi (FG)) are left time-varying, as recorded.
    single_category = normalized.groupby(result[code_col]).transform("nunique") == 1
    result.loc[single_category, category_col] = normalized.loc[single_category]

    override_mask = result[code_col].isin(EXPLICIT_CATEGORY_OVERRIDES)
    result.loc[override_mask, category_col] = result.loc[override_mask, code_col].map(
        EXPLICIT_CATEGORY_OVERRIDES
    )

    return result


AGG_SPEC = {"Kuantitas": "sum", "Kategori Barang": "first", "Nama Barang": "first", "Satuan": "first"}

# Kebab Saudagar is a different brand that briefly issued goods through this
# system (2025-12-20..2025-12-31, 137 rows). Data owner confirmed 2026-08-15
# that it is no longer operating and its data is not needed.
EXCLUDED_BRANCHES = {"Kebab Saudagar - Kutabumi"}


# ── UNIT CONVERSION & EXCLUSIONS ─────────────────────────────────────────────

def exclude_branches(
    df: pd.DataFrame, branches: set[str] = EXCLUDED_BRANCHES, branch_col: str = "Nama Cabang"
) -> pd.DataFrame:
    """Buang semua baris dari cabang yang ada di daftar EXCLUDED_BRANCHES."""
    return df[~df[branch_col].isin(branches)].reset_index(drop=True)


GRAM_TO_PORSI_FACTORS: dict[str, int] = {
    "xxx.FGS.00070": 40,  # Santan Cendol
    "xxx.FGS.00071": 30,  # Gula Cendol
}


def convert_gram_items_to_porsi(
    df: pd.DataFrame,
    factors: dict[str, int] = GRAM_TO_PORSI_FACTORS,
    code_col: str = "Kode Barang",
    unit_col: str = "Satuan",
    qty_col: str = "Kuantitas",
) -> pd.DataFrame:
    """Konversi kuantitas item tertentu dari Gram ke Porsi menggunakan faktor konversi.

    Item seperti Santan/Gula Cendol dicatat dalam gram di sumber, tetapi
    secara operasional lebih bermakna dalam porsi. Konversi dilakukan hanya
    pada baris dengan satuan 'Gr' supaya baris yang sudah benar tidak tersentuh.
    """
    result = df.copy()
    result[qty_col] = result[qty_col].astype(float)
    for code, factor in factors.items():
        mask = (result[code_col] == code) & (result[unit_col] == "Gr")
        result.loc[mask, qty_col] = result.loc[mask, qty_col] / factor
        result.loc[mask, unit_col] = "Porsi"
    return result


EXCLUDED_ITEMS = {"xxx.FGS.00066", "xxx.FGS.00067", "xxx.FGS.00068", "xxx.FGS.00069"}


def exclude_items(
    df: pd.DataFrame, items: set[str] = EXCLUDED_ITEMS, code_col: str = "Kode Barang"
) -> pd.DataFrame:
    """Buang semua baris dari item yang ada di daftar EXCLUDED_ITEMS."""
    return df[~df[code_col].isin(items)].reset_index(drop=True)


EXPLICIT_ITEM_RENAMES: dict[str, tuple[str, str]] = {}


def apply_item_renames(
    df: pd.DataFrame,
    renames: dict[str, tuple[str, str]] = EXPLICIT_ITEM_RENAMES,
    code_col: str = "Kode Barang",
    name_col: str = "Nama Barang",
) -> pd.DataFrame:
    """Ganti kode dan nama item tertentu menggunakan peta rename eksplisit.

    Berbeda dari normalisasi otomatis: rename eksplisit dipakai ketika kode
    lama dan baru tidak bisa disatukan oleh aturan regex (mis. penggantian
    kode total oleh pemilik data).
    """
    result = df.copy()
    matched = result[code_col].isin(renames)
    raw_codes = result.loc[matched, code_col]
    result.loc[matched, code_col] = raw_codes.map(lambda c: renames[c][0])
    result.loc[matched, name_col] = raw_codes.map(lambda c: renames[c][1])
    return result



# ── PIPELINE ENTRY POINT ─────────────────────────────────────────────────────

def reaggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Agregasi ulang ke level harian per (item, cabang) setelah normalisasi kode.

    Normalisasi kode bisa menggabungkan beberapa kode yang tadinya berbeda
    menjadi satu. Tanpa agregasi ulang ini, hasil penggabungan akan punya
    banyak baris duplikat per (item, cabang, tanggal).
    """
    return df.groupby(["Kode Barang", "Tanggal", "Nama Cabang"], as_index=False).agg(AGG_SPEC)


def load_and_normalize(path: str = RAW_DATA_FILE) -> pd.DataFrame:
    """Muat dataset CSV mentah lalu jalankan seluruh pipeline normalisasi item.

    Urutan langkah dipentingkan:
    1. Exclude cabang & rename eksplisit — sebelum normalisasi kode supaya
       mapping tidak mencampur item dari cabang yang akan dibuang.
    2. Konversi unit & exclude item — sebelum normalisasi kode.
    3. Normalisasi kode (xxx-prefix + separator) — menyatukan kode yang sama.
    4. Kanonikalisasi nama & kategori — setelah kode final diketahui.
    5. Re-agregasi harian — setelah normalisasi kode bisa menggabungkan baris.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format=DATE_FORMAT)
    df = exclude_branches(df)
    df = apply_item_renames(df)
    df = convert_gram_items_to_porsi(df)
    df = exclude_items(df)
    df = apply_item_normalization(df)
    df = canonicalize_item_names(df)
    df = canonicalize_item_categories(df)
    df = reaggregate_daily(df)
    return df
