"""Refresh gate: after normalization, no SKU may carry more than one category.

Run after every data refresh, before anything downstream is rebuilt:

    .venv/bin/python3 -m utils.eda.verify_category_consistency

Exits 1 when the gate fails, so it can sit in a refresh script.

Two layers, answering different questions:

  SOURCE       dataset.csv as exported. Reported for context only -- never
               fatal. The 2026-08-22 relabel of ten SKUs is applied in
               normalize_items.EXPLICIT_CATEGORY_OVERRIDES rather than by
               re-exporting the source, so WIP-2 rows are expected to remain
               in dataset.csv indefinitely. A gate that failed on them would
               fail on every future refresh and stop being read.

  NORMALIZED   after normalize_items.load_and_normalize(). This is the gate.
               It cannot mask a source problem: canonicalize_item_categories()
               deliberately leaves genuinely different categories time-varying,
               so anything the overrides do not cover surfaces here.
"""

import sys
from pathlib import Path

import pandas as pd

from utils.data_preprocessing import normalize_items

RETIRED_CATEGORY = "Barang Semi FG (WIP-2)"



# ── PEMERIKSAAN KONSISTENSI ───────────────────────────────────────────────────────

def find_multi_category_skus(
    df: pd.DataFrame,
    code_col: str = "Kode Barang",
    category_col: str = "Kategori Barang",
    apply_synonyms: bool = False,
) -> dict[str, list[str]]:
    """Kode Barang -> kategori yang tercatat, untuk SKU yang punya lebih dari satu.

    `apply_synonyms` menyatukan varian penamaan (Minuman / Minuman - FG) lebih
    dulu. Dipakai pada lapis sumber, di mana varian itu masih ada dan bukan
    perbedaan kategori yang sesungguhnya.
    """
    categories = df[category_col]
    if apply_synonyms:
        categories = categories.replace(normalize_items.CATEGORY_SYNONYMS)
    per_sku = categories.groupby(df[code_col]).unique()
    return {code: sorted(values) for code, values in per_sku.items() if len(values) > 1}


def build_report(raw: pd.DataFrame, normalized: pd.DataFrame) -> dict:
    """Susun laporan hasil pemeriksaan dari sumber data mentah dan ternormalisasi."""
    normalized_multi = find_multi_category_skus(normalized)
    return {
        "source_wip2_rows": int((raw["Kategori Barang"] == RETIRED_CATEGORY).sum()),
        "source_multi_category": find_multi_category_skus(raw, apply_synonyms=True),
        "normalized_multi_category": normalized_multi,
        "categories": sorted(normalized["Kategori Barang"].unique()),
        # Only the normalized layer decides. See the module docstring.
        "passed": not normalized_multi,
    }



# ── PELAPORAN & I/O ───────────────────────────────────────────────────────────────

def format_report(report: dict) -> str:
    """Format hasil laporan pemeriksaan menjadi string terstruktur."""
    lines = ["=== Lapis sumber (informasional, tidak menggagalkan) ==="]
    lines.append(
        f"  baris {RETIRED_CATEGORY}: {report['source_wip2_rows']:,}"
        "  -- normal, ditangani di lapisan normalisasi"
    )
    source_multi = report["source_multi_category"]
    lines.append(f"  SKU dengan kategori bervariasi di sumber: {len(source_multi)}")

    lines.append("\n=== Lapis normalisasi (gerbang) ===")
    normalized_multi = report["normalized_multi_category"]
    if normalized_multi:
        lines.append(
            f"  GAGAL - {len(normalized_multi)} SKU masih punya kategori bervariasi"
            " setelah normalisasi:"
        )
        for code, categories in sorted(normalized_multi.items()):
            lines.append(f"    {code:<16} {' | '.join(categories)}")
        lines.append(
            "\n  Reklasifikasi belum sepenuhnya tertangani. Periksa apakah SKU di"
            "\n  atas perlu masuk normalize_items.EXPLICIT_CATEGORY_OVERRIDES, atau"
            "\n  konfirmasikan ke pemilik data."
        )
    else:
        lines.append("  LULUS - setiap SKU hanya punya satu kategori.")

    lines.append(
        f"\n  kategori tersisa ({len(report['categories'])}): "
        f"{', '.join(report['categories'])}"
    )
    if RETIRED_CATEGORY in report["categories"]:
        lines.append(f"  CATATAN: {RETIRED_CATEGORY} masih muncul setelah normalisasi.")
    return "\n".join(lines)


def main(raw_path: Path = None) -> int:
    """Entry point: baca data, jalankan gate, kembalikan exit code (0 sukses, 1 gagal)."""
    raw_path = raw_path or normalize_items.RAW_DATA_FILE
    raw = pd.read_csv(
        raw_path,
        sep=";",
        encoding="utf-8-sig",
        usecols=["Kode Barang", "Kategori Barang"],
    )
    normalized = normalize_items.load_and_normalize(str(raw_path))
    report = build_report(raw, normalized)
    print(format_report(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
