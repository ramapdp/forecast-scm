"""
Menghasilkan dataset/item_cost_margin.csv berisi SATU BARIS PER Kode Barang
AKTUAL dari dataset Anda (bukan tebakan), dengan kolom biaya dikosongkan
untuk diisi tim SCM.

Jalankan sebagai modul dari root repo:
    .venv/bin/python3 -m utils.generate_item_cost_margin_template

Aman dijalankan berkali-kali: jika dataset/item_cost_margin.csv sudah ada,
skrip TIDAK menimpa baris yang sudah terisi -- ia hanya MENAMBAHKAN baris
untuk Kode Barang baru yang belum ada di berkas, supaya progres pengisian
tim SCM tidak pernah hilang saat SKU baru muncul di pemutakhiran data.
"""

import csv
from pathlib import Path

import pandas as pd

# Berkas ini ada di utils/, jadi root repo adalah dua tingkat di atasnya --
# sama seperti seluruh modul lain di paket ini.
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_INPUT_PATH = BASE_DIR / "dataset/model_ready/model_input.parquet"
OUTPUT_PATH = BASE_DIR / "dataset/item_cost_margin.csv"

COLUMNS = [
    "Kode Barang",
    "unit_cost",
    "unit_margin",
    "shelf_life_days",
    "salvage_value_ratio",
    "cost_source",
    "cost_confidence",
    "last_updated",
    "shelf_life_rank_override",
]

# Sentinel shelf_life_days untuk kategori yang praktis tidak kedaluwarsa,
# dipakai sebagai DEFAULT AWAL supaya tim SCM tidak perlu mengisi 365 secara
# manual untuk setiap SKU Packaging/Barang Umum -- tetap bisa ditimpa manual
# jika ada item spesifik yang berbeda.
LONG_SHELF_LIFE_CATEGORIES = {"Packaging", "Barang Umum"}
LONG_SHELF_LIFE_SENTINEL_DAYS = 365


def resolve_latest_category(
    df: pd.DataFrame,
    code_col: str = "Kode Barang",
    category_col: str = "Kategori Barang",
    date_col: str = "Tanggal",
) -> dict[str, str]:
    """Kode Barang -> kategori TERBARU yang tercatat untuk SKU itu.

    Sebagian SKU berpindah kategori sepanjang riwayatnya (mis. dicatat
    sebagai WIP-2 di awal 2024 lalu direklasifikasi jadi FG). Mengambil
    kemunculan pertama membuat hasilnya bergantung urutan baris di parquet;
    kategori pada tanggal terakhir adalah yang berlaku sekarang.
    """
    ordered = df.sort_values(date_col, kind="stable")
    return ordered.groupby(code_col)[category_col].last().to_dict()


def load_sku_to_category(path: Path = MODEL_INPUT_PATH) -> dict[str, str]:
    """Kode Barang -> Kategori Barang, dibaca dari model_input.parquet
    (bukan category_mapping.json, karena mapping.json hanya menyimpan
    indeks, bukan nama kategori per SKU)."""
    df = pd.read_parquet(
        path, columns=["Kode Barang", "Kategori Barang", "Tanggal"]
    )
    mapping = resolve_latest_category(df)
    if not mapping:
        raise ValueError(
            f"Tidak ada baris terbaca dari {path} -- "
            "periksa apakah path/kolom masih sesuai."
        )
    return mapping


def load_existing_rows(path: Path = OUTPUT_PATH) -> dict[str, dict]:
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        return {row["Kode Barang"]: row for row in reader}


# Kandidat override peringkat masa simpan untuk SKU yang secara fisik tidak
# representatif terhadap kategorinya (lihat "Catatan heterogenitas" di
# 2026-08-22-segmented-quantile-allocation-design.md, Bagian 3). Dua entri
# bertanda [SEMENTARA] masih menunggu konfirmasi tim SCM.
SHELF_LIFE_RANK_OVERRIDES = {
    "FGS-00054": 3,  # India Salaam Basmati Rice @1kg -- beras mentah
    "FGS-00012": 4,  # Samosa Beef Original (RM) -- beku kemasan
    "FGS-00013": 4,  # Samosa Beef Spicy (RM) -- beku kemasan
    "FGS-00011": 3,  # Saus Extra Delmonte @8gr -- sachet pabrik
    "FGS-00065": 3,  # Saus Tomat Delmonte @8gr -- sachet pabrik
    "FGS.00055": 4,  # Saus Lemon -- [SEMENTARA] perlu konfirmasi SCM
    "FGS.00056": 4,  # Saus Spicy -- [SEMENTARA] perlu konfirmasi SCM
}


def build_blank_row(kode_barang: str, kategori: str) -> dict:
    shelf_life = (
        str(LONG_SHELF_LIFE_SENTINEL_DAYS)
        if kategori in LONG_SHELF_LIFE_CATEGORIES
        else ""
    )
    return {
        "Kode Barang": kode_barang,
        "unit_cost": "",
        "unit_margin": "",
        "shelf_life_days": shelf_life,
        "salvage_value_ratio": "",
        "cost_source": "",
        "cost_confidence": "rendah",  # default aman: jatuh ke jalur proksi
        "last_updated": "",
        "shelf_life_rank_override": str(
            SHELF_LIFE_RANK_OVERRIDES.get(kode_barang, "")
        ),
    }


def detect_shelf_life_mismatches(
    existing: dict[str, dict], sku_to_category: dict[str, str]
) -> list[dict]:
    """Baris lama yang shelf_life_days-nya tidak lagi cocok dengan kategori
    terkini SKU tersebut.

    Mengembalikan temuan alih-alih mencetaknya, mengikuti pola
    detect_unrecorded_gaps() di outlet_features.py: deteksi memberi
    peringatan, manusia yang memutuskan. Skrip ini tidak pernah menimpa
    baris yang sudah ada, jadi sentinel baris lama tidak pernah dihitung
    ulang -- tanpa deteksi ini, SKU yang berpindah kategori akan diam-diam
    membawa sentinel yang salah.

    Dua kondisi yang dianggap tidak sesuai:

    - kategori sekarang berumur simpan panjang tetapi shelf_life_days KOSONG
      (sentinel tidak pernah terpasang);
    - shelf_life_days masih berisi sentinel padahal kategorinya sekarang
      bukan lagi kategori berumur simpan panjang (sentinel usang).

    Nilai non-kosong selain sentinel TIDAK pernah ditandai: sentinel hanya
    default awal, dan angka yang diisi tim SCM adalah jawaban, bukan galat.
    """
    findings = []
    for kode_barang in sorted(existing):
        kategori = sku_to_category.get(kode_barang)
        if kategori is None:
            continue  # SKU tidak ada di dataset -- di luar wewenang fungsi ini
        shelf_life = (existing[kode_barang].get("shelf_life_days") or "").strip()
        sentinel_hilang = kategori in LONG_SHELF_LIFE_CATEGORIES and shelf_life == ""
        sentinel_usang = (
            kategori not in LONG_SHELF_LIFE_CATEGORIES
            and shelf_life == str(LONG_SHELF_LIFE_SENTINEL_DAYS)
        )
        if not (sentinel_hilang or sentinel_usang):
            continue
        findings.append({
            "Kode Barang": kode_barang,
            "kategori_sekarang": kategori,
            "shelf_life_days": shelf_life,
            "alasan": "sentinel hilang" if sentinel_hilang else "sentinel usang",
        })
    return findings


def report_mismatches(findings: list[dict]) -> None:
    if not findings:
        return
    print(
        f"\nPERINGATAN: {len(findings)} baris lama punya shelf_life_days yang "
        "tidak cocok dengan kategori terkininya."
    )
    print("Tidak ada yang diubah -- putuskan sendiri apakah perlu disunting:\n")
    for finding in findings:
        print(
            f"  {finding['Kode Barang']:<12} "
            f"shelf_life_days={finding['shelf_life_days'] or '(kosong)':<9} "
            f"kategori sekarang={finding['kategori_sekarang']:<30} "
            f"({finding['alasan']})"
        )


def main(
    model_input_path: Path = MODEL_INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> None:
    sku_to_category = load_sku_to_category(model_input_path)
    existing = load_existing_rows(output_path)

    # Diperiksa SEBELUM baris baru ditambahkan, supaya hanya baris lama yang
    # dinilai -- baris baru selalu lahir dengan sentinel yang benar.
    mismatches = detect_shelf_life_mismatches(existing, sku_to_category)

    added = 0
    for kode_barang, kategori in sorted(sku_to_category.items()):
        if kode_barang not in existing:
            existing[kode_barang] = build_blank_row(kode_barang, kategori)
            added += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        for kode_barang in sorted(existing):
            writer.writerow(existing[kode_barang])

    print(
        f"{output_path}: {len(existing)} baris total, "
        f"{added} baris baru ditambahkan, "
        f"{len(existing) - added} baris lama dipertahankan apa adanya."
    )
    report_mismatches(mismatches)


if __name__ == "__main__":
    main()
