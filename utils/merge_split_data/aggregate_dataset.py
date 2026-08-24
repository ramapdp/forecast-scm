from . import merge_dataset

GROUP_FIELD_COUNT = 6  # Tanggal, Kategori Barang, Kode Barang, Nama Barang, Nama Cabang, Satuan


# ── PARSING ──────────────────────────────────────────────────────────────────

def parse_kuantitas(value: str) -> float:
    """Ubah string kuantitas (koma sebagai desimal) menjadi float.

    Dataset sumber menggunakan koma sebagai pemisah desimal ('1,5' bukan '1.5'),
    sehingga perlu diganti sebelum dikonversi ke float.
    """
    return float(value.replace(",", "."))


# ── AGGREGASI ────────────────────────────────────────────────────────────────

def aggregate_rows(rows: list[list[str]]) -> list[list[str]]:
    """Jumlahkan kuantitas untuk baris-baris dengan kunci (item, cabang, tanggal) yang sama.

    Baris duplikat muncul ketika satu item terjual beberapa kali di cabang yang
    sama pada tanggal yang sama dan dicatat sebagai baris terpisah di file sumber.
    Fungsi ini melipatnya menjadi satu baris per kunci unik.

    GROUP_FIELD_COUNT = 6 kolom pertama membentuk kunci unik; kolom ke-7
    adalah Kuantitas yang dijumlahkan.
    """
    totals: dict[tuple[str, ...], float] = {}
    for row in rows:
        key = tuple(row[:GROUP_FIELD_COUNT])
        totals[key] = totals.get(key, 0.0) + parse_kuantitas(row[GROUP_FIELD_COUNT])
    return [list(key) + [str(round(total, 1))] for key, total in totals.items()]


INPUT_FILE = merge_dataset.OUTPUT_FILE  # "dataset/dataset.csv"


# ── ENTRY POINT ──────────────────────────────────────────────────────────────

def main(path=INPUT_FILE) -> None:
    """Baca dataset hasil merge, agregasi baris duplikat, lalu tulis kembali ke file yang sama."""
    rows = merge_dataset.read_rows(path)
    aggregated = aggregate_rows(rows)
    merge_dataset.write_rows(aggregated, path)
    print(f"Aggregated {len(rows)} rows into {len(aggregated)} rows, wrote to {path}")


if __name__ == "__main__":
    main()
