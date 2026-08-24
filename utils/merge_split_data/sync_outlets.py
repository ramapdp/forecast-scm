import csv
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

SOURCE_FILE = str(BASE_DIR / "dataset/outlets.json")
OUTPUT_FILE = str(BASE_DIR / "dataset/outlets.csv")

FIELDNAMES = ["Nama Outlet", "Alamat", "Kecamatan", "Kota", "has_shopee", "has_gofood", "has_grabfood"]


def load_outlets(path: str = SOURCE_FILE) -> list[dict]:
    """Muat data mentah outlet dari file JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_outlets(outlets: list[dict], path: str = OUTPUT_FILE) -> None:
    """Simpan data outlet ke CSV agar mudah digabungkan dengan dataset utama."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, delimiter=";")
        writer.writeheader()
        writer.writerows(outlets)


def main(source_path: str = SOURCE_FILE, output_path: str = OUTPUT_FILE) -> None:
    """Entry point: konversi JSON menjadi CSV (delimiter titik koma, utf-8-sig)."""
    outlets = load_outlets(source_path)
    write_outlets(outlets, output_path)
    print(f"Wrote {len(outlets)} outlets to {output_path}")


if __name__ == "__main__":
    main()
