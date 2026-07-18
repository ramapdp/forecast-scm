import merge_dataset

GROUP_FIELD_COUNT = 6  # Tanggal, Kategori Barang, Kode Barang, Nama Barang, Nama Cabang, Satuan


def aggregate_rows(rows: list[list[str]]) -> list[list[str]]:
    totals: dict[tuple[str, ...], int] = {}
    for row in rows:
        key = tuple(row[:GROUP_FIELD_COUNT])
        totals[key] = totals.get(key, 0) + int(row[GROUP_FIELD_COUNT])
    return [list(key) + [str(total)] for key, total in totals.items()]


INPUT_FILE = merge_dataset.OUTPUT_FILE  # "dataset/dataset.csv"


def main(path=INPUT_FILE) -> None:
    rows = merge_dataset.read_rows(path)
    aggregated = aggregate_rows(rows)
    merge_dataset.write_rows(aggregated, path)
    print(f"Aggregated {len(rows)} rows into {len(aggregated)} rows, wrote to {path}")


if __name__ == "__main__":
    main()
