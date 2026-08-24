import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from utils.eda import generate_item_cost_margin_template as gen


def _write_model_input(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows: (Kode Barang, Kategori Barang, Tanggal)."""
    pd.DataFrame(
        {
            "Kode Barang": [r[0] for r in rows],
            "Kategori Barang": [r[1] for r in rows],
            "Tanggal": pd.to_datetime([r[2] for r in rows]),
        }
    ).to_parquet(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=gen.COLUMNS, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["Kode Barang"]: r for r in csv.DictReader(f, delimiter=";")}


class TestResolveLatestCategory(unittest.TestCase):
    def test_time_varying_sku_resolves_to_latest_category(self):
        # FGS-00002 was recorded as WIP-2 in early 2024 and reclassified to FG
        # later. The latest recorded category is the one that counts.
        df = pd.DataFrame(
            {
                "Kode Barang": ["FGS-00002", "FGS-00002"],
                "Kategori Barang": ["Barang Semi FG (WIP-2)", "Barang Jadi (FG)"],
                "Tanggal": pd.to_datetime(["2024-01-15", "2025-06-01"]),
            }
        )
        self.assertEqual(
            gen.resolve_latest_category(df), {"FGS-00002": "Barang Jadi (FG)"}
        )

    def test_result_does_not_depend_on_row_order(self):
        # The old drop_duplicates() implementation took whichever row came
        # first, so shuffling the frame changed the answer.
        rows = {
            "Kode Barang": ["FGS-00002", "FGS-00002"],
            "Kategori Barang": ["Barang Jadi (FG)", "Barang Semi FG (WIP-2)"],
            "Tanggal": pd.to_datetime(["2025-06-01", "2024-01-15"]),
        }
        self.assertEqual(
            gen.resolve_latest_category(pd.DataFrame(rows)),
            {"FGS-00002": "Barang Jadi (FG)"},
        )

    def test_stable_sku_keeps_its_only_category(self):
        df = pd.DataFrame(
            {
                "Kode Barang": ["PCG-00001", "PCG-00001"],
                "Kategori Barang": ["Packaging", "Packaging"],
                "Tanggal": pd.to_datetime(["2024-03-01", "2025-09-09"]),
            }
        )
        self.assertEqual(gen.resolve_latest_category(df), {"PCG-00001": "Packaging"})


class TestDetectShelfLifeMismatches(unittest.TestCase):
    def _row(self, kode: str, shelf_life: str) -> dict:
        row = gen.build_blank_row(kode, "Barang Jadi (FG)")
        row["shelf_life_days"] = shelf_life
        return row

    def test_flags_empty_sentinel_when_category_is_now_long_shelf_life(self):
        existing = {"FGS-00070": self._row("FGS-00070", "")}
        findings = gen.detect_shelf_life_mismatches(
            existing, {"FGS-00070": "Packaging"}
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["Kode Barang"], "FGS-00070")
        self.assertEqual(findings[0]["kategori_sekarang"], "Packaging")
        self.assertEqual(findings[0]["shelf_life_days"], "")

    def test_flags_stale_sentinel_when_category_is_no_longer_long_shelf_life(self):
        existing = {"FGS-00071": self._row("FGS-00071", "365")}
        findings = gen.detect_shelf_life_mismatches(
            existing, {"FGS-00071": "Barang Jadi (FG)"}
        )
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["Kode Barang"], "FGS-00071")
        self.assertEqual(findings[0]["shelf_life_days"], "365")

    def test_no_finding_when_sentinel_matches_category(self):
        existing = {
            "PCG-00001": self._row("PCG-00001", "365"),
            "FGS-00001": self._row("FGS-00001", ""),
        }
        findings = gen.detect_shelf_life_mismatches(
            existing, {"PCG-00001": "Packaging", "FGS-00001": "Barang Jadi (FG)"}
        )
        self.assertEqual(findings, [])

    def test_manual_value_on_long_shelf_life_category_is_not_flagged(self):
        # The sentinel is a starting default the SCM team may deliberately
        # override; a real number is an answer, not a mismatch.
        existing = {"PCG-00002": self._row("PCG-00002", "180")}
        findings = gen.detect_shelf_life_mismatches(
            existing, {"PCG-00002": "Packaging"}
        )
        self.assertEqual(findings, [])

    def test_ignores_rows_whose_sku_is_absent_from_the_dataset(self):
        existing = {"FGS-99999": self._row("FGS-99999", "365")}
        findings = gen.detect_shelf_life_mismatches(existing, {})
        self.assertEqual(findings, [])


class TestMainLeavesMismatchedRowsUntouched(unittest.TestCase):
    def test_warns_about_mismatch_without_rewriting_the_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_input = tmp / "model_input.parquet"
            output = tmp / "item_cost_margin.csv"

            # FGS-00001 is now FG, so its 365 is stale. PCG-00001 is new.
            _write_model_input(
                model_input,
                [
                    ("FGS-00001", "Barang Semi FG (WIP-2)", "2024-01-15"),
                    ("FGS-00001", "Barang Jadi (FG)", "2025-06-01"),
                    ("PCG-00001", "Packaging", "2025-06-01"),
                ],
            )
            stale = gen.build_blank_row("FGS-00001", "Packaging")
            stale["unit_cost"] = "12500"
            _write_csv(output, [stale])

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                gen.main(model_input_path=model_input, output_path=output)
            printed = buffer.getvalue()

            written = _read_csv(output)
            self.assertEqual(written["FGS-00001"], stale)
            self.assertEqual(written["FGS-00001"]["shelf_life_days"], "365")
            self.assertEqual(written["FGS-00001"]["unit_cost"], "12500")

            self.assertIn("FGS-00001", printed)
            self.assertIn("Barang Jadi (FG)", printed)

            # The new SKU still gets its row, with the sentinel for Packaging.
            self.assertEqual(written["PCG-00001"]["shelf_life_days"], "365")
            self.assertEqual(len(written), 2)

    def test_silent_when_every_existing_row_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            model_input = tmp / "model_input.parquet"
            output = tmp / "item_cost_margin.csv"

            _write_model_input(model_input, [("PCG-00001", "Packaging", "2025-06-01")])
            _write_csv(output, [gen.build_blank_row("PCG-00001", "Packaging")])

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                gen.main(model_input_path=model_input, output_path=output)

            self.assertNotIn("PERINGATAN", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
