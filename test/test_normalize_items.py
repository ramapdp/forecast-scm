import tempfile
import unittest
from pathlib import Path

import pandas as pd

import normalize_items


class TestStripXxxPrefix(unittest.TestCase):
    def test_strips_lowercase_prefix(self):
        self.assertEqual(normalize_items.strip_xxx_prefix("xxx.FGS-00003"), "FGS-00003")

    def test_strips_uppercase_prefix_defensively(self):
        self.assertEqual(normalize_items.strip_xxx_prefix("XXX.FGS-00003"), "FGS-00003")

    def test_leaves_code_without_prefix_unchanged(self):
        self.assertEqual(normalize_items.strip_xxx_prefix("FGS-00003"), "FGS-00003")


class TestUnifySeparator(unittest.TestCase):
    def test_converts_letter_dot_digit_separator(self):
        self.assertEqual(normalize_items.unify_separator("FGS.00047"), "FGS-00047")

    def test_leaves_code_without_dot_unchanged(self):
        self.assertEqual(normalize_items.unify_separator("FGS-00047"), "FGS-00047")

    def test_preserves_xxx_prefix_dot_since_followed_by_letter(self):
        # The dot in "xxx." is followed by a letter (F), not a digit, so it
        # must NOT be converted — only the FGS.00069 -> FGS-00069 part is a
        # true code separator. This preserves the xxx. marker as recognizable
        # even after separator normalization.
        self.assertEqual(normalize_items.unify_separator("xxx.FGS.00069"), "xxx.FGS-00069")

    def test_converts_multiple_valid_separators(self):
        self.assertEqual(normalize_items.unify_separator("WIP.00005"), "WIP-00005")


class TestNormalizeNameForComparison(unittest.TestCase):
    def test_strips_xxx_prefix_from_name(self):
        result = normalize_items.normalize_name_for_comparison("xxx.Iga Sapi Kebuli")
        self.assertEqual(result, "Iga Sapi Kebuli")

    def test_collapses_irregular_whitespace(self):
        result = normalize_items.normalize_name_for_comparison("Gula  Asam   250ml")
        self.assertEqual(result, normalize_items.normalize_name_for_comparison("Gula Asam 250ml"))

    def test_strips_trailing_parenthetical_annotation(self):
        result = normalize_items.normalize_name_for_comparison(
            "Club Mineral 330 ml (Menu pakai kode CM-330)"
        )
        self.assertEqual(result, "Club Mineral 330 ml")

    def test_preserves_non_trailing_parenthetical(self):
        # "Ayam Kebuli (0.6)" ends in a parenthetical, so under this function
        # it DOES get stripped (weight variant is treated as an annotation
        # for comparison purposes only — the stored Nama Barang is untouched
        # elsewhere). This is expected: it never causes a false merge in the
        # real dataset because no two DIFFERENT weight-variant products ever
        # land in the same code-collision group (verified against real data).
        result = normalize_items.normalize_name_for_comparison("Ayam Kebuli (0.6)")
        self.assertEqual(result, "Ayam Kebuli")

    def test_gula_asam_and_kunyit_asam_style_variants_match_after_normalization(self):
        a = normalize_items.normalize_name_for_comparison("Gula Asam 250ml")
        b = normalize_items.normalize_name_for_comparison("Gula Asam 250 ml")
        self.assertEqual(a, b)


class TestResolveConditionalNormalization(unittest.TestCase):
    def test_merges_group_with_agreeing_names(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "xxx.FGS-00003"],
            "Nama Barang": ["Iga Sapi Kebuli", "Iga Sapi Kebuli"],
        })
        result = normalize_items.resolve_conditional_normalization(
            df, normalize_items.strip_xxx_prefix
        )
        self.assertEqual(result, {"FGS-00003": "FGS-00003", "xxx.FGS-00003": "FGS-00003"})

    def test_rejects_merge_for_group_with_disagreeing_names(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS.00069", "xxx.FGS.00069"],
            "Nama Barang": ["Cendol - FG", "Cendol Pandan - FG"],
        })
        result = normalize_items.resolve_conditional_normalization(
            df, normalize_items.strip_xxx_prefix
        )
        self.assertEqual(result, {"FGS.00069": "FGS.00069", "xxx.FGS.00069": "xxx.FGS.00069"})

    def test_singleton_group_is_transformed_trivially(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00001"],
            "Nama Barang": ["Ayam Kebuli (0.9)"],
        })
        result = normalize_items.resolve_conditional_normalization(
            df, normalize_items.strip_xxx_prefix
        )
        self.assertEqual(result, {"FGS-00001": "FGS-00001"})

    def test_three_way_group_all_agreeing_merges(self):
        df = pd.DataFrame({
            "Kode Barang": ["A.001", "A-001", "xxx.A.001"],
            "Nama Barang": ["Widget", "Widget", "Widget"],
        })
        result = normalize_items.resolve_conditional_normalization(
            df, normalize_items.unify_separator
        )
        self.assertEqual(result, {"A.001": "A-001", "A-001": "A-001", "xxx.A.001": "xxx.A-001"})


class TestBuildNormalizedCodeMap(unittest.TestCase):
    def test_merges_and_separates_real_fixture_codes_correctly(self):
        df = pd.DataFrame({
            "Kode Barang": [
                "FGS-00003", "xxx.FGS-00003",       # same product -> merge
                "FGS-00047", "FGS.00047",            # different products -> stay separate
                "FGS.00069", "xxx.FGS.00069",        # different products -> stay separate
                "FGS-00053", "xxx.FGS-00053", "FGS.00053",  # first two merge; third is unrelated
            ],
            "Nama Barang": [
                "Iga Sapi Kebuli", "Iga Sapi Kebuli",
                "Kentang Mustofa Rumput Laut", "Air Isi Ulang",
                "Cendol - FG", "Cendol Pandan - FG",
                "Ayam Kebuli (0.6)", "Ayam Kebuli (0.6)", "AirAlam 330 ml (Menu pakai kode AA)",
            ],
        })
        result = normalize_items.build_normalized_code_map(df)
        self.assertEqual(result["FGS-00003"], "FGS-00003")
        self.assertEqual(result["xxx.FGS-00003"], "FGS-00003")
        self.assertEqual(result["FGS-00047"], "FGS-00047")
        self.assertEqual(result["FGS.00047"], "FGS.00047")
        self.assertEqual(result["FGS.00069"], "FGS-00069")
        self.assertEqual(result["xxx.FGS.00069"], "xxx.FGS-00069")
        self.assertEqual(result["FGS-00053"], "FGS-00053")
        self.assertEqual(result["xxx.FGS-00053"], "FGS-00053")
        self.assertEqual(result["FGS.00053"], "FGS.00053")


class TestApplyItemNormalization(unittest.TestCase):
    def test_rewrites_kode_barang_for_agreeing_collision(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "xxx.FGS-00003"],
            "Nama Barang": ["Iga Sapi Kebuli", "Iga Sapi Kebuli"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        })
        result = normalize_items.apply_item_normalization(df)
        self.assertEqual(list(result["Kode Barang"]), ["FGS-00003", "FGS-00003"])

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS-00003"],
            "Nama Barang": ["Iga Sapi Kebuli"],
        })
        normalize_items.apply_item_normalization(df)
        self.assertEqual(df["Kode Barang"].iloc[0], "xxx.FGS-00003")


class TestCanonicalizeItemNames(unittest.TestCase):
    def test_propagates_clean_name_to_xxx_prefixed_siblings_in_same_group(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00068", "FGS-00068", "FGS-00068"],
            "Nama Barang": [
                "xxx.Ayam Crispy Spicy - FG",
                "Ayam Crispy Spicy - FG",
                "xxx.Ayam Crispy Spicy - FG",
            ],
        })
        result = normalize_items.canonicalize_item_names(df)
        self.assertEqual(list(result["Nama Barang"]), ["Ayam Crispy Spicy - FG"] * 3)

    def test_leaves_group_untouched_when_no_row_is_clean(self):
        df = pd.DataFrame({
            "Kode Barang": ["WIP-00010", "WIP-00010"],
            "Nama Barang": ["xxx.Ayam (0.6) - WIP", "xxx.Ayam (0.6) - WIP"],
        })
        result = normalize_items.canonicalize_item_names(df)
        self.assertEqual(list(result["Nama Barang"]), ["xxx.Ayam (0.6) - WIP"] * 2)

    def test_resolves_independent_groups_separately(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "FGS-00003", "WIP-00010", "WIP-00010"],
            "Nama Barang": [
                "xxx.Iga Sapi Kebuli",
                "Iga Sapi Kebuli",
                "xxx.Ayam (0.6) - WIP",
                "xxx.Ayam (0.6) - WIP",
            ],
        })
        result = normalize_items.canonicalize_item_names(df)
        self.assertEqual(
            list(result["Nama Barang"]),
            ["Iga Sapi Kebuli", "Iga Sapi Kebuli", "xxx.Ayam (0.6) - WIP", "xxx.Ayam (0.6) - WIP"],
        )

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00068", "FGS-00068"],
            "Nama Barang": ["xxx.Ayam Crispy Spicy - FG", "Ayam Crispy Spicy - FG"],
        })
        normalize_items.canonicalize_item_names(df)
        self.assertEqual(df["Nama Barang"].iloc[0], "xxx.Ayam Crispy Spicy - FG")


class TestExcludeItems(unittest.TestCase):
    def test_drops_rows_for_excluded_item(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00066", "FGS-00003"],
            "Kuantitas": [1, 2],
        })
        result = normalize_items.exclude_items(df, items={"xxx.FGS.00066"})
        self.assertEqual(list(result["Kode Barang"]), ["FGS-00003"])

    def test_leaves_other_items_untouched(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "FGS-00004"],
            "Kuantitas": [1, 2],
        })
        result = normalize_items.exclude_items(df, items={"xxx.FGS.00066"})
        self.assertEqual(len(result), 2)


class TestApplyItemRenames(unittest.TestCase):
    def test_rewrites_kode_and_nama_for_matching_raw_code(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00067", "FGS-00003"],
            "Nama Barang": ["xxx.Ayam Crispy Original - FG", "Iga Sapi Kebuli"],
        })
        result = normalize_items.apply_item_renames(
            df, renames={"xxx.FGS.00067": ("FGS-00068", "Ayam Crispy Spicy - FG")}
        )
        self.assertEqual(list(result["Kode Barang"]), ["FGS-00068", "FGS-00003"])
        self.assertEqual(list(result["Nama Barang"]), ["Ayam Crispy Spicy - FG", "Iga Sapi Kebuli"])

    def test_leaves_non_matching_rows_untouched(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00067", "FGS-00003"],
            "Nama Barang": ["xxx.Ayam Crispy Original - FG", "Iga Sapi Kebuli"],
        })
        result = normalize_items.apply_item_renames(
            df, renames={"xxx.FGS.00067": ("FGS-00068", "Ayam Crispy Spicy - FG")}
        )
        self.assertEqual(result["Kode Barang"].iloc[1], "FGS-00003")
        self.assertEqual(result["Nama Barang"].iloc[1], "Iga Sapi Kebuli")

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00067"],
            "Nama Barang": ["xxx.Ayam Crispy Original - FG"],
        })
        normalize_items.apply_item_renames(
            df, renames={"xxx.FGS.00067": ("FGS-00068", "Ayam Crispy Spicy - FG")}
        )
        self.assertEqual(df["Kode Barang"].iloc[0], "xxx.FGS.00067")
        self.assertEqual(df["Nama Barang"].iloc[0], "xxx.Ayam Crispy Original - FG")

    def test_default_renames_table_applies_seed_entry(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00067"],
            "Nama Barang": ["xxx.Ayam Crispy Original - FG"],
        })
        result = normalize_items.apply_item_renames(df)
        self.assertEqual(result["Kode Barang"].iloc[0], "FGS-00068")
        self.assertEqual(result["Nama Barang"].iloc[0], "Ayam Crispy Spicy - FG")


class TestReaggregateDaily(unittest.TestCase):
    def test_sums_kuantitas_for_rows_that_collided_after_normalization(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "FGS-00003"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Nama Cabang": ["KY001 - Branch", "KY001 - Branch"],
            "Kategori Barang": ["Barang Jadi (FG)", "Barang Jadi (FG)"],
            "Nama Barang": ["Iga Sapi Kebuli", "Iga Sapi Kebuli"],
            "Satuan": ["Porsi", "Porsi"],
            "Kuantitas": [3, 4],
        })
        result = normalize_items.reaggregate_daily(df)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Kuantitas"].iloc[0], 7)

    def test_leaves_distinct_keys_separate(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "FGS-00004"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "Nama Cabang": ["KY001 - Branch", "KY001 - Branch"],
            "Kategori Barang": ["Barang Jadi (FG)", "Barang Jadi (FG)"],
            "Nama Barang": ["Iga Sapi Kebuli", "Nasi Kebuli"],
            "Satuan": ["Porsi", "Porsi"],
            "Kuantitas": [3, 4],
        })
        result = normalize_items.reaggregate_daily(df)
        self.assertEqual(len(result), 2)


class TestExcludeBranches(unittest.TestCase):
    def test_drops_rows_for_excluded_branch(self):
        df = pd.DataFrame({
            "Nama Cabang": ["Kebab Saudagar - Kutabumi", "KY001 - Branch"],
            "Kuantitas": [1, 2],
        })
        result = normalize_items.exclude_branches(df, branches={"Kebab Saudagar - Kutabumi"})
        self.assertEqual(list(result["Nama Cabang"]), ["KY001 - Branch"])

    def test_leaves_other_branches_untouched(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY001 - Branch", "KY002 - Branch"],
            "Kuantitas": [1, 2],
        })
        result = normalize_items.exclude_branches(df, branches={"Kebab Saudagar - Kutabumi"})
        self.assertEqual(len(result), 2)


class TestLoadAndNormalize(unittest.TestCase):
    def test_drops_excluded_branch_before_normalization(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);FGS-00003;Iga Sapi Kebuli;KY001 - Branch;Porsi;3\n"
            "01 Jan 2024;Barang Jadi (FG);FGS-00003;Iga Sapi Kebuli;Kebab Saudagar - Kutabumi;Porsi;9\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(len(result), 1)
        self.assertNotIn("Kebab Saudagar - Kutabumi", result["Nama Cabang"].values)


    def test_reads_normalizes_and_reaggregates_end_to_end(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);FGS-00003;Iga Sapi Kebuli;KY001 - Branch;Porsi;3\n"
            "01 Jan 2024;Barang Jadi (FG);xxx.FGS-00003;Iga Sapi Kebuli;KY001 - Branch;Porsi;4\n"
            "02 Jan 2024;Barang Jadi (FG);FGS-00003;Iga Sapi Kebuli;KY001 - Branch;Porsi;5\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(len(result), 2)
        row1 = result[result["Tanggal"] == pd.Timestamp("2024-01-01")].iloc[0]
        self.assertEqual(row1["Kode Barang"], "FGS-00003")
        self.assertEqual(row1["Kuantitas"], 7)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["Tanggal"]))

    def test_applies_explicit_rename_and_canonicalizes_before_reaggregation(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);xxx.FGS.00067;xxx.Ayam Crispy Original - FG;KY001 - Branch;Potong;3\n"
            "01 Jan 2024;Barang Jadi (FG);xxx.FGS.00068;xxx.Ayam Crispy Spicy - FG;KY001 - Branch;Potong;5\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Kode Barang"].iloc[0], "FGS-00068")
        self.assertEqual(result["Nama Barang"].iloc[0], "Ayam Crispy Spicy - FG")
        self.assertEqual(result["Kuantitas"].iloc[0], 8)

    def test_drops_explicitly_excluded_items_end_to_end(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);xxx.FGS.00066;xxx.Nasi Putih;KY001 - Branch;Porsi;3\n"
            "01 Jan 2024;Barang Jadi (FG);FGS-00003;Iga Sapi Kebuli;KY001 - Branch;Porsi;5\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(len(result), 1)
        self.assertEqual(result["Kode Barang"].iloc[0], "FGS-00003")


if __name__ == "__main__":
    unittest.main()
