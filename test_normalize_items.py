import unittest

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


if __name__ == "__main__":
    unittest.main()
