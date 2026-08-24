import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.data_preprocessing import normalize_items


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


class TestCanonicalizeItemCategories(unittest.TestCase):
    def test_maps_all_rows_to_category_of_most_recent_date(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00006", "FGS-00006"],
            "Kategori Barang": ["Minuman", "Minuman - FG"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-03-01"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(list(result["Kategori Barang"]), ["Minuman - FG", "Minuman - FG"])

    def test_leaves_single_category_group_unchanged(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003", "FGS-00003"],
            "Kategori Barang": ["Barang Jadi (FG)", "Barang Jadi (FG)"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(list(result["Kategori Barang"]), ["Barang Jadi (FG)", "Barang Jadi (FG)"])

    def test_resolves_independent_codes_separately(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00006", "FGS-00006", "FGS-00055", "FGS-00055"],
            "Kategori Barang": ["Minuman", "Minuman - FG", "Snack", "Snack (FG)"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-03-01", "2024-01-01", "2024-03-01"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(
            list(result["Kategori Barang"]),
            ["Minuman - FG", "Minuman - FG", "Snack (FG)", "Snack (FG)"],
        )

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00006", "FGS-00006"],
            "Kategori Barang": ["Minuman", "Minuman - FG"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-03-01"]),
        })
        normalize_items.canonicalize_item_categories(df)
        self.assertEqual(df["Kategori Barang"].iloc[0], "Minuman")

    def test_leaves_wip_to_fg_transition_time_varying_not_a_rename(self):
        # The general rule: WIP-2 and Barang Jadi (FG) are genuinely different
        # categories (data owner, 2026-08-10), unlike Minuman/Snack which are
        # the same category relabeled — so this pair must NOT be collapsed to
        # the latest category by the synonym rule.
        #
        # The rule still holds, but it is no longer the whole story. The data
        # owner later confirmed (2026-08-22) that for ten specific SKUs the
        # WIP-2 label was administrative only, and those are handled by name
        # in EXPLICIT_CATEGORY_OVERRIDES — the exception list, not a change to
        # the rule below. This test therefore uses a code that is deliberately
        # NOT on that list, so it keeps testing the general rule; the override
        # path has its own test.
        code = "FGS-00099"
        self.assertNotIn(code, normalize_items.EXPLICIT_CATEGORY_OVERRIDES)
        df = pd.DataFrame({
            "Kode Barang": [code, code],
            "Kategori Barang": ["Barang Semi FG (WIP-2)", "Barang Jadi (FG)"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-03-01"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(
            list(result["Kategori Barang"]),
            ["Barang Semi FG (WIP-2)", "Barang Jadi (FG)"],
        )

    def test_override_applies_to_earliest_rows_not_just_later_ones(self):
        # The ten SKUs confirmed on 2026-08-22 carry the WIP-2 label only in
        # their earliest rows. The override has to reach those rows too —
        # rewriting just the later ones would leave the history split, which
        # is the exact condition this confirmation removes.
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00001"] * 3,
            "Kategori Barang": [
                "Barang Semi FG (WIP-2)",
                "Barang Semi FG (WIP-2)",
                "Barang Jadi (FG)",
            ],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(list(result["Kategori Barang"]), ["Barang Jadi (FG)"] * 3)

    def test_every_administratively_relabeled_sku_resolves_to_fg(self):
        # All ten at once, so adding or dropping one from the override map
        # cannot pass unnoticed.
        codes = [
            "FGS-00001", "FGS-00002", "FGS-00003", "FGS-00004", "FGS-00005",
            "FGS-00012", "FGS-00013", "FGS-00018", "FGS-00049", "FGS-00053",
        ]
        df = pd.DataFrame({
            "Kode Barang": [c for c in codes for _ in range(2)],
            "Kategori Barang": ["Barang Semi FG (WIP-2)", "Barang Jadi (FG)"] * len(codes),
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-03-01"] * len(codes)),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(set(result["Kategori Barang"]), {"Barang Jadi (FG)"})

    def test_administrative_relabel_does_not_touch_club_mineral_override(self):
        # FGS-00014 is a different case: WIP-2 -> Minuman - FG, not FG.
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00014", "FGS-00001"],
            "Kategori Barang": ["Barang Semi FG (WIP-2)", "Barang Semi FG (WIP-2)"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(
            list(result["Kategori Barang"]), ["Minuman - FG", "Barang Jadi (FG)"]
        )

    def test_applies_explicit_override_for_club_mineral_600ml(self):
        # FGS-00014 (Club Mineral 600ml) was recorded as WIP-2 early on but
        # is actually a drink — confirmed by data owner (2026-08-10) it
        # should be Minuman - FG for its entire history, not time-varying.
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00014", "FGS-00014"],
            "Kategori Barang": ["Barang Semi FG (WIP-2)", "Minuman - FG"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2024-03-01"]),
        })
        result = normalize_items.canonicalize_item_categories(df)
        self.assertEqual(
            list(result["Kategori Barang"]), ["Minuman - FG", "Minuman - FG"]
        )


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


class TestConvertGramItemsToPorsi(unittest.TestCase):
    def test_converts_kuantitas_and_satuan_for_matching_gram_items(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00070", "xxx.FGS.00071"],
            "Satuan": ["Gr", "Gr"],
            "Kuantitas": [80, 60],
        })
        result = normalize_items.convert_gram_items_to_porsi(
            df, factors={"xxx.FGS.00070": 40, "xxx.FGS.00071": 30}
        )
        self.assertEqual(list(result["Satuan"]), ["Porsi", "Porsi"])
        self.assertEqual(list(result["Kuantitas"]), [2.0, 2.0])

    def test_leaves_non_matching_items_untouched(self):
        df = pd.DataFrame({
            "Kode Barang": ["FGS-00003"],
            "Satuan": ["Kg"],
            "Kuantitas": [5],
        })
        result = normalize_items.convert_gram_items_to_porsi(
            df, factors={"xxx.FGS.00070": 40}
        )
        self.assertEqual(result["Satuan"].iloc[0], "Kg")
        self.assertEqual(result["Kuantitas"].iloc[0], 5.0)

    def test_leaves_matching_item_untouched_when_satuan_is_not_gr(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00070"],
            "Satuan": ["Porsi"],
            "Kuantitas": [4],
        })
        result = normalize_items.convert_gram_items_to_porsi(
            df, factors={"xxx.FGS.00070": 40}
        )
        self.assertEqual(result["Satuan"].iloc[0], "Porsi")
        self.assertEqual(result["Kuantitas"].iloc[0], 4.0)

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00070"],
            "Satuan": ["Gr"],
            "Kuantitas": [80],
        })
        normalize_items.convert_gram_items_to_porsi(df, factors={"xxx.FGS.00070": 40})
        self.assertEqual(df["Satuan"].iloc[0], "Gr")
        self.assertEqual(df["Kuantitas"].iloc[0], 80)


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

    def test_default_renames_table_is_empty(self):
        df = pd.DataFrame({
            "Kode Barang": ["xxx.FGS.00067"],
            "Nama Barang": ["xxx.Ayam Crispy Original - FG"],
        })
        result = normalize_items.apply_item_renames(df)
        self.assertEqual(result["Kode Barang"].iloc[0], "xxx.FGS.00067")
        self.assertEqual(result["Nama Barang"].iloc[0], "xxx.Ayam Crispy Original - FG")


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

    def test_drops_discontinued_ayam_crispy_original_and_spicy_by_default(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Barang Jadi (FG);xxx.FGS.00067;xxx.Ayam Crispy Original - FG;KY001 - Branch;Potong;3\n"
            "01 Jan 2024;Barang Jadi (FG);xxx.FGS.00068;xxx.Ayam Crispy Spicy - FG;KY001 - Branch;Potong;5\n"
            "01 Jan 2024;Barang Jadi (FG);FGS-00003;Iga Sapi Kebuli;KY001 - Branch;Porsi;5\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(list(result["Kode Barang"]), ["FGS-00003"])

    def test_canonicalizes_category_relabel_across_time_end_to_end(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "01 Jan 2024;Minuman;FGS-00006;Club Mineral 330 ml;KY001 - Branch;Botol;2\n"
            "01 Mar 2024;Minuman - FG;FGS-00006;Club Mineral 330 ml;KY001 - Branch;Botol;5\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(set(result["Kategori Barang"]), {"Minuman - FG"})

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

    def test_drops_discontinued_cendol_pandan_by_default(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "24 Jan 2025;Barang Jadi (FG);xxx.FGS.00069;xxx.Cendol Pandan - FG;KY001 - Branch;Gr;250\n"
            "03 Jun 2025;Barang Jadi (FG);FGS.00069;Cendol - FG;KY001 - Branch;Porsi;3\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(list(result["Kode Barang"]), ["FGS-00069"])

    def test_merges_santan_and_gula_cendol_across_xxx_prefix_gap(self):
        content = (
            "Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"
            "24 Jan 2025;Barang Jadi (FG);xxx.FGS.00070;xxx.Santan Cendol - FG;KY001 - Branch;Gr;80\n"
            "03 Jun 2025;Barang Jadi (FG);FGS.00070;Santan Cendol - FG;KY001 - Branch;Porsi;4\n"
            "24 Jan 2025;Barang Jadi (FG);xxx.FGS.00071;xxx.Gula Cendol - FG;KY001 - Branch;Gr;60\n"
            "03 Jun 2025;Barang Jadi (FG);FGS.00071;Gula Cendol - FG;KY001 - Branch;Porsi;6\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.csv"
            path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            result = normalize_items.load_and_normalize(str(path))
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["Kode Barang"]), {"FGS-00070", "FGS-00071"})
        self.assertTrue((result["Satuan"] == "Porsi").all())
        santan = result[result["Kode Barang"] == "FGS-00070"]
        self.assertEqual(set(santan["Nama Barang"]), {"Santan Cendol - FG"})
        self.assertEqual(sorted(santan["Kuantitas"]), [2.0, 4.0])


if __name__ == "__main__":
    unittest.main()
