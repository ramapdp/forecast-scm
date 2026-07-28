import math
import unittest

import pandas as pd

import outlet_features


def _outlets(rows):
    return pd.DataFrame(rows, columns=["Nama Outlet", "Alamat", "Kecamatan", "Kota", "has_shopee", "has_gofood", "has_grabfood"])


def _overrides(rows):
    return pd.DataFrame(rows, columns=["Nama Cabang", "Nama Outlet", "Kota Override"])


class TestMatchBranchToOutlet(unittest.TestCase):
    def test_exact_match_after_prefix_strip(self):
        outlets = _outlets([["KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes"]])
        overrides = _overrides([])
        row, kota_override = outlet_features.match_branch_to_outlet(
            "KY007 - Kebuli Yaman Cibubur", outlets, overrides
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["Kota"], "Jakarta Timur")
        self.assertIsNone(kota_override)

    def test_substring_match_for_no_code_outlet(self):
        outlets = _outlets([["Kebuli Yaman Cadas", "addr", "Sepatan", "Kabupaten Tangerang", "No", "No", "Yes"]])
        overrides = _overrides([])
        row, _ = outlet_features.match_branch_to_outlet("KY070 - Kebuli Yaman Cadas", outlets, overrides)
        self.assertIsNotNone(row)
        self.assertEqual(row["Nama Outlet"], "Kebuli Yaman Cadas")

    def test_override_lookup_wins_and_supplies_kota_override(self):
        outlets = _outlets([["KY001 - Kebuli Yaman Kutabumi (Pusat)", "addr", "Jatiuwung", "Banten", "Yes", "Yes", "Yes"]])
        overrides = _overrides([
            ["KY001 - Kebuli Yaman Kutabumi (Pusat)", "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Kota Tangerang"],
        ])
        row, kota_override = outlet_features.match_branch_to_outlet(
            "KY001 - Kebuli Yaman Kutabumi (Pusat)", outlets, overrides
        )
        self.assertIsNotNone(row)
        self.assertEqual(kota_override, "Kota Tangerang")

    def test_override_resolves_duplicate_branch_code(self):
        outlets = _outlets([["KY011 - Kebuli Yaman Bekasi Galaxy", "addr", "Bekasi Selatan", "Kota Bekasi", "No", "Yes", "No"]])
        overrides = _overrides([
            ["KY069 - Kebuli Yaman Bekasi Galaxy", "KY011 - Kebuli Yaman Bekasi Galaxy", ""],
        ])
        row, _ = outlet_features.match_branch_to_outlet("KY069 - Kebuli Yaman Bekasi Galaxy", outlets, overrides)
        self.assertIsNotNone(row)
        self.assertEqual(row["Nama Outlet"], "KY011 - Kebuli Yaman Bekasi Galaxy")

    def test_override_resolves_tod_m1_bandara_legacy_name(self):
        outlets = _outlets([["KY051 - kebuli Yaman TOD M1 Bandara", "addr", "Neglasari", "Kota Tangerang", "No", "Yes", "No"]])
        overrides = _overrides([
            ["TOD M1 Bandara", "KY051 - kebuli Yaman TOD M1 Bandara", ""],
        ])
        row, _ = outlet_features.match_branch_to_outlet("TOD M1 Bandara", outlets, overrides)
        self.assertIsNotNone(row)
        self.assertEqual(row["Nama Outlet"], "KY051 - kebuli Yaman TOD M1 Bandara")

    def test_no_outlet_counterpart_is_unmatched(self):
        outlets = _outlets([["KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes"]])
        overrides = _overrides([])
        row, kota_override = outlet_features.match_branch_to_outlet(
            "KY020 - Kebuli Yaman Tambun", outlets, overrides
        )
        self.assertIsNone(row)
        self.assertIsNone(kota_override)

    def test_ambiguous_substring_is_unmatched_not_guessed(self):
        outlets = _outlets([
            ["KY001 - Kebuli Yaman Cadas", "addr", "A", "Kota A", "Yes", "Yes", "Yes"],
            ["KY002 - Kebuli Yaman Cadas Baru", "addr", "B", "Kota B", "Yes", "Yes", "Yes"],
        ])
        overrides = _overrides([])
        row, _ = outlet_features.match_branch_to_outlet("KY003 - Kebuli Yaman Cadas", outlets, overrides)
        self.assertIsNone(row)


class TestNormalizeKota(unittest.TestCase):
    def test_override_wins_over_outlet_kota(self):
        self.assertEqual(outlet_features.normalize_kota("Banten", "Kota Tangerang"), "Kota Tangerang")

    def test_no_override_keeps_kota_and_kabupaten_distinction(self):
        self.assertEqual(outlet_features.normalize_kota("Kota Bogor", None), "Kota Bogor")
        self.assertEqual(outlet_features.normalize_kota("Kabupaten Bogor", None), "Kabupaten Bogor")

    def test_missing_kota_becomes_unknown(self):
        self.assertEqual(outlet_features.normalize_kota(None, None), "Unknown")
        self.assertEqual(outlet_features.normalize_kota(float("nan"), None), "Unknown")


class TestBuildOutletFeatures(unittest.TestCase):
    def setUp(self):
        self.outlets = _outlets([
            ["KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes"],
            ["KY067 - Kebuli Yaman Metland", "addr", "Cileungsi", "Kabupaten Bogor", "No", "No", "No"],
            ["KY001 - Kebuli Yaman Kutabumi (Pusat)", "addr", "Jatiuwung", "Banten", "Yes", "Yes", "Yes"],
        ])
        self.overrides = _overrides([
            ["KY001 - Kebuli Yaman Kutabumi (Pusat)", "KY001 - Kebuli Yaman Kutabumi (Pusat)", "Kota Tangerang"],
        ])

    def test_matched_branch_carries_channel_flags_and_kota(self):
        result = outlet_features.build_outlet_features(
            ["KY007 - Kebuli Yaman Cibubur"], self.outlets, self.overrides
        )
        row = result.iloc[0]
        self.assertEqual(row["kota"], "Jakarta Timur")
        self.assertTrue(row["has_shopee"])
        self.assertTrue(row["has_gofood"])
        self.assertTrue(row["has_grabfood"])

    def test_can_order_online_true_when_any_channel_yes(self):
        result = outlet_features.build_outlet_features(
            ["KY007 - Kebuli Yaman Cibubur"], self.outlets, self.overrides
        )
        self.assertTrue(result.iloc[0]["can_order_online"])

    def test_can_order_online_false_when_all_channels_no(self):
        result = outlet_features.build_outlet_features(
            ["KY067 - Kebuli Yaman Metland"], self.outlets, self.overrides
        )
        self.assertFalse(result.iloc[0]["can_order_online"])

    def test_can_order_online_nan_when_unmatched(self):
        result = outlet_features.build_outlet_features(
            ["KY020 - Kebuli Yaman Tambun"], self.outlets, self.overrides
        )
        row = result.iloc[0]
        self.assertEqual(row["kota"], "Unknown")
        self.assertTrue(math.isnan(row["can_order_online"]))
        self.assertTrue(math.isnan(row["has_shopee"]))

    def test_kota_override_applied_on_top_of_matched_outlet(self):
        result = outlet_features.build_outlet_features(
            ["KY001 - Kebuli Yaman Kutabumi (Pusat)"], self.outlets, self.overrides
        )
        self.assertEqual(result.iloc[0]["kota"], "Kota Tangerang")

    def test_one_row_per_branch_in_input_order(self):
        branches = ["KY007 - Kebuli Yaman Cibubur", "KY020 - Kebuli Yaman Tambun", "KY067 - Kebuli Yaman Metland"]
        result = outlet_features.build_outlet_features(branches, self.outlets, self.overrides)
        self.assertEqual(list(result["Nama Cabang"]), branches)


class TestCanonicalizeBranchNames(unittest.TestCase):
    def test_override_matched_branch_renamed_to_canonical_outlet_name(self):
        outlets = _outlets([["KY051 - kebuli Yaman TOD M1 Bandara", "addr", "Neglasari", "Kota Tangerang", "No", "Yes", "No"]])
        overrides = _overrides([["TOD M1 Bandara", "KY051 - kebuli Yaman TOD M1 Bandara", ""]])
        df = pd.DataFrame({"Nama Cabang": ["TOD M1 Bandara"], "Kuantitas": [1]})
        result = outlet_features.canonicalize_branch_names(df, outlets, overrides)
        self.assertEqual(result["Nama Cabang"].iloc[0], "KY051 - kebuli Yaman TOD M1 Bandara")

    def test_two_raw_spellings_collapse_to_same_canonical_name(self):
        outlets = _outlets([["KY051 - kebuli Yaman TOD M1 Bandara", "addr", "Neglasari", "Kota Tangerang", "No", "Yes", "No"]])
        overrides = _overrides([["TOD M1 Bandara", "KY051 - kebuli Yaman TOD M1 Bandara", ""]])
        df = pd.DataFrame({
            "Nama Cabang": ["TOD M1 Bandara", "KY051 - kebuli Yaman TOD M1 Bandara"],
            "Kuantitas": [1, 2],
        })
        result = outlet_features.canonicalize_branch_names(df, outlets, overrides)
        self.assertEqual(result["Nama Cabang"].nunique(), 1)
        self.assertEqual(result["Nama Cabang"].iloc[0], "KY051 - kebuli Yaman TOD M1 Bandara")

    def test_automatic_fallback_match_renamed_to_outlet_name(self):
        outlets = _outlets([["Kebuli Yaman Cadas", "addr", "Sepatan", "Kabupaten Tangerang", "No", "No", "Yes"]])
        overrides = _overrides([])
        df = pd.DataFrame({"Nama Cabang": ["KY070 - Kebuli Yaman Cadas"], "Kuantitas": [1]})
        result = outlet_features.canonicalize_branch_names(df, outlets, overrides)
        self.assertEqual(result["Nama Cabang"].iloc[0], "Kebuli Yaman Cadas")

    def test_already_canonical_name_left_unchanged(self):
        outlets = _outlets([["KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes"]])
        overrides = _overrides([])
        df = pd.DataFrame({"Nama Cabang": ["KY007 - Kebuli Yaman Cibubur"], "Kuantitas": [1]})
        result = outlet_features.canonicalize_branch_names(df, outlets, overrides)
        self.assertEqual(result["Nama Cabang"].iloc[0], "KY007 - Kebuli Yaman Cibubur")

    def test_unmatched_branch_left_unchanged(self):
        outlets = _outlets([["KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes"]])
        overrides = _overrides([])
        df = pd.DataFrame({"Nama Cabang": ["KY020 - Kebuli Yaman Tambun"], "Kuantitas": [1]})
        result = outlet_features.canonicalize_branch_names(df, outlets, overrides)
        self.assertEqual(result["Nama Cabang"].iloc[0], "KY020 - Kebuli Yaman Tambun")


class TestFilterMatchedBranches(unittest.TestCase):
    def setUp(self):
        self.outlets = _outlets([
            ["KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes"],
        ])
        self.overrides = _overrides([
            ["KY069 - Kebuli Yaman Bekasi Galaxy", "KY007 - Kebuli Yaman Cibubur", ""],
        ])

    def test_drops_rows_for_unmatched_branch(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY007 - Kebuli Yaman Cibubur", "KY020 - Kebuli Yaman Tambun"],
            "Kuantitas": [1, 2],
        })
        result = outlet_features.filter_matched_branches(df, self.outlets, self.overrides)
        self.assertEqual(list(result["Nama Cabang"]), ["KY007 - Kebuli Yaman Cibubur"])

    def test_keeps_rows_for_matched_branch(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY007 - Kebuli Yaman Cibubur", "KY007 - Kebuli Yaman Cibubur"],
            "Kuantitas": [1, 2],
        })
        result = outlet_features.filter_matched_branches(df, self.outlets, self.overrides)
        self.assertEqual(len(result), 2)

    def test_keeps_rows_for_override_matched_branch(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY069 - Kebuli Yaman Bekasi Galaxy"],
            "Kuantitas": [1],
        })
        result = outlet_features.filter_matched_branches(df, self.outlets, self.overrides)
        self.assertEqual(len(result), 1)

    def test_all_unmatched_yields_empty_result(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY020 - Kebuli Yaman Tambun"],
            "Kuantitas": [1],
        })
        result = outlet_features.filter_matched_branches(df, self.outlets, self.overrides)
        self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
