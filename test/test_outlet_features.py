import math
import tempfile
import unittest

import pandas as pd

from utils import outlet_features


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


def _region_mapping(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "old_name", "new_name", "Alamat", "Kecamatan", "Kota",
            "has_shopee", "has_gofood", "has_grabfood", "kawasan", "hari_pengiriman",
        ],
    )


class TestParseDeliveryDays(unittest.TestCase):
    def test_senin_dan_kamis(self):
        self.assertEqual(outlet_features.parse_delivery_days("Senin dan Kamis"), {0, 3})

    def test_selasa_dan_jumat(self):
        self.assertEqual(outlet_features.parse_delivery_days("Selasa dan Jumat"), {1, 4})

    def test_unrecognized_token_raises(self):
        with self.assertRaises(ValueError):
            outlet_features.parse_delivery_days("Senin dan Blah")


class TestComputeLeadTimeDays(unittest.TestCase):
    # Verified by direct computation, not hand-derived: for each day_of_week
    # (0=Senin..6=Minggu), the smallest number of days strictly forward to
    # the next delivery day.
    KAWASAN_1 = {0, 3}  # Senin & Kamis
    KAWASAN_2 = {1, 4}  # Selasa & Jumat
    EXPECTED_KAWASAN_1 = [3, 2, 1, 4, 3, 2, 1]
    EXPECTED_KAWASAN_2 = [1, 3, 2, 1, 4, 3, 2]

    def test_kawasan_1_full_week_matrix(self):
        for day_of_week, expected in enumerate(self.EXPECTED_KAWASAN_1):
            with self.subTest(day_of_week=day_of_week):
                self.assertEqual(
                    outlet_features.compute_lead_time_days(day_of_week, self.KAWASAN_1), expected
                )

    def test_kawasan_2_full_week_matrix(self):
        for day_of_week, expected in enumerate(self.EXPECTED_KAWASAN_2):
            with self.subTest(day_of_week=day_of_week):
                self.assertEqual(
                    outlet_features.compute_lead_time_days(day_of_week, self.KAWASAN_2), expected
                )


class TestApplyRegionFeatures(unittest.TestCase):
    def setUp(self):
        self.region = _region_mapping([
            ["KY007 - Kebuli Yaman Cibubur", "KY007 - Kebuli Yaman Cibubur", "addr", "Ciracas", "Jakarta Timur", "Yes", "Yes", "Yes", 2, "Selasa dan Jumat"],
            ["KY054 - Kebuli Yaman Jagakarsa", "KY054 - Kebuli Yaman Jagakarsa", "addr", "Jagakarsa", "Jakarta Selatan", "Yes", "No", "Yes", 1, "Senin dan Kamis"],
        ])

    def test_joins_kawasan_and_hari_pengiriman_by_canonical_branch_name(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY007 - Kebuli Yaman Cibubur"],
            "Tanggal": pd.to_datetime(["2025-08-04"]),
            "Kuantitas": [1],
        })
        result = outlet_features.apply_region_features(df, self.region)
        self.assertEqual(result.iloc[0]["kawasan"], 2)
        self.assertEqual(result.iloc[0]["hari_pengiriman"], "Selasa dan Jumat")

    def test_lead_time_days_varies_by_kawasan_and_day_of_week(self):
        # 2025-08-04 = Monday, 2025-08-05 = Tuesday, 2025-08-07 = Thursday.
        df = pd.DataFrame({
            "Nama Cabang": [
                "KY007 - Kebuli Yaman Cibubur", "KY007 - Kebuli Yaman Cibubur",
                "KY054 - Kebuli Yaman Jagakarsa", "KY054 - Kebuli Yaman Jagakarsa",
            ],
            "Tanggal": pd.to_datetime(["2025-08-04", "2025-08-05", "2025-08-04", "2025-08-07"]),
            "Kuantitas": [1, 2, 3, 4],
        })
        result = outlet_features.apply_region_features(df, self.region)
        self.assertEqual(result["lead_time_days"].tolist(), [1, 3, 3, 4])

    def test_unmatched_branch_gets_nan_kawasan_and_lead_time(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY999 - Unknown Branch"],
            "Tanggal": pd.to_datetime(["2025-08-04"]),
            "Kuantitas": [1],
        })
        result = outlet_features.apply_region_features(df, self.region)
        self.assertTrue(math.isnan(result.iloc[0]["kawasan"]))
        self.assertTrue(math.isnan(result.iloc[0]["lead_time_days"]))

    def test_one_row_per_input_row_no_fanout(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY007 - Kebuli Yaman Cibubur", "KY007 - Kebuli Yaman Cibubur"],
            "Tanggal": pd.to_datetime(["2025-08-04", "2025-08-05"]),
            "Kuantitas": [1, 2],
        })
        result = outlet_features.apply_region_features(df, self.region)
        self.assertEqual(len(result), 2)


class TestAddRelocationFeature(unittest.TestCase):
    def test_negative_for_rows_before_relocation_date(self):
        df = pd.DataFrame({
            "Nama Cabang": ["Kebuli Yaman Cadas"],
            "Tanggal": pd.to_datetime(["2025-09-28"]),
        })
        dates = {"Kebuli Yaman Cadas": pd.Timestamp("2025-10-03")}
        result = outlet_features.add_relocation_feature(df, dates)
        self.assertEqual(result.iloc[0]["days_since_relocation"], -5)

    def test_zero_on_relocation_date_itself(self):
        df = pd.DataFrame({
            "Nama Cabang": ["Kebuli Yaman Cadas"],
            "Tanggal": pd.to_datetime(["2025-10-03"]),
        })
        dates = {"Kebuli Yaman Cadas": pd.Timestamp("2025-10-03")}
        result = outlet_features.add_relocation_feature(df, dates)
        self.assertEqual(result.iloc[0]["days_since_relocation"], 0)

    def test_positive_for_rows_after_relocation_date(self):
        df = pd.DataFrame({
            "Nama Cabang": ["Kebuli Yaman Cadas"],
            "Tanggal": pd.to_datetime(["2025-10-10"]),
        })
        dates = {"Kebuli Yaman Cadas": pd.Timestamp("2025-10-03")}
        result = outlet_features.add_relocation_feature(df, dates)
        self.assertEqual(result.iloc[0]["days_since_relocation"], 7)

    def test_nan_for_branch_not_in_relocation_dates(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY007 - Kebuli Yaman Cibubur"],
            "Tanggal": pd.to_datetime(["2025-10-10"]),
        })
        dates = {"Kebuli Yaman Cadas": pd.Timestamp("2025-10-03")}
        result = outlet_features.add_relocation_feature(df, dates)
        self.assertTrue(math.isnan(result.iloc[0]["days_since_relocation"]))

    def test_resolves_independent_branches_separately(self):
        df = pd.DataFrame({
            "Nama Cabang": ["Kebuli Yaman Cadas", "Kebuli Yaman Bintara"],
            "Tanggal": pd.to_datetime(["2025-10-03", "2025-11-28"]),
        })
        dates = {
            "Kebuli Yaman Cadas": pd.Timestamp("2025-10-03"),
            "Kebuli Yaman Bintara": pd.Timestamp("2025-11-28"),
        }
        result = outlet_features.add_relocation_feature(df, dates)
        self.assertEqual(result["days_since_relocation"].tolist(), [0, 0])

    def test_does_not_mutate_original_dataframe(self):
        df = pd.DataFrame({
            "Nama Cabang": ["Kebuli Yaman Cadas"],
            "Tanggal": pd.to_datetime(["2025-10-03"]),
        })
        dates = {"Kebuli Yaman Cadas": pd.Timestamp("2025-10-03")}
        outlet_features.add_relocation_feature(df, dates)
        self.assertNotIn("days_since_relocation", df.columns)


class TestLoadClosures(unittest.TestCase):
    def _write(self, body):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        )
        tmp.write("Nama Outlet;tanggal_tutup;tanggal_buka;alasan\n" + body)
        tmp.close()
        return tmp.name

    def test_parses_closed_and_reopened_interval(self):
        path = self._write("Cabang A;2024-03-01;2025-07-18;tutup\n")
        result = outlet_features.load_closures(path)
        self.assertEqual(list(result), ["Cabang A"])
        start, end = result["Cabang A"][0]
        self.assertEqual(start, pd.Timestamp("2024-03-01"))
        self.assertEqual(end, pd.Timestamp("2025-07-18"))

    def test_empty_tanggal_buka_means_still_closed(self):
        path = self._write("Cabang A;2025-12-01;;masih tutup\n")
        result = outlet_features.load_closures(path)
        self.assertIsNone(result["Cabang A"][0][1])

    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(outlet_features.load_closures("/tmp/tidak-ada-file.csv"), {})

    def test_unparseable_date_raises(self):
        path = self._write("Cabang A;01 Mar 2024;2025-07-18;salah format\n")
        with self.assertRaises(ValueError):
            outlet_features.load_closures(path)

    def test_reopen_before_close_raises(self):
        path = self._write("Cabang A;2025-07-18;2024-03-01;terbalik\n")
        with self.assertRaises(ValueError):
            outlet_features.load_closures(path)

    def test_overlapping_intervals_raise(self):
        path = self._write(
            "Cabang A;2024-01-01;2024-06-01;satu\nCabang A;2024-05-01;2024-08-01;dua\n"
        )
        with self.assertRaises(ValueError):
            outlet_features.load_closures(path)

    def test_real_file_has_the_three_confirmed_closures(self):
        result = outlet_features.load_closures()
        self.assertIn("KY011 - Kebuli Yaman Bekasi Galaxy", result)
        self.assertIn("KY056 - Kebuli Yaman Tigaraksa", result)
        self.assertIn("Kebuli Yaman Cikarang Pusat", result)
        self.assertIsNone(result["Kebuli Yaman Cikarang Pusat"][0][1])


if __name__ == "__main__":
    unittest.main()
