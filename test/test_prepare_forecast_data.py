import unittest
import tempfile
from pathlib import Path

import pandas as pd

from utils import prepare_forecast_data
from utils import normalize_items
from utils import build_panel
from utils import outlet_features


def _pair_series(qtys, start="2025-01-01", pair=("A", "X")):
    n = len(qtys)
    return pd.DataFrame({
        "Kode Barang": [pair[0]] * n, "Nama Cabang": [pair[1]] * n,
        "Tanggal": pd.date_range(start, periods=n, freq="D"),
        "Kuantitas": qtys,
    })


class TestAddTargets(unittest.TestCase):
    def test_all_horizons_populated_with_correct_future_values(self):
        df = _pair_series(list(range(1, 11)))  # 10 days: 1..10
        result = prepare_forecast_data.add_targets(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        for h in range(1, 8):
            self.assertEqual(day1[f"target_h{h}"], h + 1)  # day1 + h -> value h+1

    def test_target_h1_is_next_day_not_current_day(self):
        df = _pair_series([10, 20, 30])
        result = prepare_forecast_data.add_targets(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        self.assertEqual(day1["target_h1"], 20)
        self.assertNotEqual(day1["target_h1"], 10)

    def test_targets_beyond_available_data_are_nan(self):
        df = _pair_series([1, 2, 3])  # only 3 days
        result = prepare_forecast_data.add_targets(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        self.assertEqual(day1["target_h1"], 2)
        self.assertEqual(day1["target_h2"], 3)
        self.assertTrue(pd.isna(day1["target_h3"]))
        self.assertTrue(pd.isna(day1["target_h7"]))


class TestAddLagFeatures(unittest.TestCase):
    def test_all_lags_populated_with_correct_past_values(self):
        df = _pair_series(list(range(1, 30)))  # 29 days: 1..29
        result = prepare_forecast_data.add_lag_features(df)
        day29 = result[result["Tanggal"] == pd.Timestamp("2025-01-29")].iloc[0]
        self.assertEqual(day29["lag_1"], 28)
        self.assertEqual(day29["lag_7"], 22)
        self.assertEqual(day29["lag_28"], 1)

    def test_higher_lags_are_nan_when_insufficient_history(self):
        df = _pair_series(list(range(1, 6)))  # only 5 days
        result = prepare_forecast_data.add_lag_features(df)
        day5 = result[result["Tanggal"] == pd.Timestamp("2025-01-05")].iloc[0]
        self.assertEqual(day5["lag_1"], 4)  # 1 day back is available
        self.assertTrue(pd.isna(day5["lag_7"]))
        self.assertTrue(pd.isna(day5["lag_28"]))


class TestAddRollingFeatures(unittest.TestCase):
    def test_rolling_mean_and_std_exclude_current_day(self):
        # Days 1..10 with Kuantitas 1..10. On day 8, the trailing 7-day
        # window (days 1-7, i.e. values 1..7) must be used — NOT days 2-8.
        df = _pair_series(list(range(1, 11)))
        result = prepare_forecast_data.add_rolling_features(df)
        day8 = result[result["Tanggal"] == pd.Timestamp("2025-01-08")].iloc[0]
        self.assertAlmostEqual(day8["roll_mean_7"], 4.0)  # mean(1..7)
        self.assertAlmostEqual(day8["roll_std_7"], pd.Series(range(1, 8)).std())

    def test_early_rows_are_nan_for_windows_larger_than_available_history(self):
        df = _pair_series(list(range(1, 6)))  # only 5 days
        result = prepare_forecast_data.add_rolling_features(df)
        day5 = result[result["Tanggal"] == pd.Timestamp("2025-01-05")].iloc[0]
        self.assertTrue(pd.isna(day5["roll_mean_7"]))
        self.assertTrue(pd.isna(day5["roll_mean_14"]))
        self.assertTrue(pd.isna(day5["roll_mean_28"]))


class TestAddLeadTimeTarget(unittest.TestCase):
    def test_cumulative_sum_for_fixed_3_day_window(self):
        df = _pair_series(list(range(1, 11)))  # 1..10
        df["lead_time_days"] = 3
        result = prepare_forecast_data.add_lead_time_target(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        self.assertEqual(day1["target_lead_time_cumulative"], 2 + 3 + 4)

    def test_cumulative_sum_for_fixed_4_day_window(self):
        df = _pair_series(list(range(1, 11)))  # 1..10
        df["lead_time_days"] = 4
        result = prepare_forecast_data.add_lead_time_target(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]
        self.assertEqual(day1["target_lead_time_cumulative"], 2 + 3 + 4 + 5)

    def test_variable_window_per_row(self):
        df = _pair_series(list(range(1, 11)))  # 1..10
        df["lead_time_days"] = [3, 4, 3, 4, 3, 4, 3, 4, 3, 4]
        result = prepare_forecast_data.add_lead_time_target(df)
        day1 = result[result["Tanggal"] == pd.Timestamp("2025-01-01")].iloc[0]  # window 3
        day2 = result[result["Tanggal"] == pd.Timestamp("2025-01-02")].iloc[0]  # window 4
        self.assertEqual(day1["target_lead_time_cumulative"], 2 + 3 + 4)
        self.assertEqual(day2["target_lead_time_cumulative"], 3 + 4 + 5 + 6)

    def test_window_excludes_current_and_prior_days(self):
        qtys = [10] * 10
        qtys[5] = 1000  # 2025-01-06, a large outlier
        df = _pair_series(qtys)
        df["lead_time_days"] = 2
        result = prepare_forecast_data.add_lead_time_target(df)
        spike_day = result[result["Tanggal"] == pd.Timestamp("2025-01-06")].iloc[0]
        self.assertEqual(spike_day["target_lead_time_cumulative"], 20)  # excludes its own 1000
        day_before_spike = result[result["Tanggal"] == pd.Timestamp("2025-01-05")].iloc[0]
        self.assertEqual(day_before_spike["target_lead_time_cumulative"], 1000 + 10)  # includes it

    def test_nan_at_end_of_data_boundary(self):
        df = _pair_series(list(range(1, 6)))  # only 5 days
        df["lead_time_days"] = 3
        result = prepare_forecast_data.add_lead_time_target(df)
        day5 = result[result["Tanggal"] == pd.Timestamp("2025-01-05")].iloc[0]
        day4 = result[result["Tanggal"] == pd.Timestamp("2025-01-04")].iloc[0]
        self.assertTrue(pd.isna(day5["target_lead_time_cumulative"]))
        self.assertTrue(pd.isna(day4["target_lead_time_cumulative"]))

    def test_nan_propagates_when_lead_time_days_is_nan(self):
        df = _pair_series(list(range(1, 6)))
        df["lead_time_days"] = [3, 3, float("nan"), 3, 3]
        result = prepare_forecast_data.add_lead_time_target(df)
        day3 = result[result["Tanggal"] == pd.Timestamp("2025-01-03")].iloc[0]
        self.assertTrue(pd.isna(day3["target_lead_time_cumulative"]))

    def test_multiple_pairs_independent(self):
        pair_a = _pair_series([1, 2, 3, 4, 5], pair=("A", "X"))
        pair_b = _pair_series([100, 200, 300, 400, 500], pair=("B", "Y"))
        df = pd.concat([pair_a, pair_b], ignore_index=True)
        df["lead_time_days"] = 2
        result = prepare_forecast_data.add_lead_time_target(df)
        a_day1 = result[(result["Kode Barang"] == "A") & (result["Tanggal"] == pd.Timestamp("2025-01-01"))].iloc[0]
        b_day1 = result[(result["Kode Barang"] == "B") & (result["Tanggal"] == pd.Timestamp("2025-01-01"))].iloc[0]
        self.assertEqual(a_day1["target_lead_time_cumulative"], 2 + 3)
        self.assertEqual(b_day1["target_lead_time_cumulative"], 200 + 300)


class TestComputeBranchStats(unittest.TestCase):
    def test_stats_computed_only_from_pre_cutoff_rows(self):
        train = pd.DataFrame({
            "Nama Cabang": ["X"] * 10,
            "Tanggal": pd.date_range("2025-11-01", periods=10, freq="D"),
            "Kuantitas": [10] * 10,  # steady 10/day
        })
        test_period = pd.DataFrame({
            "Nama Cabang": ["X"] * 5,
            "Tanggal": pd.date_range("2025-12-01", periods=5, freq="D"),
            "Kuantitas": [99999] * 5,  # extreme test-period values must not leak in
        })
        df = pd.concat([train, test_period], ignore_index=True)
        result = prepare_forecast_data.compute_branch_stats(df, cutoff=pd.Timestamp("2025-12-01"))
        branch_x = result[result["Nama Cabang"] == "X"].iloc[0]
        self.assertAlmostEqual(branch_x["branch_avg_daily_qty"], 10.0)

    def test_changing_test_period_values_does_not_change_output(self):
        train = pd.DataFrame({
            "Nama Cabang": ["X"] * 10,
            "Tanggal": pd.date_range("2025-11-01", periods=10, freq="D"),
            "Kuantitas": [10] * 10,
        })
        cutoff = pd.Timestamp("2025-12-01")
        test_a = pd.concat([train, pd.DataFrame({
            "Nama Cabang": ["X"], "Tanggal": [cutoff], "Kuantitas": [1],
        })], ignore_index=True)
        test_b = pd.concat([train, pd.DataFrame({
            "Nama Cabang": ["X"], "Tanggal": [cutoff], "Kuantitas": [999999],
        })], ignore_index=True)
        result_a = prepare_forecast_data.compute_branch_stats(test_a, cutoff=cutoff)
        result_b = prepare_forecast_data.compute_branch_stats(test_b, cutoff=cutoff)
        pd.testing.assert_frame_equal(result_a, result_b)

    def test_branch_volume_tier_ranks_distinct_branches(self):
        rows = []
        for branch, daily_qty in [("Small", 5), ("Medium", 50), ("Large", 500), ("Flagship", 5000)]:
            rows.append(pd.DataFrame({
                "Nama Cabang": [branch] * 10,
                "Tanggal": pd.date_range("2025-09-01", periods=10, freq="D"),
                "Kuantitas": [daily_qty] * 10,
            }))
        df = pd.concat(rows, ignore_index=True)
        result = prepare_forecast_data.compute_branch_stats(df, cutoff=pd.Timestamp("2025-12-01"))
        tiers = result.set_index("Nama Cabang")["branch_volume_tier"]
        self.assertNotEqual(tiers["Small"], tiers["Flagship"])


class TestApplyBranchStats(unittest.TestCase):
    def test_stats_appear_identically_on_train_and_test_rows(self):
        df = pd.DataFrame({
            "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2025-11-01", "2025-12-15"]),  # one train, one test row
        })
        branch_stats = pd.DataFrame({
            "Nama Cabang": ["X"], "branch_avg_daily_qty": [42.0],
            "branch_demand_cv": [0.5], "branch_volume_tier": ["large"],
        })
        result = prepare_forecast_data.apply_branch_stats(df, branch_stats)
        self.assertEqual(result["branch_avg_daily_qty"].iloc[0], result["branch_avg_daily_qty"].iloc[1])
        self.assertEqual(result["branch_avg_daily_qty"].iloc[0], 42.0)


class TestAddBranchAgeDays(unittest.TestCase):
    def test_first_date_has_zero_age(self):
        df = pd.DataFrame({
            "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2024-03-01", "2024-03-11"]),
        })
        result = prepare_forecast_data.add_branch_age_days(df)
        self.assertEqual(result["branch_age_days"].iloc[0], 0)
        self.assertEqual(result["branch_age_days"].iloc[1], 10)

    def test_large_but_correct_age_for_later_date(self):
        df = pd.DataFrame({
            "Nama Cabang": ["X", "X"],
            "Tanggal": pd.to_datetime(["2024-01-01", "2025-12-15"]),
        })
        result = prepare_forecast_data.add_branch_age_days(df)
        expected = (pd.Timestamp("2025-12-15") - pd.Timestamp("2024-01-01")).days
        self.assertEqual(result["branch_age_days"].iloc[1], expected)

    def test_computed_independently_per_branch(self):
        df = pd.DataFrame({
            "Nama Cabang": ["X", "Y"],
            "Tanggal": pd.to_datetime(["2024-01-10", "2024-01-10"]),
        })
        # X's first date is unknown here (only one row for X shown), but
        # Y's age must not be affected by X's data at all.
        result = prepare_forecast_data.add_branch_age_days(df)
        self.assertEqual(result["branch_age_days"].iloc[1], 0)  # Y's own first date


class TestApplyOutletFeatures(unittest.TestCase):
    def test_joins_outlet_columns_identically_onto_matching_rows(self):
        df = pd.DataFrame({
            "Nama Cabang": ["KY007 - Kebuli Yaman Cibubur", "KY007 - Kebuli Yaman Cibubur"],
            "Tanggal": pd.to_datetime(["2025-11-01", "2025-12-15"]),
        })
        outlets_df = pd.DataFrame({
            "Nama Outlet": ["KY007 - Kebuli Yaman Cibubur"],
            "Alamat": ["addr"], "Kecamatan": ["Ciracas"], "Kota": ["Jakarta Timur"],
            "has_shopee": ["Yes"], "has_gofood": ["Yes"], "has_grabfood": ["Yes"],
        })
        overrides_df = pd.DataFrame(columns=["Nama Cabang", "Nama Outlet", "Kota Override"])
        result = prepare_forecast_data.apply_outlet_features(df, outlets_df, overrides_df)
        self.assertEqual(list(result["kota"]), ["Jakarta Timur", "Jakarta Timur"])
        self.assertTrue(result["can_order_online"].all())

    def test_unmatched_branch_gets_unknown_kota_and_nan_flags(self):
        df = pd.DataFrame({"Nama Cabang": ["KY999 - No Such Outlet"], "Tanggal": pd.to_datetime(["2025-11-01"])})
        outlets_df = pd.DataFrame({
            "Nama Outlet": ["KY007 - Kebuli Yaman Cibubur"],
            "Alamat": ["addr"], "Kecamatan": ["Ciracas"], "Kota": ["Jakarta Timur"],
            "has_shopee": ["Yes"], "has_gofood": ["Yes"], "has_grabfood": ["Yes"],
        })
        overrides_df = pd.DataFrame(columns=["Nama Cabang", "Nama Outlet", "Kota Override"])
        result = prepare_forecast_data.apply_outlet_features(df, outlets_df, overrides_df)
        self.assertEqual(result["kota"].iloc[0], "Unknown")
        self.assertTrue(pd.isna(result["can_order_online"].iloc[0]))


class TestSplitTrainTest(unittest.TestCase):
    def test_boundary_dates_split_correctly(self):
        df = pd.DataFrame({
            "Tanggal": pd.to_datetime(["2025-11-30", "2025-12-01", "2025-12-31"]),
            "Kuantitas": [1, 2, 3],
        })
        train, test = prepare_forecast_data.split_train_test(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(list(train["Tanggal"]), [pd.Timestamp("2025-11-30")])
        self.assertEqual(
            list(test["Tanggal"]), [pd.Timestamp("2025-12-01"), pd.Timestamp("2025-12-31")]
        )

    def test_date_after_available_data_stays_in_test(self):
        df = pd.DataFrame({"Tanggal": pd.to_datetime(["2026-01-05"]), "Kuantitas": [1]})
        train, test = prepare_forecast_data.split_train_test(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(len(test), 1)
        self.assertEqual(len(train), 0)


class TestExportSplits(unittest.TestCase):
    def test_writes_and_round_trips_parquet_files(self):
        train = pd.DataFrame({
            "Tanggal": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "Kuantitas": [1, 2],
        })
        test = pd.DataFrame({
            "Tanggal": pd.to_datetime(["2025-12-01"]),
            "Kuantitas": [3],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_forecast_data.export_splits(train, test, output_dir=tmpdir)
            train_path = Path(tmpdir) / "train.parquet"
            test_path = Path(tmpdir) / "test.parquet"
            self.assertTrue(train_path.exists())
            self.assertTrue(test_path.exists())
            round_tripped_train = pd.read_parquet(train_path)
            round_tripped_test = pd.read_parquet(test_path)
        self.assertEqual(len(round_tripped_train), 2)
        self.assertEqual(len(round_tripped_test), 1)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(round_tripped_train["Tanggal"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(round_tripped_train["Kuantitas"]))


def _write_outlets_fixture(tmpdir, outlet_names):
    path = Path(tmpdir) / "outlets.csv"
    lines = ["Nama Outlet;Alamat;Kecamatan;Kota;has_shopee;has_gofood;has_grabfood\n"]
    for name in outlet_names:
        lines.append(f"{name};addr;Kec;Kota A;Yes;Yes;Yes\n")
    path.write_bytes(b"\xef\xbb\xbf" + "".join(lines).encode("utf-8"))
    return path


def _write_empty_overrides_fixture(tmpdir):
    path = Path(tmpdir) / "outlet_name_overrides.csv"
    path.write_bytes(b"\xef\xbb\xbfNama Cabang;Nama Outlet;Kota Override\n")
    return path


def _write_overrides_fixture(tmpdir, rows):
    path = Path(tmpdir) / "outlet_name_overrides.csv"
    lines = ["Nama Cabang;Nama Outlet;Kota Override\n"]
    for nama_cabang, nama_outlet, kota_override in rows:
        lines.append(f"{nama_cabang};{nama_outlet};{kota_override}\n")
    path.write_bytes(b"\xef\xbb\xbf" + "".join(lines).encode("utf-8"))
    return path


def _write_region_mapping_fixture(tmpdir, rows):
    path = Path(tmpdir) / "outlet_mapping.csv"
    lines = ["old_name;new_name;Alamat;Kecamatan;Kota;has_shopee;has_gofood;has_grabfood;kawasan;hari_pengiriman\n"]
    for old_name, new_name, kawasan, hari_pengiriman in rows:
        lines.append(f"{old_name};{new_name};addr;Kec;Kota A;Yes;Yes;Yes;{kawasan};{hari_pengiriman}\n")
    path.write_bytes(b"\xef\xbb\xbf" + "".join(lines).encode("utf-8"))
    return path


def _branch_rows(branch, start, periods):
    lines = []
    for i, date in enumerate(pd.date_range(start, periods=periods, freq="D")):
        lines.append(
            f"{date.strftime('%d %b %Y')};Barang Jadi (FG);FGS-00001;Widget;"
            f"{branch};Porsi;{i + 1}\n"
        )
    return lines


def _branch_rows_with_quantities(branch, start, quantities):
    lines = []
    for date, qty in zip(pd.date_range(start, periods=len(quantities), freq="D"), quantities):
        lines.append(
            f"{date.strftime('%d %b %Y')};Barang Jadi (FG);FGS-00001;Widget;"
            f"{branch};Porsi;{qty}\n"
        )
    return lines


class TestBuildFeaturedDataset(unittest.TestCase):
    def test_returns_full_unsplit_range_with_all_expected_columns(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        rows += _branch_rows("KY001 - Branch", "2025-08-01", 90)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_empty_overrides_fixture(tmpdir)
            region_path = _write_region_mapping_fixture(
                tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
            )
            cutoff = pd.Timestamp("2025-10-01")
            df = prepare_forecast_data.build_featured_dataset(
                input_path=input_path,
                min_history_days=60,
                cutoff=cutoff,
                outlets_path=outlets_path,
                overrides_path=overrides_path,
                region_path=region_path,
            )
        # Unsplit: unlike train/test, both sides of the cutoff are present.
        self.assertTrue((df["Tanggal"] < cutoff).any())
        self.assertTrue((df["Tanggal"] >= cutoff).any())
        for col in [
            "target_h1", "lag_1", "is_ramadan", "branch_avg_daily_qty", "kota",
            "can_order_online", "kawasan", "lead_time_days", "target_lead_time_cumulative",
        ]:
            self.assertIn(col, df.columns)


class TestCappedLeadTimeTarget(unittest.TestCase):
    """A second target built from Kuantitas_capped.

    The business needs the model for demand *outside* pre-orders, which the
    team handles separately, but Kuantitas mixes both. Capped spikes are the
    only available proxy for the pre-order component, so a capped target
    approximates the scope the model is actually responsible for. The raw
    target stays for full-accountability reporting.
    """

    def _frame(self, quantities, capped):
        n = len(quantities)
        return pd.DataFrame({
            "Kode Barang": ["A"] * n, "Nama Cabang": ["X"] * n,
            "Tanggal": pd.date_range("2025-01-01", periods=n, freq="D"),
            "Kuantitas": quantities, "Kuantitas_capped": capped,
            "lead_time_days": [2] * n,
        })

    def test_named_target_column_is_produced(self):
        df = self._frame([1, 2, 3, 4], [1, 2, 3, 4])
        result = prepare_forecast_data.add_lead_time_target(
            df, qty_col="Kuantitas_capped", target_col="target_capped"
        )
        self.assertIn("target_capped", result.columns)

    def test_capped_target_sums_the_capped_quantities(self):
        df = self._frame([1, 2, 3, 4], [1, 2, 3, 4])
        result = prepare_forecast_data.add_lead_time_target(
            df, qty_col="Kuantitas_capped", target_col="target_capped"
        ).sort_values("Tanggal")
        self.assertEqual(result["target_capped"].iloc[0], 5)  # 2 + 3

    def test_capped_target_never_exceeds_the_raw_target(self):
        df = self._frame([1, 100, 3, 4], [1, 5, 3, 4])
        result = prepare_forecast_data.add_lead_time_target(df)
        result = prepare_forecast_data.add_lead_time_target(
            result, qty_col="Kuantitas_capped", target_col="target_capped"
        )
        both = result.dropna(subset=["target_lead_time_cumulative", "target_capped"])
        self.assertTrue(
            (both["target_capped"] <= both["target_lead_time_cumulative"]).all()
        )

    def test_capping_a_spike_lowers_the_target_that_covers_it(self):
        df = self._frame([1, 100, 3, 4], [1, 5, 3, 4])
        result = prepare_forecast_data.add_lead_time_target(
            df, qty_col="Kuantitas_capped", target_col="target_capped"
        ).sort_values("Tanggal")
        self.assertEqual(result["target_capped"].iloc[0], 8)  # 5 + 3, not 100 + 3

    def test_default_target_name_is_unchanged(self):
        df = self._frame([1, 2, 3, 4], [1, 2, 3, 4])
        result = prepare_forecast_data.add_lead_time_target(df)
        self.assertIn("target_lead_time_cumulative", result.columns)


class TestSplitTrainTestPurging(unittest.TestCase):
    """Train rows whose label reaches into the test month must be dropped."""

    def _frame(self):
        dates = pd.date_range("2025-11-25", periods=12, freq="D")  # 25 Nov - 6 Dec
        return pd.DataFrame({
            "Tanggal": dates,
            "lead_time_days": [4] * len(dates),
            "target_lead_time_cumulative": range(len(dates)),
        })

    def test_train_rows_reaching_into_test_are_removed(self):
        train, _ = prepare_forecast_data.split_train_test(
            self._frame(), cutoff=pd.Timestamp("2025-12-01")
        )
        # 25 and 26 Nov end on 29 and 30 Nov; 27-30 Nov all reach 1 Dec or past.
        self.assertEqual(
            list(train["Tanggal"]),
            [pd.Timestamp("2025-11-25"), pd.Timestamp("2025-11-26")],
        )

    def test_test_side_is_never_purged(self):
        _, test = prepare_forecast_data.split_train_test(
            self._frame(), cutoff=pd.Timestamp("2025-12-01")
        )
        self.assertEqual(len(test), 6)  # 1-6 Dec, all kept
        self.assertEqual(test["Tanggal"].min(), pd.Timestamp("2025-12-01"))

    def test_purge_false_restores_the_previous_behaviour(self):
        train, _ = prepare_forecast_data.split_train_test(
            self._frame(), cutoff=pd.Timestamp("2025-12-01"), purge=False
        )
        self.assertEqual(len(train), 6)  # 25-30 Nov

    def test_frame_without_lead_time_column_is_left_intact(self):
        """featured.parquet always carries lead_time_days, but fixtures and
        older callers may not — they must not silently lose rows."""
        df = pd.DataFrame({
            "Tanggal": pd.date_range("2025-11-25", periods=10, freq="D"),
            "Kuantitas": range(10),
        })
        train, test = prepare_forecast_data.split_train_test(
            df, cutoff=pd.Timestamp("2025-12-01")
        )
        self.assertEqual(len(train) + len(test), 10)


class TestBuildFeaturedDatasetRelocationSegments(unittest.TestCase):
    """A relocation date must split the pair's history into two segments."""

    def _run(self, relocation_dates):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        rows += _branch_rows("KY001 - Branch", "2025-06-01", 120)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + "".join(rows).encode("utf-8"))
            return prepare_forecast_data.build_featured_dataset(
                input_path=input_path,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=_write_outlets_fixture(tmpdir, ["KY001 - Branch"]),
                overrides_path=_write_empty_overrides_fixture(tmpdir),
                region_path=_write_region_mapping_fixture(
                    tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
                ),
                relocation_dates=relocation_dates,
            )

    def test_relocation_date_splits_the_pair_into_two_segments(self):
        df = self._run({"KY001 - Branch": pd.Timestamp("2025-08-01")})
        self.assertEqual(sorted(df["segment_id"].unique()), [1, 2])
        moved = df[df["Tanggal"] == pd.Timestamp("2025-08-01")]
        self.assertEqual(moved["segment_id"].iloc[0], 2)

    def test_no_relocation_leaves_a_single_segment(self):
        df = self._run({})
        self.assertEqual(list(df["segment_id"].unique()), [1])

    def test_lag_does_not_reach_across_the_relocation(self):
        df = self._run({"KY001 - Branch": pd.Timestamp("2025-08-01")})
        first_after = df[df["Tanggal"] == pd.Timestamp("2025-08-01")]
        self.assertTrue(pd.isna(first_after["lag_1"].iloc[0]))

    def test_target_does_not_reach_across_the_relocation(self):
        df = self._run({"KY001 - Branch": pd.Timestamp("2025-08-01")})
        last_before = df[df["Tanggal"] == pd.Timestamp("2025-07-31")]
        self.assertTrue(pd.isna(last_before["target_h1"].iloc[0]))


class TestFeaturedDatasetNewColumns(unittest.TestCase):
    """The three columns added for scope-correct evaluation must reach the
    exported parquet, not just exist as functions."""

    def _featured(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        rows += _branch_rows("KY001 - Branch", "2025-06-01", 120)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + "".join(rows).encode("utf-8"))
            return prepare_forecast_data.build_featured_dataset(
                input_path=input_path,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=_write_outlets_fixture(tmpdir, ["KY001 - Branch"]),
                overrides_path=_write_empty_overrides_fixture(tmpdir),
                region_path=_write_region_mapping_fixture(
                    tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
                ),
            )

    def test_all_three_new_columns_are_present(self):
        df = self._featured()
        for col in [
            "target_lead_time_cumulative_capped",
            "is_delivery_day",
            "target_window_weekend_days",
        ]:
            self.assertIn(col, df.columns)

    def test_featured_columns_constant_lists_them(self):
        for col in [
            "target_lead_time_cumulative_capped",
            "is_delivery_day",
            "target_window_weekend_days",
        ]:
            self.assertIn(col, prepare_forecast_data.FEATURED_COLUMNS)

    def test_capped_target_never_exceeds_the_raw_target(self):
        df = self._featured().dropna(
            subset=["target_lead_time_cumulative", "target_lead_time_cumulative_capped"]
        )
        self.assertTrue(
            (
                df["target_lead_time_cumulative_capped"]
                <= df["target_lead_time_cumulative"]
            ).all()
        )

    def test_delivery_days_are_the_branch_shipping_weekdays(self):
        df = self._featured()
        flagged = df.loc[df["is_delivery_day"], "Tanggal"].dt.weekday.unique()
        self.assertEqual(sorted(flagged), [0, 3])  # Senin dan Kamis

    def test_lead_time_on_delivery_days_is_only_three_or_four(self):
        """The Monday shipment covers Tue-Thu, the Thursday one Fri-Mon. Rows
        with lead time 1-2 are never a shipment decision."""
        df = self._featured()
        self.assertEqual(
            sorted(df.loc[df["is_delivery_day"], "lead_time_days"].unique()), [3, 4]
        )

    def test_qa_rejects_a_capped_target_above_the_raw_one(self):
        df = self._featured().copy()
        df.loc[df.index[0], "target_lead_time_cumulative_capped"] = 1e9
        with self.assertRaisesRegex(AssertionError, "capped"):
            prepare_forecast_data.run_qa_checks(df)


class TestObservedRelocationDates(unittest.TestCase):
    """Only relocations visible inside the data may break a segment.

    The five lower-bound dates describe moves that happened after coverage
    ends, so there is no break in the data to mark -- forcing one would carve
    a one-day segment out of the test window and strip its lag features.
    """

    def test_observed_dates_are_a_subset_of_all_relocation_dates(self):
        self.assertTrue(
            set(outlet_features.OBSERVED_RELOCATION_DATES)
            <= set(outlet_features.RELOCATION_DATES)
        )

    def test_lower_bound_relocations_are_excluded(self):
        for branch in outlet_features.LOWER_BOUND_RELOCATIONS:
            self.assertNotIn(branch, outlet_features.OBSERVED_RELOCATION_DATES)

    def test_observed_dates_keep_the_same_timestamps(self):
        for branch, date in outlet_features.OBSERVED_RELOCATION_DATES.items():
            self.assertEqual(date, outlet_features.RELOCATION_DATES[branch])

    def test_every_lower_bound_branch_is_a_known_relocation(self):
        self.assertTrue(
            set(outlet_features.LOWER_BOUND_RELOCATIONS)
            <= set(outlet_features.RELOCATION_DATES)
        )


class TestExportFeatured(unittest.TestCase):
    def test_writes_and_round_trips_parquet_file(self):
        df = pd.DataFrame({
            "Tanggal": pd.to_datetime(["2025-01-01", "2025-12-15"]),
            "Kuantitas": [1, 2],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            prepare_forecast_data.export_featured(df, output_dir=tmpdir)
            featured_path = Path(tmpdir) / "featured.parquet"
            self.assertTrue(featured_path.exists())
            round_tripped = pd.read_parquet(featured_path)
        self.assertEqual(len(round_tripped), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(round_tripped["Tanggal"]))


class TestMain(unittest.TestCase):
    def test_main_writes_train_and_test_parquet_end_to_end(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        # 90 days of daily activity for one pair: 2025-08-01 .. 2025-10-29.
        # Cutoff 2025-10-01 leaves 61 pre-cutoff days (>= the 60-day minimum)
        # and 29 post-cutoff days, so both train and test end up non-empty.
        rows += _branch_rows("KY001 - Branch", "2025-08-01", 90)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            output_dir = Path(tmpdir) / "model_ready"
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_empty_overrides_fixture(tmpdir)
            region_path = _write_region_mapping_fixture(
                tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
            )
            prepare_forecast_data.main(
                input_path=input_path,
                output_dir=output_dir,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=outlets_path,
                overrides_path=overrides_path,
                region_path=region_path,
            )
            train = pd.read_parquet(output_dir / "train.parquet")
            test = pd.read_parquet(output_dir / "test.parquet")
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertIn("target_h1", train.columns)
        self.assertIn("lag_1", train.columns)
        self.assertIn("is_ramadan", train.columns)
        self.assertIn("branch_avg_daily_qty", train.columns)
        self.assertIn("kota", train.columns)
        self.assertIn("can_order_online", train.columns)
        self.assertIn("kawasan", train.columns)
        self.assertIn("lead_time_days", train.columns)
        self.assertIn("target_lead_time_cumulative", train.columns)

    def test_main_drops_rows_for_branch_with_no_outlet_match(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        rows += _branch_rows("KY001 - Branch", "2025-08-01", 90)
        rows += _branch_rows("KY999 - Ghost Branch", "2025-08-01", 90)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            output_dir = Path(tmpdir) / "model_ready"
            # Only "KY001 - Branch" has an outlet counterpart; "KY999 - Ghost
            # Branch" must be dropped entirely rather than kept as "Unknown".
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_empty_overrides_fixture(tmpdir)
            region_path = _write_region_mapping_fixture(
                tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
            )
            prepare_forecast_data.main(
                input_path=input_path,
                output_dir=output_dir,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=outlets_path,
                overrides_path=overrides_path,
                region_path=region_path,
            )
            train = pd.read_parquet(output_dir / "train.parquet")
            test = pd.read_parquet(output_dir / "test.parquet")
        self.assertNotIn("KY999 - Ghost Branch", train["Nama Cabang"].values)
        self.assertNotIn("KY999 - Ghost Branch", test["Nama Cabang"].values)
        self.assertIn("KY001 - Branch", train["Nama Cabang"].values)

    def test_main_merges_legacy_branch_spelling_into_canonical_history(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        # Mirrors the real TOD M1 Bandara / KY051 split: a legacy short name
        # used for an early period, then the canonical "KY0NN - ..." name
        # used from then on, with no date overlap between the two. Each side
        # individually clears the 60-day min-history bar, so if canonicalization
        # were NOT happening, "Legacy Name" would survive filter_min_history
        # and show up as its own separate branch rather than being dropped —
        # making this a real test of merging, not a false pass via the
        # min-history filter incidentally dropping the shorter series.
        rows += _branch_rows("Legacy Name", "2025-01-01", 65)
        rows += _branch_rows("KY001 - Branch", "2025-03-10", 65)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            output_dir = Path(tmpdir) / "model_ready"
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_overrides_fixture(tmpdir, [("Legacy Name", "KY001 - Branch", "")])
            region_path = _write_region_mapping_fixture(
                tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
            )
            prepare_forecast_data.main(
                input_path=input_path,
                output_dir=output_dir,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=outlets_path,
                overrides_path=overrides_path,
                region_path=region_path,
            )
            train = pd.read_parquet(output_dir / "train.parquet")
            test = pd.read_parquet(output_dir / "test.parquet")
        combined = pd.concat([train, test])
        self.assertEqual(set(combined["Nama Cabang"].unique()), {"KY001 - Branch"})
        self.assertNotIn("Legacy Name", combined["Nama Cabang"].values)

    def test_main_sums_kuantitas_when_duplicate_spellings_overlap_same_date(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        # Both raw spellings report on the exact same dates for the exact
        # same item — a regression check for the reaggregate_daily step
        # that must run after canonicalization, since build_dense_panel's
        # reindex would otherwise raise on duplicate (item, date) rows.
        rows += _branch_rows("Legacy Name", "2025-08-01", 90)
        rows += _branch_rows("KY001 - Branch", "2025-08-01", 90)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            output_dir = Path(tmpdir) / "model_ready"
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_overrides_fixture(tmpdir, [("Legacy Name", "KY001 - Branch", "")])
            region_path = _write_region_mapping_fixture(
                tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
            )
            prepare_forecast_data.main(
                input_path=input_path,
                output_dir=output_dir,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=outlets_path,
                overrides_path=overrides_path,
                region_path=region_path,
            )
            train = pd.read_parquet(output_dir / "train.parquet")
            test = pd.read_parquet(output_dir / "test.parquet")
        combined = pd.concat([train, test])
        self.assertEqual(set(combined["Nama Cabang"].unique()), {"KY001 - Branch"})
        first_day = combined[combined["Tanggal"] == pd.Timestamp("2025-08-01")].iloc[0]
        self.assertEqual(first_day["Kuantitas"], 2)  # both fixtures' day-1 Kuantitas is 1, summed

    def test_main_caps_lag_input_but_leaves_target_and_flags_spike(self):
        rows = ["Tanggal;Kategori Barang;Kode Barang;Nama Barang;Nama Cabang;Satuan;Kuantitas\n"]
        # 90 days, 2025-08-01..2025-10-29, steady Kuantitas=10, except a
        # single spike of 1000 on 2025-08-31 (not a calendar event date).
        # Cutoff 2025-10-01 gives 61 pre-cutoff real-transaction days for
        # "KY001 - Branch", comfortably above MIN_PAIR_HISTORY (30) and
        # min_history_days (60).
        quantities = [10] * 90
        quantities[30] = 1000  # 2025-08-01 + 30 days = 2025-08-31
        rows += _branch_rows_with_quantities("KY001 - Branch", "2025-08-01", quantities)
        content = "".join(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "dataset.csv"
            input_path.write_bytes(b"\xef\xbb\xbf" + content.encode("utf-8"))
            output_dir = Path(tmpdir) / "model_ready"
            outlets_path = _write_outlets_fixture(tmpdir, ["KY001 - Branch"])
            overrides_path = _write_empty_overrides_fixture(tmpdir)
            region_path = _write_region_mapping_fixture(
                tmpdir, [("KY001 - Branch", "KY001 - Branch", 1, "Senin dan Kamis")]
            )
            prepare_forecast_data.main(
                input_path=input_path,
                output_dir=output_dir,
                min_history_days=60,
                cutoff=pd.Timestamp("2025-10-01"),
                outlets_path=outlets_path,
                overrides_path=overrides_path,
                region_path=region_path,
            )
            train = pd.read_parquet(output_dir / "train.parquet")

        self.assertIn("Kuantitas_capped", train.columns)
        self.assertIn("baseline_ratio", train.columns)
        self.assertIn("is_spike", train.columns)

        spike_day = train[train["Tanggal"] == pd.Timestamp("2025-08-31")].iloc[0]
        self.assertTrue(spike_day["is_spike"])
        self.assertEqual(spike_day["Kuantitas_capped"], 50.0)  # 10 * SPIKE_RATIO_THRESHOLD

        day_before_spike = train[train["Tanggal"] == pd.Timestamp("2025-08-30")].iloc[0]
        self.assertEqual(day_before_spike["target_h1"], 1000)  # target uses RAW Kuantitas

        day_after_spike = train[train["Tanggal"] == pd.Timestamp("2025-09-01")].iloc[0]
        self.assertEqual(day_after_spike["lag_1"], 50.0)  # lag uses CAPPED Kuantitas

        branch_stats_row = train[train["Tanggal"] == pd.Timestamp("2025-09-15")].iloc[0]
        # Uncapped, the spike would pull the average toward ~26; capped it
        # stays close to the steady 10/day baseline.
        self.assertLess(branch_stats_row["branch_avg_daily_qty"], 15.0)


class TestEngineerFeaturesContract(unittest.TestCase):
    def test_featured_columns_constant_has_67_entries(self):
        # 64 before 2026-08-15, plus target_lead_time_cumulative_capped,
        # is_delivery_day, and target_window_weekend_days.
        self.assertEqual(len(prepare_forecast_data.FEATURED_COLUMNS), 67)

    def test_featured_columns_includes_days_since_relocation(self):
        self.assertIn("days_since_relocation", prepare_forecast_data.FEATURED_COLUMNS)

    def test_featured_columns_includes_segment_id(self):
        self.assertIn("segment_id", prepare_forecast_data.FEATURED_COLUMNS)

    def test_build_featured_dataset_delegates_to_engineer_features(self):
        self.assertTrue(hasattr(prepare_forecast_data, "engineer_features"))
        self.assertTrue(callable(prepare_forecast_data.engineer_features))


def _qa_frame():
    """Minimal frame that satisfies every run_qa_checks assertion."""
    return pd.DataFrame({
        "Kode Barang": ["FGS-00001", "FGS-00001"],
        "Nama Cabang": ["KY001 - A", "KY001 - A"],
        "Tanggal": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Kuantitas": [10.0, 12.0],
        "Kuantitas_capped": [10.0, 12.0],
        "kota": ["Kota Tangerang", "Kota Tangerang"],
        "kawasan": [1, 1],
        "segment_id": [1, 1],
    })


class TestRunQaChecks(unittest.TestCase):
    def test_passes_on_a_clean_frame(self):
        self.assertIsNone(prepare_forecast_data.run_qa_checks(_qa_frame()))

    def test_rejects_negative_kuantitas(self):
        df = _qa_frame()
        df.loc[0, "Kuantitas"] = -1.0
        with self.assertRaisesRegex(AssertionError, "negatif"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_duplicate_pair_date_rows(self):
        df = _qa_frame()
        df.loc[1, "Tanggal"] = df.loc[0, "Tanggal"]
        with self.assertRaisesRegex(AssertionError, "duplikat"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_capped_exceeding_raw(self):
        df = _qa_frame()
        df.loc[0, "Kuantitas_capped"] = 999.0
        with self.assertRaisesRegex(AssertionError, "capped"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_unknown_kota(self):
        df = _qa_frame()
        df.loc[0, "kota"] = "Unknown"
        with self.assertRaisesRegex(AssertionError, "kota"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_null_kawasan(self):
        df = _qa_frame()
        df.loc[0, "kawasan"] = None
        with self.assertRaisesRegex(AssertionError, "kawasan"):
            prepare_forecast_data.run_qa_checks(df)

    def test_rejects_branch_mapped_to_two_cities(self):
        df = _qa_frame()
        df.loc[1, "kota"] = "Kota Bekasi"
        with self.assertRaisesRegex(AssertionError, "lebih dari satu kota"):
            prepare_forecast_data.run_qa_checks(df)


class TestSegmentAwareFeatures(unittest.TestCase):
    def _panel(self):
        # One pair, two segments: 2024-01-01..2024-01-05 and 2024-03-01..2024-03-05.
        dates = list(pd.date_range("2024-01-01", periods=5, freq="D"))
        dates += list(pd.date_range("2024-03-01", periods=5, freq="D"))
        return pd.DataFrame({
            "Kode Barang": ["A"] * 10, "Nama Cabang": ["X"] * 10,
            "Tanggal": dates, "Kuantitas": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "segment_id": [1] * 5 + [2] * 5,
        })

    def test_targets_do_not_jump_across_a_segment_boundary(self):
        result = prepare_forecast_data.add_targets(
            self._panel(), pair_cols=prepare_forecast_data.SEGMENT_COLS
        )
        last_of_segment_1 = result[
            (result["segment_id"] == 1) & (result["Tanggal"] == pd.Timestamp("2024-01-05"))
        ]
        self.assertTrue(pd.isna(last_of_segment_1["target_h1"].iloc[0]))

    def test_targets_would_jump_without_segment_cols(self):
        # Guards the guard: with plain PAIR_COLS the target silently reaches
        # across the closure, which is the bug segmentation exists to prevent.
        result = prepare_forecast_data.add_targets(
            self._panel(), pair_cols=prepare_forecast_data.PAIR_COLS
        )
        last_of_segment_1 = result[result["Tanggal"] == pd.Timestamp("2024-01-05")]
        self.assertEqual(last_of_segment_1["target_h1"].iloc[0], 6)

    def test_lags_do_not_cross_a_segment_boundary(self):
        result = prepare_forecast_data.add_lag_features(
            self._panel(), pair_cols=prepare_forecast_data.SEGMENT_COLS
        )
        first_of_segment_2 = result[result["Tanggal"] == pd.Timestamp("2024-03-01")]
        self.assertTrue(pd.isna(first_of_segment_2["lag_1"].iloc[0]))

    def test_rolling_does_not_cross_a_segment_boundary(self):
        panel = self._panel()
        result = prepare_forecast_data.add_rolling_features(
            panel, pair_cols=prepare_forecast_data.SEGMENT_COLS, windows=[3]
        )
        first_of_segment_2 = result[result["Tanggal"] == pd.Timestamp("2024-03-01")]
        self.assertTrue(pd.isna(first_of_segment_2["roll_mean_3"].iloc[0]))


class TestSegmentQaChecks(unittest.TestCase):
    def _minimal_featured(self):
        dates = list(pd.date_range("2024-01-01", periods=3, freq="D"))
        return pd.DataFrame({
            "Kode Barang": ["A"] * 3, "Nama Cabang": ["X"] * 3,
            "Tanggal": dates, "Kuantitas": [1.0, 2.0, 3.0],
            "Kuantitas_capped": [1.0, 2.0, 3.0], "segment_id": [1, 1, 1],
            "kota": ["Kota Tangerang"] * 3, "kawasan": [1, 1, 1],
        })

    def test_clean_frame_passes(self):
        prepare_forecast_data.run_qa_checks(self._minimal_featured(), closures={})

    def test_row_inside_a_closure_fails(self):
        df = self._minimal_featured()
        closures = {"X": [(pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"))]}
        with self.assertRaises(AssertionError):
            prepare_forecast_data.run_qa_checks(df, closures=closures)

    def test_segment_ids_not_starting_at_one_fail(self):
        df = self._minimal_featured()
        df["segment_id"] = [2, 2, 2]
        with self.assertRaises(AssertionError):
            prepare_forecast_data.run_qa_checks(df, closures={})

    def test_date_gap_inside_one_segment_fails(self):
        df = self._minimal_featured()
        df.loc[2, "Tanggal"] = pd.Timestamp("2024-01-10")
        with self.assertRaises(AssertionError):
            prepare_forecast_data.run_qa_checks(df, closures={})


if __name__ == "__main__":
    unittest.main()
