import unittest

import numpy as np
import pandas as pd

from utils import build_panel, modeling_prep


def _event_items(rows):
    return pd.DataFrame(rows, columns=["Kode Barang", "is_event_driven"])


class TestAddEventFlag(unittest.TestCase):
    def test_marks_true_for_listed_event_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_marks_false_for_ordinary_sku(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00002", "true"], ["PCG-00001", "false"]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertFalse(bool(result.iloc[0]["is_event_driven"]))

    def test_is_case_and_whitespace_insensitive(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00002"]})
        items = _event_items([["PCG-00002", "  TRUE "]])
        result = modeling_prep.add_event_flag(df, items)
        self.assertTrue(bool(result.iloc[0]["is_event_driven"]))

    def test_raises_when_a_sku_is_missing_from_the_list(self):
        df = pd.DataFrame({"Kode Barang": ["FGS-99999"]})
        items = _event_items([["PCG-00002", "true"]])
        with self.assertRaisesRegex(ValueError, "FGS-99999"):
            modeling_prep.add_event_flag(df, items)

    def test_does_not_mutate_the_input_frame(self):
        df = pd.DataFrame({"Kode Barang": ["PCG-00001"]})
        items = _event_items([["PCG-00001", "false"]])
        modeling_prep.add_event_flag(df, items)
        self.assertNotIn("is_event_driven", df.columns)


def _series_frame(quantities, start="2024-01-01", item="I1", branch="B1"):
    return pd.DataFrame({
        "Kode Barang": [item] * len(quantities),
        "Nama Cabang": [branch] * len(quantities),
        "Tanggal": pd.date_range(start, periods=len(quantities), freq="D"),
        "Kuantitas": [float(q) for q in quantities],
    })


class TestClassifyPairs(unittest.TestCase):
    def test_daily_stable_demand_is_smooth(self):
        df = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "smooth")

    def test_daily_but_wildly_varying_demand_is_erratic(self):
        df = _series_frame([1, 50, 2, 80, 3, 90, 1, 70])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "erratic")

    def test_rare_but_consistent_demand_is_intermittent(self):
        df = _series_frame([10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "intermittent")

    def test_rare_and_bulky_demand_is_lumpy(self):
        df = _series_frame([5, 0, 0, 0, 0, 0, 0, 200, 0, 0, 0, 0, 0, 0, 90])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "lumpy")

    def test_pair_that_never_moved_is_lumpy(self):
        df = _series_frame([0, 0, 0, 0])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result.iloc[0]["demand_segment"], "lumpy")

    def test_segment_ignores_rows_at_or_after_the_cutoff(self):
        """The whole point of computing segments on train only: post-cutoff
        behaviour must not change the label."""
        train_only = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        with_future = pd.concat([
            train_only,
            _series_frame([0] * 60, start="2025-12-01"),
        ], ignore_index=True)
        cutoff = pd.Timestamp("2025-12-01")
        a = modeling_prep.classify_pairs(train_only, cutoff=cutoff).iloc[0]["demand_segment"]
        b = modeling_prep.classify_pairs(with_future, cutoff=cutoff).iloc[0]["demand_segment"]
        self.assertEqual(a, b)

    def test_every_row_of_a_pair_gets_the_same_label(self):
        df = _series_frame([10, 11, 10, 9, 10, 11, 10, 10])
        result = modeling_prep.classify_pairs(df, cutoff=pd.Timestamp("2025-12-01"))
        self.assertEqual(result["demand_segment"].nunique(), 1)


class TestAssignFolds(unittest.TestCase):
    def _frame(self, dates):
        return pd.DataFrame({"Tanggal": pd.to_datetime(dates)})

    def test_july_2025_is_fold_1(self):
        result = modeling_prep.assign_folds(self._frame(["2025-07-15"]))
        self.assertEqual(result.iloc[0]["fold_id"], 1)

    def test_november_2025_is_fold_5(self):
        result = modeling_prep.assign_folds(self._frame(["2025-11-15"]))
        self.assertEqual(result.iloc[0]["fold_id"], 5)

    def test_rows_before_july_2025_have_no_fold(self):
        result = modeling_prep.assign_folds(self._frame(["2024-05-01", "2025-06-30"]))
        self.assertTrue(result["fold_id"].isna().all())

    def test_december_2025_has_no_fold_because_it_is_the_locked_test_set(self):
        result = modeling_prep.assign_folds(self._frame(["2025-12-15"]))
        self.assertTrue(result["fold_id"].isna().all())

    def test_month_boundaries_land_in_the_right_fold(self):
        result = modeling_prep.assign_folds(
            self._frame(["2025-07-01", "2025-07-31", "2025-08-01"])
        )
        self.assertEqual(list(result["fold_id"]), [1, 1, 2])

    def test_train_mask_excludes_the_validation_month_itself(self):
        df = self._frame(["2025-06-30", "2025-07-15", "2025-08-15"])
        mask = modeling_prep.fold_train_mask(df, fold_id=1)
        self.assertEqual(list(mask), [True, False, False])

    def test_train_mask_expands_for_later_folds(self):
        df = self._frame(["2025-06-30", "2025-07-15", "2025-08-15"])
        mask = modeling_prep.fold_train_mask(df, fold_id=2)
        self.assertEqual(list(mask), [True, True, False])

    def test_no_validation_row_ever_appears_in_its_own_training_mask(self):
        df = self._frame([
            "2025-06-15", "2025-07-15", "2025-08-15",
            "2025-09-15", "2025-10-15", "2025-11-15",
        ])
        folded = modeling_prep.assign_folds(df)
        for fold in range(1, 6):
            train = modeling_prep.fold_train_mask(df, fold_id=fold)
            valid = folded["fold_id"] == fold
            self.assertFalse((train & valid).any(), f"kebocoran di fold {fold}")


class TestFeatureCols(unittest.TestCase):
    """The canonical feature list all three models train on.

    Without one written down, each model script picks its own columns and the
    comparison stops being a comparison.
    """

    def test_baseline_ratio_is_excluded(self):
        """baseline_ratio is Kuantitas_H divided by a per-pair constant, so a
        tree that can identify the pair can recover today's own demand from
        it. Every lag and rolling feature deliberately stops at H-1; keeping
        this one would make 'what the model knows at prediction time' mean two
        different things in the same row."""
        self.assertNotIn("baseline_ratio", modeling_prep.FEATURE_COLS)

    def test_is_spike_is_excluded(self):
        self.assertNotIn("is_spike", modeling_prep.FEATURE_COLS)

    def test_no_target_column_leaks_into_the_features(self):
        for col in modeling_prep.FEATURE_COLS:
            self.assertFalse(
                col.startswith("target_lead_time") or col.startswith("target_h"),
                f"{col} adalah target, bukan fitur",
            )

    def test_target_window_composition_is_a_feature_not_a_target(self):
        """It describes which weekdays the window covers, all knowable in
        advance from the calendar."""
        self.assertIn("target_window_weekend_days", modeling_prep.FEATURE_COLS)

    def test_raw_quantity_columns_are_excluded(self):
        for col in ["Kuantitas", "Kuantitas_capped"]:
            self.assertNotIn(col, modeling_prep.FEATURE_COLS)

    def test_identifier_columns_are_excluded(self):
        for col in ["Tanggal", "segment_id", "Nama Barang", "fold_id"]:
            self.assertNotIn(col, modeling_prep.FEATURE_COLS)

    def test_categoricals_appear_only_as_encoded_indices(self):
        for col in modeling_prep.CATEGORICAL_COLS:
            self.assertNotIn(col, modeling_prep.FEATURE_COLS)
            self.assertIn(f"{col}_idx", modeling_prep.FEATURE_COLS)

    def test_lag_and_rolling_features_are_all_included(self):
        for col in modeling_prep.HISTORY_COLS:
            self.assertIn(col, modeling_prep.FEATURE_COLS)

    def test_imputation_indicators_are_included(self):
        for col in ["was_relocated", "has_full_history", "missing_history_count"]:
            self.assertIn(col, modeling_prep.FEATURE_COLS)

    def test_list_has_no_duplicates(self):
        self.assertEqual(
            len(modeling_prep.FEATURE_COLS), len(set(modeling_prep.FEATURE_COLS))
        )


class TestFoldTrainMaskPurging(unittest.TestCase):
    """Each fold boundary needs the same purge as the train/test boundary."""

    def _frame(self):
        dates = pd.date_range("2025-06-25", periods=10, freq="D")  # 25 Jun - 4 Jul
        return pd.DataFrame({
            "Tanggal": dates,
            "lead_time_days": [4] * len(dates),
        })

    def test_rows_whose_label_enters_the_validation_month_are_excluded(self):
        mask = modeling_prep.fold_train_mask(self._frame(), 1)
        kept = self._frame().loc[mask, "Tanggal"]
        # Fold 1 validates July; 25 and 26 Jun end on 29 and 30 Jun and stay.
        self.assertEqual(
            list(kept), [pd.Timestamp("2025-06-25"), pd.Timestamp("2025-06-26")]
        )

    def test_purge_false_restores_the_plain_date_filter(self):
        mask = modeling_prep.fold_train_mask(self._frame(), 1, purge=False)
        self.assertEqual(mask.sum(), 6)  # 25-30 Jun

    def test_validation_month_rows_are_still_excluded_from_training(self):
        mask = modeling_prep.fold_train_mask(self._frame(), 1)
        kept = self._frame().loc[mask, "Tanggal"]
        self.assertTrue((kept < pd.Timestamp("2025-07-01")).all())

    def test_frame_without_lead_time_column_falls_back_to_the_date_filter(self):
        df = pd.DataFrame({"Tanggal": pd.date_range("2025-06-25", periods=10, freq="D")})
        self.assertEqual(modeling_prep.fold_train_mask(df, 1).sum(), 6)

    def test_invalid_fold_id_still_raises(self):
        with self.assertRaises(ValueError):
            modeling_prep.fold_train_mask(self._frame(), 9)


class TestCutoffSingleSource(unittest.TestCase):
    def test_test_start_is_the_same_object_as_build_panel_s(self):
        """Two independent literals drift: moving the cutoff for a refresh
        would silently split the panel and the model input onto two dates."""
        self.assertIs(modeling_prep.TEST_START, build_panel.TEST_START)


class TestEncodeCategoricals(unittest.TestCase):
    def _frame(self, kota, dates=None):
        dates = dates or ["2024-01-01"] * len(kota)
        return pd.DataFrame({"kota": kota, "Tanggal": pd.to_datetime(dates)})

    def test_unknown_token_is_index_zero(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        self.assertEqual(mapping["kota"][modeling_prep.UNKNOWN_TOKEN], 0)

    def test_known_values_get_stable_sorted_indices(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Depok", "Kota Bekasi"]), cols=["kota"]
        )
        self.assertEqual(mapping["kota"]["Kota Bekasi"], 1)
        self.assertEqual(mapping["kota"]["Kota Depok"], 2)

    def test_mapping_ignores_values_seen_only_after_the_cutoff(self):
        df = self._frame(
            ["Kota Bekasi", "Kota Baru"], ["2024-01-01", "2025-12-15"]
        )
        mapping = modeling_prep.build_category_mapping(
            df, cutoff=pd.Timestamp("2025-12-01"), cols=["kota"]
        )
        self.assertNotIn("Kota Baru", mapping["kota"])

    def test_unseen_value_encodes_to_unknown_index(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        result = modeling_prep.encode_categoricals(
            self._frame(["Kota Antah Berantah"]), mapping, cols=["kota"]
        )
        self.assertEqual(result.iloc[0]["kota_idx"], modeling_prep.UNKNOWN_INDEX)

    def test_adding_a_new_branch_does_not_shift_existing_indices(self):
        """A 60th branch opening must not renumber the other 59, or every
        previously trained model silently breaks."""
        before = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok"]), cols=["kota"]
        )
        frame = self._frame(["Kota Bekasi", "Kota Depok", "Kota Antah Berantah"])
        after = modeling_prep.encode_categoricals(frame, before, cols=["kota"])
        self.assertEqual(after.iloc[0]["kota_idx"], before["kota"]["Kota Bekasi"])
        self.assertEqual(after.iloc[1]["kota_idx"], before["kota"]["Kota Depok"])

    def test_encoded_column_is_integer_dtype(self):
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        result = modeling_prep.encode_categoricals(
            self._frame(["Kota Bekasi"]), mapping, cols=["kota"]
        )
        self.assertTrue(pd.api.types.is_integer_dtype(result["kota_idx"]))

    def test_extending_a_mapping_appends_new_values_after_the_existing_ones(self):
        """A monthly refresh moves the cutoff, so a SKU that was only in the
        test period enters the training period and joins the mapping. Sorting
        the whole set again would renumber every value that sorts after it."""
        existing = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok"]), cols=["kota"]
        )
        extended = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok", "Kota Ambon"]),
            cols=["kota"],
            existing=existing,
        )
        self.assertEqual(extended["kota"]["Kota Bekasi"], 1)
        self.assertEqual(extended["kota"]["Kota Depok"], 2)
        self.assertEqual(extended["kota"]["Kota Ambon"], 3)

    def test_extending_keeps_a_value_that_disappeared_from_the_data(self):
        # A discontinued SKU must not free up its index for a new one, or
        # every model trained on the old mapping reads the wrong embedding.
        existing = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok"]), cols=["kota"]
        )
        extended = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Ambon"]), cols=["kota"], existing=existing
        )
        self.assertEqual(extended["kota"]["Kota Depok"], 2)
        self.assertEqual(extended["kota"]["Kota Ambon"], 3)

    def test_extending_leaves_the_unknown_token_at_zero(self):
        existing = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        extended = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Ambon"]), cols=["kota"], existing=existing
        )
        self.assertEqual(extended["kota"][modeling_prep.UNKNOWN_TOKEN], 0)

    def test_extending_an_absent_column_falls_back_to_a_fresh_mapping(self):
        extended = modeling_prep.build_category_mapping(
            self._frame(["Kota Depok", "Kota Bekasi"]), cols=["kota"], existing={}
        )
        self.assertEqual(extended["kota"]["Kota Bekasi"], 1)
        self.assertEqual(extended["kota"]["Kota Depok"], 2)

    def test_existing_mapping_is_empty_when_no_file_has_been_saved_yet(self):
        import os, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "belum-ada.json")
            self.assertEqual(modeling_prep.load_existing_mapping(path), {})

    def test_existing_mapping_is_read_back_when_the_file_is_there(self):
        import os, tempfile
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi"]), cols=["kota"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mapping.json")
            modeling_prep.save_category_mapping(mapping, path)
            self.assertEqual(modeling_prep.load_existing_mapping(path), mapping)

    def test_mapping_survives_a_save_load_round_trip(self):
        import tempfile, os
        mapping = modeling_prep.build_category_mapping(
            self._frame(["Kota Bekasi", "Kota Depok"]), cols=["kota"]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mapping.json")
            modeling_prep.save_category_mapping(mapping, path)
            self.assertEqual(modeling_prep.load_category_mapping(path), mapping)


class TestImputeFeatures(unittest.TestCase):
    def _frame(self, **overrides):
        base = {col: [np.nan] for col in modeling_prep.EVENT_PROXIMITY_COLS}
        base["days_since_relocation"] = [np.nan]
        base["baseline_ratio"] = [np.nan]
        for col in modeling_prep.HISTORY_COLS:
            base[col] = [np.nan]
        base.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(base)

    def test_event_proximity_nulls_become_the_sentinel_not_zero(self):
        result = modeling_prep.impute_features(self._frame())
        for col in modeling_prep.EVENT_PROXIMITY_COLS:
            self.assertEqual(
                result.iloc[0][col], modeling_prep.EVENT_PROXIMITY_SENTINEL,
                f"{col} salah diimputasi",
            )
            self.assertNotEqual(result.iloc[0][col], 0.0)

    def test_sentinel_is_above_the_largest_real_value(self):
        """days_until_ramadan reaches 70 in the real data, so any sentinel at
        or below that would be indistinguishable from a genuine observation."""
        self.assertGreater(modeling_prep.EVENT_PROXIMITY_SENTINEL, 70)

    def test_there_are_ten_event_proximity_columns(self):
        self.assertEqual(len(modeling_prep.EVENT_PROXIMITY_COLS), 10)

    def test_real_event_proximity_values_are_left_alone(self):
        result = modeling_prep.impute_features(self._frame(days_until_ramadan=5.0))
        self.assertEqual(result.iloc[0]["days_until_ramadan"], 5.0)

    def test_missing_relocation_becomes_zero_with_a_false_indicator(self):
        result = modeling_prep.impute_features(self._frame())
        self.assertEqual(result.iloc[0]["days_since_relocation"], 0.0)
        self.assertFalse(bool(result.iloc[0]["was_relocated"]))

    def test_relocation_day_zero_is_distinguishable_from_never_relocated(self):
        """0 is a legitimate value meaning 'relocated today'; without the
        indicator it would be identical to 'never relocated'."""
        relocated = modeling_prep.impute_features(self._frame(days_since_relocation=0.0))
        never = modeling_prep.impute_features(self._frame())
        self.assertEqual(
            relocated.iloc[0]["days_since_relocation"],
            never.iloc[0]["days_since_relocation"],
        )
        self.assertTrue(bool(relocated.iloc[0]["was_relocated"]))
        self.assertFalse(bool(never.iloc[0]["was_relocated"]))

    def test_missing_baseline_ratio_becomes_one_with_a_false_indicator(self):
        result = modeling_prep.impute_features(self._frame())
        self.assertEqual(result.iloc[0]["baseline_ratio"], 1.0)
        self.assertFalse(bool(result.iloc[0]["has_baseline"]))

    def test_no_nulls_remain_in_the_imputed_columns(self):
        result = modeling_prep.impute_features(self._frame())
        targets = modeling_prep.EVENT_PROXIMITY_COLS + [
            "days_since_relocation", "baseline_ratio",
        ]
        self.assertFalse(result[targets].isna().any().any())


class TestImputeHistoryFeatures(unittest.TestCase):
    """Lag and rolling nulls must be filled too, or the LSTM cannot train.

    to_tabular() only ever exposes prediction rows, whose lags are complete.
    An LSTM window reaches 28 days further back and pulls in warm-up rows that
    still carry nulls, so the sequence tensor arrives with NaNs while the
    tabular matrix is clean -- the two models then see different information.
    """

    def _frame(self, **overrides):
        base = {col: [np.nan] for col in modeling_prep.EVENT_PROXIMITY_COLS}
        base["days_since_relocation"] = [np.nan]
        base["baseline_ratio"] = [np.nan]
        for col in modeling_prep.HISTORY_COLS:
            base[col] = [np.nan]
        base.update({k: [v] for k, v in overrides.items()})
        return pd.DataFrame(base)

    def test_history_cols_cover_every_lag_and_rolling_feature(self):
        self.assertEqual(
            sorted(modeling_prep.HISTORY_COLS),
            sorted([
                "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
                "roll_mean_7", "roll_std_7", "roll_mean_14", "roll_std_14",
                "roll_mean_28", "roll_std_28",
            ]),
        )

    def test_missing_lag_and_rolling_values_become_zero(self):
        result = modeling_prep.impute_features(self._frame())
        for col in modeling_prep.HISTORY_COLS:
            self.assertEqual(result.iloc[0][col], 0.0, f"{col} tidak diimputasi")

    def test_real_lag_values_are_left_alone(self):
        result = modeling_prep.impute_features(self._frame(lag_7=13.0))
        self.assertEqual(result.iloc[0]["lag_7"], 13.0)

    def test_all_history_present_sets_the_full_history_indicator(self):
        complete = {col: 1.0 for col in modeling_prep.HISTORY_COLS}
        result = modeling_prep.impute_features(self._frame(**complete))
        self.assertTrue(bool(result.iloc[0]["has_full_history"]))
        self.assertEqual(result.iloc[0]["missing_history_count"], 0)

    def test_partial_history_is_counted_not_just_flagged(self):
        """A row 14 days into its segment has more usable history than one 3
        days in; a bare boolean would collapse the two into the same value."""
        result = modeling_prep.impute_features(self._frame(lag_1=1.0, lag_2=1.0))
        self.assertFalse(bool(result.iloc[0]["has_full_history"]))
        self.assertEqual(
            result.iloc[0]["missing_history_count"], len(modeling_prep.HISTORY_COLS) - 2
        )

    def test_a_genuine_zero_lag_still_counts_as_present(self):
        """Zero demand yesterday is a real observation, not a missing one."""
        complete = {col: 0.0 for col in modeling_prep.HISTORY_COLS}
        result = modeling_prep.impute_features(self._frame(**complete))
        self.assertTrue(bool(result.iloc[0]["has_full_history"]))

    def test_no_history_nulls_remain(self):
        result = modeling_prep.impute_features(self._frame())
        self.assertFalse(result[modeling_prep.HISTORY_COLS].isna().any().any())


def _pair_frame(n_rows, item="I1", branch="B1", start="2024-01-01"):
    return pd.DataFrame({
        "Kode Barang": [item] * n_rows,
        "Nama Cabang": [branch] * n_rows,
        "Tanggal": pd.date_range(start, periods=n_rows, freq="D"),
        "feat_a": np.arange(n_rows, dtype=float),
        "target_lead_time_cumulative": np.arange(n_rows, dtype=float) * 2,
        "fold_id": [np.nan] * n_rows,
    })


class TestDropWarmupRows(unittest.TestCase):
    def test_drops_exactly_the_first_lookback_rows_of_each_pair(self):
        df = _pair_frame(40)
        result = modeling_prep.drop_warmup_rows(df, lookback=28)
        self.assertEqual(len(result), 12)

    def test_first_surviving_row_is_at_position_lookback(self):
        df = _pair_frame(40)
        result = modeling_prep.drop_warmup_rows(df, lookback=28)
        self.assertEqual(
            result.iloc[0]["Tanggal"], pd.Timestamp("2024-01-01") + pd.Timedelta(days=28)
        )

    def test_a_pair_shorter_than_the_lookback_disappears_entirely(self):
        result = modeling_prep.drop_warmup_rows(_pair_frame(10), lookback=28)
        self.assertEqual(len(result), 0)

    def test_each_pair_gets_its_own_warmup_cut(self):
        df = pd.concat([_pair_frame(40, item="I1"), _pair_frame(40, item="I2")],
                       ignore_index=True)
        result = modeling_prep.drop_warmup_rows(df, lookback=28)
        self.assertEqual(len(result), 24)
        self.assertEqual(result.groupby("Kode Barang").size().tolist(), [12, 12])


class TestToTabular(unittest.TestCase):
    def test_returns_the_expected_keys(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(set(out), {"X", "y", "keys", "fold_id"})

    def test_x_contains_only_the_requested_features(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(list(out["X"].columns), ["feat_a"])

    def test_all_parts_have_the_same_length(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(len(out["X"]), 12)
        self.assertEqual(len(out["y"]), 12)
        self.assertEqual(len(out["keys"]), 12)
        self.assertEqual(len(out["fold_id"]), 12)

    def test_keys_identify_pair_and_date(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(
            list(out["keys"].columns), ["Kode Barang", "Nama Cabang", "Tanggal"]
        )


class TestScaler(unittest.TestCase):
    def test_scaled_column_has_zero_mean_and_unit_std(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        scaler = modeling_prep.fit_scaler(df, ["a"])
        scaled = modeling_prep.apply_scaler(df, scaler, ["a"])
        self.assertAlmostEqual(scaled["a"].mean(), 0.0, places=6)
        self.assertAlmostEqual(scaled["a"].std(ddof=0), 1.0, places=6)

    def test_constant_column_does_not_divide_by_zero(self):
        df = pd.DataFrame({"a": [7.0, 7.0, 7.0]})
        scaler = modeling_prep.fit_scaler(df, ["a"])
        scaled = modeling_prep.apply_scaler(df, scaler, ["a"])
        self.assertTrue(np.isfinite(scaled["a"]).all())

    def test_scaler_fit_on_train_is_applied_unchanged_to_validation(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        valid = pd.DataFrame({"a": [10.0]})
        scaler = modeling_prep.fit_scaler(train, ["a"])
        scaled = modeling_prep.apply_scaler(valid, scaler, ["a"])
        mean, std = scaler["a"]
        self.assertAlmostEqual(scaled.iloc[0]["a"], (10.0 - mean) / std, places=6)

    def test_scaler_survives_a_save_load_round_trip(self):
        import tempfile, os
        scaler = modeling_prep.fit_scaler(pd.DataFrame({"a": [1.0, 2.0]}), ["a"])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scaler.json")
            modeling_prep.save_scaler(scaler, path)
            self.assertEqual(modeling_prep.load_scaler(path), scaler)


class TestToSequences(unittest.TestCase):
    def test_tensor_has_shape_samples_lookback_features(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].shape, (12, 28, 1))

    def test_window_ends_at_the_prediction_row_inclusive(self):
        """The prediction row's own features (lags, calendar) are known at
        prediction time, so the window must include them."""
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        # feat_a == row position, so the first sample predicts position 28 and
        # its window must be positions 1..28.
        self.assertEqual(out["X"][0, -1, 0], 28.0)
        self.assertEqual(out["X"][0, 0, 0], 1.0)

    def test_target_matches_the_prediction_row(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["y"][0], 56.0)  # position 28 * 2

    def test_keys_identify_the_prediction_row_not_the_window_start(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(
            out["keys"].iloc[0]["Tanggal"],
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=28),
        )

    def test_a_pair_shorter_than_the_lookback_produces_no_samples(self):
        out = modeling_prep.to_sequences(_pair_frame(10), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].shape[0], 0)

    def test_windows_never_span_two_pairs(self):
        df = pd.concat([_pair_frame(40, item="I1"), _pair_frame(40, item="I2")],
                       ignore_index=True)
        out = modeling_prep.to_sequences(df, feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].shape[0], 24)
        self.assertTrue((out["X"] <= 39.0).all())

    def test_tensor_is_float32(self):
        out = modeling_prep.to_sequences(_pair_frame(40), feature_cols=["feat_a"], lookback=28)
        self.assertEqual(out["X"].dtype, np.float32)


class TestLogTarget(unittest.TestCase):
    def test_log_target_transforms_y_in_both_adapters(self):
        df = _pair_frame(40)
        tab = modeling_prep.to_tabular(df, feature_cols=["feat_a"], log_target=True)
        seq = modeling_prep.to_sequences(df, feature_cols=["feat_a"], log_target=True)
        self.assertAlmostEqual(float(tab["y"].iloc[0]), float(np.log1p(56.0)), places=5)
        self.assertAlmostEqual(float(seq["y"][0]), float(np.log1p(56.0)), places=5)

    def test_log_target_defaults_to_off(self):
        out = modeling_prep.to_tabular(_pair_frame(40), feature_cols=["feat_a"])
        self.assertEqual(float(out["y"].iloc[0]), 56.0)

    def test_inverse_returns_the_original_scale(self):
        original = np.array([0.0, 5.0, 488.0, 3067.0])
        restored = modeling_prep.inverse_log_target(np.log1p(original))
        self.assertTrue(np.allclose(restored, original))

    def test_contract_still_holds_when_both_adapters_use_log(self):
        df = _pair_frame(40)
        tab = modeling_prep.to_tabular(df, feature_cols=["feat_a"], log_target=True)
        seq = modeling_prep.to_sequences(df, feature_cols=["feat_a"], log_target=True)
        self.assertIsNone(modeling_prep.validate_contract(tab, seq))


class TestValidateContract(unittest.TestCase):
    def _pair(self):
        df = _pair_frame(40)
        tabular = modeling_prep.to_tabular(df, feature_cols=["feat_a"])
        sequences = modeling_prep.to_sequences(df, feature_cols=["feat_a"])
        return tabular, sequences

    def test_matching_adapters_pass(self):
        tabular, sequences = self._pair()
        self.assertIsNone(modeling_prep.validate_contract(tabular, sequences))

    def test_rejects_differing_row_counts(self):
        tabular, sequences = self._pair()
        tabular["keys"] = tabular["keys"].iloc[:-1]
        with self.assertRaisesRegex(AssertionError, "jumlah baris"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_same_count_but_different_dates(self):
        """Equal lengths are not enough — the actual rows must match."""
        tabular, sequences = self._pair()
        tabular["keys"] = tabular["keys"].copy()
        tabular["keys"].loc[0, "Tanggal"] = pd.Timestamp("1999-01-01")
        with self.assertRaisesRegex(AssertionError, "baris berbeda"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_differing_targets(self):
        tabular, sequences = self._pair()
        tabular["y"] = tabular["y"].copy()
        tabular["y"].iloc[0] = -12345.0
        with self.assertRaisesRegex(AssertionError, "target"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_differing_fold_assignments(self):
        tabular, sequences = self._pair()
        tabular["fold_id"] = tabular["fold_id"].copy()
        tabular["fold_id"].iloc[0] = 3.0
        with self.assertRaisesRegex(AssertionError, "fold"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_nan_in_the_sequence_tensor(self):
        """The failure this contract exists to catch: an LSTM given NaN
        produces NaN loss, so it would be quietly patched at training time and
        stop seeing what the tree models see."""
        tabular, sequences = self._pair()
        sequences["X"] = sequences["X"].copy()
        sequences["X"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, "NaN"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_rejects_nan_in_the_tabular_matrix(self):
        tabular, sequences = self._pair()
        tabular["X"] = tabular["X"].copy()
        tabular["X"].iloc[0, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, "NaN"):
            modeling_prep.validate_contract(tabular, sequences)

    def test_require_finite_false_allows_nan_for_tree_only_runs(self):
        """XGBoost handles NaN natively; a run that is not comparing against
        the LSTM may legitimately want them left in."""
        tabular, sequences = self._pair()
        sequences["X"] = sequences["X"].copy()
        sequences["X"][0, 0, 0] = np.nan
        self.assertIsNone(
            modeling_prep.validate_contract(tabular, sequences, require_finite=False)
        )

    def test_clean_adapters_pass_the_finiteness_check(self):
        tabular, sequences = self._pair()
        self.assertIsNone(modeling_prep.validate_contract(tabular, sequences))


class TestSegmentAwareAdapters(unittest.TestCase):
    def _frame(self):
        dates = list(pd.date_range("2024-01-01", periods=6, freq="D"))
        dates += list(pd.date_range("2024-06-01", periods=6, freq="D"))
        return pd.DataFrame({
            "Kode Barang": ["A"] * 12, "Nama Cabang": ["X"] * 12,
            "Tanggal": dates, "segment_id": [1] * 6 + [2] * 6,
            "feat": list(range(12)), "target_lead_time_cumulative": list(range(12)),
            "fold_id": [float("nan")] * 12,
        })

    def test_warmup_is_cut_per_segment(self):
        result = modeling_prep.drop_warmup_rows(self._frame(), lookback=3)
        # 3 rows survive in each of the two segments, not 9 across one series.
        self.assertEqual(len(result), 6)
        self.assertEqual(sorted(result["segment_id"].unique()), [1, 2])
        self.assertEqual(result["Tanggal"].min(), pd.Timestamp("2024-01-04"))

    def test_frames_without_segment_id_still_group_by_pair(self):
        df = self._frame().drop(columns=["segment_id"])
        result = modeling_prep.drop_warmup_rows(df, lookback=3)
        self.assertEqual(len(result), 9)

    def test_sequences_never_bridge_two_segments(self):
        result = modeling_prep.to_sequences(
            self._frame(), feature_cols=["feat"], lookback=3
        )
        self.assertEqual(len(result["X"]), 6)
        for window in result["X"]:
            values = [int(v) for v in window[:, 0]]
            self.assertTrue(all(b - a == 1 for a, b in zip(values, values[1:])))

    def test_adapters_agree_on_segmented_input(self):
        df = self._frame()
        tabular = modeling_prep.to_tabular(df, feature_cols=["feat"], lookback=3)
        sequences = modeling_prep.to_sequences(df, feature_cols=["feat"], lookback=3)
        modeling_prep.validate_contract(tabular, sequences)
        self.assertIn("segment_id", tabular["keys"].columns)


if __name__ == "__main__":
    unittest.main()
