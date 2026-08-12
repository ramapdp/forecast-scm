import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep


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


if __name__ == "__main__":
    unittest.main()
