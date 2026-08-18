import unittest

import numpy as np
import pandas as pd

from utils import modeling_prep, walk_forward


def _panel(n_days=245, pairs=(("I1", "B1"), ("I2", "B1")), start="2025-05-01"):
    """Two pairs spanning 2025-05-01..2025-12-31, so every fold and the locked
    December window are represented.
    """
    rows = []
    for item, branch in pairs:
        for i, date in enumerate(pd.date_range(start, periods=n_days, freq="D")):
            rows.append({
                "Kode Barang": item,
                "Nama Cabang": branch,
                "segment_id": 1,
                "Tanggal": date,
                "target_lead_time_cumulative": float(i % 7),
                "lead_time_days": 3.0,
                "lag_1": float(i % 5),
                "roll_mean_7": float(i % 4),
                "demand_segment": "smooth",
                "is_delivery_day": bool(i % 2),
                "feat_a": float(i),
                "feat_b": float(i % 3),
            })
    return modeling_prep.assign_folds(pd.DataFrame(rows))


FEATURES = ["feat_a", "feat_b"]


class TestEligibleRows(unittest.TestCase):
    def test_drops_the_first_28_days_of_each_pair(self):
        result = walk_forward.eligible_rows(_panel())
        first = result[result["Kode Barang"] == "I1"]["Tanggal"].min()
        self.assertEqual(first, pd.Timestamp("2025-05-29"))

    def test_drops_every_december_row(self):
        result = walk_forward.eligible_rows(_panel())
        self.assertEqual(len(result[result["Tanggal"] >= modeling_prep.TEST_START]), 0)

    def test_drops_rows_with_a_null_target(self):
        panel = _panel()
        panel.loc[panel["Tanggal"] == pd.Timestamp("2025-07-15"), "target_lead_time_cumulative"] = np.nan
        result = walk_forward.eligible_rows(panel)
        self.assertEqual(len(result[result["Tanggal"] == pd.Timestamp("2025-07-15")]), 0)

    def test_keeps_every_original_column(self):
        panel = _panel()
        result = walk_forward.eligible_rows(panel)
        self.assertEqual(set(panel.columns), set(result.columns))

    def test_matches_to_tabular_row_for_row(self):
        """The contract: the tabular adapter and the runner must agree on the
        row set exactly, or a cross-model comparison compares different data.
        """
        panel = _panel()
        pre_december = panel[panel["Tanggal"] < modeling_prep.TEST_START]
        expected = modeling_prep.to_tabular(pre_december, FEATURES)["keys"]
        result = walk_forward.eligible_rows(panel)
        key_cols = ["Kode Barang", "Nama Cabang", "segment_id", "Tanggal"]
        self.assertEqual(
            set(map(tuple, expected[key_cols].to_numpy())),
            set(map(tuple, result[key_cols].to_numpy())),
        )

    def test_does_not_mutate_the_input_frame(self):
        panel = _panel()
        before = len(panel)
        walk_forward.eligible_rows(panel)
        self.assertEqual(len(panel), before)


class TestPrepareFold(unittest.TestCase):
    def test_validation_is_exactly_the_folds_month(self):
        prepared = walk_forward.prepare_fold(_panel(), 1)
        valid = prepared["valid"]
        self.assertEqual(valid["Tanggal"].min(), pd.Timestamp("2025-07-01"))
        self.assertEqual(valid["Tanggal"].max(), pd.Timestamp("2025-07-31"))

    def test_no_training_row_reaches_the_fold_boundary(self):
        for fold_id in walk_forward.FOLDS:
            prepared = walk_forward.prepare_fold(_panel(), fold_id)
            boundary = modeling_prep.FOLD_STARTS[fold_id - 1]
            self.assertLess(prepared["train"]["Tanggal"].max(), boundary)

    def test_purging_removes_rows_whose_target_window_crosses_the_boundary(self):
        """lead_time_days is 3, so 2025-06-30 sums demand through 2025-07-03 —
        three days of the validation month.
        """
        prepared = walk_forward.prepare_fold(_panel(), 1)
        last = prepared["train"]["Tanggal"].max()
        self.assertLessEqual(last, pd.Timestamp("2025-06-27"))

    def test_no_fold_ever_sees_december(self):
        for fold_id in walk_forward.FOLDS:
            prepared = walk_forward.prepare_fold(_panel(), fold_id)
            for part in ("train", "valid"):
                late = prepared[part]["Tanggal"] >= modeling_prep.TEST_START
                self.assertEqual(int(late.sum()), 0, f"fold {fold_id} {part}")

    def test_training_window_expands_with_the_fold_number(self):
        sizes = [len(walk_forward.prepare_fold(_panel(), f)["train"]) for f in walk_forward.FOLDS]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_train_and_valid_never_share_a_row(self):
        prepared = walk_forward.prepare_fold(_panel(), 3)
        overlap = set(prepared["train"].index) & set(prepared["valid"].index)
        self.assertEqual(overlap, set())

    def test_prepared_flag_skips_the_second_cut(self):
        panel = _panel()
        once = walk_forward.eligible_rows(panel)
        from_raw = walk_forward.prepare_fold(panel, 2)["valid"]
        from_prepared = walk_forward.prepare_fold(once, 2, prepared=True)["valid"]
        pd.testing.assert_frame_equal(from_raw, from_prepared)

    def test_rejects_an_out_of_range_fold(self):
        with self.assertRaises(ValueError):
            walk_forward.prepare_fold(_panel(), 9)


if __name__ == "__main__":
    unittest.main()
