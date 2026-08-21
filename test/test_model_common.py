import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils import model_common, modeling_prep


FEATURES = ["feat_a", "feat_b", "cat_idx"]

SPACE = {"alpha": [1, 2, 3], "beta": ["x", "y"]}
DEFAULTS = {"alpha": 1, "beta": "x", "pinned": 99}


def _panel(periods=245):
    """One pair's daily series, long enough that the 28-day warm-up cut leaves rows."""
    rows = []
    for i, date in enumerate(pd.date_range("2025-05-01", periods=periods, freq="D")):
        rows.append({
            "Kode Barang": "I1", "Nama Cabang": "B1", "segment_id": 1,
            "Tanggal": date,
            "target_lead_time_cumulative": float(i % 7),
            "lead_time_days": 3.0, "lag_1": float(i % 5),
            "roll_mean_7": float(i % 4), "demand_segment": "smooth",
            "is_delivery_day": bool(i % 2),
            "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
        })
    return modeling_prep.assign_folds(pd.DataFrame(rows))


def _mean_fit_predict(params, feature_cols=None, quantile=0.9):
    """A stand-in model: no library, no fitting, one number per validation row."""
    def fit_predict(train, valid):
        return np.full(len(valid), float(params["alpha"]))
    return fit_predict


class TestSampleSearchSpace(unittest.TestCase):
    def test_returns_the_requested_number_of_candidates(self):
        self.assertEqual(len(model_common.sample_search_space(SPACE, DEFAULTS, 4)), 4)

    def test_the_same_seed_reproduces_the_same_list(self):
        first = model_common.sample_search_space(SPACE, DEFAULTS, 4, seed=7)
        second = model_common.sample_search_space(SPACE, DEFAULTS, 4, seed=7)
        self.assertEqual(first, second)

    def test_candidates_are_distinct(self):
        drawn = model_common.sample_search_space(SPACE, DEFAULTS, 6)
        signatures = {(c["alpha"], c["beta"]) for c in drawn}
        self.assertEqual(len(signatures), 6)

    def test_defaults_fill_the_unsearched_keys(self):
        for candidate in model_common.sample_search_space(SPACE, DEFAULTS, 3):
            self.assertEqual(candidate["pinned"], 99)

    def test_no_screen_rejects_nothing(self):
        self.assertEqual(
            len(model_common.sample_search_space(SPACE, DEFAULTS, 6, screen=None)), 6
        )

    def test_an_injected_screen_rejects_exactly_what_it_says(self):
        drawn = model_common.sample_search_space(
            SPACE, DEFAULTS, 2, screen=lambda params: params["alpha"] == 1
        )
        self.assertEqual({c["alpha"] for c in drawn}, {1})

    def test_a_screen_that_admits_nothing_raises_naming_the_screen(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            model_common.sample_search_space(
                SPACE, DEFAULTS, 4,
                screen=lambda params: False, screen_label="budget 3.0 GB",
            )


class TestRunSearch(unittest.TestCase):
    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2)]

    def test_scores_a_space_it_has_never_seen(self):
        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        self.assertEqual(list(results["candidate_id"]), [0, 1])
        self.assertTrue(results["pinball"].notna().all())

    def test_records_the_searched_keys_of_that_space(self):
        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        for key in SPACE:
            self.assertIn(key, results.columns)

    def test_a_failing_candidate_is_recorded_not_raised(self):
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise ValueError("meledak")
            return fit_predict

        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=exploding,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(results["pinball"].isna().all())
        self.assertTrue(results["error"].str.contains("meledak").all())

    def test_an_uncaught_exception_type_propagates(self):
        """The catch list is deliberately narrow: a bug must not become a NaN row."""
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise KeyError("bug")
            return fit_predict

        with self.assertRaises(KeyError):
            model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=exploding,
                search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
                feature_cols=FEATURES, verbose=False,
            )

    def test_a_widened_catch_list_records_the_new_type(self):
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise KeyError("bug")
            return fit_predict

        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=exploding,
            search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False, catch=(KeyError,),
        )
        self.assertTrue(results["pinball"].isna().all())


class TestRunSearchCheckpoint(unittest.TestCase):
    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2, 3)]

    def _run(self, path, candidates=None, resume=True):
        return model_common.run_search(
            _panel(), candidates or self._candidates(),
            make_fit_predict=_mean_fit_predict, search_space=SPACE,
            folds=(1,), alpha=0.9, model_name="toy", feature_cols=FEATURES,
            verbose=False, checkpoint_path=path, resume=resume,
        )

    def test_writes_a_row_per_candidate_as_it_finishes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path)
            self.assertEqual(len(pd.read_csv(path)), 3)

    def test_a_finished_candidate_is_not_recomputed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path)
            calls = []

            def counting(params, feature_cols=None, quantile=0.9):
                calls.append(params["alpha"])
                return _mean_fit_predict(params)

            model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=counting,
                search_space=SPACE, folds=(1,), alpha=0.9, model_name="toy",
                feature_cols=FEATURES, verbose=False, checkpoint_path=path,
            )
            self.assertEqual(calls, [])

    def test_results_come_back_in_candidate_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            results = self._run(path)
            self.assertEqual(list(results["candidate_id"]), [0, 1, 2])

    def test_a_checkpoint_from_a_different_space_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(path)
            other = [{**DEFAULTS, "alpha": a} for a in (7, 8, 9)]
            with self.assertRaisesRegex(ValueError, "tidak cocok"):
                self._run(path, candidates=other)


class TestBundleIO(unittest.TestCase):
    def test_a_bundle_round_trips(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "nested" / "bundle.joblib")
            model_common.save_bundle({"columns": ["a", "b"]}, path)
            self.assertEqual(model_common.load_bundle(path)["columns"], ["a", "b"])

    def test_best_params_are_written_sorted_and_readable(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "params.json")
            model_common.save_best_params({"b": 2, "a": 1}, path)
            written = Path(path).read_text(encoding="utf-8")
            self.assertLess(written.index('"a"'), written.index('"b"'))
            self.assertEqual(json.loads(written), {"a": 1, "b": 2})


class TestSplitEarlyStopping(unittest.TestCase):
    """The mechanism moved here from model_xgboost because it is not
    XGBoost-specific: any model that must choose its own capacity without
    reading the validation fold needs a purged training tail.
    """

    def _dated_frame(self, n=200, lead_time=3.0):
        rng = np.random.default_rng(11)
        return pd.DataFrame({
            "Tanggal": pd.date_range("2025-01-01", periods=n, freq="D"),
            "feat_a": rng.normal(size=n),
            "target_lead_time_cumulative": np.abs(rng.normal(size=n)) * 10,
            "lead_time_days": lead_time,
            "Kode Barang": "FGS-00001",
            "Nama Cabang": "KY001",
            "segment_id": 1,
        })

    def test_the_tail_is_the_last_thirty_days(self):
        train = self._dated_frame()
        _, es_rows = model_common.split_early_stopping(train, tail_days=30)
        self.assertEqual(len(es_rows), 30)
        self.assertEqual(es_rows["Tanggal"].max(), train["Tanggal"].max())

    def test_no_es_date_precedes_a_fit_date(self):
        fit_rows, es_rows = model_common.split_early_stopping(self._dated_frame())
        self.assertLess(fit_rows["Tanggal"].max(), es_rows["Tanggal"].min())

    def test_fit_rows_whose_label_window_crosses_the_tail_are_purged(self):
        # lead_time_days=3 means the label at H sums H+1..H+3, so the last
        # three days before the tail carry a label built inside the tail.
        train = self._dated_frame(lead_time=3.0)
        fit_rows, es_rows = model_common.split_early_stopping(train, tail_days=30)
        es_start = es_rows["Tanggal"].min()
        self.assertLessEqual(fit_rows["Tanggal"].max(),
                             es_start - pd.Timedelta(days=4))

    def test_an_empty_training_frame_raises(self):
        with self.assertRaises(ValueError):
            model_common.split_early_stopping(self._dated_frame().iloc[0:0])

    def test_a_window_too_short_for_the_tail_raises(self):
        with self.assertRaises(ValueError):
            model_common.split_early_stopping(self._dated_frame(n=20), tail_days=30)

if __name__ == "__main__":
    unittest.main()


class TestRunSearchCost(unittest.TestCase):
    """What a candidate cost is as much a result as what it scored.

    The LSTM search overran its eight-hour ceiling by roughly 2x because
    `best_epoch` varies across the space while the budget formula treats it as
    a constant measured once. Neither the epoch count nor the wall time was
    recorded, so the overrun could only be reconstructed afterwards from the
    modification timestamps of the checkpoint file.
    """

    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2)]

    def _reporting(self, attribute, per_fold=(4, 6)):
        """A model that reports its own chosen capacity the way the real ones
        do: one value appended to a list on the callable, per fold."""
        def make(params, feature_cols=None, quantile=0.9):
            inner = _mean_fit_predict(params)

            def fit_predict(train, valid):
                getattr(fit_predict, attribute).append(
                    per_fold[len(getattr(fit_predict, attribute))])
                return inner(train, valid)

            setattr(fit_predict, attribute, [])
            return fit_predict
        return make

    def _run(self, make, folds=(1, 2), candidates=None, **kwargs):
        return model_common.run_search(
            _panel(), candidates or self._candidates(), make_fit_predict=make,
            search_space=SPACE, folds=folds, alpha=0.9, model_name="toy",
            feature_cols=FEATURES, verbose=False, **kwargs,
        )

    def test_the_epochs_an_lstm_chose_land_in_best_epoch_one_per_fold(self):
        results = self._run(self._reporting("best_epochs"))
        self.assertEqual(list(results["best_epoch"]), ["4,6", "4,6"])

    def test_the_rounds_an_xgboost_chose_land_there_too(self):
        """Same column, same meaning — the capacity early stopping picked."""
        results = self._run(self._reporting("best_iterations", per_fold=(30, 45)))
        self.assertEqual(list(results["best_epoch"]), ["30,45"] * 2)

    def test_a_model_that_reports_no_capacity_leaves_the_column_empty(self):
        results = self._run(_mean_fit_predict)
        self.assertTrue(results["best_epoch"].isna().all())

    def test_every_candidate_records_its_wall_time(self):
        results = self._run(_mean_fit_predict)
        self.assertTrue(results["elapsed_seconds"].notna().all())
        self.assertTrue((results["elapsed_seconds"] >= 0).all())

    def test_a_failed_candidate_still_records_what_it_burned(self):
        """A candidate that dies after forty minutes is the most expensive row
        in the table, and the one a budget post-mortem most needs."""
        def exploding(params, feature_cols=None, quantile=0.9):
            def fit_predict(train, valid):
                raise ValueError("meledak")
            return fit_predict

        results = self._run(exploding)
        self.assertTrue(results["pinball"].isna().all())
        self.assertTrue(results["elapsed_seconds"].notna().all())
        self.assertTrue(results["best_epoch"].isna().all())

    def test_a_checkpoint_written_before_these_columns_existed_still_resumes(self):
        """Three searches already have checkpoints on disk in the old shape,
        one of them mid-run. Resuming must not demand columns they lack.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "checkpoint.csv")
            self._run(_mean_fit_predict, checkpoint_path=path)
            old = pd.read_csv(path).drop(columns=["best_epoch", "elapsed_seconds"])
            old.to_csv(path, index=False)

            three = [{**DEFAULTS, "alpha": a} for a in (1, 2, 3)]
            results = self._run(self._reporting("best_epochs"), candidates=three,
                                checkpoint_path=path)

            self.assertEqual(list(results["candidate_id"]), [0, 1, 2])
            self.assertTrue(results.loc[:1, "elapsed_seconds"].isna().all())
            self.assertEqual(results.loc[2, "best_epoch"], "4,6")
