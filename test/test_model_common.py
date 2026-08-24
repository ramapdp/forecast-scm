import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.modelling import model_common, modeling_prep


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
            "target_lead_time_cumulative_capped": float(i % 7),
            "lead_time_days": 3.0, "lag_1": float(i % 5),
            "roll_mean_7": float(i % 4), "demand_segment": "smooth",
            "is_delivery_day": bool(i % 2),
            "feat_a": float(i), "feat_b": float(i % 3), "cat_idx": i % 3,
        })
    return modeling_prep.assign_folds(pd.DataFrame(rows))


QUANTILES = (0.1, 0.5, 0.9)


def _mean_fit_predict(params, feature_cols=None, quantiles=QUANTILES):
    """A stand-in model: no library, no fitting, one row per validation row and
    one column per quantile — the shape `walk_forward` now requires."""
    def fit_predict(train, valid):
        flat = np.full(len(valid), float(params["alpha"]))
        return np.repeat(flat[:, None], len(quantiles), axis=1)
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
            search_space=SPACE, folds=(1,), quantiles=QUANTILES, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        self.assertEqual(list(results["candidate_id"]), [0, 1])
        self.assertTrue(results["pinball"].notna().all())

    def test_records_the_searched_keys_of_that_space(self):
        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        for key in SPACE:
            self.assertIn(key, results.columns)

    def test_a_failing_candidate_is_recorded_not_raised(self):
        def exploding(params, feature_cols=None, quantiles=QUANTILES):
            def fit_predict(train, valid):
                raise ValueError("meledak")
            return fit_predict

        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=exploding,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(results["pinball"].isna().all())
        self.assertTrue(results["error"].str.contains("meledak").all())

    def test_an_uncaught_exception_type_propagates(self):
        """The catch list is deliberately narrow: a bug must not become a NaN row."""
        def exploding(params, feature_cols=None, quantiles=QUANTILES):
            def fit_predict(train, valid):
                raise KeyError("bug")
            return fit_predict

        with self.assertRaises(KeyError):
            model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=exploding,
                search_space=SPACE, folds=(1,), quantiles=QUANTILES, model_name="toy",
                feature_cols=FEATURES, verbose=False,
            )

    def test_a_widened_catch_list_records_the_new_type(self):
        def exploding(params, feature_cols=None, quantiles=QUANTILES):
            def fit_predict(train, valid):
                raise KeyError("bug")
            return fit_predict

        results = model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=exploding,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES, model_name="toy",
            feature_cols=FEATURES, verbose=False, catch=(KeyError,),
        )
        self.assertTrue(results["pinball"].isna().all())


class TestRunSearchMetrics(unittest.TestCase):
    def _results(self):
        return model_common.run_search(
            _panel(), [{**DEFAULTS, "alpha": a} for a in (1, 2)],
            make_fit_predict=_mean_fit_predict, search_space=SPACE,
            folds=(1,), quantiles=QUANTILES, model_name="toy",
            feature_cols=FEATURES, verbose=False,
        )

    def test_pinball_is_k1_not_a_single_point(self):
        """The selection column has to be the criterion the methodology
        defines, or select_best() picks on something else than K1."""
        from utils.modelling import walk_forward
        frame = walk_forward.eligible_rows(_panel())
        fit_predict = _mean_fit_predict({**DEFAULTS, "alpha": 1})
        scored = walk_forward.run_fold(frame, 1, fit_predict, model_name="toy",
                                       quantiles=QUANTILES, prepared=True)
        expected = walk_forward.pooled_k1(scored, "toy")
        self.assertAlmostEqual(float(self._results().iloc[0]["pinball"]), expected)

    def test_the_headline_quantile_is_recorded_beside_its_metrics(self):
        """`mae` alone would be ambiguous once there are nineteen of them."""
        results = self._results()
        for column in ["mae_headline", "coverage_headline", "fill_rate_headline",
                       "headline_quantile"]:
            self.assertIn(column, results.columns)
        self.assertTrue((results["headline_quantile"] == 0.9).all())

    def test_calibration_and_crossing_are_recorded_for_k2(self):
        results = self._results()
        self.assertIn("coverage_gap", results.columns)
        self.assertIn("crossing_rate", results.columns)
        self.assertTrue(results["coverage_gap"].notna().all())

    def test_a_one_dimensional_prediction_fails_the_candidate(self):
        """Recorded, not raised: a model wired to the old scalar contract
        should cost one candidate, not a multi-hour run."""
        def flat(params, feature_cols=None, quantiles=QUANTILES):
            return lambda train, valid: np.full(len(valid), 1.0)

        results = model_common.run_search(
            _panel(), [{**DEFAULTS, "alpha": 1}], make_fit_predict=flat,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES,
            model_name="toy", feature_cols=FEATURES, verbose=False,
        )
        self.assertTrue(results["pinball"].isna().all())
        self.assertIn("bentuk", results.iloc[0]["error"])


class TestStaleSchemaCheckpoint(unittest.TestCase):
    """The three search CSVs on disk were written by the single-quantile runs
    against the same space and the same seed. The parameter guard accepts them;
    only the schema guard stops a resume from reporting pinball@0.9 as K1."""

    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2)]

    def _old_checkpoint(self, path):
        pd.DataFrame([
            {"candidate_id": 0, "alpha": 1, "beta": "x", "pinball": 0.5,
             "mae": 1.0, "coverage": 0.9, "fill_rate": 0.9,
             "best_epoch": None, "elapsed_seconds": 1.0, "error": None},
        ]).to_csv(path, index=False)

    def test_a_single_quantile_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "old.csv")
            self._old_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "kuantil tunggal"):
                model_common.run_search(
                    _panel(), self._candidates(),
                    make_fit_predict=_mean_fit_predict, search_space=SPACE,
                    folds=(1,), quantiles=QUANTILES, model_name="toy",
                    feature_cols=FEATURES, verbose=False, checkpoint_path=path,
                )

    def test_resume_false_ignores_it_and_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "old.csv")
            self._old_checkpoint(path)
            results = model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
                search_space=SPACE, folds=(1,), quantiles=QUANTILES,
                model_name="toy", feature_cols=FEATURES, verbose=False,
                checkpoint_path=path, resume=False,
            )
            self.assertIn("headline_quantile", results.columns)
            self.assertEqual(len(results), 2)


class TestRunSearchCheckpoint(unittest.TestCase):
    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2, 3)]

    def _run(self, path, candidates=None, resume=True):
        return model_common.run_search(
            _panel(), candidates or self._candidates(),
            make_fit_predict=_mean_fit_predict, search_space=SPACE,
            folds=(1,), quantiles=QUANTILES, model_name="toy", feature_cols=FEATURES,
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

            def counting(params, feature_cols=None, quantiles=QUANTILES):
                calls.append(params["alpha"])
                return _mean_fit_predict(params)

            model_common.run_search(
                _panel(), self._candidates(), make_fit_predict=counting,
                search_space=SPACE, folds=(1,), quantiles=QUANTILES, model_name="toy",
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
            "target_lead_time_cumulative_capped": np.abs(rng.normal(size=n)) * 10,
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
        def make(params, feature_cols=None, quantiles=QUANTILES):
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
            search_space=SPACE, folds=folds, quantiles=QUANTILES, model_name="toy",
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
        def exploding(params, feature_cols=None, quantiles=QUANTILES):
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


class TestTrainTarget(unittest.TestCase):
    """Satu seam untuk label latih, dipakai ketiga model.

    Sebelumnya tiap model membaca kolom targetnya sendiri-sendiri, jadi
    "latih di capped" harus benar di tiga tempat terpisah dan tidak ada yang
    memeriksa ketiganya sepakat. Helper ini yang diperiksa, sekali.
    """

    def _frame(self):
        return pd.DataFrame({
            "target_lead_time_cumulative": [10.0, 20.0, 30.0],
            "target_lead_time_cumulative_capped": [10.0, 20.0, 15.0],
        })

    def test_reads_the_capped_target(self):
        values = model_common.train_target(self._frame())
        np.testing.assert_allclose(values, [10.0, 20.0, 15.0])

    def test_applies_log1p_when_asked(self):
        values = model_common.train_target(self._frame(), log_target=True)
        np.testing.assert_allclose(values, np.log1p([10.0, 20.0, 15.0]))

    def test_refuses_a_frame_without_the_capped_target(self):
        frame = self._frame().drop(columns=["target_lead_time_cumulative_capped"])
        with self.assertRaises(KeyError) as ctx:
            model_common.train_target(frame)
        self.assertIn("target_lead_time_cumulative_capped", str(ctx.exception))


class TestRunSearchOnly(unittest.TestCase):
    """Sharding: satu mesin menjalankan sebagian candidate_id tanpa menggeser
    penomorannya, supaya dua shard bisa disatukan lewat id-nya nanti."""

    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2, 3)]

    def _run(self, only, **kwargs):
        return model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES,
            model_name="toy", feature_cols=FEATURES, verbose=False,
            only=only, **kwargs)

    def test_only_runs_the_named_candidates(self):
        self.assertEqual(list(self._run(only=[0, 2])["candidate_id"]), [0, 2])

    def test_none_runs_every_candidate(self):
        self.assertEqual(list(self._run(only=None)["candidate_id"]), [0, 1, 2])

    def test_ids_keep_their_absolute_position(self):
        """Memotong daftar kandidat di sisi pemanggil akan menomori ulang;
        `only` tidak boleh, atau dua shard tidak akan bisa disatukan."""
        shard = self._run(only=[2])
        whole = self._run(only=None)
        expected = whole[whole["candidate_id"] == 2].iloc[0]
        self.assertEqual(int(shard.iloc[0]["candidate_id"]), 2)
        self.assertEqual(shard.iloc[0]["alpha"], expected["alpha"])
        self.assertAlmostEqual(float(shard.iloc[0]["pinball"]),
                               float(expected["pinball"]))

    def test_an_id_out_of_range_raises(self):
        """Salah tulis batas shard adalah kesalahan yang paling mungkin terjadi
        dan paling mahal: ia baru ketahuan saat merge, berjam-jam kemudian."""
        with self.assertRaisesRegex(ValueError, "di luar 3 kandidat"):
            self._run(only=[1, 3])

    def test_a_negative_id_raises(self):
        with self.assertRaisesRegex(ValueError, "di luar 3 kandidat"):
            self._run(only=[-1])

    def test_an_empty_selection_raises(self):
        """Shard kosong selesai dalam sedetik dan menulis CSV kosong — dari luar
        ia tampak persis seperti shard yang berhasil."""
        with self.assertRaisesRegex(ValueError, "kosong"):
            self._run(only=[])

    def test_only_skips_candidates_already_in_the_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "shard.csv")
            first = self._run(only=[0], checkpoint_path=path)
            second = self._run(only=[0, 1], checkpoint_path=path)
            self.assertEqual(list(second["candidate_id"]), [0, 1])
            self.assertEqual(float(second.iloc[0]["elapsed_seconds"]),
                             float(first.iloc[0]["elapsed_seconds"]))

    def test_a_candidate_outside_only_is_never_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "shard.csv")
            self._run(only=[0, 1], checkpoint_path=path)
            written = pd.read_csv(path)
            self.assertEqual(list(written["candidate_id"]), [0, 1])


class TestRunSearchProvenance(unittest.TestCase):
    """Sebuah baris shard adalah bukti. Angka yang tidak dapat ditelusuri ke
    mesin dan versi kode yang melahirkannya tidak reprodusibel — dan mesin yang
    menjalankan shard ini sifatnya sementara."""

    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2)]

    def _run(self, provenance, only=None, **kwargs):
        return model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES,
            model_name="toy", feature_cols=FEATURES, verbose=False,
            provenance=provenance, only=only, **kwargs)

    def test_every_row_carries_the_provenance_columns(self):
        results = self._run({"device": "cuda:0", "commit": "abc1234"})
        self.assertEqual(list(results["device"]), ["cuda:0", "cuda:0"])
        self.assertEqual(list(results["commit"]), ["abc1234", "abc1234"])

    def test_none_adds_no_columns(self):
        results = self._run(None)
        self.assertNotIn("device", results.columns)

    def test_resumed_rows_keep_the_machine_that_produced_them(self):
        """Sebuah shard yang dilanjutkan di mesin lain harus menunjukkan kedua
        mesin itu, bukan menimpa yang lama dengan yang sekarang."""
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "shard.csv")
            self._run({"device": "cpu"}, only=[0], checkpoint_path=path)
            results = self._run({"device": "cuda:0"}, only=[0, 1],
                                checkpoint_path=path)
            by_id = results.set_index("candidate_id")["device"]
            self.assertEqual(by_id.loc[0], "cpu")
            self.assertEqual(by_id.loc[1], "cuda:0")

    def test_a_key_that_collides_with_a_searched_parameter_raises(self):
        with self.assertRaisesRegex(ValueError, "bertabrakan"):
            self._run({"alpha": 5})

    def test_a_key_that_collides_with_candidate_id_raises(self):
        with self.assertRaisesRegex(ValueError, "bertabrakan"):
            self._run({"candidate_id": 5})


class TestCurrentCommit(unittest.TestCase):
    def test_returns_a_short_hash_inside_this_repository(self):
        commit = model_common.current_commit()
        self.assertRegex(commit, r"^[0-9a-f]{7,40}$")

    def test_returns_the_default_outside_a_repository(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(
                model_common.current_commit(default="tidak-diketahui",
                                            cwd=folder),
                "tidak-diketahui")


class TestMergeShards(unittest.TestCase):
    """Penggabungan harus membuktikan dirinya sendiri: sebuah shard yang
    tertukar antar mesin atau lahir dari ruang pencarian lain tidak boleh
    lolos jadi baris yang tampak wajar di CSV gabungan."""

    def _candidates(self):
        return [{**DEFAULTS, "alpha": a} for a in (1, 2, 3)]

    def _shard(self, folder, name, only):
        path = str(Path(folder) / name)
        model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES,
            model_name="toy", feature_cols=FEATURES, verbose=False,
            only=only, checkpoint_path=path)
        return path

    def _whole(self):
        return model_common.run_search(
            _panel(), self._candidates(), make_fit_predict=_mean_fit_predict,
            search_space=SPACE, folds=(1,), quantiles=QUANTILES,
            model_name="toy", feature_cols=FEATURES, verbose=False)

    def test_two_shards_reproduce_the_whole_run(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [self._shard(folder, "a.csv", [0]),
                     self._shard(folder, "b.csv", [1, 2])]
            merged = model_common.merge_shards(paths, self._candidates(), SPACE)
            whole = self._whole()
            self.assertEqual(list(merged["candidate_id"]),
                             list(whole["candidate_id"]))
            self.assertEqual(list(merged["alpha"]), list(whole["alpha"]))
            for left, right in zip(merged["pinball"], whole["pinball"]):
                self.assertAlmostEqual(float(left), float(right))

    def test_the_merged_frame_is_ordered_by_candidate_id(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [self._shard(folder, "b.csv", [2]),
                     self._shard(folder, "a.csv", [0, 1])]
            merged = model_common.merge_shards(paths, self._candidates(), SPACE)
            self.assertEqual(list(merged["candidate_id"]), [0, 1, 2])

    def test_a_duplicate_candidate_id_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [self._shard(folder, "a.csv", [0, 1]),
                     self._shard(folder, "b.csv", [1, 2])]
            with self.assertRaisesRegex(ValueError, "ganda"):
                model_common.merge_shards(paths, self._candidates(), SPACE)

    def test_a_hole_in_the_coverage_is_refused(self):
        """Satu shard yang gagal diam-diam adalah pencarian yang menyusut."""
        with tempfile.TemporaryDirectory() as folder:
            paths = [self._shard(folder, "a.csv", [0, 2])]
            with self.assertRaisesRegex(ValueError, r"tidak menutup.*\[1\]"):
                model_common.merge_shards(paths, self._candidates(), SPACE)

    def test_a_shard_from_a_different_space_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [self._shard(folder, "a.csv", [0]),
                     self._shard(folder, "b.csv", [1, 2])]
            tampered = pd.read_csv(paths[0])
            tampered.loc[0, "alpha"] = 99
            tampered.to_csv(paths[0], index=False)
            with self.assertRaisesRegex(ValueError, "ruang pencarian"):
                model_common.merge_shards(paths, self._candidates(), SPACE)

    def test_a_single_quantile_shard_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "old.csv")
            pd.DataFrame([{"candidate_id": 0, "alpha": 1, "beta": "x",
                           "pinball": 0.5}]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "kuantil tunggal"):
                model_common.merge_shards([path], [self._candidates()[0]], SPACE)

    def test_no_shards_at_all_is_refused(self):
        with self.assertRaisesRegex(ValueError, "tidak ada shard"):
            model_common.merge_shards([], self._candidates(), SPACE)

    def test_provenance_columns_survive_the_merge(self):
        with tempfile.TemporaryDirectory() as folder:
            path_a = str(Path(folder) / "a.csv")
            path_b = str(Path(folder) / "b.csv")
            for path, only, device in ((path_a, [0], "cuda:0"),
                                       (path_b, [1, 2], "cpu")):
                model_common.run_search(
                    _panel(), self._candidates(),
                    make_fit_predict=_mean_fit_predict, search_space=SPACE,
                    folds=(1,), quantiles=QUANTILES, model_name="toy",
                    feature_cols=FEATURES, verbose=False, only=only,
                    provenance={"device": device}, checkpoint_path=path)
            merged = model_common.merge_shards([path_a, path_b],
                                               self._candidates(), SPACE)
            self.assertEqual(list(merged["device"]), ["cuda:0", "cpu", "cpu"])
