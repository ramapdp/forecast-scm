import unittest

import numpy as np
import torch

from utils import model_common, model_lstm, modeling_prep, sequence_windows, walk_forward


class TestPinballLoss(unittest.TestCase):
    def test_under_prediction_is_penalised_nine_times_harder(self):
        """alpha=0.9 means a shortfall costs 0.9 per unit and an overstock
        0.1 per unit — the asymmetry the whole project is built on.
        """
        target = torch.tensor([10.0])
        under = model_lstm.pinball_loss(torch.tensor([9.0]), target, 0.9)
        over = model_lstm.pinball_loss(torch.tensor([11.0]), target, 0.9)
        self.assertAlmostEqual(float(under), 0.9, places=5)
        self.assertAlmostEqual(float(over), 0.1, places=5)

    def test_a_perfect_prediction_costs_nothing(self):
        loss = model_lstm.pinball_loss(torch.tensor([5.0, 7.0]),
                                       torch.tensor([5.0, 7.0]), 0.9)
        self.assertAlmostEqual(float(loss), 0.0, places=6)


class TestEmbeddingSizes(unittest.TestCase):
    def test_sizes_come_from_the_mapping_not_from_observed_values(self):
        """num_embeddings must cover every index the saved mapping can emit,
        including UNKNOWN=0. A branch that opens after training would
        otherwise index out of bounds months later.
        """
        mapping = {
            "Kode Barang": {"<UNKNOWN>": 0, "A": 1, "B": 2},
            "Nama Cabang": {"<UNKNOWN>": 0, "KY001": 1},
        }
        sizes = model_lstm.embedding_sizes(
            mapping, idx_cols=["Kode Barang_idx", "Nama Cabang_idx"])
        self.assertEqual(sizes, [(3, 2), (2, 1)])

    def test_the_dimension_is_capped_at_sixteen(self):
        mapping = {"Kode Barang": {str(i): i for i in range(200)}}
        sizes = model_lstm.embedding_sizes(mapping, idx_cols=["Kode Barang_idx"])
        self.assertEqual(sizes, [(200, 16)])


class TestQuantileLSTM(unittest.TestCase):
    def _model(self, num_layers=2, dropout=0.2):
        return model_lstm.QuantileLSTM(
            n_dynamic=4, sizes=[(5, 3), (3, 2)],
            hidden_size=8, num_layers=num_layers, dropout=dropout)

    def test_forward_returns_one_value_per_row(self):
        model = self._model()
        x = torch.randn(6, 28, 4)
        c = torch.zeros(6, 2, dtype=torch.long)
        self.assertEqual(model(x, c).shape, (6,))

    def test_a_single_layer_model_still_applies_dropout_in_the_head(self):
        """nn.LSTM ignores dropout when num_layers=1, so the flag would be
        meaningless across half the search space if the head did not use it.
        """
        model = self._model(num_layers=1, dropout=0.5)
        self.assertEqual(model.lstm.dropout, 0.0)
        self.assertTrue(any(isinstance(layer, torch.nn.Dropout)
                            for layer in model.head))

    def test_the_highest_category_index_is_in_range(self):
        model = self._model()
        x = torch.randn(2, 28, 4)
        c = torch.tensor([[4, 2], [0, 0]], dtype=torch.long)
        self.assertEqual(model(x, c).shape, (2,))


class TestBuildModel(unittest.TestCase):
    def test_the_same_seed_produces_identical_initial_weights(self):
        params = {**model_lstm.DEFAULT_PARAMS, "hidden_size": 8, "num_layers": 1}
        a = model_lstm.build_model(params, n_dynamic=4, sizes=[(5, 3)], seed=42)
        b = model_lstm.build_model(params, n_dynamic=4, sizes=[(5, 3)], seed=42)
        for left, right in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(left, right)


class TestCandidateBudget(unittest.TestCase):
    def test_a_cheap_configuration_is_capped_at_twenty(self):
        self.assertEqual(
            model_lstm.candidate_budget(sec_per_epoch=1.0, best_epoch=2), 20)

    def test_the_formula_divides_the_budget_by_two_folds_of_two_fits(self):
        # per_fit = 60 * (2*10 + 5) = 1500s; two folds = 3000s; 28800/3000 = 9
        self.assertEqual(
            model_lstm.candidate_budget(sec_per_epoch=60.0, best_epoch=10), 9)

    def test_a_configuration_too_slow_for_six_candidates_raises(self):
        """Below six draws it is not a search. Clamping N up to six would
        silently overrun the 8-hour ceiling, so the operator is told to
        shrink the space instead.
        """
        with self.assertRaises(ValueError) as caught:
            model_lstm.candidate_budget(sec_per_epoch=600.0, best_epoch=20)
        self.assertIn("perkecil ruang search", str(caught.exception))


def _tiny_index(n=80, seed=3):
    """A one-pair panel small enough to train on in a test."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    feat = rng.normal(size=n).astype("float64")
    panel = pd.DataFrame({
        "Tanggal": pd.date_range("2025-01-01", periods=n, freq="D"),
        "Kode Barang": "FGS-00001",
        "Nama Cabang": "KY001",
        "segment_id": 1,
        "feat_a": feat,
        "feat_b": rng.normal(size=n),
        "cat_idx": rng.integers(0, 3, size=n),
        "target_lead_time_cumulative": np.abs(feat * 5 + 10),
    })
    index = sequence_windows.build_index(
        panel, feature_cols=["feat_a", "feat_b", "cat_idx"], lookback=7)
    return panel, index


class TestScaleValues(unittest.TestCase):
    def test_each_column_is_standardised_by_its_own_statistics(self):
        _, index = _tiny_index()
        scaler = {"feat_a": (1.0, 2.0), "feat_b": (0.0, 1.0)}
        scaled = model_lstm.scale_values(index["values"], scaler,
                                         index["dynamic_cols"])
        np.testing.assert_allclose(scaled[:, 0],
                                   (index["values"][:, 0] - 1.0) / 2.0,
                                   rtol=1e-6)
        np.testing.assert_allclose(scaled[:, 1], index["values"][:, 1],
                                   rtol=1e-6)
        self.assertEqual(scaled.dtype, np.dtype("float32"))


class TestTrainingLoop(unittest.TestCase):
    def _setup(self):
        panel, index = _tiny_index()
        ends = np.arange(7, len(panel))
        targets = panel["target_lead_time_cumulative"].to_numpy("float32")[ends]
        params = {**model_lstm.DEFAULT_PARAMS, "hidden_size": 8,
                  "num_layers": 1, "batch_size": 16}
        model = model_lstm.build_model(params, n_dynamic=2, sizes=[(3, 2)], seed=42)
        return index, ends, targets, params, model

    def test_one_epoch_returns_a_finite_mean_loss(self):
        index, ends, targets, params, model = self._setup()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss = model_lstm.run_epoch(
            model, optimizer, index["values"], index["cats"], ends, targets,
            params, quantile=0.9,
            generator=torch.Generator().manual_seed(42),
            device=torch.device("cpu"), lookback=7)
        self.assertTrue(np.isfinite(loss))

    def test_a_non_finite_loss_raises_rather_than_poisoning_the_search(self):
        index, ends, targets, params, model = self._setup()
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        poisoned = targets.copy()
        poisoned[0] = np.inf
        with self.assertRaises(ValueError) as caught:
            model_lstm.run_epoch(
                model, optimizer, index["values"], index["cats"], ends, poisoned,
                params, quantile=0.9,
                generator=torch.Generator().manual_seed(42),
                device=torch.device("cpu"), lookback=7)
        self.assertIn("NaN", str(caught.exception))

    def test_predict_returns_one_value_per_end(self):
        index, ends, _, params, model = self._setup()
        prediction = model_lstm.predict(
            model, index["values"], index["cats"], ends,
            device=torch.device("cpu"), lookback=7, batch_size=16)
        self.assertEqual(prediction.shape, (len(ends),))

    def test_early_stopping_reports_an_epoch_within_the_cap(self):
        index, ends, targets, params, model = self._setup()
        fit_ends, es_ends = ends[:50], ends[50:]
        fitted, best_epoch = model_lstm.fit_with_early_stopping(
            params, index, fit_ends, targets[:50], es_ends, targets[50:],
            quantile=0.9, sizes=[(3, 2)], device=torch.device("cpu"),
            max_epochs=6, patience=2, lookback=7)
        self.assertGreaterEqual(best_epoch, 1)
        self.assertLessEqual(best_epoch, 6)
        self.assertIsInstance(fitted, model_lstm.QuantileLSTM)

    def test_fit_epochs_runs_exactly_the_requested_number_of_epochs(self):
        index, ends, targets, params, model = self._setup()
        fitted = model_lstm.fit_epochs(
            params, index, ends, targets, epochs=3, quantile=0.9,
            sizes=[(3, 2)], device=torch.device("cpu"), lookback=7)
        self.assertEqual(fitted.epochs_run, 3)

    def test_the_same_seed_produces_identical_predictions(self):
        index, ends, targets, params, _ = self._setup()
        outputs = []
        for _ in range(2):
            fitted = model_lstm.fit_epochs(
                params, index, ends, targets, epochs=2, quantile=0.9,
                sizes=[(3, 2)], device=torch.device("cpu"), lookback=7)
            outputs.append(model_lstm.predict(
                fitted, index["values"], index["cats"], ends,
                device=torch.device("cpu"), lookback=7))
        np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5)

import pandas as pd


def _fold_panel(n_days=200, n_pairs=2, seed=7):
    """A panel long enough to reach fold 5 and survive a 7-day warm-up."""
    rng = np.random.default_rng(seed)
    parts = []
    for pair in range(n_pairs):
        feat = rng.normal(size=n_days)
        parts.append(pd.DataFrame({
            "Tanggal": pd.date_range("2025-05-01", periods=n_days, freq="D"),
            "Kode Barang": f"FGS-0000{pair}",
            "Nama Cabang": "KY001",
            "segment_id": 1,
            "feat_a": feat,
            "feat_b": rng.normal(size=n_days),
            "cat_idx": rng.integers(0, 3, size=n_days),
            "lead_time_days": 3.0,
            "demand_segment": "smooth",
            "is_delivery_day": True,
            "target_lead_time_cumulative": np.abs(feat * 5 + 10),
        }))
    panel = pd.concat(parts, ignore_index=True)
    return modeling_prep.assign_folds(panel)


FOLD_FEATURES = ["feat_a", "feat_b", "cat_idx"]
SMALL = {**model_lstm.DEFAULT_PARAMS, "hidden_size": 8, "num_layers": 1,
         "batch_size": 64}


class TestFitPredict(unittest.TestCase):
    def _index_and_split(self):
        panel = _fold_panel()
        index = sequence_windows.build_index(panel, feature_cols=FOLD_FEATURES,
                                             lookback=7)
        frame = walk_forward.eligible_rows(panel, lookback=7)
        split = walk_forward.prepare_fold(frame, 5, prepared=True)
        return panel, index, split["train"], split["valid"]

    def _fit_predict(self, index, **kwargs):
        # sizes is passed explicitly: the fixture's `cat_idx` has no entry in
        # the real category_mapping.json that embedding_sizes() would read.
        return model_lstm.make_fit_predict(
            SMALL, index=index, quantile=0.9, tail_days=30,
            sizes=[(3, 2)],
            max_epochs=kwargs.pop("max_epochs", 3),
            patience=kwargs.pop("patience", 2), **kwargs)

    def test_it_returns_one_prediction_per_validation_row(self):
        _, index, train, valid = self._index_and_split()
        prediction = self._fit_predict(index)(train, valid)
        self.assertEqual(prediction.shape, (len(valid),))

    def test_predictions_are_never_negative(self):
        _, index, train, valid = self._index_and_split()
        prediction = self._fit_predict(index)(train, valid)
        self.assertTrue((prediction >= 0).all())

    def test_no_training_row_is_dated_inside_the_validation_month(self):
        """G3. This checks the position mapping, not fold_train_mask —
        window_ends() returning wrong positions is the new failure mode.
        """
        _, index, train, valid = self._index_and_split()
        train_ends = sequence_windows.window_ends(index, train)
        self.assertLess(index["dates"][train_ends].max(),
                        np.datetime64(valid["Tanggal"].min(), "D"))

    def test_the_early_stopping_tail_is_absent_from_the_first_fit(self):
        """G4."""
        _, index, train, _ = self._index_and_split()
        fit_rows, es_rows = model_common.split_early_stopping(train, tail_days=30)
        fit_ends = set(sequence_windows.window_ends(index, fit_rows).tolist())
        es_ends = set(sequence_windows.window_ends(index, es_rows).tolist())
        self.assertEqual(fit_ends & es_ends, set())

    def test_no_window_reaches_into_december(self):
        """G5."""
        _, index, train, valid = self._index_and_split()
        ends = np.concatenate([
            sequence_windows.window_ends(index, train),
            sequence_windows.window_ends(index, valid),
        ])
        self.assertLess(index["dates"][ends].max(),
                        np.datetime64(modeling_prep.TEST_START, "D"))

    def test_the_best_epoch_of_each_fold_is_recorded(self):
        _, index, train, valid = self._index_and_split()
        fit_predict = self._fit_predict(index)
        fit_predict(train, valid)
        self.assertEqual(len(fit_predict.best_epochs), 1)
        self.assertGreaterEqual(fit_predict.best_epochs[0], 1)

    def test_log_target_round_trips_to_the_original_scale(self):
        _, index, train, valid = self._index_and_split()
        logged = model_lstm.make_fit_predict(
            {**SMALL, "log_target": True}, index=index, quantile=0.9,
            sizes=[(3, 2)], max_epochs=2, patience=2)(train, valid)
        self.assertTrue(np.isfinite(logged).all())
        self.assertTrue((logged >= 0).all())


class TestBindPanel(unittest.TestCase):
    def test_it_matches_run_searchs_expected_signature(self):
        panel = _fold_panel()
        make = model_lstm.bind_panel(panel, feature_cols=FOLD_FEATURES,
                                     lookback=7, sizes=[(3, 2)])
        fit_predict = make(SMALL, feature_cols=FOLD_FEATURES, quantile=0.9)
        self.assertTrue(callable(fit_predict))

    def test_the_index_is_built_once_and_reused(self):
        panel = _fold_panel()
        make = model_lstm.bind_panel(panel, feature_cols=FOLD_FEATURES,
                                     lookback=7, sizes=[(3, 2)])
        first = make(SMALL, quantile=0.9)
        second = make(SMALL, quantile=0.9)
        self.assertIs(first.index, second.index)

    def test_a_different_feature_list_raises(self):
        panel = _fold_panel()
        make = model_lstm.bind_panel(panel, feature_cols=FOLD_FEATURES,
                                     lookback=7, sizes=[(3, 2)])
        with self.assertRaises(ValueError) as caught:
            make(SMALL, feature_cols=["feat_a"], quantile=0.9)
        self.assertIn("berbeda dari yang dipakai membangun indeks",
                      str(caught.exception))

class TestFitFinalAndBundle(unittest.TestCase):
    def _bundle(self):
        panel = _fold_panel()
        bundle = model_lstm.fit_final(
            panel, SMALL, feature_cols=FOLD_FEATURES, lookback=7,
            sizes=[(3, 2)], max_epochs=3, patience=2)
        return panel, bundle

    def test_the_bundle_records_everything_needed_to_reload(self):
        _, bundle = self._bundle()
        for key in ("state_dict", "params", "feature_cols", "dynamic_cols",
                    "idx_cols", "embedding_sizes", "scaler", "log_target",
                    "best_epoch", "quantile", "n_train", "lookback"):
            self.assertIn(key, bundle)
        self.assertEqual(bundle["quantile"], 0.9)
        self.assertEqual(bundle["feature_cols"], FOLD_FEATURES)

    def test_no_training_row_reaches_december(self):
        """fit_final takes its rows from walk_forward.eligible_rows(), which
        cuts December before anything else — so this checks the cut survived
        the extra purge and the window mapping.
        """
        panel, bundle = self._bundle()
        self.assertGreater(bundle["n_train"], 0)
        eligible = walk_forward.eligible_rows(panel, lookback=7)
        self.assertLess(eligible["Tanggal"].max(), modeling_prep.TEST_START)
        self.assertLessEqual(bundle["n_train"], len(eligible))

    def test_predict_bundle_returns_non_negative_values_per_row(self):
        panel, bundle = self._bundle()
        frame = walk_forward.eligible_rows(panel, lookback=7).head(20)
        prediction = model_lstm.predict_bundle(bundle, panel, frame)
        self.assertEqual(prediction.shape, (20,))
        self.assertTrue((prediction >= 0).all())

    def test_a_column_shuffled_frame_produces_identical_predictions(self):
        """The bundle forces the recorded column order. A model reloaded
        against a different layout does not fail — it predicts confidently
        from the wrong features, which is worse.
        """
        panel, bundle = self._bundle()
        frame = walk_forward.eligible_rows(panel, lookback=7).head(20)
        shuffled_panel = panel[list(reversed(panel.columns))]
        straight = model_lstm.predict_bundle(bundle, panel, frame)
        shuffled = model_lstm.predict_bundle(bundle, shuffled_panel, frame)
        np.testing.assert_allclose(straight, shuffled, rtol=1e-5)


class TestSearchWrappers(unittest.TestCase):
    def test_the_same_seed_reproduces_the_identical_candidate_list(self):
        first = model_lstm.sample_search_space(5, seed=42)
        second = model_lstm.sample_search_space(5, seed=42)
        self.assertEqual(first, second)

    def test_every_candidate_carries_the_defaults_it_did_not_draw(self):
        for candidate in model_lstm.sample_search_space(5, seed=42):
            self.assertEqual(candidate["random_state"], 42)
            self.assertEqual(candidate["grad_clip"], 1.0)
            self.assertIn(candidate["hidden_size"], [64, 128, 256])

    def test_the_search_folds_match_the_other_two_models(self):
        self.assertEqual(model_lstm.SEARCH_FOLDS, (3, 5))

if __name__ == "__main__":
    unittest.main()
