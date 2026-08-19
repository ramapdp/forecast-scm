import unittest

import numpy as np
import torch

from utils import model_lstm


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


if __name__ == "__main__":
    unittest.main()
