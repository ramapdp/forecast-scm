"""Konfigurasi run dari environment, dibaca sekali di sel pertama notebook.

Aturan yang mengikat seluruh berkas ini: **tanpa satu pun env var, tiap fungsi
mengembalikan persis apa yang notebook lakukan sebelum modul ini ada.** Sebuah
run lokal tidak boleh berubah perilakunya hanya karena jalur cloud ditambahkan.
"""

import unittest

from utils.modelling import modeling_prep, run_config


class TestShard(unittest.TestCase):
    def test_unset_means_every_candidate(self):
        self.assertIsNone(run_config.shard(env={}))

    def test_a_range_is_inclusive_at_both_ends(self):
        """Batas shard ditulis manusia di dua mesin berbeda. Kalau satu sisi
        membacanya eksklusif, kandidat di sambungannya hilang atau dinilai dua
        kali — dan itu baru ketahuan saat merge."""
        self.assertEqual(run_config.shard(env={"FORECAST_SHARD": "0-4"}),
                         [0, 1, 2, 3, 4])

    def test_a_comma_list_is_taken_literally(self):
        self.assertEqual(run_config.shard(env={"FORECAST_SHARD": "0,3,7"}),
                         [0, 3, 7])

    def test_ranges_and_singles_can_be_mixed(self):
        self.assertEqual(run_config.shard(env={"FORECAST_SHARD": "0-2,9"}),
                         [0, 1, 2, 9])

    def test_whitespace_is_tolerated(self):
        self.assertEqual(run_config.shard(env={"FORECAST_SHARD": " 1 - 3 , 8 "}),
                         [1, 2, 3, 8])

    def test_the_result_is_sorted_and_deduplicated(self):
        self.assertEqual(run_config.shard(env={"FORECAST_SHARD": "5,1,5"}),
                         [1, 5])

    def test_an_empty_value_raises_rather_than_meaning_unset(self):
        """`export FORECAST_SHARD=` adalah salah ketik, dan cabang diamnya
        mahal: kedua mesin menjalankan ke-30 kandidat."""
        with self.assertRaisesRegex(ValueError, "kosong"):
            run_config.shard(env={"FORECAST_SHARD": "  "})

    def test_a_descending_range_raises(self):
        with self.assertRaisesRegex(ValueError, "menurun"):
            run_config.shard(env={"FORECAST_SHARD": "14-0"})

    def test_a_non_numeric_value_raises(self):
        with self.assertRaisesRegex(ValueError, "tidak dapat dibaca"):
            run_config.shard(env={"FORECAST_SHARD": "0-x"})


class TestShardLabel(unittest.TestCase):
    def test_unset_is_labelled_full(self):
        self.assertEqual(run_config.shard_label(env={}), "full")

    def test_a_range_keeps_its_shape(self):
        self.assertEqual(run_config.shard_label(env={"FORECAST_SHARD": "0-14"}),
                         "0-14")

    def test_commas_become_underscores_so_it_can_be_a_filename(self):
        self.assertEqual(run_config.shard_label(env={"FORECAST_SHARD": "0,3,7"}),
                         "0_3_7")


class TestDevice(unittest.TestCase):
    def test_unset_returns_the_default_it_was_given(self):
        self.assertEqual(run_config.device("cpu", env={}), "cpu")

    def test_the_environment_overrides_the_default(self):
        self.assertEqual(run_config.device("cpu", env={"FORECAST_DEVICE": "cuda:1"}),
                         "cuda:1")

    def test_an_empty_value_raises(self):
        with self.assertRaisesRegex(ValueError, "kosong"):
            run_config.device("cpu", env={"FORECAST_DEVICE": " "})


class TestModelInputPath(unittest.TestCase):
    def test_unset_is_the_repository_copy(self):
        self.assertEqual(run_config.model_input_path(env={}),
                         modeling_prep.MODEL_INPUT_FILE)

    def test_the_environment_overrides_it(self):
        self.assertEqual(
            run_config.model_input_path(env={"FORECAST_MODEL_INPUT": "/kaggle/input/x.parquet"}),
            "/kaggle/input/x.parquet")


class TestCheckpointPath(unittest.TestCase):
    def test_unset_writes_beside_the_other_model_artefacts(self):
        self.assertEqual(run_config.checkpoint_path("a.csv", env={}),
                         str(modeling_prep.MODEL_READY_DIR + "/a.csv"))

    def test_the_environment_moves_the_whole_folder(self):
        self.assertEqual(
            run_config.checkpoint_path("a.csv", env={"FORECAST_CHECKPOINT_DIR": "/kaggle/working"}),
            "/kaggle/working/a.csv")


class TestSearchCheckpoint(unittest.TestCase):
    def test_an_unsharded_run_keeps_todays_filename(self):
        """Run lokal tanpa env var harus menulis ke berkas yang sama persis
        seperti sebelum modul ini ada — termasuk namanya."""
        self.assertEqual(run_config.search_checkpoint("xgb", env={}),
                         str(modeling_prep.MODEL_READY_DIR + "/xgb_search_results.csv"))

    def test_a_shard_is_named_after_itself(self):
        """Dua shard yang dikumpulkan di satu folder tidak boleh saling
        menimpa."""
        self.assertEqual(
            run_config.search_checkpoint("xgb", env={"FORECAST_SHARD": "0-14"}),
            str(modeling_prep.MODEL_READY_DIR + "/xgb_search_results.shard-0-14.csv"))

    def test_the_checkpoint_folder_still_applies(self):
        self.assertEqual(
            run_config.search_checkpoint("lstm", env={"FORECAST_SHARD": "15-29",
                                                      "FORECAST_CHECKPOINT_DIR": "/kaggle/working"}),
            "/kaggle/working/lstm_search_results.shard-15-29.csv")


class TestProvenance(unittest.TestCase):
    def test_it_records_the_device_and_a_commit(self):
        record = run_config.provenance("cuda:0", env={})
        self.assertEqual(record["device"], "cuda:0")
        self.assertRegex(record["commit"], r"^[0-9a-f]{7,40}$")

    def test_the_keys_are_exactly_what_run_search_will_write(self):
        self.assertEqual(set(run_config.provenance("cpu", env={})), {"device", "commit"})


class TestDescribe(unittest.TestCase):
    def test_it_names_every_setting_that_is_in_force(self):
        line = run_config.describe("cuda:1", env={"FORECAST_SHARD": "0-14",
                                                  "FORECAST_CHECKPOINT_DIR": "/kaggle/working"})
        self.assertIn("cuda:1", line)
        self.assertIn("0-14", line)
        self.assertIn("/kaggle/working", line)

    def test_a_plain_local_run_says_so(self):
        self.assertIn("seluruh kandidat", run_config.describe("cpu", env={}))
