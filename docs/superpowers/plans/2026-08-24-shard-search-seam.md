# Seam `only=` dan Penggabungan Shard — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Memungkinkan satu pencarian hyperparameter dijalankan terpecah di beberapa mesin dan disatukan kembali tanpa kehilangan identitas kandidat, sesuai Bagian 5 `docs/superpowers/specs/2026-08-24-distributed-gpu-training-design.md`.

**Architecture:** Empat tambahan kecil di `utils/modelling/model_common.py` — parameter `only=` yang menjalankan subset `candidate_id` **tanpa menggeser penomorannya**, parameter `provenance=` yang menuliskan mesin dan versi kode ke tiap baris, helper `current_commit()`, dan `merge_shards()` yang menyatukan CSV shard sambil **memakai ulang** guard `_assert_checkpoint_matches()` yang sudah ada. Ketiga pembungkus `run_search` per model meneruskan dua parameter baru itu supaya tanda tangannya tetap seragam.

**Tech Stack:** Python 3.9.6, pandas 2.3.3, unittest (bukan pytest), `.venv` di root repo.

## Global Constraints

- **Python 3.9.6.** Anotasi tipe memakai `Optional[...]` / `Iterable[...]` dari `typing`. Sintaks `X | Y` **tidak** tersedia dan akan gagal saat impor.
- **Semua perintah dijalankan dari root repo** (`/Users/ramapdp/Project/Personal/forecast-scm`), memakai `.venv/bin/python3`. Paket-paket `utils` adalah namespace package tanpa `__init__.py` di root; menjalankan tes dari direktori lain akan gagal impor.
- **Test runner adalah `unittest`, bukan pytest**: `.venv/bin/python3 -m unittest test.test_model_common -v`.
- **Pesan error ditulis dalam Bahasa Indonesia**, docstring boleh Inggris — ikuti pola yang sudah ada di `model_common.py`.
- **`_assert_checkpoint_matches()` dipakai ulang, tidak ditulis ulang.** Ia sudah mencocokkan nilai parameter tiap baris dengan `candidate_id` yang diklaimnya, dan sudah menolak file dari run kuantil tunggal lewat `CHECKPOINT_SCHEMA_COL`. Penggabungan shard yang menulis pengecekan tandingannya sendiri akan punya dua definisi "cocok" yang bisa berbeda diam-diam.
- **Tidak ada perubahan metodologi.** Set kuantil, target, purging, split, dan anggaran kandidat tidak disentuh sama sekali oleh rencana ini.
- **Baseline tes: 752 lolos** (per `docs/todolist-proyek.md`). Setiap task menambah tes, tidak boleh ada yang merah.
- Kerjakan di branch baru: `git checkout -b feat/shard-search-seam` (dari `main`).

---

## File Structure

| Berkas | Tanggung jawab | Perubahan |
|---|---|---|
| `utils/modelling/model_common.py` | Protokol pencarian, checkpoint, format bundle | **Modify** — tambah `_selected()`, `only=`, `provenance=`, `current_commit()`, `merge_shards()` |
| `utils/modelling/model_xgboost.py:439` | Pembungkus `run_search` XGBoost | **Modify** — teruskan `only`/`provenance` |
| `utils/modelling/model_lstm.py:749` | Pembungkus `run_search` LSTM | **Modify** — teruskan `only`/`provenance` |
| `utils/modelling/model_random_forest.py:195` | Pembungkus `run_search` RF | **Modify** — teruskan `only`/`provenance` |
| `test/test_model_common.py` | Tes seam | **Modify** — tiga kelas tes baru |
| `test/test_model_{xgboost,lstm,random_forest}.py` | Tes penerusan | **Modify** — satu tes per berkas |
| `docs/todolist-proyek.md` | Status proyek | **Modify** — butir 0c menunjuk mekanisme baru |

---

### Task 1: Parameter `only=` — menjalankan subset tanpa menggeser `candidate_id`

**Files:**
- Modify: `utils/modelling/model_common.py` (impor `typing`, helper baru sebelum `run_search`, dan badan `run_search` di sekitar baris 236–320)
- Test: `test/test_model_common.py` (kelas baru, letakkan sesudah `class TestRunSearchCheckpoint`)

**Interfaces:**
- Consumes: `model_common.run_search(...)` dengan tanda tangannya yang sekarang; `sample_search_space(space, defaults, n, seed=42)` yang deterministik.
- Produces: `run_search(..., only: Optional[Iterable[int]] = None)` dan `_selected(candidates: list, only: Optional[Iterable[int]]) -> set`. Task 3 mengandalkan bahwa `candidate_id` di hasil shard adalah posisi absolut di `candidates`.

- [ ] **Step 1: Tulis tes yang gagal**

Tambahkan di akhir `test/test_model_common.py`:

```python
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
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv/bin/python3 -m unittest test.test_model_common.TestRunSearchOnly -v`
Expected: FAIL — `TypeError: run_search() got an unexpected keyword argument 'only'`

- [ ] **Step 3: Tulis implementasi minimal**

Di `utils/modelling/model_common.py`, ubah baris impor `typing`:

```python
from typing import Callable, Iterable, Optional
```

Tambahkan helper tepat sebelum `def run_search(`:

```python
def _selected(candidates: list, only: Optional[Iterable[int]]) -> set:
    """Id kandidat yang menjadi tanggung jawab proses ini.

    Alasan keberadaannya adalah pemecahan pekerjaan antar mesin. `run_search`
    menomori kandidat lewat posisinya di `candidates`, jadi memotong daftar itu
    di sisi pemanggil akan menomori ulang dan dua shard tidak akan pernah bisa
    disatukan lewat id. `only` menjaga penomoran tetap absolut terhadap
    `sample_search_space(seed=...)`, yang deterministik di mesin mana pun.

    Seleksi di luar jangkauan atau kosong dilempar, bukan dijalankan sebagai
    no-op: shard kosong selesai dalam sedetik dan menulis CSV kosong, yang dari
    luar tidak dapat dibedakan dari shard yang berhasil. Lubangnya baru muncul
    saat penggabungan, berjam-jam sesudahnya.
    """
    if only is None:
        return set(range(len(candidates)))
    selected = {int(value) for value in only}
    out_of_range = sorted(value for value in selected
                          if value < 0 or value >= len(candidates))
    if out_of_range:
        raise ValueError(
            f"only memuat candidate_id di luar {len(candidates)} kandidat "
            f"saat ini: {out_of_range}"
        )
    if not selected:
        raise ValueError("only kosong — tidak ada kandidat untuk dijalankan")
    return selected
```

Tambahkan parameter di tanda tangan `run_search`, tepat sesudah `resume: bool = True,`:

```python
    only: Optional[Iterable[int]] = None,
```

Di badan `run_search`, ganti baris `frame = walk_forward.eligible_rows(df)` menjadi:

```python
    selected = _selected(candidates, only)
    frame = walk_forward.eligible_rows(df)
```

dan ganti awal perulangan:

```python
    for candidate_id, candidate in enumerate(candidates):
        if candidate_id in completed:
            continue
```

menjadi:

```python
    for candidate_id, candidate in enumerate(candidates):
        if candidate_id not in selected or candidate_id in completed:
            continue
```

Tambahkan dua kalimat di docstring `run_search`, sesudah paragraf `resume`:

```
    `only` memecah satu pencarian ke beberapa mesin: tiap mesin menjalankan
    subset candidate_id-nya sendiri sementara penomorannya tetap absolut,
    sehingga hasilnya dapat disatukan oleh `merge_shards()`.
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv/bin/python3 -m unittest test.test_model_common -v`
Expected: OK — seluruh kelas di berkas itu, termasuk `TestRunSearchOnly` (8 tes baru), lolos.

- [ ] **Step 5: Commit**

```bash
git add utils/modelling/model_common.py test/test_model_common.py
git commit -m "feat: jalankan subset candidate_id lewat only= tanpa menggeser penomorannya"
```

---

### Task 2: `provenance=` dan `current_commit()` — tiap baris shard tahu asalnya

**Files:**
- Modify: `utils/modelling/model_common.py` (impor `subprocess`, helper `current_commit()` di dekat `_ordered()`, dan `record` di dalam `run_search`)
- Test: `test/test_model_common.py` (kelas baru sesudah `TestRunSearchOnly`)

**Interfaces:**
- Consumes: `run_search(..., only=...)` dari Task 1.
- Produces: `run_search(..., provenance: Optional[dict] = None)` yang menyalin tiap pasangan kunci–nilai `provenance` ke setiap baris hasil **yang dijalankan proses ini**; `current_commit(default: str = "unknown", cwd: Optional[str] = None) -> str`. Task 3 mengandalkan bahwa kolom tambahan ini tidak mengganggu penggabungan.

- [ ] **Step 1: Tulis tes yang gagal**

Tambahkan di akhir `test/test_model_common.py`:

```python
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
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv/bin/python3 -m unittest test.test_model_common.TestRunSearchProvenance test.test_model_common.TestCurrentCommit -v`
Expected: FAIL — `TypeError: run_search() got an unexpected keyword argument 'provenance'` dan `AttributeError: module ... has no attribute 'current_commit'`

- [ ] **Step 3: Tulis implementasi minimal**

Tambahkan `subprocess` ke blok impor `model_common.py` (urutan alfabetis: sesudah `random`):

```python
import subprocess
```

Tambahkan helper tepat sesudah `def _ordered(rows: list) -> pd.DataFrame:` beserta badannya:

```python
def current_commit(default: str = "unknown",
                   cwd: Optional[str] = None) -> str:
    """Hash git pendek dari pohon kerja yang menjalankan proses ini.

    Ditulis ke tiap baris shard karena sebuah baris shard adalah bukti: angka
    yang tidak dapat ditelusuri ke versi kode yang melahirkannya tidak
    reprodusibel, dan mesin cloud yang menjalankan shard ini sifatnya
    sementara — ia tidak akan ada lagi saat angkanya dibaca.

    Mengembalikan `default` alih-alih melempar ketika git tidak tersedia sama
    sekali: kegagalan mencatat asal-usul tidak boleh menggagalkan pencarian
    delapan jam yang selain itu baik-baik saja. Yang hilang tercatat sebagai
    nilai `default` yang kasat mata, bukan sebagai sel kosong.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
            cwd=cwd or str(Path(__file__).resolve().parents[2]),
        )
    except (OSError, subprocess.SubprocessError):
        return default
    return completed.stdout.strip() or default
```

Tambahkan parameter di tanda tangan `run_search`, tepat sesudah `only: Optional[Iterable[int]] = None,`:

```python
    provenance: Optional[dict] = None,
```

Di badan `run_search`, tepat sesudah baris `selected = _selected(candidates, only)`, sisipkan:

```python
    provenance = provenance or {}
    reserved = {"candidate_id", *search_space}
    collisions = sorted(reserved & set(provenance))
    if collisions:
        raise ValueError(
            f"kunci provenance bertabrakan dengan kolom pencarian: "
            f"{collisions}"
        )
```

Ganti pembentukan `record`:

```python
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(search_space)}}
```

menjadi:

```python
        record = {"candidate_id": candidate_id,
                  **{key: candidate[key] for key in sorted(search_space)},
                  **provenance}
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv/bin/python3 -m unittest test.test_model_common -v`
Expected: OK — termasuk 7 tes baru (`TestRunSearchProvenance` 5, `TestCurrentCommit` 2).

- [ ] **Step 5: Commit**

```bash
git add utils/modelling/model_common.py test/test_model_common.py
git commit -m "feat: catat mesin dan versi kode di tiap baris shard lewat provenance="
```

---

### Task 3: `merge_shards()` — penggabungan yang diverifikasi, bukan dipercaya

**Files:**
- Modify: `utils/modelling/model_common.py` (fungsi baru sesudah `_assert_checkpoint_matches()`, sebelum `select_best()`)
- Test: `test/test_model_common.py` (kelas baru sesudah `TestCurrentCommit`)

**Interfaces:**
- Consumes: `_assert_checkpoint_matches(prior, candidates, search_space, path)` yang sudah ada; `_ordered(rows)`; hasil `run_search(..., only=...)` dari Task 1.
- Produces: `merge_shards(paths: list, candidates: list, search_space: dict) -> pd.DataFrame` — satu frame terurut `candidate_id` yang siap diberikan ke `select_best(results, candidates)`.

- [ ] **Step 1: Tulis tes yang gagal**

Tambahkan di akhir `test/test_model_common.py`:

```python
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
```

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv/bin/python3 -m unittest test.test_model_common.TestMergeShards -v`
Expected: FAIL — `AttributeError: module 'utils.modelling.model_common' has no attribute 'merge_shards'`

- [ ] **Step 3: Tulis implementasi minimal**

Tambahkan di `utils/modelling/model_common.py`, sesudah `_assert_checkpoint_matches()` dan sebelum `select_best()`:

```python
def merge_shards(paths: list, candidates: list, search_space: dict) -> pd.DataFrame:
    """Satu frame pencarian dari beberapa CSV shard, diverifikasi bukan dipercaya.

    Empat pemeriksaan, dan tiap satunya membalas satu cara pemecahan pekerjaan
    bisa gagal tanpa suara:

    1. `candidate_id` ganda — dua mesin diberi rentang yang tumpang tindih,
       sehingga satu kandidat dinilai dua kali dan `select_best()` memilih di
       antara baris kembar.
    2. Cakupan berlubang — satu shard tidak pernah selesai, dan yang dilaporkan
       sebagai "pencarian 30 kandidat" sebenarnya 22.
    3. Parameter yang tidak cocok dengan id yang diklaimnya — shard tertukar,
       atau lahir dari `seed` / ruang pencarian yang berbeda.
    4. Skema kuantil tunggal — CSV pra-2026-08-24 yang angkanya pinball@0,9,
       bukan K1.

    Pemeriksaan 3 dan 4 tidak ditulis ulang di sini: keduanya sudah menjadi
    `_assert_checkpoint_matches()`, dan dua definisi "cocok" yang hidup
    berdampingan pasti akan berbeda diam-diam suatu hari.

    Kolom di luar `search_space` — `device`, hash commit, metrik — dibawa apa
    adanya; ia justru yang membuat baris hasil dapat ditelusuri ke mesinnya.
    """
    if not paths:
        raise ValueError("tidak ada shard untuk digabungkan")

    merged = pd.concat([pd.read_csv(path) for path in paths],
                       ignore_index=True)
    ids = [int(value) for value in merged["candidate_id"]]

    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"candidate_id ganda di gabungan shard: {duplicates} — dua mesin "
            f"diberi rentang yang tumpang tindih"
        )

    missing = sorted(set(range(len(candidates))) - set(ids))
    if missing:
        raise ValueError(
            f"gabungan shard tidak menutup seluruh {len(candidates)} kandidat: "
            f"{missing} hilang — satu shard belum selesai atau tidak ikut "
            f"digabungkan"
        )

    _assert_checkpoint_matches(merged, candidates, search_space,
                               path="gabungan shard")
    return _ordered(merged.to_dict("records"))
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv/bin/python3 -m unittest test.test_model_common -v`
Expected: OK — termasuk 8 tes `TestMergeShards`.

- [ ] **Step 5: Commit**

```bash
git add utils/modelling/model_common.py test/test_model_common.py
git commit -m "feat: satukan CSV shard lewat merge_shards() dengan guard yang sudah ada"
```

---

### Task 4: Penerusan di ketiga pembungkus model

**Files:**
- Modify: `utils/modelling/model_xgboost.py:439-467`
- Modify: `utils/modelling/model_lstm.py:749-780`
- Modify: `utils/modelling/model_random_forest.py:195-213`
- Test: `test/test_model_xgboost.py`, `test/test_model_lstm.py`, `test/test_model_random_forest.py` (satu kelas tes baru di tiap berkas, letakkan di akhir)

**Interfaces:**
- Consumes: `model_common.run_search(..., only=..., provenance=...)` dari Task 1 dan 2.
- Produces: `model_{xgboost,lstm,random_forest}.run_search(..., only=None, provenance=None)` — tiga tanda tangan yang seragam, sehingga notebook mana pun memanggilnya dengan cara yang sama.

**Catatan:** Random Forest berjalan utuh di satu mesin menurut spec dan tidak akan dipecah. Ia tetap ikut karena tiga pembungkus bersaudara dengan tanda tangan yang berbeda-beda lebih mahal daripada satu kwarg yang tak terpakai — alasan yang sama yang sudah tertulis di komentar `partial(make_fit_predict, device=device)` di `model_xgboost.py`.

- [ ] **Step 1: Tulis tes yang gagal**

Tambahkan di akhir `test/test_model_xgboost.py`:

```python
class TestRunSearchForwarding(unittest.TestCase):
    """Notebook memanggil pembungkus per model, bukan model_common. Kalau
    `only` berhenti di sini, pemecahan shard tidak pernah sampai ke mesinnya."""

    def test_only_and_provenance_reach_model_common(self):
        from unittest import mock
        with mock.patch.object(model_xgboost.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            model_xgboost.run_search(pd.DataFrame(), [{"a": 1}],
                                     only=[3, 4],
                                     provenance={"device": "cuda:0"})
        self.assertEqual(spy.call_args.kwargs["only"], [3, 4])
        self.assertEqual(spy.call_args.kwargs["provenance"],
                         {"device": "cuda:0"})

    def test_the_defaults_stay_none(self):
        from unittest import mock
        with mock.patch.object(model_xgboost.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            model_xgboost.run_search(pd.DataFrame(), [{"a": 1}])
        self.assertIsNone(spy.call_args.kwargs["only"])
        self.assertIsNone(spy.call_args.kwargs["provenance"])
```

Tambahkan di akhir `test/test_model_lstm.py`. **Perhatikan bedanya:**
`model_lstm.run_search` memanggil `bind_panel(df, ...)` di dalam daftar
argumennya, jadi ia dijalankan sungguhan sebelum `model_common.run_search`
sempat di-mock — dan `bind_panel` pada DataFrame kosong akan gagal. Karena itu
`bind_panel` ikut di-patch:

```python
class TestRunSearchForwarding(unittest.TestCase):
    """Notebook memanggil pembungkus per model, bukan model_common. Kalau
    `only` berhenti di sini, pemecahan shard tidak pernah sampai ke mesinnya."""

    def test_only_and_provenance_reach_model_common(self):
        from unittest import mock
        with mock.patch.object(model_lstm, "bind_panel",
                               return_value=lambda *a, **k: None), \
             mock.patch.object(model_lstm.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            model_lstm.run_search(pd.DataFrame(), [{"a": 1}],
                                  only=[3, 4],
                                  provenance={"device": "cuda:0"})
        self.assertEqual(spy.call_args.kwargs["only"], [3, 4])
        self.assertEqual(spy.call_args.kwargs["provenance"],
                         {"device": "cuda:0"})

    def test_the_defaults_stay_none(self):
        from unittest import mock
        with mock.patch.object(model_lstm, "bind_panel",
                               return_value=lambda *a, **k: None), \
             mock.patch.object(model_lstm.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            model_lstm.run_search(pd.DataFrame(), [{"a": 1}])
        self.assertIsNone(spy.call_args.kwargs["only"])
        self.assertIsNone(spy.call_args.kwargs["provenance"])
```

Tambahkan di akhir `test/test_model_random_forest.py`:

```python
class TestRunSearchForwarding(unittest.TestCase):
    """Notebook memanggil pembungkus per model, bukan model_common. Kalau
    `only` berhenti di sini, pemecahan shard tidak pernah sampai ke mesinnya."""

    def test_only_and_provenance_reach_model_common(self):
        from unittest import mock
        with mock.patch.object(model_random_forest.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            model_random_forest.run_search(pd.DataFrame(), [{"a": 1}],
                                           only=[3, 4],
                                           provenance={"device": "cpu"})
        self.assertEqual(spy.call_args.kwargs["only"], [3, 4])
        self.assertEqual(spy.call_args.kwargs["provenance"], {"device": "cpu"})

    def test_the_defaults_stay_none(self):
        from unittest import mock
        with mock.patch.object(model_random_forest.model_common, "run_search",
                               return_value=pd.DataFrame()) as spy:
            model_random_forest.run_search(pd.DataFrame(), [{"a": 1}])
        self.assertIsNone(spy.call_args.kwargs["only"])
        self.assertIsNone(spy.call_args.kwargs["provenance"])
```

Ketiga berkas tes itu sudah mengimpor `unittest`, `pandas as pd`, dan modul
modelnya masing-masing — tidak ada impor baru yang perlu ditambahkan di
tingkat berkas.

- [ ] **Step 2: Jalankan tes, pastikan gagal**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost.TestRunSearchForwarding test.test_model_lstm.TestRunSearchForwarding test.test_model_random_forest.TestRunSearchForwarding -v`
Expected: FAIL — `TypeError: run_search() got an unexpected keyword argument 'only'` di ketiganya.

- [ ] **Step 3: Tulis implementasi minimal**

**`utils/modelling/model_xgboost.py`** — ubah baris 19:

```python
from typing import Callable, Iterable, Optional
```

Tambahkan dua parameter di tanda tangan `run_search` (baris 439), tepat
sesudah `resume: bool = True,`:

```python
    only: Optional[Iterable[int]] = None,
    provenance: Optional[dict] = None,
```

dan di panggilan `model_common.run_search(...)` di dalamnya, tepat sesudah
`checkpoint_path=checkpoint_path, resume=resume,`:

```python
        only=only, provenance=provenance,
```

**`utils/modelling/model_lstm.py`** — ubah baris 14:

```python
from typing import Iterable, Optional
```

Tambahkan dua parameter di tanda tangan `run_search` (baris 749), tepat
sesudah `resume: bool = True,` dan **sebelum** `device_name: str = "cpu",`:

```python
    only: Optional[Iterable[int]] = None,
    provenance: Optional[dict] = None,
```

dan di panggilan `model_common.run_search(...)` di dalamnya, tepat sesudah
baris `resume=resume,`:

```python
        only=only,
        provenance=provenance,
```

(Perhatikan gaya berkas ini: satu argumen per baris, bukan digabung seperti di
`model_xgboost.py`.)

**`utils/modelling/model_random_forest.py`** — ubah baris 19:

```python
from typing import Callable, Iterable, Optional
```

Tambahkan dua parameter di tanda tangan `run_search` (baris 195), tepat
sesudah `resume: bool = True,`:

```python
    only: Optional[Iterable[int]] = None,
    provenance: Optional[dict] = None,
```

dan di panggilan `model_common.run_search(...)` di dalamnya, tepat sesudah
`checkpoint_path=checkpoint_path, resume=resume,`:

```python
        only=only, provenance=provenance,
```

- [ ] **Step 4: Jalankan tes, pastikan lolos**

Run: `.venv/bin/python3 -m unittest test.test_model_xgboost test.test_model_lstm test.test_model_random_forest -v`
Expected: OK — 6 tes baru lolos, tidak ada yang lama merah.

- [ ] **Step 5: Commit**

```bash
git add utils/modelling/model_xgboost.py utils/modelling/model_lstm.py utils/modelling/model_random_forest.py test/test_model_xgboost.py test/test_model_lstm.py test/test_model_random_forest.py
git commit -m "feat: teruskan only= dan provenance= lewat ketiga pembungkus run_search"
```

---

### Task 5: Catat mekanismenya di todolist, lalu jalankan seluruh suite

**Files:**
- Modify: `docs/todolist-proyek.md` (bagian `### 0c — Menjalankan ulang ketiga notebook`)

**Interfaces:**
- Consumes: seluruh seam dari Task 1–4.
- Produces: tidak ada antarmuka kode; hanya catatan status supaya sesi berikutnya tidak menyangka mekanisme ini belum ada.

- [ ] **Step 1: Jalankan seluruh suite lebih dulu, sebagai baseline yang sudah hijau**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py"`
Expected: OK — 752 tes lama + 29 tes baru (8 + 7 + 8 + 6) = **781 tes**, nol gagal.

- [ ] **Step 2: Tambahkan catatan di todolist**

Di `docs/todolist-proyek.md`, tepat sesudah baris judul `### 0c — Menjalankan ulang ketiga notebook ⬜ 🔒 menunggu izin` dan paragraf anggarannya, sisipkan:

```markdown
**Mekanisme eksekusi terdistribusi sudah terpasang (2026-08-24).** Pencarian
dapat dipecah antar mesin lewat `model_common.run_search(..., only=[...],
provenance={...})` dan disatukan kembali oleh `model_common.merge_shards()`,
yang memakai ulang guard `_assert_checkpoint_matches()` sehingga shard yang
tertukar, berlubang, atau berasal dari run kuantil tunggal tertolak. Rencana
mesin per tahap, probe paritas device yang mengesahkan pemecahan itu, dan
alasan walk-forward tetap di Mac ada di
`docs/superpowers/specs/2026-08-24-distributed-gpu-training-design.md`.
Ini **bukan** izin menjalankan 0c; ia hanya menghapus kode sebagai penghalang.
```

- [ ] **Step 3: Jalankan seluruh suite sekali lagi**

Run: `.venv/bin/python3 -m unittest discover -p "test_*.py"`
Expected: OK — 781 tes, nol gagal. (Perubahan dokumentasi tidak boleh mengubah angka ini; kalau berubah, ada yang salah di Task 1–4.)

- [ ] **Step 4: Commit**

```bash
git add docs/todolist-proyek.md
git commit -m "docs: catat seam shard di butir 0c"
```

---

## Out of scope

Dua hal yang **sengaja** tidak ada di rencana ini, supaya tidak ada yang menyangka keduanya sudah selesai:

1. **Perubahan notebook.** Ketiga notebook modeling belum punya sel yang membaca `only`/`device` dari environment, sehingga satu notebook yang sama bisa dijalankan di Kaggle dan Colab dengan shard berbeda. Itu pekerjaan terpisah, dan ia menyentuh berkas yang butir 0c sedang menunggu izin untuk dijalankan.
2. **Menjalankan Fase 3 itu sendiri.** Butir 0c masih 🔒. Rencana ini hanya menghapus kode sebagai penghalang.
