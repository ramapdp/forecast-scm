"""Model Random Forest kuantil untuk peramalan permintaan.

**Apa yang disediakan modul ini.** Satu model kandidat lengkap dengan empat
tahap pemakaiannya: `make_fit_predict()` (fungsi latih-dan-prediksi yang
disuntikkan ke mesin evaluasi), `sample_search_space()` + `run_search()`
(pencarian hyperparameter), `fit_final()` (pelatihan model akhir), dan
`predict_bundle()` (inferensi dari model tersimpan).

**Mengapa bukan Random Forest biasa.** Random Forest sklearn meminimalkan
galat kuadrat atau absolut; keduanya menaksir *pusat* distribusi, sehingga
ramalannya kehabisan stok kira-kira separuh waktu. *Quantile regression
forest* menyimpan seluruh nilai target yang jatuh di tiap leaf, lalu membaca
kuantil yang diminta dari distribusi empiris tersebut — sehingga satu model
yang sama dapat menjawab 19 titik kuantil sekaligus.

**Konsekuensi biayanya.** Penyimpanan nilai per-leaf itu berbentuk array padat
berukuran `(n_estimators, max_node_count, n_outputs, max_samples_leaf)`,
sehingga kebutuhan memorinya sudah tertentu dari hyperparameter *sebelum* satu
pohon pun dibangun — lihat `estimate_leaf_memory_bytes()`. Pohon dalam dengan
leaf sangat kecil karenanya tidak terjangkau di sini, dan pembatasan itu
kebetulan sejalan dengan alasan statistiknya: leaf berisi satu sampel tidak
dapat menaksir kuantil sama sekali.
"""

from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from . import evaluation, model_common, modeling_prep, purging, walk_forward

# Service level yang dijanjikan ke bisnis (B-9), dibaca lapisan alokasi
# produksi. Sengaja *bukan* besaran yang sama dengan QUANTILES di bawah: yang
# satu komitmen ke outlet, yang lain grid tempat perbandingan model dinilai.
# Menyamakan namanya adalah cara keduanya diam-diam menjadi satu angka lagi.
QUANTILE = 0.9

# Grid evaluasi, diambil dari evaluation.py alih-alih ditulis ulang di sini,
# supaya peralihan Tahap A -> Tahap B tidak menyisakan modul ini menilai di
# grid yang lama.
QUANTILES = evaluation.QUANTILE_SET_A

# Kebutuhan penyimpanan leaf di atas ambang ini ditolak sebelum fit dimulai.
# Alternatifnya adalah menemukan batas itu lewat OOM killer, dua puluh menit
# setelah fit berjalan.
MEMORY_BUDGET_BYTES = 3 * 1024 ** 3

# Diekspor ulang supaya pemanggil modul ini — test suite dan modeling_rf.ipynb
# — tetap jalan tanpa diubah setelah ekstraksi. Definisinya tinggal di
# model_common.py karena XGBoost dan LSTM juga memakainya.
IDX_COLS = model_common.IDX_COLS
assert_no_nan = model_common.assert_no_nan
expand_one_hot = model_common.expand_one_hot
select_best = model_common.select_best
load_bundle = model_common.load_bundle

# Titik awal yang dipakai benchmark, dan dasar bagi setiap kandidat pencarian:
# kunci yang tidak ditarik acak diisi dari sini.
DEFAULT_PARAMS = {
    "n_estimators": 200,
    "max_depth": 16,
    "min_samples_leaf": 50,
    "max_samples_leaf": 20,
    "max_features": "sqrt",
    "max_samples": None,
    "log_target": False,
    "one_hot": False,
    "random_state": 42,
}

# Ruang pencarian: 1.152 kombinasi (3 x 4 x 3 x 4 x 2 x 2 x 2).
#
# `n_estimators` sengaja tidak ada di sini. Mutu forest monoton terhadap jumlah
# pohon, jadi mencarinya membelanjakan anggaran untuk pertanyaan yang jawabannya
# sudah diketahui; ia dipatok selama pencarian dan dinaikkan saat fit final.
#
# `max_depth=None` dan `min_samples_leaf` di bawah 20 juga tidak ada, dengan
# alasan yang dijelaskan di docstring modul: keduanya melanggar budget memori
# sekaligus merusak taksiran kuantil.
SEARCH_SPACE = {
    "max_depth": [12, 16, 20],
    "min_samples_leaf": [20, 50, 100, 200],
    "max_samples_leaf": [1, 20, 50],
    "max_features": ["sqrt", 0.3, 0.5, 1.0],
    "max_samples": [None, 0.5],
    "log_target": [False, True],
    "one_hot": [False, True],
}

# Kunci yang diteruskan ke estimator; sisanya (`log_target`, `one_hot`)
# ditangani modul ini sendiri, bukan oleh quantile-forest.
ESTIMATOR_KEYS = ("n_estimators", "max_depth", "min_samples_leaf",
                  "max_samples_leaf", "max_features", "max_samples",
                  "random_state")


def estimate_leaf_memory_bytes(params: dict, n_train: int) -> int:
    """Menaksir batas atas memori penyimpanan leaf, dalam byte.

    Dipakai menyaring kandidat sebelum data dimuat, sehingga konfigurasi yang
    tidak terjangkau ditolak tanpa membangun pohon.

        byte = n_estimators x jumlah_node x max_samples_leaf x 8

    Jumlah node dibatasi dua kali: oleh kedalaman, karena pohon berkedalaman d
    memuat paling banyak 2^(d+1) node; dan oleh ukuran leaf, karena n baris
    yang terbagi ke leaf berisi minimal L baris menghasilkan paling banyak
    2n/L node termasuk node internal. Batas yang lebih ketat yang dipakai.
    """
    fraction = params.get("max_samples") or 1.0
    n_bootstrap = n_train * fraction
    depth_bound = 2.0 ** (params["max_depth"] + 1)
    leaf_bound = 2.0 * n_bootstrap / params["min_samples_leaf"]
    node_count = min(depth_bound, leaf_bound)
    return int(params["n_estimators"] * node_count * params["max_samples_leaf"] * 8)


def build_estimator(params: dict) -> RandomForestQuantileRegressor:
    """Menyusun estimator quantile-forest dari satu set hyperparameter."""
    kwargs = {key: params[key] for key in ESTIMATOR_KEYS if key in params}
    return RandomForestQuantileRegressor(n_jobs=-1, **kwargs)


def make_fit_predict(
    params: Optional[dict] = None,
    feature_cols: Optional[list] = None,
    quantiles: tuple = QUANTILES,
    memory_budget: int = MEMORY_BUDGET_BYTES,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Membentuk fungsi latih-dan-prediksi yang disuntikkan ke mesin evaluasi.

    Fungsi yang dikembalikan menerima `(train, valid)` dan mengembalikan matriks
    prediksi berukuran `(len(valid), len(quantiles))`. Di dalamnya, berurutan:
    cek NaN, penyaringan budget memori, pemilihan fitur, ekspansi one-hot bila
    diminta, transformasi target, fit, prediksi seluruh grid kuantil, lalu
    pembalikan transformasi dan pemotongan nilai negatif.

    Pemilihan fitur, ekspansi one-hot, dan transformasi target sengaja tinggal
    di sini alih-alih di `walk_forward`, karena ketiganya adalah *pilihan
    model* — persis hal yang seharusnya diperbandingkan antar ketiga kandidat.

    Seluruh grid kuantil keluar dari satu kali fit: leaf sudah memuat distribusi
    training, sehingga setiap titik tambahan hanya berongkos satu pembacaan lagi
    atas distribusi yang memang sudah tersimpan. Itulah sebabnya model ini tidak
    perlu dicari ulang saat kriteria berpindah ke multi-kuantil, sementara dua
    model lainnya perlu.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS
    quantiles = tuple(quantiles)

    def fit_predict(train: pd.DataFrame, valid: pd.DataFrame) -> np.ndarray:
        assert_no_nan(train, feature_cols)
        assert_no_nan(valid, feature_cols)

        needed = estimate_leaf_memory_bytes(params, len(train))
        if needed > memory_budget:
            raise MemoryError(
                f"leaf storage {needed / 1024 ** 3:.1f} GB melebihi budget "
                f"{memory_budget / 1024 ** 3:.1f} GB untuk {params}"
            )

        train_X, valid_X = train[feature_cols], valid[feature_cols]
        if params["one_hot"]:
            train_X, valid_X = expand_one_hot(train_X, valid_X)

        y_train = model_common.train_target(train, log_target=params["log_target"])

        model = build_estimator(params)
        model.fit(train_X.to_numpy(dtype=np.float32), y_train)
        # Satu panggilan, bukan perulangan: mengoper seluruh grid menelusuri
        # leaf sekali, sementara sembilan belas panggilan akan menelusurinya
        # sembilan belas kali untuk jawaban yang sama persis.
        prediction = model.predict(valid_X.to_numpy(dtype=np.float32),
                                   quantiles=list(quantiles))
        prediction = np.asarray(prediction, dtype=float).reshape(len(valid_X),
                                                                 len(quantiles))
        if params["log_target"]:
            prediction = modeling_prep.inverse_log_target(prediction)
        # Kuantitas kirim negatif tidak punya makna fisik.
        return np.clip(prediction, 0.0, None)

    return fit_predict


# Fold tempat kandidat dinilai. Hanya dua dari lima, karena pencarian bertugas
# memeringkat kandidat, bukan melaporkan hasil; laporannya datang dari
# walk-forward lima fold dengan fold 1/2/4 sebagai potongan yang bersih dari
# pengaruh seleksi.
SEARCH_FOLDS = (3, 5)

# Jumlah baris training yang cukup representatif untuk menakar penyaringan
# memori sebelum data dimuat — fold 5 melatih di kisaran angka ini.
TYPICAL_N_TRAIN = 1_280_000


def sample_search_space(
    n_candidates: int = 18,
    n_train: int = TYPICAL_N_TRAIN,
    seed: int = 42,
    memory_budget: int = MEMORY_BUDGET_BYTES,
    space: Optional[dict] = None,
) -> list:
    """Menarik acak sejumlah kandidat hyperparameter yang unik dan terjangkau.

    Penyaringan budget-lah yang membuat pembungkus ini layak ada: quantile-forest
    menentukan ukuran array nilai per-leaf dari hyperparameter sebelum satu pohon
    pun dibangun, sehingga kandidat yang tidak terjangkau dapat ditolak tanpa
    memuat data sama sekali.
    """
    def screen(candidate: dict) -> bool:
        return estimate_leaf_memory_bytes(candidate, n_train) <= memory_budget

    return model_common.sample_search_space(
        space=SEARCH_SPACE if space is None else space,
        defaults=DEFAULT_PARAMS,
        n_candidates=n_candidates,
        seed=seed,
        screen=screen,
        screen_label=f"budget {memory_budget / 1024 ** 3:.1f} GB",
    )


def run_search(
    df: pd.DataFrame,
    candidates: list,
    folds: tuple = SEARCH_FOLDS,
    quantiles: tuple = QUANTILES,
    model_name: str = "random_forest",
    feature_cols: Optional[list] = None,
    verbose: bool = True,
    checkpoint_path: Optional[str] = None,
    resume: bool = True,
    only: Optional[Iterable[int]] = None,
    provenance: Optional[dict] = None,
) -> pd.DataFrame:
    """Menilai setiap kandidat di fold pencarian, satu baris hasil per kandidat.

    Protokolnya milik `model_common.run_search()` — termasuk checkpoint yang
    ditulis tiap kandidat selesai dan guard yang menolak melanjutkan dari
    checkpoint yang lahir di ruang pencarian atau grid kuantil berbeda —
    sehingga ketiga model dinilai lewat mesin yang sama persis.
    """
    return model_common.run_search(
        df, candidates, make_fit_predict=make_fit_predict,
        search_space=SEARCH_SPACE, folds=folds, quantiles=quantiles,
        model_name=model_name, feature_cols=feature_cols, verbose=verbose,
        checkpoint_path=checkpoint_path, resume=resume,
        only=only, provenance=provenance,
    )


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_FILE = str(BASE_DIR / "models/random_forest_q90.joblib")
BEST_PARAMS_FILE = str(BASE_DIR / "dataset/model_ready/rf_best_params.json")
SEARCH_FILE = str(BASE_DIR / "dataset/model_ready/rf_search_results.csv")
RESULTS_FILE = str(BASE_DIR / "dataset/model_ready/rf_walk_forward_results.csv")

# Dinaikkan dari 200 yang dipakai saat pencarian. Mutu forest monoton terhadap
# jumlah pohon, jadi model akhir membeli pengurangan varians yang tidak
# terjangkau saat 18 kandidat harus dinilai berurutan.
FINAL_N_ESTIMATORS = 400


def fit_final(
    df: pd.DataFrame,
    params: dict,
    feature_cols: Optional[list] = None,
    n_estimators: int = FINAL_N_ESTIMATORS,
    quantiles: tuple = QUANTILES,
    date_col: str = modeling_prep.DATE_COL,
    test_start: pd.Timestamp = modeling_prep.TEST_START,
) -> dict:
    """Melatih model akhir pada seluruh baris layak sebelum Desember.

    Mengembalikan *bundle*: model terlatih beserta hyperparameter, daftar dan
    urutan kolom training, grid kuantil, jumlah baris, dan provenance target —
    segala yang dibutuhkan `predict_bundle()` untuk mengulang perlakuan yang
    sama saat inferensi.

    Kelayakan baris datang dari `walk_forward.eligible_rows()`, bukan dari
    penyaringan tanggal yang ditulis di sini. Baris yang melatih model akhir
    harus sama dengan baris yang menilainya, dan pemotongan saat penilaian
    bukan hanya soal tanggal: 28 hari pertama tiap segmen belum punya jendela
    lag yang penuh, dan beberapa hari terakhir tidak punya target sama sekali
    karena penjumlahan lead-time melewati ujung data. Menyeleksi baris secara
    terpisah di sini pernah membuat model yang dikirim dilatih pada populasi
    yang berbeda dari yang dilaporkan metriknya — dan, karena pemotongan target
    ikut hilang, pada label yang bernilai NaN.

    `purging.lookahead_safe_mask()` memotong di batas Desember supaya tidak ada
    baris training yang targetnya menjangkau ke dalam test set.

    Bundle mencatat urutan kolom training bersama modelnya. Forest yang dimuat
    ulang pekan depan dengan urutan kolom berbeda tidak gagal — ia meramal
    dengan percaya diri dari fitur yang salah, dan itu lebih buruk.
    """
    params = {**DEFAULT_PARAMS, **params, "n_estimators": n_estimators}
    feature_cols = feature_cols or modeling_prep.FEATURE_COLS

    frame = walk_forward.eligible_rows(df, date_col=date_col, test_start=test_start)
    frame = frame[purging.lookahead_safe_mask(frame, test_start, date_col=date_col)]
    assert_no_nan(frame, feature_cols)

    train_X = frame[feature_cols]
    if params["one_hot"]:
        train_X, _ = expand_one_hot(train_X, train_X)

    y_train = model_common.train_target(frame, log_target=params["log_target"])

    model = build_estimator(params)
    model.fit(train_X.to_numpy(dtype=np.float32), y_train)
    return {
        "model": model,
        "params": params,
        "feature_cols": feature_cols,
        "columns": list(train_X.columns),
        "quantiles": tuple(quantiles),
        "n_train": int(len(frame)),
        **model_common.target_provenance(),
    }


def predict_bundle(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Meramal dengan bundle terlatih, memaksa urutan kolom yang tercatat.

    Grid kuantilnya dibaca dari bundle, bukan dari konstanta modul ini. Forest
    yang dimuat ulang setelah peralihan Tahap A -> Tahap B harus menjawab di
    titik-titik tempat ia dilaporkan, bukan di grid apa pun yang sedang berlaku.
    """
    params = bundle["params"]
    quantiles = tuple(bundle["quantiles"])
    features = frame[bundle["feature_cols"]]
    if params["one_hot"]:
        features, _ = expand_one_hot(features, features)
    features = features.reindex(columns=bundle["columns"], fill_value=0)
    prediction = bundle["model"].predict(
        features.to_numpy(dtype=np.float32), quantiles=list(quantiles)
    )
    prediction = np.asarray(prediction, dtype=float).reshape(len(features),
                                                             len(quantiles))
    if params["log_target"]:
        prediction = modeling_prep.inverse_log_target(prediction)
    return np.clip(prediction, 0.0, None)


def save_bundle(bundle: dict, path: str = MODEL_FILE) -> None:
    """Menyimpan bundle model akhir ke berkas joblib."""
    model_common.save_bundle(bundle, path)


def save_best_params(params: dict, path: str = BEST_PARAMS_FILE) -> None:
    """Menyimpan hyperparameter pemenang pencarian ke berkas JSON."""
    model_common.save_best_params(params, path)
