"""Ke mana sebuah run menulis, di device apa ia berjalan, dan kandidat mana
yang menjadi tanggung jawabnya — dibaca dari environment, sekali, di sel
pertama notebook.

Ada karena Fase 3 dijalankan di tiga tempat sekaligus (Mac, Kaggle, Colab)
lewat notebook yang sama, dan tiap tempat menaruh berkasnya di folder yang
berbeda: `dataset/` tidak ikut `git clone` karena gitignored, `/kaggle/working`
lenyap kalau tidak di-commit, dan Drive baru ada setelah di-mount. Tanpa lapis
ini, tiga notebook × empat setelan = dua belas tempat parsing yang harus
sepakat — dan kalau "0-14" dibaca inklusif di satu notebook dan eksklusif di
notebook lain, kandidat 14 hilang atau dinilai dua kali. Kegagalan itu tidak
kelihatan sampai penggabungan shard, berjam-jam sesudahnya. Alasannya sama
persis dengan alasan `model_common` ada.

**Aturan yang mengikat seluruh modul: tanpa satu pun env var, tiap fungsi di
sini mengembalikan persis apa yang notebook lakukan sebelum modul ini ada** —
termasuk nama berkasnya. Sebuah run lokal tidak boleh berubah perilakunya
hanya karena jalur cloud ditambahkan.

Nilai kosong (`export FORECAST_SHARD=`) ditolak, bukan diperlakukan sebagai
"tidak diset". Itu salah ketik yang lazim, dan cabang diamnya adalah cabang
yang mahal: kedua mesin menjalankan seluruh kandidat.
"""

import os
from typing import Optional

from . import model_common, modeling_prep

SHARD_ENV = "FORECAST_SHARD"
DEVICE_ENV = "FORECAST_DEVICE"
MODEL_INPUT_ENV = "FORECAST_MODEL_INPUT"
CHECKPOINT_DIR_ENV = "FORECAST_CHECKPOINT_DIR"


def _read(name: str, env: Optional[dict]) -> Optional[str]:
    """Nilai env var, atau None kalau tidak diset. Kosong = salah, bukan diam."""
    source = os.environ if env is None else env
    if name not in source:
        return None
    value = source[name].strip()
    if not value:
        raise ValueError(
            f"{name} diset tapi kosong — hapus variabelnya kalau memang tidak "
            f"dipakai, jangan dikosongkan"
        )
    return value


def shard(env: Optional[dict] = None) -> Optional[list]:
    """`only=` untuk `run_search`, dibaca dari `FORECAST_SHARD`.

    Bentuk yang diterima: `"0-14"` (rentang, **inklusif di kedua ujung**),
    `"0,3,7"` (daftar), atau campurannya `"0-2,9"`. Inklusif karena batasnya
    ditulis manusia di dua mesin yang berbeda, dan setengah-terbuka adalah
    ejaan yang paling mudah membuat sambungan dua shard bocor satu kandidat.

    Jangkauannya tidak diperiksa di sini: `model_common._selected()` sudah
    menolak id di luar jumlah kandidat, dan satu-satunya tempat yang tahu
    berapa jumlah itu adalah pemanggil `run_search`.
    """
    raw = _read(SHARD_ENV, env)
    if raw is None:
        return None

    selected = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            if "-" in piece:
                low, high = (int(part) for part in piece.split("-", 1))
                if low > high:
                    raise ValueError(
                        f"{SHARD_ENV}={raw!r} memuat rentang menurun {piece!r}"
                    )
                selected.update(range(low, high + 1))
            else:
                selected.add(int(piece))
        except ValueError as failure:
            if "menurun" in str(failure):
                raise
            raise ValueError(
                f"{SHARD_ENV}={raw!r} tidak dapat dibaca — tulis "
                f"'0-14', '0,3,7', atau '0-2,9'"
            ) from failure
    return sorted(selected)


def shard_label(env: Optional[dict] = None) -> str:
    """Potongan nama berkas yang menyebut shard ini. `"full"` kalau tidak dipecah."""
    raw = _read(SHARD_ENV, env)
    if raw is None:
        return "full"
    return raw.replace(" ", "").replace(",", "_")


def device(default: str, env: Optional[dict] = None) -> str:
    """Device yang dipakai, dengan `FORECAST_DEVICE` menimpa `default`.

    `default` datang dari pemanggil, bukan dari modul ini: XGBoost memberinya
    `DEFAULT_DEVICE`, sementara notebook LSTM memberinya device tercepat yang
    baru saja **diukur** benchmark. Penimpaan tetap dibolehkan di kedua kasus
    karena dua shard di satu sesi T4×2 harus dapat dipin ke `cuda:0` dan
    `cuda:1`, dan benchmark tidak tahu apa-apa soal shard — tetapi benchmark
    tetap dijalankan dan angkanya tetap dicetak, supaya penimpaan yang
    ternyata lebih lambat terbaca di output alih-alih hilang diam-diam.
    """
    return _read(DEVICE_ENV, env) or default


def model_input_path(env: Optional[dict] = None) -> str:
    """`model_input.parquet` — di repo, atau di mana pun mesin ini menaruhnya."""
    return _read(MODEL_INPUT_ENV, env) or modeling_prep.MODEL_INPUT_FILE


def checkpoint_path(filename: str, env: Optional[dict] = None) -> str:
    """Sebuah berkas keluaran, di folder yang bisa ditulis mesin ini."""
    folder = _read(CHECKPOINT_DIR_ENV, env) or modeling_prep.MODEL_READY_DIR
    return f"{folder.rstrip('/')}/{filename}"


def search_checkpoint(model: str, env: Optional[dict] = None) -> str:
    """Checkpoint pencarian, dinamai menurut shard-nya.

    Run tanpa shard memakai nama hari ini apa adanya
    (`xgb_search_results.csv`), sehingga run lokal tidak berubah sama sekali.
    Run bershard menyisipkan labelnya, supaya dua shard yang dikumpulkan ke
    satu folder untuk digabungkan tidak pernah bisa saling menimpa — dan
    supaya memindahkan pekerjaan dari satu mesin ke mesin lain cukup dengan
    menyalin satu berkas yang namanya sudah menyebut isinya.
    """
    label = shard_label(env)
    suffix = "" if label == "full" else f".shard-{label}"
    return checkpoint_path(f"{model}_search_results{suffix}.csv", env)


def provenance(device_name: str, env: Optional[dict] = None) -> dict:
    """Kolom asal-usul yang ditempelkan `run_search` ke tiap baris hasil."""
    return {"device": device_name, "commit": model_common.current_commit()}


def describe(device_name: str, env: Optional[dict] = None) -> str:
    """Satu baris untuk dicetak sel pertama notebook.

    Dicetak, bukan disimpan diam-diam: setelan yang salah di sini berongkos
    berjam-jam, dan satu-satunya saat murah untuk menyadarinya adalah sebelum
    sel berikutnya dijalankan.
    """
    ids = shard(env)
    cakupan = ("seluruh kandidat" if ids is None
               else f"shard {shard_label(env)} ({len(ids)} kandidat)")
    return (f"device: {device_name} | {cakupan} | "
            f"input: {model_input_path(env)} | "
            f"checkpoint: {checkpoint_path('', env).rstrip('/')}")
