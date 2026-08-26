"""Jalankan sebagian sel sebuah notebook, satu kernel, simpan setelah tiap sel.

Ada karena `jupyter nbconvert --execute` hanya mengenal "seluruh notebook".
Pencarian XGBoost berjalan berhari-hari sementara walk-forward dan fit final
tidak punya checkpoint sama sekali, jadi menjalankan keduanya dalam satu
proses berarti satu kill di jam ke-120 menghapus semuanya. Skrip ini memecah
notebook yang sama menjadi beberapa proses tanpa mengedit notebooknya.

Pemakaian:
    python3 run_cells.py <notebook.ipynb> <spec>

`spec` adalah indeks sel absolut, inklusif di kedua ujung, dipisah koma —
`"2-12"`, `"2-10,14"`. Sel markdown di dalam rentang dilewati diam-diam.
Indeksnya dicetak oleh `--list`.

Output tiap sel diteruskan ke stderr selagi berjalan (nbclient normalnya
menelannya sampai sel selesai) dan tetap tersimpan ke notebook seperti biasa.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


class EchoingClient(NotebookClient):
    """NotebookClient yang ikut mencetak output sel ke stderr saat berjalan."""

    def output(self, outs, msg, display_id, cell_index):
        out = super().output(outs, msg, display_id, cell_index)
        if out is not None:
            text = out.get("text") or "".join(
                out.get("data", {}).get("text/plain", "")
            )
            if out.get("output_type") == "error":
                text = "\n".join(out.get("traceback", []))
            if text:
                print(text, end="" if text.endswith("\n") else "\n",
                      file=sys.stderr, flush=True)
        return out


def parse_spec(spec: str) -> list:
    """`"2-10,14"` -> [2..10, 14]. Inklusif, karena batasnya ditulis manusia."""
    picked = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            low, high = (int(part) for part in piece.split("-", 1))
            if low > high:
                raise SystemExit(f"rentang menurun: {piece!r}")
            picked.extend(range(low, high + 1))
        else:
            picked.append(int(piece))
    return sorted(set(picked))


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1]).resolve()
    nb = nbformat.read(path, as_version=4)

    if sys.argv[2] == "--list":
        for i, cell in enumerate(nb.cells):
            head = (cell.source.splitlines() or [""])[0][:70]
            print(f"{i:2d} {cell.cell_type:8s} {head}")
        return 0

    indices = parse_spec(sys.argv[2])
    bad = [i for i in indices if i >= len(nb.cells)]
    if bad:
        raise SystemExit(f"indeks di luar notebook ({len(nb.cells)} sel): {bad}")

    client = EchoingClient(nb, timeout=-1, kernel_name="python3",
                           allow_errors=False,
                           resources={"metadata": {"path": str(path.parent)}})
    started = time.time()
    with client.setup_kernel():
        client.reset_execution_trackers()
        for index in indices:
            cell = nb.cells[index]
            if cell.cell_type != "code":
                continue
            head = (cell.source.splitlines() or [""])[0][:60]
            print(f"\n=== sel {index} | {time.strftime('%H:%M:%S')} | {head}",
                  file=sys.stderr, flush=True)
            cell_started = time.time()
            try:
                client.execute_cell(cell, index)
            except CellExecutionError as failure:
                nbformat.write(nb, path)
                print(f"\n!!! sel {index} GAGAL setelah "
                      f"{(time.time() - cell_started) / 60:.1f} mnt\n{failure}",
                      file=sys.stderr, flush=True)
                return 1
            # Disimpan per sel: sesi yang mati di sel berikutnya tetap
            # meninggalkan output sel-sel yang sudah selesai.
            nbformat.write(nb, path)
            print(f"--- sel {index} selesai dalam "
                  f"{(time.time() - cell_started) / 60:.1f} mnt",
                  file=sys.stderr, flush=True)

    print(f"\n=== SELESAI: {len(indices)} sel, "
          f"{(time.time() - started) / 3600:.2f} jam", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
