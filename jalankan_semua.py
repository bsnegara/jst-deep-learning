"""Jalankan semua skrip listing buku dan cocokkan keluarannya dengan
angka yang tercetak di buku (folder harapan/).

Pemakaian:
    python jalankan_semua.py        # semua bab
    python jalankan_semua.py 06     # satu bab

Status:
    COCOK   keluaran sama persis dengan buku
    BEDA    keluaran berbeda (lihat catatan di README)
    MODUL   berkas berisi fungsi/kelas untuk diimpor skrip lain
    LEWATI  butuh pustaka yang belum terpasang (mis. PyTorch)
    GALAT   skrip berhenti dengan galat
"""
import importlib.util
import pathlib
import subprocess
import sys

akar = pathlib.Path(__file__).resolve().parent
pola = sys.argv[1] if len(sys.argv) > 1 else "*"
punya = {m: importlib.util.find_spec(m) is not None
         for m in ["torch", "sklearn"]}
hitung = {}
for berkas in sorted(akar.glob(f"bab{pola}/*.py")):
    rel = berkas.relative_to(akar).as_posix()
    teks = berkas.read_text(encoding="utf-8")
    kurang = [m for m, ada in punya.items()
              if not ada and (f"import {m}" in teks or f"from {m}" in teks)]
    if kurang:
        status = f"LEWATI (butuh {', '.join(kurang)})"
    else:
        hasil = subprocess.run([sys.executable, berkas.name],
                               cwd=berkas.parent, capture_output=True,
                               text=True, timeout=1800)
        harapan = akar / "harapan" / (rel[:-3] + ".txt")
        if hasil.returncode != 0:
            status = "GALAT"
            print(hasil.stderr[-800:])
        elif not harapan.exists():
            status = "MODUL" if not hasil.stdout.strip() else "TANPA HARAPAN"
        elif hasil.stdout.strip() == harapan.read_text(
                encoding="utf-8").strip():
            status = "COCOK"
        else:
            status = "BEDA"
            print(hasil.stdout)
    kunci = status.split()[0]
    hitung[kunci] = hitung.get(kunci, 0) + 1
    print(f"[{status}] {rel}", flush=True)

print("Ringkasan:", ", ".join(f"{k} {v}" for k, v in sorted(hitung.items())))
