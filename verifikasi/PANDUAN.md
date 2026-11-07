# Panduan verifikasi kode

Angka di buku diambil dari Kaggle (CPU, NumPy 2.1). Panduan ini menjalankan
semua listing dan mencocokkannya dengan folder `harapan/`.

## Google Colab

1. Buka https://colab.research.google.com → **New notebook**.
2. Tempel sel berikut, lalu jalankan (Shift+Enter):

```python
!git clone -q https://github.com/bsnegara/jst-deep-learning.git
%cd jst-deep-learning
import numpy, torch
print("numpy", numpy.__version__, "| torch", torch.__version__)
!python jalankan_semua.py
```

## Kaggle

1. kaggle.com → **Create** → **New Notebook**.
2. Panel kanan → **Session options** → **Internet: On** (perlu verifikasi
   nomor telepon). Accelerator: **None**.
3. Tempel sel yang sama seperti untuk Colab, lalu jalankan.

Tanpa akses internet: unduh ZIP repositori dari GitHub
(**Code → Download ZIP**), unggah sebagai *dataset* pribadi, tambahkan sebagai
*input* notebook, lalu jalankan:

```python
import glob, os, shutil
src = os.path.dirname(glob.glob("/kaggle/input/**/jalankan_semua.py",
                                recursive=True)[0])
shutil.copytree(src, "/kaggle/working/jst", dirs_exist_ok=True)
os.chdir("/kaggle/working/jst")
!python jalankan_semua.py
```

## Membaca hasil

Baris terakhir merangkum hasilnya, misalnya
`Ringkasan: COCOK 70, MODUL 6`. Lihat bagian *Tentang status BEDA* di
`README.md` bila ada skrip yang berbeda.
