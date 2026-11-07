from dropout_wd import latih

catatan = []
latih(catat=catatan)                       # tanpa regularisasi  #<1>
for t, a_latih, a_val in catatan[::5]:
    print(f"langkah {t:4d}  akurasi latih = {a_latih:.3f}"
          f"  validasi = {a_val:.3f}")
t_baik, _, a_baik = max(catatan, key=lambda c: c[2])   #<2>
print(f"Terbaik: langkah {t_baik}, akurasi validasi {a_baik:.3f}")
