import contextlib, io
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):     #<1>
    import difusi_1d as d      # latih ulang, keluaran disembunyikan

def ddim(langkah, n=4000):
    ts = np.linspace(d.T - 1, 0, langkah).round().astype(int)
    x = np.random.default_rng(1).normal(size=(n, 1))
    for i, t in enumerate(ts):
        ab = d.alfa_bar[t]
        e = d.net.maju(d.fitur(x, np.full(n, t)))
        x0 = (x - np.sqrt(1 - ab) * e) / np.sqrt(ab)     # tebak x0  #<2>
        ab2 = d.alfa_bar[ts[i + 1]] if i + 1 < len(ts) else 1.0
        x = np.sqrt(ab2) * x0 + np.sqrt(1 - ab2) * e   # tanpa derau  #<3>
    return x[:, 0]

print("langkah   dekat -2   dekat +2   lainnya   std per mode")
for S in [100, 20, 10, 5]:
    s = ddim(S)
    kiri, kanan = np.mean(abs(s + 2) < 1), np.mean(abs(s - 2) < 1)
    lain = 1 - kiri - kanan
    print(f"{S:7d}   {kiri:8.2f}   {kanan:8.2f}   {lain:7.2f}"
          f"   {s[s < 0].std():.2f}, {s[s > 0].std():.2f}")
