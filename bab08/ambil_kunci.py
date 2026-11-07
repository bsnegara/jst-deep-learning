import numpy as np
from atensi import atensi
from atensi_mundur import atensi_mundur

rng = np.random.default_rng(1)
d, m = 8, 10                           # dimensi kunci, ukuran kamus

def contoh():                          # kamus acak + kueri berderau
    K, v = rng.normal(size=(m, d)), rng.normal(size=(m, 1))
    j = rng.integers(m)
    q = K[j] + rng.normal(0, 0.3, d)   #<1>
    return q[None], K, v, j

Wq = rng.normal(0, 0.1, (d, d))        # proyeksi yang dipelajari  #<2>
Wk = rng.normal(0, 0.1, (d, d))

def evaluasi(n=1000):
    mse = tepat = 0.0
    for _ in range(n):
        q, K, v, j = contoh()
        o, P = atensi(q @ Wq, K @ Wk, v)
        mse += ((o - v[j]) ** 2).item() / n
        tepat += (P.argmax() == j) / n
    return mse, tepat

mse, tepat = evaluasi()
print(f"sebelum: MSE = {mse:.3f}, atensi tepat sasaran = {tepat:.3f}")
for langkah in range(10000):
    q, K, v, j = contoh()
    Qp, Kp = q @ Wq, K @ Wk
    o, P = atensi(Qp, Kp, v)
    dO = 2 * (o - v[j])                # turunan galat kuadrat
    dQ, dK, _ = atensi_mundur(dO, Qp, Kp, v, P)
    Wq -= 0.05 * q.T @ dQ              # rantai ke proyeksi  #<3>
    Wk -= 0.05 * K.T @ dK
mse, tepat = evaluasi()
print(f"sesudah: MSE = {mse:.3f}, atensi tepat sasaran = {tepat:.3f}")
