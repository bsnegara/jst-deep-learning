import numpy as np
from konvolusi import konv2d
from konv_lapisan import Konv2D

rng = np.random.default_rng(0)
lapis = Konv2D(c_in=2, c_out=3, k=3, p=1, rng=rng)
X = rng.normal(size=(2, 2, 5, 5))
R = rng.normal(size=(2, 3, 5, 5))            # loss L = sum(R * Y)

def loss():
    return np.sum(R * lapis.maju(X))

def galat_relatif(analitis, param):          # beda hingga terpusat
    num = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        s = param[idx]
        param[idx] = s + 1e-6; lp = loss()
        param[idx] = s - 1e-6; lm = loss()
        param[idx] = s
        num[idx] = (lp - lm) / 2e-6
    return np.abs(analitis - num).max() / np.abs(num).max()

lapis.maju(X); gX = lapis.mundur(R)
print(f"Galat relatif grad W: {galat_relatif(lapis.gW, lapis.W):.1e}")
print(f"Galat relatif grad X: {galat_relatif(gX, X):.1e}")

# Gradien masukan = konvolusi penuh G dgn filter diputar 180 deg
K, G = rng.normal(size=(3, 3)), rng.normal(size=(4, 4))  #<1>
satu = Konv2D(1, 1, 3, 0, rng); satu.W[0, 0] = K
satu.maju(np.zeros((1, 1, 6, 6)))
dX = satu.mundur(G[None, None])[0, 0]
transpos = konv2d(np.pad(G, 2), np.flip(K))               #<2>
selisih = np.abs(dX - transpos).max()
print(f"Selisih dengan konvolusi transpos: {selisih:.1e}")
