import numpy as np
from konv_lapisan import Konv2D

def gambar_garis(n, kolom, rng, S=10):
    """Garis datar, tegak, atau miring sepanjang 4 piksel + derau."""
    X, y = np.zeros((n, 1, S, S)), rng.integers(0, 3, n)
    for i in range(n):
        r = rng.integers(0, S - 3)
        c = rng.integers(*kolom)                   # posisi kolom
        for t in range(4):
            dr, dc = [(0, t), (t, 0), (t, t)][y[i]]
            X[i, 0, r + dr, min(c + dc, S - 1)] = 1
    return X + rng.normal(0, 0.3, X.shape), y

rng = np.random.default_rng(0)
X, y = gambar_garis(600, (0, 3), rng)        # latih: kolom 0-2  #<1>
Xu, yu = gambar_garis(600, (6, 7), rng)      # uji: kolom 6

def softmax_ce(Z, y):
    P = np.exp(Z - Z.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    G = P.copy(); G[np.arange(len(y)), y] -= 1
    return P, G / len(y)

def latih_cnn(langkah=400, laju=0.5):
    konv = Konv2D(1, 4, 3, 1, rng)
    V, c = rng.normal(0, 0.5, (4, 3)), np.zeros(3)
    def maju(X):
        A = konv.maju(X); H = np.maximum(A, 0)
        return A, H, H.max(axis=(2, 3))            # global max pool  #<2>
    for _ in range(langkah):
        A, H, g = maju(X)
        P, G = softmax_ce(g @ V + c, y)
        gV, gc = g.T @ G, G.sum(0)
        Gg = G @ V.T                               # ke fitur pool
        juara = H == g[:, :, None, None]           # posisi maksimum
        GH = Gg[:, :, None, None] * juara          # hanya ke juara  #<3>
        konv.mundur(GH * (A > 0))                  # lewat ReLU
        konv.W -= laju * konv.gW; konv.b -= laju * konv.gb
        V -= laju * gV; c -= laju * gc
    def akurasi(X, y):
        return np.mean((maju(X)[2] @ V + c).argmax(1) == y)
    return akurasi(X, y), akurasi(Xu, yu), konv.W

def latih_mlp(langkah=400, laju=0.5):
    F, Fu = X.reshape(len(X), -1), Xu.reshape(len(Xu), -1)
    W, b = np.zeros((100, 3)), np.zeros(3)         # regresi softmax
    for _ in range(langkah):
        _, G = softmax_ce(F @ W + b, y)
        W -= laju * F.T @ G; b -= laju * G.sum(0)
    akurasi = lambda F, y: np.mean((F @ W + b).argmax(1) == y)
    return akurasi(F, y), akurasi(Fu, yu)

a, au = latih_mlp()
print(f"Lapisan penuh (303 param): latih = {a:.3f}  uji = {au:.3f}")
a, au, filter_ = latih_cnn()
print(f"CNN kecil      (55 param): latih = {a:.3f}  uji = {au:.3f}")
