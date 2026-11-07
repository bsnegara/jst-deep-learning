import numpy as np

def spiral(n, rng, derau=0.35):
    X, y = [], []
    for k in range(3):
        r = rng.uniform(0.05, 1, n)
        t = r * 4.2 + k * 4.2 + rng.normal(0, derau, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [k] * n
    return np.vstack(X), np.array(y)

rng = np.random.default_rng(0)
X, y = spiral(50, rng)                       # 150 contoh latih
acak = rng.random(len(y)) < 0.2             # 20% label dibalik  #<1>
y = np.where(acak, (y + rng.integers(1, 3, len(y))) % 3, y)
Xu, yu = spiral(1000, rng)                   # 3000 contoh uji
Xv, yv = spiral(100, rng)                    # 300 contoh validasi

def latih(p_drop=0.0, lam=0.0, langkah=6000, laju=0.3, catat=None):
    r = np.random.default_rng(1)
    dims = [2, 256, 256, 3]
    W = [r.normal(0, np.sqrt(2 / a), (a, b))
         for a, b in zip(dims[:-1], dims[1:])]
    b = [np.zeros(k) for k in dims[1:]]

    def maju(H, latihan):
        cache = []
        for k in range(2):
            A = H @ W[k] + b[k]
            M = (A > 0).astype(float)
            if latihan and p_drop > 0:       # dropout terbalik  #<2>
                M *= (r.random(A.shape) > p_drop) / (1 - p_drop)
            cache.append((H, M)); H = A * M
        return H @ W[2] + b[2], cache, H

    for t in range(langkah + 1):
        Z, cache, H = maju(X, True)
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(len(y)), y] -= 1; G /= len(y)
        gW, gb = [None] * 3, [None] * 3
        gW[2], gb[2] = H.T @ G, G.sum(0)
        G = G @ W[2].T
        for k in (1, 0):
            Hin, M = cache[k]
            G = G * M                        # ReLU dan masker dropout
            gW[k], gb[k] = Hin.T @ G, G.sum(0)
            G = G @ W[k].T
        for k in range(3):
            W[k] -= laju * (gW[k] + lam * W[k])  # weight decay  #<3>
            b[k] -= laju * gb[k]
        if catat is not None and t % 200 == 0:
            catat.append((t, akurasi(maju, X, y),
                          akurasi(maju, Xv, yv)))
    return akurasi(maju, X, y), akurasi(maju, Xu, yu)

def akurasi(maju, A, c):
    return np.mean(maju(A, False)[0].argmax(1) == c)

if __name__ == "__main__":
    for nama, kw in [("tanpa regularisasi", {}),
                     ("weight decay 1e-3", {"lam": 1e-3}),
                     ("dropout p=0.5", {"p_drop": 0.5})]:
        a_latih, a_uji = latih(**kw)
        print(f"{nama:19s} akurasi latih = {a_latih:.3f}"
              f"  uji = {a_uji:.3f}")
