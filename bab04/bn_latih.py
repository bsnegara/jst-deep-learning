import numpy as np
from batchnorm import bn_maju, bn_mundur

def spiral(n, K, rng):
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.05, 1, n)
        t = np.linspace(k * 4.2, (k + 1) * 4.2, n)
        t = t + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [k] * n
    return np.vstack(X), np.array(y)

def latih(pakai_bn, laju, langkah=2000, lebar=64, L=6):
    rng = np.random.default_rng(0)
    X, y = spiral(100, 3, rng)
    dims = [2] + [lebar] * L + [3]
    W = [rng.normal(0, np.sqrt(1 / a), (a, b))      # Xavier
         for a, b in zip(dims[:-1], dims[1:])]
    b = [np.zeros(d) for d in dims[1:]]
    g = [np.ones(d) for d in dims[1:-1]]          # gamma BN
    be = [np.zeros(d) for d in dims[1:-1]]        # beta BN
    sig = lambda a: 1 / (1 + np.exp(-a))
    for t in range(langkah + 1):
        H, cache = X, []
        for k in range(L):                 # lapisan tersembunyi
            A = H @ W[k] + b[k]
            c = None
            if pakai_bn:
                A, c = bn_maju(A, g[k], be[k])    #<1>
            Hn = sig(A)
            cache.append((H, c, Hn)); H = Hn
        Z = H @ W[L] + b[L]
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        loss = -np.mean(np.log(P[np.arange(len(y)), y]))
        G = P.copy(); G[np.arange(len(y)), y] -= 1; G /= len(y)
        gW, gb = [None] * (L + 1), [None] * (L + 1)
        gW[L], gb[L] = H.T @ G, G.sum(0)
        G = G @ W[L].T
        for k in reversed(range(L)):
            Hin, c, Hout = cache[k]
            G = G * Hout * (1 - Hout)              # turunan sigmoid
            if pakai_bn:
                G, gg, gbe = bn_mundur(G, c)
                g[k] -= laju * gg; be[k] -= laju * gbe
            gW[k], gb[k] = Hin.T @ G, G.sum(0)
            G = G @ W[k].T
        for k in range(L + 1):
            W[k] -= laju * gW[k]; b[k] -= laju * gb[k]
    akurasi = np.mean(P.argmax(1) == y)
    return loss, akurasi

for pakai_bn in (False, True):
    for laju in (0.1, 1.0):
        loss, akurasi = latih(pakai_bn, laju)
        print(f"BN={str(pakai_bn):5s} laju={laju:3.1f}  "
              f"loss={loss:.3f}  akurasi={akurasi:.3f}")
