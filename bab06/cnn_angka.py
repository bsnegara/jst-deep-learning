import numpy as np
from sklearn.datasets import load_digits
from konv_lapisan import Konv2D
from pool import MaksPool2

def data_angka():
    d = load_digits()                      # 1797 gambar 8x8
    X, y = d.images[:, None] / 16.0, d.target
    idx = np.random.default_rng(0).permutation(len(X))
    X, y = X[idx], y[idx]
    return X[:1300], y[:1300], X[1300:], y[1300:]

class CNN:
    def __init__(self, rng):
        self.k1 = Konv2D(1, 8, 3, 1, rng)              # 8x8   #<1>
        self.p1 = MaksPool2()                          # -> 4x4
        self.k2 = Konv2D(8, 16, 3, 1, rng)             # 4x4
        self.p2 = MaksPool2()                          # -> 2x2
        self.V = rng.normal(0, np.sqrt(2 / 64), (64, 10))
        self.c = np.zeros(10)

    def maju(self, X):
        self.A1 = self.k1.maju(X)
        H1 = self.p1.maju(np.maximum(self.A1, 0))
        self.A2 = self.k2.maju(H1)
        H2 = self.p2.maju(np.maximum(self.A2, 0))
        self.f = H2.reshape(len(X), -1)                # ratakan
        return self.f @ self.V + self.c

    def mundur(self, G):                               #<2>
        self.gV, self.gc = self.f.T @ G, G.sum(0)
        G = (G @ self.V.T).reshape(-1, 16, 2, 2)
        G = self.k2.mundur(self.p2.mundur(G) * (self.A2 > 0))
        self.k1.mundur(self.p1.mundur(G) * (self.A1 > 0))

    def perbarui(self, laju):
        for lap in (self.k1, self.k2):
            lap.W -= laju * lap.gW; lap.b -= laju * lap.gb
        self.V -= laju * self.gV; self.c -= laju * self.gc

def ce(Z, y):                              # loss dan gradien logit
    Z = Z - Z.max(1, keepdims=True)
    P = np.exp(Z); P /= P.sum(1, keepdims=True)
    loss = -np.log(P[np.arange(len(y)), y]).mean()
    G = P.copy(); G[np.arange(len(y)), y] -= 1
    return loss, G / len(y)

def latih(model, X, y, epoch=20, laju=0.1, B=32, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(epoch):
        idx = rng.permutation(len(X))
        for i in range(0, len(X), B):
            j = idx[i:i + B]
            _, G = ce(model.maju(X[j]), y[j])
            model.mundur(G); model.perbarui(laju)
    return model

def akurasi(model, X, y):
    return np.mean(model.maju(X).argmax(1) == y)

if __name__ == "__main__":
    X, y, Xu, yu = data_angka()
    rng = np.random.default_rng(1)
    m = CNN(rng)
    Xc, yc = X[:5], y[:5]                  # pemeriksaan gradien  #<3>
    m.mundur(ce(m.maju(Xc), yc)[1])
    W, i, h = m.k1.W, (3, 0, 1, 2), 1e-6
    analitis = m.k1.gW[i]
    W[i] += h; lp = ce(m.maju(Xc), yc)[0]
    W[i] -= 2 * h; lm = ce(m.maju(Xc), yc)[0]
    W[i] += h
    galat = abs(analitis - (lp - lm) / (2 * h)) / abs(analitis)
    print(f"Gradien lolos pemeriksaan (galat < 1e-6): {galat < 1e-6}")
    n = sum(p.size for p in [m.k1.W, m.k1.b, m.k2.W, m.k2.b,
                             m.V, m.c])
    latih(m, X, y)
    a = akurasi(m, Xu, yu)
    print(f"CNN ({n} parameter): akurasi uji = {a:.2f}")
