import numpy as np
from cnn_angka import CNN, data_angka, latih, akurasi

class MLP:                                 # 64 -> 30 -> 10
    def __init__(self, rng):
        self.W1 = rng.normal(0, np.sqrt(2 / 64), (64, 30))
        self.b1 = np.zeros(30)
        self.W2 = rng.normal(0, np.sqrt(2 / 30), (30, 10))
        self.b2 = np.zeros(10)
    def maju(self, X):
        self.x = X.reshape(len(X), -1)
        self.a = self.x @ self.W1 + self.b1
        self.h = np.maximum(self.a, 0)
        return self.h @ self.W2 + self.b2
    def mundur(self, G):
        self.g = [self.h.T @ G, G.sum(0)]
        Ga = (G @ self.W2.T) * (self.a > 0)
        self.g = [self.x.T @ Ga, Ga.sum(0)] + self.g
    def perbarui(self, laju):
        for p, g in zip([self.W1, self.b1, self.W2, self.b2], self.g):
            p -= laju * g

def geser(X, s):                     # geser mendatar s piksel  #<1>
    Y = np.roll(X, s, axis=-1)
    if s > 0: Y[..., :s] = 0
    if s < 0: Y[..., s:] = 0
    return Y

class Augmentasi:                   # bungkus: geser acak saat latih  #<2>
    def __init__(self, model, seed=0):
        self.m, self.rng = model, np.random.default_rng(seed)
        self.latih = True
    def maju(self, X):
        if self.latih:
            X = geser(X, int(self.rng.integers(-2, 3)))
        return self.m.maju(X)
    def mundur(self, G): self.m.mundur(G)
    def perbarui(self, laju): self.m.perbarui(laju)

X, y, Xu, yu = data_angka()
mlp = latih(MLP(np.random.default_rng(1)), X, y)
cnn = latih(CNN(np.random.default_rng(1)), X, y)
aug = latih(Augmentasi(CNN(np.random.default_rng(1))), X, y)
aug.latih = False
print("geser    MLP    CNN    CNN+augmentasi")
for s in [0, 1, 2]:
    a = [akurasi(m, geser(Xu, s), yu) for m in (mlp, cnn, aug)]
    print(f"  {s}     {a[0]:.2f}   {a[1]:.2f}   {a[2]:.2f}")
