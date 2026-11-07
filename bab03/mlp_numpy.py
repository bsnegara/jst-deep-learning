import numpy as np

class Padat:                                  # lapisan dense
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out))
        self.b = np.zeros(n_out)
    def maju(self, X):
        self.X = X                     # simpan untuk mundur  #<1>
        return X @ self.W + self.b
    def mundur(self, G):               # G = dL/dkeluaran
        self.gW = self.X.T @ G         #<2>
        self.gb = G.sum(axis=0)
        return G @ self.W.T            # dL/dmasukan

class ReLU:
    def maju(self, A):
        self.aktif = A > 0
        return A * self.aktif
    def mundur(self, G):
        return G * self.aktif          # Phi'(a) per elemen

def softmax_ce(Z, y):                # loss dan dL/dZ sekaligus
    Z = Z - Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.mean(np.log(P[np.arange(n), y]))
    G = P.copy(); G[np.arange(n), y] -= 1
    return loss, G / n

def spiral(n, K, rng):                        # data spiral K lengan
    X, y = [], []
    for k in range(K):
        r = np.linspace(0.05, 1, n)
        t = np.linspace(k * 4.2, (k + 1) * 4.2, n)
        t = t + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)]); y += [k] * n
    return np.vstack(X), np.array(y)

rng = np.random.default_rng(0)
X, y = spiral(100, 3, rng)
lapisan = [Padat(2, 64, rng), ReLU(), Padat(64, 64, rng), ReLU(),
           Padat(64, 3, rng)]

def loss_dan_mundur():
    H = X
    for l in lapisan:
        H = l.maju(H)
    loss, G = softmax_ce(H, y)
    for l in reversed(lapisan):               # rekursi delta  #<3>
        G = l.mundur(G)
    return loss

# Periksa gradien: galat relatif analitis vs beda hingga  #<4>
loss_dan_mundur()
W = lapisan[0].W
analitis = lapisan[0].gW.copy()
numerik = np.zeros_like(W)
for idx in np.ndindex(W.shape):
    simpan = W[idx]
    W[idx] = simpan + 1e-5; lp = loss_dan_mundur()
    W[idx] = simpan - 1e-5; lm = loss_dan_mundur()
    W[idx] = simpan
    numerik[idx] = (lp - lm) / 2e-5
rel = (np.abs(analitis - numerik).max()
       / max(np.abs(analitis).max(), np.abs(numerik).max()))
print(f"Galat relatif gradien W1: {rel:.1e}")

for langkah in range(3001):
    loss = loss_dan_mundur()
    for l in lapisan:
        if isinstance(l, Padat):
            l.W -= 0.5 * l.gW; l.b -= 0.5 * l.gb
    if langkah % 1000 == 0:
        print(f"langkah {langkah:4d}  loss = {loss:.4f}")
H = X
for l in lapisan:
    H = l.maju(H)
print(f"Akurasi latih: {np.mean(H.argmax(axis=1) == y):.3f}")
