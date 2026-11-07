import numpy as np
from lstm import LSTM

class RNN:                                    # RNN tanh biasa
    def __init__(self, d, H, rng):
        self.H = H
        self.W = rng.normal(0, 1 / np.sqrt(d + H), (H + d, H))
        self.b = np.zeros(H)
    def maju(self, X):
        h, self.cache = np.zeros((len(X), self.H)), []
        for t in range(X.shape[1]):
            hx = np.hstack([h, X[:, t]])
            h = np.tanh(hx @ self.W + self.b)
            self.cache.append((hx, h))
        return h
    def mundur(self, dh):
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        for hx, h in reversed(self.cache):
            da = dh * (1 - h ** 2)
            self.gW += hx.T @ da; self.gb += da.sum(0)
            dh = (da @ self.W.T)[:, :self.H]
        return dh

def data(B, T, rng):          # simbol hanya di t=0, lalu derau saja
    y = rng.integers(0, 2, B)                 #<1>
    X = np.zeros((B, T, 3)); X[np.arange(B), 0, y] = 1
    X[:, :, 2] = rng.normal(0, 1, (B, T))     # derau pengalih
    return X, y

def latih(Kelas, T, langkah=1500, H=16, seed=0, **kw):
    rng = np.random.default_rng(seed)
    sel, V = Kelas(3, H, rng, **kw), rng.normal(0, 0.3, (H, 2))
    param = [sel.W, sel.b, V]
    m = [np.zeros_like(p) for p in param]
    v = [np.zeros_like(p) for p in param]
    for t in range(1, langkah + 1):           # Adam
        X, y = data(64, T, rng)
        h = sel.maju(X); Z = h @ V
        P = np.exp(Z - Z.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(64), y] -= 1; G /= 64
        gV = h.T @ G; sel.mundur(G @ V.T)
        gs = [sel.gW, sel.gb, gV]
        nrm = np.sqrt(sum((g * g).sum() for g in gs))     #<2>
        gs = [g / max(1.0, nrm) for g in gs]
        for p, g, mi, vi in zip(param, gs, m, v):
            mi[:] = 0.9 * mi + 0.1 * g
            vi[:] = 0.999 * vi + 0.001 * g * g
            m_hat, v_hat = mi / (1 - 0.9**t), vi / (1 - 0.999**t)
            p -= 0.003 * m_hat / (np.sqrt(v_hat) + 1e-8)
    X, y = data(2000, T, rng)
    return np.mean((sel.maju(X) @ V).argmax(1) == y)

def aliran(Kelas, T, H=16, seed=0, **kw):    # gradien di awal  #<3>
    rng = np.random.default_rng(seed)
    sel = Kelas(3, H, rng, **kw)
    X, _ = data(256, T, rng)
    sel.maju(X)
    dh0 = sel.mundur(np.ones((256, H)))
    return np.linalg.norm(dh0) / np.linalg.norm(np.ones((256, H)))

print("Norma gradien yang sampai ke t=0 (sebelum pelatihan):")
print("   T   RNN biasa   LSTM b_f=1   LSTM b_f=3")
for T in [10, 30, 100]:
    print(f"{T:4d}   {aliran(RNN, T):9.1e}   {aliran(LSTM, T):10.1e}"
          f"   {aliran(LSTM, T, bias_lupa=3.0):10.1e}")

print("Berhasil (akurasi > 0.9) dari 5 seed, T = 100:")      #<4>
for nama, Kelas, kw in [("RNN biasa ", RNN, {}),
                        ("LSTM b_f=3", LSTM, {"bias_lupa": 3.0})]:
    n = sum(latih(Kelas, 100, seed=s, **kw) > 0.9 for s in range(5))
    print(f"  {nama} : {n}/5")
