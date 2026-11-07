import numpy as np

class Linear:
    def __init__(self, a, b, rng):
        self.W = rng.normal(0, a ** -0.5, (a, b))
        self.b = np.zeros(b)
    def maju(self, X):
        self.X = X; return X @ self.W + self.b
    def mundur(self, G):
        X2, G2 = self.X.reshape(-1, self.X.shape[-1]), G.reshape(
            -1, G.shape[-1])
        self.grad = [X2.T @ G2, G2.sum(0)]
        return G @ self.W.T
    def param(self): return [self.W, self.b]

class LayerNorm:
    def __init__(self, d):
        self.g, self.b = np.ones(d), np.zeros(d)
    def maju(self, X):
        mu, var = X.mean(-1, keepdims=True), X.var(-1, keepdims=True)
        self.s = np.sqrt(var + 1e-5)
        self.Xh = (X - mu) / self.s
        return self.g * self.Xh + self.b
    def mundur(self, G):                              #<1>
        Xh, dXh = self.Xh, G * self.g
        self.grad = [(G * Xh).reshape(-1, len(self.g)).sum(0),
                     G.reshape(-1, len(self.g)).sum(0)]
        return (dXh - dXh.mean(-1, keepdims=True)
                - Xh * (dXh * Xh).mean(-1, keepdims=True)) / self.s
    def param(self): return [self.g, self.b]

class AtensiKausal:
    def __init__(self, d, h, rng):
        self.h = h
        self.q, self.k, self.v, self.o = (Linear(d, d, rng)
                                          for _ in range(4))
    def pisah(self, X):                     # (B,n,d) -> (B,h,n,dk)
        B, n, d = X.shape
        X = X.reshape(B, n, self.h, d // self.h)
        return X.transpose(0, 2, 1, 3)
    def gabung(self, X):                    # kebalikan pisah
        B, h, n, dk = X.shape
        return X.transpose(0, 2, 1, 3).reshape(B, n, h * dk)
    def maju(self, X):
        Q, K, V = (self.pisah(l.maju(X))
                   for l in (self.q, self.k, self.v))
        n, dk = X.shape[1], Q.shape[-1]
        S = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dk)
        S = np.where(np.tril(np.ones((n, n), bool)), S, -np.inf)  #<2>
        P = np.exp(S - S.max(-1, keepdims=True))
        self.P = P / P.sum(-1, keepdims=True)
        self.QKV = Q, K, V
        return self.o.maju(self.gabung(self.P @ V))
    def mundur(self, G):                    # rumus atensi_mundur  #<3>
        Q, K, V, P = *self.QKV, self.P
        dO = self.pisah(self.o.mundur(G))
        dV = P.transpose(0, 1, 3, 2) @ dO
        dP = dO @ V.transpose(0, 1, 3, 2)
        dS = P * (dP - (dP * P).sum(-1, keepdims=True)) / np.sqrt(
            Q.shape[-1])
        dQ, dK = dS @ K, dS.transpose(0, 1, 3, 2) @ Q
        return sum(l.mundur(self.gabung(D)) for l, D in
                   zip((self.q, self.k, self.v), (dQ, dK, dV)))
    def anak(self): return [self.q, self.k, self.v, self.o]

class Blok:                                 # pre-LN, dua koneksi sisa
    def __init__(self, d, h, rng):
        self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)
        self.atn = AtensiKausal(d, h, rng)
        self.f1 = Linear(d, 4 * d, rng)       # FFN selebar 4d
        self.f2 = Linear(4 * d, d, rng)
    def maju(self, X):
        X = X + self.atn.maju(self.ln1.maju(X))
        self.A = self.f1.maju(self.ln2.maju(X))
        return X + self.f2.maju(np.maximum(self.A, 0))
    def mundur(self, G):                # sisa: gradien bercabang  #<4>
        dF = self.f1.mundur(self.f2.mundur(G) * (self.A > 0))
        G = G + self.ln2.mundur(dF)
        return G + self.ln1.mundur(self.atn.mundur(G))
    def anak(self):
        return [self.ln1, self.ln2, *self.atn.anak(),
                self.f1, self.f2]

class Adam:
    def __init__(self, lapisan, laju=3e-3):
        self.lap, self.laju, self.t = lapisan, laju, 0
        nol = lambda: [[np.zeros_like(p) for p in l.param()]
                       for l in lapisan]
        self.m, self.v = nol(), nol()
    def langkah(self):
        self.t += 1
        for l, ms, vs in zip(self.lap, self.m, self.v):
            for p, g, m, v in zip(l.param(), l.grad, ms, vs):
                m[:] = 0.9 * m + 0.1 * g
                v[:] = 0.999 * v + 0.001 * g * g
                mh = m / (1 - 0.9 ** self.t)
                vh = v / (1 - 0.999 ** self.t)
                p -= self.laju * mh / (np.sqrt(vh) + 1e-8)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    blok = Blok(8, 2, rng)
    X, R = rng.normal(size=(2, 5, 8)), rng.normal(size=(2, 5, 8))
    loss = lambda: np.sum(R * blok.maju(X))
    blok.maju(X); dX = blok.mundur(R)
    galat = 0.0
    for l in blok.anak():
        for p, g in zip(l.param(), l.grad):
            for idx in list(np.ndindex(p.shape))[:6]:   # sampel
                s = p[idx]
                p[idx] = s + 1e-6; lp = loss()
                p[idx] = s - 1e-6; lm = loss(); p[idx] = s
                galat = max(galat, abs(g[idx] - (lp - lm) / 2e-6))
    for idx in [(0, 1, 2), (1, 4, 7)]:
        s = X[idx]
        X[idx] = s + 1e-6; lp = loss()
        X[idx] = s - 1e-6; lm = loss(); X[idx] = s
        galat = max(galat, abs(dX[idx] - (lp - lm) / 2e-6))
    print(f"Gradien blok lolos pemeriksaan: {galat < 1e-6}")
