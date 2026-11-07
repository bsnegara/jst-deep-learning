"""MLP kecil dengan backpropagation manual dan Adam (Bab 3-4)."""
import numpy as np

class MLP:
    def __init__(self, dims, rng, akt="tanh"):
        self.W = [rng.normal(0, np.sqrt(1 / a), (a, b))
                  for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros(b) for b in dims[1:]]
        self.akt = akt

    def _phi(self, A):
        return np.tanh(A) if self.akt == "tanh" else np.maximum(A, 0)

    def _dphi(self, H):
        return 1 - H ** 2 if self.akt == "tanh" else (H > 0) * 1.0

    def maju(self, X):
        self.H = [X]
        for k, (W, b) in enumerate(zip(self.W, self.b)):
            A = self.H[-1] @ W + b
            last = k == len(self.W) - 1
            self.H.append(A if last else self._phi(A))
        return self.H[-1]                     # keluaran linear

    def mundur(self, G):                      # G = dL/dkeluaran
        self.gW, self.gb = [], []
        for k in reversed(range(len(self.W))):
            self.gW.insert(0, self.H[k].T @ G)
            self.gb.insert(0, G.sum(0))
            G = G @ self.W[k].T
            if k > 0:
                G = G * self._dphi(self.H[k])
        return G                              # dL/dmasukan

    def param_grad(self):
        return list(zip(self.W + self.b, self.gW + self.gb))

class Adam:
    def __init__(self, laju=1e-3, b1=0.9, b2=0.999):
        self.laju, self.b1, self.b2 = laju, b1, b2
        self.t, self.s = 0, {}

    def langkah(self, pasangan):
        self.t += 1
        for p, g in pasangan:
            m, v = self.s.get(id(p), (0, 0))
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.s[id(p)] = (m, v)
            mh = m / (1 - self.b1 ** self.t)
            vh = v / (1 - self.b2 ** self.t)
            p -= self.laju * mh / (np.sqrt(vh) + 1e-8)
