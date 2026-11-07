import numpy as np
from lapisan_tf import Blok, LayerNorm, Linear, Adam

class Embedding:
    def __init__(self, V, n, d, rng):
        self.E = rng.normal(0, 0.1, (V, d))    # token
        self.P = rng.normal(0, 0.1, (n, d))    # posisi, dipelajari  #<1>
    def maju(self, idx):
        self.idx = idx
        return self.E[idx] + self.P[:idx.shape[1]]
    def mundur(self, G):
        gE = np.zeros_like(self.E)
        np.add.at(gE, self.idx, G)             # token kembar dijumlah
        gP = np.zeros_like(self.P); gP[:G.shape[1]] = G.sum(0)
        self.grad = [gE, gP]
    def param(self): return [self.E, self.P]

class GPT:
    def __init__(self, V, n, d=32, h=4, L=2, seed=0):
        rng = np.random.default_rng(seed)
        self.emb = Embedding(V, n, d, rng)
        self.blok = [Blok(d, h, rng) for _ in range(L)]
        self.ln, self.kepala = LayerNorm(d), Linear(d, V, rng)
        lap = [self.emb, self.ln, self.kepala]
        for b in self.blok: lap += b.anak()
        self.opt = Adam(lap)

    def maju(self, idx):                       # idx: (B, n) -> logit
        X = self.emb.maju(idx)
        for b in self.blok: X = b.maju(X)
        return self.kepala.maju(self.ln.maju(X))

    def langkah(self, idx, y, bobot=None):     # satu langkah Adam
        Z = self.maju(idx)
        Z = Z - Z.max(-1, keepdims=True)
        P = np.exp(Z); P /= P.sum(-1, keepdims=True)
        w = np.ones(y.shape) if bobot is None else bobot   #<2>
        loss = -(w * np.log(np.take_along_axis(
            P, y[..., None], -1)[..., 0])).sum() / w.sum()
        G = (P - np.eye(P.shape[-1])[y]) * w[..., None] / w.sum()
        G = self.ln.mundur(self.kepala.mundur(G))
        for b in reversed(self.blok): G = b.mundur(G)
        self.emb.mundur(G)
        self.opt.langkah()
        return loss
