import numpy as np
from mlp_kecil import MLP, Adam

rng = np.random.default_rng(0)
def busur(n):                        # data 2D dari vae_mini.py
    t = rng.uniform(0, 1.5 * np.pi, n)
    return np.c_[np.cos(t), np.sin(t)] + rng.normal(0, 0.05, (n, 2))

class Kopling:                       # kopling afin: ubah dimensi u  #<1>
    def __init__(self, c):
        self.c, self.u = c, 1 - c
        self.net = MLP([1, 32, 32, 2], rng)
    def maju(self, X):               # x -> z, simpan untuk mundur
        self.X = X
        o = self.net.maju(X[:, [self.c]])
        self.s, t = np.tanh(o[:, 0]), o[:, 1]
        Z = X.copy()
        Z[:, self.u] = X[:, self.u] * np.exp(self.s) + t
        return Z, self.s                       # s = log|det J|
    def mundur(self, dZ, B):         # dZ = dL/dz; L memuat -sum(s)
        e, xu = np.exp(self.s), self.X[:, self.u]
        ds = dZ[:, self.u] * xu * e - 1 / B    # dari -log det  #<2>
        do = np.c_[ds * (1 - self.s ** 2), dZ[:, self.u]]
        dX = dZ.copy(); dX[:, self.u] = dZ[:, self.u] * e
        dX[:, self.c] += self.net.mundur(do)[:, 0]
        return dX
    def balik(self, Z):              # z -> x, untuk sampling
        o = self.net.maju(Z[:, [self.c]])
        X = Z.copy()
        X[:, self.u] = (Z[:, self.u] - o[:, 1]) * np.exp(
            -np.tanh(o[:, 0]))
        return X

lapisan = [Kopling(k % 2) for k in range(6)]
opt = Adam(1e-3)

def nll(X):                          # -log p(x), rata-rata  #<3>
    Z, logdet = X, 0
    for l in lapisan:
        Z, s = l.maju(Z); logdet = logdet + s
    return np.mean(0.5 * (Z ** 2).sum(1) + np.log(2 * np.pi)
                   - logdet), Z

for t in range(1, 4001):
    X = busur(256)
    loss, Z = nll(X)
    dZ = Z / len(X)
    for l in reversed(lapisan): dZ = l.mundur(dZ, len(X))
    opt.langkah([pg for l in lapisan for pg in l.net.param_grad()])
    if t % 2000 == 0:
        print(f"langkah {t}: NLL latih = {loss:.2f} nat")

Xu = busur(5000)                     # pembanding: satu Gauss penuh
mu, C = Xu.mean(0), np.cov(Xu.T)
D = Xu - mu
nll_g = np.mean(0.5 * np.sum(D @ np.linalg.inv(C) * D, 1)
                + np.log(2 * np.pi) + 0.5 * np.log(np.linalg.det(C)))
print(f"NLL uji: aliran = {nll(Xu)[0]:.2f}, Gauss = {nll_g:.2f}")
X = rng.normal(size=(5000, 2))       # sampling: balik urutan  #<4>
for l in reversed(lapisan): X = l.balik(X)
r = np.linalg.norm(X, axis=1)
sudut = np.mod(np.arctan2(X[:, 1], X[:, 0]), 2 * np.pi)
print(f"sampel di busur (r 0.8-1.2, 0-270 derajat): "
      f"{np.mean((abs(r - 1) < 0.2) & (sudut < 1.5 * np.pi)):.2f}")
