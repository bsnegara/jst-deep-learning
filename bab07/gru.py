import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))

class GRU:
    def __init__(self, d, H, rng, bias_z=0.0):
        self.H, s = H, 1 / np.sqrt(d + H)
        self.Wz, self.Wr, self.Wh = (rng.normal(0, s, (H + d, H))
                                     for _ in range(3))
        self.bz = np.full(H, bias_z)       # z kecil = salin  #<1>
        self.br, self.bh = np.zeros(H), np.zeros(H)

    def maju(self, X):                     # X: (B, T, d)
        h = np.zeros((X.shape[0], self.H))
        self.cache = []
        for t in range(X.shape[1]):
            x = X[:, t]; hx = np.hstack([h, x])
            z = sig(hx @ self.Wz + self.bz)
            r = sig(hx @ self.Wr + self.br)
            rhx = np.hstack([r * h, x])
            hh = np.tanh(rhx @ self.Wh + self.bh)
            self.cache.append((h, hx, rhx, z, r, hh))
            h = (1 - z) * h + z * hh       # interpolasi  #<2>
        return h

    def mundur(self, dh):
        H = self.H
        g = {k: np.zeros_like(getattr(self, k))
             for k in ["Wz", "Wr", "Wh", "bz", "br", "bh"]}
        for h, hx, rhx, z, r, hh in reversed(self.cache):
            dah = dh * z * (1 - hh ** 2)            # lewat tanh  #<3>
            daz = dh * (hh - h) * z * (1 - z)       # lewat z
            drh = (dah @ self.Wh.T)[:, :H]          # ke r*h
            dar = drh * h * r * (1 - r)             # lewat r
            g["Wh"] += rhx.T @ dah; g["bh"] += dah.sum(0)
            g["Wz"] += hx.T @ daz;  g["bz"] += daz.sum(0)
            g["Wr"] += hx.T @ dar;  g["br"] += dar.sum(0)
            dh = (dh * (1 - z) + drh * r            # empat jalur  #<4>
                  + (daz @ self.Wz.T)[:, :H]
                  + (dar @ self.Wr.T)[:, :H])
        self.g = g
        return dh

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sel = GRU(3, 5, rng)
    X, R = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 5))
    loss = lambda: np.sum(R * sel.maju(X))
    sel.maju(X); sel.mundur(R)
    galat = 0
    for k in ["Wz", "Wr", "Wh", "bz"]:
        P, G = getattr(sel, k), sel.g[k]
        for idx in np.ndindex(P.shape):
            s = P[idx]
            P[idx] = s + 1e-6; lp = loss()
            P[idx] = s - 1e-6; lm = loss(); P[idx] = s
            galat = max(galat, abs(G[idx] - (lp - lm) / 2e-6))
    print(f"Gradien GRU lolos pemeriksaan: {galat < 1e-7}")

    print("Norma gradien di t=0 (sebelum pelatihan):")
    print("   T   GRU b_z=0   GRU b_z=-3")
    for T in [10, 30, 100]:
        n = []
        for bz in [0.0, -3.0]:
            rng = np.random.default_rng(0)
            sel = GRU(3, 16, rng, bias_z=bz)
            sel.maju(rng.normal(0, 1, (256, T, 3)))
            dh0 = sel.mundur(np.ones((256, 16)))
            n.append(np.linalg.norm(dh0) / np.linalg.norm(
                np.ones((256, 16))))
        print(f"{T:4d}   {n[0]:9.1e}   {n[1]:10.1e}")
