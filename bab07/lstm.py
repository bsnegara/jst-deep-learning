import numpy as np

sig = lambda z: 1 / (1 + np.exp(-z))

class LSTM:
    def __init__(self, d, H, rng, bias_lupa=1.0):
        self.H = H
        self.W = rng.normal(0, 1 / np.sqrt(d + H), (H + d, 4 * H))
        self.b = np.zeros(4 * H)
        self.b[H:2 * H] = bias_lupa              # gerbang lupa  #<1>

    def maju(self, X):                            # X: (B, T, d)
        H, B, T = self.H, X.shape[0], X.shape[1]
        h, c = np.zeros((B, H)), np.zeros((B, H))
        self.cache = []
        for t in range(T):
            hx = np.hstack([h, X[:, t]])
            Z = hx @ self.W + self.b
            i, f = sig(Z[:, :H]), sig(Z[:, H:2 * H])
            o, g = sig(Z[:, 2 * H:3 * H]), np.tanh(Z[:, 3 * H:])
            c_lama = c
            c = f * c + i * g                     # sel memori  #<2>
            h = o * np.tanh(c)
            self.cache.append((hx, i, f, o, g, c_lama, c))
        return h                                  # keadaan terakhir

    def mundur(self, dh):                         # dh = dL/dh_T
        H = self.H
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        dc = np.zeros_like(dh)
        for hx, i, f, o, g, c_lama, c in reversed(self.cache):
            tc = np.tanh(c)
            dc = dc + dh * o * (1 - tc ** 2)
            dZ = np.hstack([dc * g * i * (1 - i),       # gerbang i
                            dc * c_lama * f * (1 - f),  # gerbang f
                            dh * tc * o * (1 - o),      # gerbang o
                            dc * i * (1 - g ** 2)])     # kandidat g
            self.gW += hx.T @ dZ
            self.gb += dZ.sum(0)
            dh = (dZ @ self.W.T)[:, :H]
            dc = dc * f                       # jalan tol gradien  #<3>
        return dh

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sel = LSTM(3, 5, rng)
    X, R = rng.normal(size=(4, 6, 3)), rng.normal(size=(4, 5))
    loss = lambda: np.sum(R * sel.maju(X))
    sel.maju(X); sel.mundur(R)
    num = np.zeros_like(sel.W)
    for idx in np.ndindex(sel.W.shape):
        s = sel.W[idx]
        sel.W[idx] = s + 1e-6; lp = loss()
        sel.W[idx] = s - 1e-6; lm = loss()
        sel.W[idx] = s; num[idx] = (lp - lm) / 2e-6
    rel = np.abs(sel.gW - num).max() / np.abs(num).max()
    ok = rel < 1e-6
    print(f"Gradien LSTM lolos pemeriksaan (galat < 1e-6): {ok}")
