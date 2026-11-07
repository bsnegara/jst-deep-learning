import numpy as np

class RNNUrutan:
    """RNN tanh yang mengembalikan keadaan di SEMUA langkah waktu."""
    def __init__(self, d, H, rng):
        self.H = H
        self.W = rng.normal(0, 1 / np.sqrt(d + H), (H + d, H))
        self.b = np.zeros(H)

    def maju(self, X, h0=None):                 # X: (B, T, d)
        B, T, _ = X.shape
        h = np.zeros((B, self.H)) if h0 is None else h0
        self.cache, Hs = [], []
        for t in range(T):
            hx = np.hstack([h, X[:, t]])
            h = np.tanh(hx @ self.W + self.b)
            self.cache.append((hx, h)); Hs.append(h)
        return np.stack(Hs, 1)                  # (B, T, H)

    def mundur(self, dHs):                      # dHs: (B, T, H)  #<1>
        self.gW = np.zeros_like(self.W)
        self.gb = np.zeros_like(self.b)
        dh = np.zeros_like(dHs[:, 0])
        for t in reversed(range(dHs.shape[1])):
            hx, h = self.cache[t]
            da = (dh + dHs[:, t]) * (1 - h ** 2)  # dari atas + kanan
            self.gW += hx.T @ da; self.gb += da.sum(0)
            dh = (da @ self.W.T)[:, :self.H]
        return dh                               # dL/dh0  #<2>

class Adam:
    def __init__(self, param, laju=0.01):
        self.p, self.laju, self.t = param, laju, 0
        self.m = [np.zeros_like(p) for p in param]
        self.v = [np.zeros_like(p) for p in param]
    def langkah(self, grad):
        self.t += 1
        for p, g, m, v in zip(self.p, grad, self.m, self.v):
            g = g / max(1.0, np.linalg.norm(g))  # clipping per tensor
            m[:] = 0.9 * m + 0.1 * g
            v[:] = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** self.t)
            vh = v / (1 - 0.999 ** self.t)
            p -= self.laju * mh / (np.sqrt(vh) + 1e-8)

def softmax_ce(Z, y):                           # Z: (..., K)
    Z = Z - Z.max(-1, keepdims=True)
    P = np.exp(Z); P /= P.sum(-1, keepdims=True)
    G = P - np.eye(Z.shape[-1])[y]              # p - e_y
    return P, G / y.size

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    rnn = RNNUrutan(3, 4, rng)
    X, R = rng.normal(size=(2, 5, 3)), rng.normal(size=(2, 5, 4))
    h0 = rng.normal(size=(2, 4))
    loss = lambda: np.sum(R * rnn.maju(X, h0))
    rnn.maju(X, h0); dh0 = rnn.mundur(R)
    i, e = (2, 1), 1e-6
    rnn.W[i] += e; lp = loss(); rnn.W[i] -= 2 * e; lm = loss()
    rnn.W[i] += e
    ok_W = abs(rnn.gW[i] - (lp - lm) / (2 * e)) < 1e-6
    h0[0, 2] += e; lp = loss(); h0[0, 2] -= 2 * e; lm = loss()
    ok_h = abs(dh0[0, 2] - (lp - lm) / (2 * e)) < 1e-6
    print(f"Gradien W benar: {ok_W}, gradien h0 benar: {ok_h}")
