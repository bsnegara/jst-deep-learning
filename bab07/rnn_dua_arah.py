import numpy as np
from rnn_urutan import RNNUrutan, Adam, softmax_ce

def data(B, T, rng):              # y_t = x_{t+2}: butuh masa depan  #<1>
    s = rng.integers(0, 2, (B, T))
    return np.eye(2)[s], s[:, 2:]   # target hanya untuk t < T-2

def latih(dua_arah, T=10, H=16, langkah=1500, seed=0):
    rng = np.random.default_rng(seed)
    maju, mundur = RNNUrutan(2, H, rng), RNNUrutan(2, H, rng)
    k = 2 if dua_arah else 1
    V, c = rng.normal(0, 0.3, (k * H, 2)), np.zeros(2)
    lap = [maju, mundur] if dua_arah else [maju]
    opt = Adam([p for l in lap for p in (l.W, l.b)] + [V, c])
    def keluaran(X):
        Hs = [maju.maju(X)]
        if dua_arah:                              # baca terbalik  #<2>
            Hs.append(mundur.maju(X[:, ::-1])[:, ::-1])
        return np.concatenate(Hs, -1)
    for _ in range(langkah):
        X, y = data(32, T, rng)
        Hc = keluaran(X)[:, :T - 2]
        _, G = softmax_ce(Hc @ V + c, y)
        dH = np.zeros((32, T, k * H)); dH[:, :T - 2] = G @ V.T
        maju.mundur(dH[..., :H])
        if dua_arah:
            mundur.mundur(dH[..., H:][:, ::-1])   # balik lagi  #<3>
        g = [p for l in lap for p in (l.gW, l.gb)]
        opt.langkah(g + [Hc.reshape(-1, k * H).T @ G.reshape(-1, 2),
                         G.sum((0, 1))])
    X, y = data(1000, T, rng)
    Z = keluaran(X)[:, :T - 2] @ V + c
    return np.mean(Z.argmax(-1) == y)

print(f"RNN satu arah : akurasi = {latih(False):.2f}")
print(f"RNN dua arah  : akurasi = {latih(True):.2f}")
