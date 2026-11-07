import numpy as np
from rnn_urutan import RNNUrutan, Adam, softmax_ce

K, H = 8, 32                      # 8 simbol; token 8 = "mulai"
rng = np.random.default_rng(0)
enc, dec = RNNUrutan(K, H, rng), RNNUrutan(K + 1, H, rng)
V, c = rng.normal(0, 0.3, (H, K)), np.zeros(K)
opt = Adam([enc.W, enc.b, dec.W, dec.b, V, c], 0.005)

def satu_panas(s, n): return np.eye(n)[s]

for langkah in range(4000):
    T = int(rng.integers(3, 13))                # panjang acak 3..12
    s = rng.integers(0, K, (32, T))
    y = s[:, ::-1]                      # sasaran: urutan dibalik
    mulai = np.full((32, 1), K)
    masuk = np.hstack([mulai, y[:, :-1]])       # teacher forcing  #<1>
    ctx = enc.maju(satu_panas(s, K))[:, -1]     # vektor tunggal c  #<2>
    Hd = dec.maju(satu_panas(masuk, K + 1), h0=ctx)
    _, G = softmax_ce(Hd @ V + c, y)
    dctx = dec.mundur(G @ V.T)                  # gradien ke c ...
    dHe = np.zeros((32, T, H)); dHe[:, -1] = dctx
    enc.mundur(dHe)                             # ... lalu ke penyandi
    opt.langkah([enc.gW, enc.gb, dec.gW, dec.gb,
                 Hd.reshape(-1, H).T @ G.reshape(-1, K),
                 G.sum((0, 1))])

def urai(s):                            # serakah, tanpa teacher  #<3>
    h = enc.maju(satu_panas(s, K))[:, -1]
    tok, hasil = np.full(len(s), K), []
    for _ in range(s.shape[1]):
        h = dec.maju(satu_panas(tok, K + 1)[:, None], h0=h)[:, 0]
        tok = (h @ V + c).argmax(1); hasil.append(tok)
    return np.stack(hasil, 1)

print(" T   akurasi token   urutan benar penuh")
for T in [3, 6, 9, 12]:
    s = np.random.default_rng(T).integers(0, K, (500, T))
    tebak = urai(s)
    print(f"{T:2d}   {np.mean(tebak == s[:, ::-1]):13.2f}"
          f"   {np.mean((tebak == s[:, ::-1]).all(1)):18.2f}")
