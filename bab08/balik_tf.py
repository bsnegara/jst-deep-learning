import numpy as np
from gpt_mini import GPT

K, SEP, PAD, n = 8, 8, 9, 25                 # 8 simbol + 2 khusus
rng = np.random.default_rng(0)
gpt = GPT(K + 2, n, d=32, h=4, L=2)

def contoh(B, T):                            # s SEP balik(s) PAD...  #<1>
    s = rng.integers(0, K, (B, T))
    X = np.full((B, n + 1), PAD)
    X[:, :T], X[:, T], X[:, T + 1:2 * T + 1] = s, SEP, s[:, ::-1]
    w = np.zeros((B, n)); w[:, T:2 * T] = 1  # loss hanya bagian balik
    return X[:, :-1], X[:, 1:], w, s

for t in range(3000):
    X, y, w, _ = contoh(32, int(rng.integers(3, 13)))
    gpt.langkah(X, y, w)

print(" T   akurasi token   urutan benar penuh")
for T in [3, 6, 9, 12]:
    X, _, _, s = contoh(500, T)
    X = X.copy(); X[:, T + 1:] = PAD         # hapus jawaban  #<2>
    for j in range(T):                       # urai serakah
        z = gpt.maju(X)[:, T + j]
        X[:, T + 1 + j] = z[:, :K].argmax(1)
    benar = X[:, T + 1:2 * T + 1] == s[:, ::-1]
    a, b = benar.mean(), benar.all(1).mean()
    print(f"{T:2d}   {a:13.2f}   {b:18.2f}")
