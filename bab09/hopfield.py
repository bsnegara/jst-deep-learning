import numpy as np

pola_teks = ["X...X.X.X...X...X.X.X...X",       # huruf X  #<1>
             "XXXXX..X....X....X....X..",       # huruf T
             "X...XX...XXXXXXX...XX...X"]       # huruf H
P = np.array([[1 if c == "X" else -1 for c in s] for s in pola_teks])
N = P.shape[1]

W = P.T @ P / N                        # aturan Hebb  #<2>
np.fill_diagonal(W, 0)

energi = lambda s: -0.5 * s @ W @ s    #<3>

def ingat(s, rng, sapuan=3):
    s = s.copy()
    for k in range(sapuan):
        for i in rng.permutation(N):    # pembaruan asinkron  #<4>
            s[i] = 1 if W[i] @ s >= 0 else -1
        print(f"  sapuan {k + 1}: energi = {energi(s):.3f}")
    return s

tampil = lambda s: "\n".join(
    "  " + "".join("#" if v > 0 else "." for v in s[r * 5:r * 5 + 5])
    for r in range(5))

rng = np.random.default_rng(0)
rusak = P[1].copy()
balik = rng.choice(N, 7, replace=False)          # 7 dari 25 piksel
rusak[balik] *= -1
print(f"Pola T yang dirusak, energi = {energi(rusak):.3f}")
print(tampil(rusak))
hasil = ingat(rusak, rng)
print("Hasil:")
print(tampil(hasil))
print("Sama dengan pola T asli?", bool(np.all(hasil == P[1])))
