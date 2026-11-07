import numpy as np

rng = np.random.default_rng(1)
n, n_uji, d = 40, 1000, 10
w_benar = rng.normal(size=d)
def data(m):
    X = rng.normal(size=(m, d))
    return X, X @ w_benar + rng.normal(0, 0.5, m)
X, y = data(n)
Xu, yu = data(n_uji)

def galat_uji(p, ulang=20):
    hasil = []
    for _ in range(ulang):
        V = rng.normal(size=(d, p)) / np.sqrt(d)   # fitur acak ReLU  #<1>
        F, Fu = np.maximum(X @ V, 0), np.maximum(Xu @ V, 0)
        w = np.linalg.pinv(F) @ y                  # norma minimum  #<2>
        hasil.append(np.mean((Fu @ w - yu) ** 2))
    return np.median(hasil)

print("  p   p/n   MSE uji")
for p in [5, 10, 20, 30, 36, 40, 44, 50, 80, 160, 400, 1000]:
    print(f"{p:4d}  {p / n:4.2f}  {galat_uji(p):8.3f}")
