import numpy as np

rng = np.random.default_rng(0)
n = 500
x1 = rng.normal(0, 1, n)                 # fitur berskala satuan
x2 = rng.normal(0, 100, n)               # fitur berskala ratusan
y = 2 * x1 + 0.03 * x2 + rng.normal(0, 0.1, n)

def langkah_sampai_konvergen(X, y, tol=1e-6, maks=200_000):
    H = X.T @ X / len(y)                 # Hessian loss kuadrat  #<1>
    lam = np.linalg.eigvalsh(H)
    laju = 1 / lam.max()                 # laju aman terbesar  #<2>
    w_opt = np.linalg.solve(H, X.T @ y / len(y))
    w = np.zeros(X.shape[1])
    for t in range(1, maks + 1):
        w -= laju * X.T @ (X @ w - y) / len(y)
        if np.linalg.norm(w - w_opt) < tol * np.linalg.norm(w_opt):
            return lam.max() / lam.min(), t
    return lam.max() / lam.min(), None

X = np.c_[x1, x2]
X = X - X.mean(0)                        # pusatkan
Xs = (X - X.mean(0)) / X.std(0)          # standardisasi  #<3>
for nama, M in [("asli", X), ("distandarkan", Xs)]:
    kappa, t = langkah_sampai_konvergen(M, y - y.mean())
    print(f"{nama:13s} kappa = {kappa:9.1f}   langkah = {t}")
