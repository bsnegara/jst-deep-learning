import numpy as np

rng = np.random.default_rng(0)
n, d = 30, 20
w_benar = np.zeros(d)
w_benar[:3] = [3, -2, 1.5]               # hanya 3 fitur berguna
def data(m):
    X = rng.normal(size=(m, d))
    return X, X @ w_benar + rng.normal(0, 1, m)
X, y = data(n)
Xu, yu = data(2000)

def ridge(lam):                       # solusi tertutup  #<1>
    return np.linalg.solve(X.T @ X + lam * n * np.eye(d), X.T @ y)

def lasso(lam, langkah=5000):         # ISTA (proksimal)  #<2>
    w = np.zeros(d)
    eta = n / np.linalg.eigvalsh(X.T @ X).max()
    for _ in range(langkah):
        w = w - eta * X.T @ (X @ w - y) / n
        w = np.sign(w) * np.maximum(np.abs(w) - eta*lam, 0)  #<3>
    return w

w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
for nama, w in [("kuadrat terkecil", w_ols),
                ("ridge (L2)", ridge(0.1)),
                ("lasso (L1)", lasso(0.1))]:
    mse = np.mean((Xu @ w - yu) ** 2)
    nol = np.sum(np.abs(w) < 1e-6)
    print(f"{nama:17s} MSE = {mse:4.2f}  nol = {nol:2d}  "
          f"w[:4] = {np.round(w[:4], 1)}")
