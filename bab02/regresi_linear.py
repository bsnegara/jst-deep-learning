import numpy as np

rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 2))
y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + rng.normal(0, 0.3, n)

# 1) Solusi tertutup: persamaan normal  #<1>
Xb = np.hstack([X, np.ones((n, 1))])          # serap bias
w_tutup = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)

# 2) Penurunan gradien pada loss kuadrat rata-rata  #<2>
w = np.zeros(3)
laju = 0.1
for langkah in range(200):
    galat = Xb @ w - y
    grad = Xb.T @ galat / n                   # turunan 1/2 MSE
    w -= laju * grad

print("Persamaan normal :", np.round(w_tutup, 4))
print("Penurunan gradien:", np.round(w, 4))
mse = np.mean((Xb @ w - y) ** 2)
print(f"MSE akhir        : {mse:.4f}")
