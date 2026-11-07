import numpy as np

rng = np.random.default_rng(2)
m, n, r = 40, 30, 3                     # pengguna, item, rank
R_benar = rng.normal(size=(m, r)) @ rng.normal(size=(r, n))
amati = rng.random((m, n)) < 0.3        # hanya 30% rating terlihat  #<1>
uji = ~amati

U = rng.normal(0, 0.1, (m, r))
V = rng.normal(0, 0.1, (n, r))
lam, laju = 0.05, 0.02
for epoch in range(2001):
    E = (U @ V.T - R_benar) * amati     # hanya sel terlihat  #<2>
    gU = E @ V + lam * U
    gV = E.T @ U + lam * V
    U -= laju * gU; V -= laju * gV
    if epoch % 500 == 0:
        rmse = lambda M: np.sqrt(np.mean((U @ V.T - R_benar)[M] ** 2))
        print(f"epoch {epoch:4d}  RMSE latih = {rmse(amati):.3f}"
              f"  RMSE uji = {rmse(uji):.3f}")
