import numpy as np
from mlp_kecil import MLP, Adam

sig = lambda a: 1 / (1 + np.exp(-a))
rng = np.random.default_rng(0)

# 1) Diskriminator optimal: D*(x) = p_data / (p_data + p_g)  #<1>
pd = lambda x: np.exp(-(x - 2) ** 2 / 2)      # data   N(2, 1)
pg = lambda x: np.exp(-x ** 2 / 2)            # model  N(0, 1)
w = np.zeros(3)                       # logistik pada (1, x, x^2)
for _ in range(3000):
    xd, xg = rng.normal(2, 1, 512), rng.normal(0, 1, 512)
    F = np.c_[np.ones(1024), np.r_[xd, xg], np.r_[xd, xg] ** 2]
    y = np.r_[np.ones(512), np.zeros(512)]
    w -= 0.1 * F.T @ (sig(F @ w) - y) / 1024
for x in [-1.0, 0.0, 1.0, 2.0, 3.0]:
    D = sig(w @ [1, x, x * x])
    print(f"x = {x:+.0f}   D terlatih = {D:.3f}   "
          f"D* teori = {pd(x) / (pd(x) + pg(x)):.3f}")

# 2) GAN kecil pada campuran dua Gaussian di -2 dan +2  #<2>
def data(n):
    pusat = rng.choice([-2, 2], (n, 1))
    return pusat + rng.normal(0, 0.3, (n, 1))
G, D = MLP([1, 32, 32, 1], rng), MLP([1, 32, 32, 1], rng)
oG, oD = Adam(1e-3, b1=0.5), Adam(1e-3, b1=0.5)
for t in range(1, 6001):
    x, xg = data(128), G.maju(rng.normal(size=(128, 1)))
    for X, y in [(x, 1.0), (xg, 0.0)]:          # langkah D  #<3>
        p = sig(D.maju(X)); D.mundur((p - y) / 128)
        oD.langkah(D.param_grad())
    xg = G.maju(rng.normal(size=(128, 1)))      # langkah G  #<4>
    p = sig(D.maju(xg))
    dx = D.mundur((p - 1.0) / 128)              # non-saturating
    G.mundur(dx); oG.langkah(G.param_grad())
    if t % 1500 == 0:
        s = G.maju(rng.normal(size=(4000, 1)))[:, 0]
        kiri, kanan = np.mean(abs(s + 2) < 1), np.mean(abs(s - 2) < 1)
        print(f"langkah {t}: dekat -2: {kiri:.2f}, dekat +2: "
              f"{kanan:.2f}, lainnya: {1 - kiri - kanan:.2f}")
