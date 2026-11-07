import numpy as np
from mlp_kecil import MLP, Adam

rng = np.random.default_rng(0)

def w1(a, b):                       # Wasserstein-1 di 1D: urutkan  #<1>
    return np.mean(np.abs(np.sort(a) - np.sort(b)))

def jsd(a, b, lebar=0.01):          # JSD dari histogram
    tepi = np.arange(min(a.min(), b.min()), max(a.max(), b.max())
                     + lebar, lebar)
    p = np.histogram(a, tepi)[0] / len(a)
    q = np.histogram(b, tepi)[0] / len(b)
    m = (p + q) / 2
    kl = lambda u: np.sum(u[u > 0] * np.log(u[u > 0] / m[u > 0]))
    return (kl(p) + kl(q)) / 2

print("geser   JSD    W1      (log 2 = 0.69)")
x = rng.uniform(0, 1, 200000)                  # data U[0, 1]
for geser in [0.5, 1.0, 2.0, 4.0]:
    y = rng.uniform(0, 1, 200000) + geser      # model U[g, g+1]
    print(f"{geser:4.1f}   {jsd(x, y):.2f}   {w1(x, y):.2f}")

def data(n):                        # dua mode, sama dengan gan_1d.py
    pusat = rng.choice([-2, 2], (n, 1))
    return pusat + rng.normal(0, 0.3, (n, 1))

G, opt = MLP([1, 32, 32, 1], rng), Adam(1e-3)
for t in range(1, 6001):            # generator meminimumkan W1  #<2>
    xd, xg = data(128)[:, 0], G.maju(rng.normal(size=(128, 1)))[:, 0]
    ig, idd = np.argsort(xg), np.argsort(xd)
    g = np.zeros(128)
    g[ig] = np.sign(xg[ig] - xd[idd]) / 128    # pasangan terurut  #<3>
    G.mundur(g[:, None]); opt.langkah(G.param_grad())
    if t % 1500 == 0:
        s = G.maju(rng.normal(size=(4000, 1)))[:, 0]
        kiri, kanan = np.mean(abs(s + 2) < 1), np.mean(abs(s - 2) < 1)
        print(f"langkah {t}: dekat -2: {kiri:.2f}, dekat +2: "
              f"{kanan:.2f}, lainnya: {1 - kiri - kanan:.2f}")
