import numpy as np
from mlp_kecil import MLP, Adam

def busur(n, rng):                      # data: busur lingkaran 2D
    t = rng.uniform(0, 1.5 * np.pi, n)
    X = np.c_[np.cos(t), np.sin(t)]
    return X + rng.normal(0, 0.05, X.shape)

rng = np.random.default_rng(0)
enc = MLP([2, 64, 64, 2], rng)          # -> (mu, log varians)  #<1>
dec = MLP([1, 64, 64, 2], rng)          # z -> x_topi
opt, s2 = Adam(2e-3), 0.05 ** 2         # s2: varians derau pengurai

for langkah in range(8001):
    X = busur(128, rng)
    E = enc.maju(X)
    mu, lv = E[:, :1], E[:, 1:]
    eps = rng.normal(size=mu.shape)
    z = mu + np.exp(0.5 * lv) * eps     # reparameterisasi  #<2>
    Xh = dec.maju(z)
    rekon = np.sum((Xh - X) ** 2, 1) / (2 * s2)
    kl = 0.5 * np.sum(mu**2 + np.exp(lv) - 1 - lv, 1)  #<3>
    n = len(X)
    dz = dec.mundur((Xh - X) / s2 / n)                  # mundur
    dmu = dz + mu / n
    dlv = (dz * eps * 0.5 * np.exp(0.5 * lv)
           + 0.5 * (np.exp(lv) - 1) / n)
    enc.mundur(np.hstack([dmu, dlv]))
    opt.langkah(enc.param_grad() + dec.param_grad())
    if langkah % 2000 == 0:
        r_, k_ = rekon.mean(), kl.mean()
        print(f"langkah {langkah:4d}  rekonstruksi {r_:7.2f}  "
              f"KL {k_:4.2f}  -ELBO {r_ + k_:7.2f}")

Xg = dec.maju(rng.normal(size=(2000, 1)))  # dari prior  #<4>
r = np.linalg.norm(Xg, axis=1)
sudut = np.mod(np.arctan2(Xg[:, 1], Xg[:, 0]), 2 * np.pi)
jarak = np.mean(np.abs(r - 1))
print(f"Sampel baru: jarak rata-rata ke lingkaran = {jarak:.3f}")
print(f"  bagian yang jatuh di busur data (0-270 derajat): "
      f"{np.mean(sudut < 1.5 * np.pi):.3f}")
