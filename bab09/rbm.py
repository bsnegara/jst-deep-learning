import itertools
import numpy as np

def bars_stripes(n=4):                      # 30 pola 4x4  #<1>
    pola = set()
    for b in itertools.product([0, 1], repeat=n):
        kolom = np.tile(b, (n, 1))           # garis tegak
        pola.add(tuple(kolom.ravel()))
        pola.add(tuple(kolom.T.ravel()))
    return np.array(sorted(pola), float)

sig = lambda z: 1 / (1 + np.exp(-z))
D = bars_stripes()
nv, nh = 16, 12
semua_v = np.array(list(itertools.product([0, 1], repeat=nv)), float)

def energi_bebas(V, W, a, b):     # F(v): h dijumlahkan habis  #<2>
    return -V @ a - np.logaddexp(0, V @ W + b).sum(1)

def log_likelihood(W, a, b):                # eksak: Z atas 65 536 v  #<3>
    F = energi_bebas(semua_v, W, a, b)
    logZ = np.logaddexp.reduce(-F)
    return np.mean(-energi_bebas(D, W, a, b)) - logZ

rng = np.random.default_rng(0)
W = rng.normal(0, 0.01, (nv, nh))
a, b = np.zeros(nv), np.zeros(nh)
print(f"pola data: {len(D)}, LL terbaik = {np.log(1 / len(D)):.3f}, "
      f"LL acak seragam = {nv * np.log(0.5):.3f}")
for epoch in range(20001):
    ph0 = sig(D @ W + b)                    # fase positif  #<4>
    h0 = (rng.random(ph0.shape) < ph0) * 1.0
    pv1 = sig(h0 @ W.T + a)                 # satu langkah Gibbs
    v1 = (rng.random(pv1.shape) < pv1) * 1.0
    ph1 = sig(v1 @ W + b)                   # fase negatif
    W += 0.05 * (D.T @ ph0 - v1.T @ ph1) / len(D)  # CD-1  #<5>
    a += 0.05 * (D - v1).mean(0)
    b += 0.05 * (ph0 - ph1).mean(0)
    if epoch % 5000 == 0:
        print(f"epoch {epoch:5d}  log-likelihood = "
              f"{log_likelihood(W, a, b):.3f}")

v = (rng.random((1000, nv)) < 0.5) * 1.0    # 1000 rantai Gibbs acak
for _ in range(500):
    h = (rng.random((1000, nh)) < sig(v @ W + b)) * 1.0
    v = (rng.random((1000, nv)) < sig(h @ W.T + a)) * 1.0
pola_sah = set(map(tuple, D))
sah = [tuple(x) for x in v if tuple(x) in pola_sah]
print(f"Sampel yang merupakan pola sah: {len(sah) / len(v):.3f}, "
      f"pola berbeda: {len(set(sah))} dari {len(D)}")
