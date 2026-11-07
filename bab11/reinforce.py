import numpy as np
from cartpole import cartpole_langkah

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
theta = np.zeros(5)                 # kebijakan logistik pada (s, 1)

def episode():
    s = rng.uniform(-0.05, 0.05, 4)
    fitur, aksi = [], []
    for t in range(500):
        f = np.r_[s, 1.0]
        a = int(rng.random() < sig(theta @ f))      # sampel aksi
        fitur.append(f); aksi.append(a)
        s, jatuh = cartpole_langkah(s, a)
        if jatuh:
            break
    return np.array(fitur), np.array(aksi)

gamma, laju = 0.99, 0.01
for it in range(1, 201):
    grad, panjang = np.zeros(5), []
    batch = [episode() for _ in range(10)]
    G_semua = []
    for F, A in batch:              # return dari t sampai akhir  #<1>
        G = np.zeros(len(A)); g = 0.0
        for t in reversed(range(len(A))):
            g = 1.0 + gamma * g; G[t] = g
        G_semua.append(G); panjang.append(len(A))
    b = np.mean(np.concatenate(G_semua))          # baseline  #<2>
    for (F, A), G in zip(batch, G_semua):
        p = sig(F @ theta)
        grad += F.T @ ((A - p) * (G - b))   # skor x keuntungan  #<3>
    theta += laju * grad / len(batch)
    if it % 40 == 0:
        print(f"iterasi {it:3d}  panjang episode rata-rata = "
              f"{np.mean(panjang):6.1f}")
