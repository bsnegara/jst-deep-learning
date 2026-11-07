import numpy as np
from konv_lapisan import Konv2D

S = 65                                   # cukup besar untuk L = 16
rng = np.random.default_rng(0)
print(" L   lebar MR   sigma numerik   sigma teori   sigma/lebar")
for L in [1, 2, 4, 8, 16]:
    lapisan = [Konv2D(1, 1, 3, 1, rng) for _ in range(L)]
    for lap in lapisan:
        lap.W[:] = 1 / 9                 # filter rata-rata  #<1>
    X = np.zeros((1, 1, S, S))
    for lap in lapisan:
        X = lap.maju(X)
    G = np.zeros_like(X); G[0, 0, S // 2, S // 2] = 1     #<2>
    for lap in reversed(lapisan):
        G = lap.mundur(G)
    g = G[0, 0].sum(0)                   # marginal satu sumbu
    g /= g.sum()
    u = np.arange(S) - S // 2
    sigma = np.sqrt((g * u ** 2).sum())  #<3>
    lebar = 2 * L + 1
    print(f"{L:2d}   {lebar:8d}   {sigma:13.3f}   "
          f"{np.sqrt(2 * L / 3):11.3f}   {sigma / lebar:11.3f}")
