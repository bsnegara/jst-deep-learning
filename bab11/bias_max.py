import numpy as np

rng = np.random.default_rng(0)
print(" K   E[max Qhat]   batas atas     penduga ganda")
for K in [2, 5, 10, 50]:
    A = rng.normal(0, 1, (100000, K))     # dugaan berderau, Q* = 0  #<1>
    B = rng.normal(0, 1, (100000, K))     # dugaan kedua, bebas
    tunggal = A.max(1).mean()
    ganda = B[np.arange(len(B)), A.argmax(1)].mean()     #<2>
    ganda = np.round(ganda, 2) + 0.0          # hindari "-0.00"
    print(f"{K:2d}   {tunggal:11.2f}   {np.sqrt(2 * np.log(K)):12.2f}"
          f"   {ganda:13.2f}")
