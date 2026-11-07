import numpy as np

rng = np.random.default_rng(0)
N = 100

def klasik(P, s, sapuan=10):                 # Hopfield klasik (Hebb)
    W = P.T @ P / N
    np.fill_diagonal(W, 0)
    for _ in range(sapuan):
        for i in rng.permutation(N):
            s[i] = 1 if W[i] @ s >= 0 else -1
    return s

def modern(P, s, beta=0.2):                  # Hopfield modern  #<1>
    for _ in range(3):
        b = np.exp(beta * (P @ s - (P @ s).max()))
        s = np.sign(P.T @ (b / b.sum()))     # = atensi + tanda  #<2>
    return s

print("  p   p/N   klasik   modern   (bagian pola yang pulih)")
for p in [5, 10, 14, 20, 40, 80]:
    P = rng.choice([-1, 1], size=(p, N))
    pulih = {"klasik": 0, "modern": 0}
    for mu in range(p):
        s0 = P[mu].copy()
        s0[rng.choice(N, 10, replace=False)] *= -1   # 10% rusak
        pulih["klasik"] += np.all(klasik(P, s0.copy()) == P[mu]) / p
        pulih["modern"] += np.all(modern(P, s0.copy()) == P[mu]) / p
    print(f"{p:3d}  {p / N:4.2f}   {pulih['klasik']:6.2f}   "
          f"{pulih['modern']:6.2f}")
