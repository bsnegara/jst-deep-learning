import numpy as np

def norma_gradien(L, sisa, d=64, n=256, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(n, d))
    Ws, As = [], []
    for _ in range(L):                            # maju
        W = rng.normal(0, 0.5 / np.sqrt(d), (d, d))  # gain 0,5
        A = H @ W
        H = H + np.tanh(A) if sisa else np.tanh(A)  #<1>
        Ws.append(W); As.append(A)
    G = np.ones_like(H) / n                        # dL/dh_L
    g_atas = np.linalg.norm(G)
    for W, A in zip(reversed(Ws), reversed(As)):  # mundur
        cabang = ((G * (1 - np.tanh(A) ** 2)) @ W.T)
        G = G + cabang if sisa else cabang         #<2>
    return np.linalg.norm(G) / g_atas

print(" L    biasa      residual")
for L in [10, 30, 50, 100]:
    print(f"{L:3d}  {norma_gradien(L, False):9.2e}  "
          f"{norma_gradien(L, True):9.2e}")
