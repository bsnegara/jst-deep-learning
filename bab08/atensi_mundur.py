import numpy as np
from atensi import atensi

def atensi_mundur(dO, Q, K, V, P):
    """Gradien atensi scaled dot-product terhadap Q, K, V."""
    d = Q.shape[-1]
    dV = P.T @ dO                                        #<1>
    dP = dO @ V.T
    dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True))  #<2>
    dQ = dS @ K / np.sqrt(d)
    dK = dS.T @ Q / np.sqrt(d)
    return dQ, dK, dV

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Q, K = rng.normal(size=(5, 4)), rng.normal(size=(6, 4))
    V = rng.normal(size=(6, 3))
    R = rng.normal(size=(5, 3))            # loss L = sum(R * O)
    O, P = atensi(Q, K, V)
    grad = dict(zip("QKV", atensi_mundur(R, Q, K, V, P)))
    for nama, M in zip("QKV", (Q, K, V)):
        num = np.zeros_like(M)
        for idx in np.ndindex(M.shape):
            s = M[idx]
            M[idx] = s + 1e-6; lp = np.sum(R * atensi(Q, K, V)[0])
            M[idx] = s - 1e-6; lm = np.sum(R * atensi(Q, K, V)[0])
            M[idx] = s; num[idx] = (lp - lm) / 2e-6
        rel = np.abs(grad[nama] - num).max() / np.abs(num).max()
        print(f"Galat relatif gradien {nama}: {rel:.1e}")
