import numpy as np

def bn_maju(X, gamma, beta, eps=1e-5):
    mu, var = X.mean(0), X.var(0)
    Xh = (X - mu) / np.sqrt(var + eps)        # normalkan per fitur
    return gamma * Xh + beta, (Xh, var, gamma, eps)

def bn_mundur(G, cache):                      # G = dL/dY  #<1>
    Xh, var, gamma, eps = cache
    n = len(G)
    gGamma, gBeta = (G * Xh).sum(0), G.sum(0)
    gXh = G * gamma
    gX = (n * gXh - gXh.sum(0) - Xh * (gXh * Xh).sum(0)) \
         / (n * np.sqrt(var + eps))           #<2>
    return gX, gGamma, gBeta

if __name__ == "__main__":
    # Periksa gradien dengan loss acak L = sum(R * Y)  #<3>
    rng = np.random.default_rng(0)
    X = rng.normal(2, 3, (16, 5))
    gamma, beta = rng.normal(size=5), rng.normal(size=5)
    R = rng.normal(size=(16, 5))
    loss = lambda X: (R * bn_maju(X, gamma, beta)[0]).sum()
    gX = bn_mundur(R, bn_maju(X, gamma, beta)[1])[0]
    num = np.zeros_like(X)
    for idx in np.ndindex(X.shape):
        Xp, Xm = X.copy(), X.copy()
        Xp[idx] += 1e-6; Xm[idx] -= 1e-6
        num[idx] = (loss(Xp) - loss(Xm)) / 2e-6
    rel = np.abs(gX - num).max() / np.abs(num).max()
    print(f"Galat relatif gradien BatchNorm: {rel:.1e}")
    print(f"Jumlah kolom gX (harus 0): {np.abs(gX.sum(0)).max():.1e}")
