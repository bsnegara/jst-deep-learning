import numpy as np

rng = np.random.default_rng(0)
# 500 titik 5D yang hampir terletak di subruang 2D  #<1>
A = rng.normal(size=(2, 5))
X = rng.normal(size=(500, 2)) @ A + 0.05 * rng.normal(size=(500, 5))
X -= X.mean(axis=0)
n = len(X)

# Autoencoder linear 5 -> 2 -> 5, dilatih dengan penurunan gradien
U = rng.normal(0, 0.1, (5, 2))          # penyandi (encoder)
V = rng.normal(0, 0.1, (2, 5))          # pengurai (decoder)
for langkah in range(3000):
    H = X @ U                           # kode 2D
    R = H @ V - X                       # galat rekonstruksi
    gV = H.T @ R / n
    gU = X.T @ (R @ V.T) / n            # aturan rantai  #<2>
    U -= 0.05 * gU; V -= 0.05 * gV
print(f"MSE autoencoder: {np.mean((X @ U @ V - X) ** 2):.5f}")

# PCA: dua vektor singular kanan teratas
_, S, Vt = np.linalg.svd(X, full_matrices=False)
P2 = Vt[:2].T @ Vt[:2]
print(f"MSE PCA        : {np.mean((X @ P2 - X) ** 2):.5f}")

# Bandingkan subruang: kosinus sudut prinsipal  #<3>
Q, _ = np.linalg.qr(V.T)
kos = np.linalg.svd(Q.T @ Vt[:2].T, compute_uv=False)
print("Kosinus sudut subruang:", np.round(kos, 5))
