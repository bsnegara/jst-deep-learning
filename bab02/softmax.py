import numpy as np

rng = np.random.default_rng(1)
pusat = np.array([[0, 2], [-2, -1], [2, -1]])
X = np.vstack([rng.normal(c, 0.8, (60, 2)) for c in pusat])
y = np.repeat([0, 1, 2], 60)
n, d, K = len(y), 2, 3

def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)    # stabil numerik  #<1>
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)

def loss_dan_grad(W, b):
    P = softmax(X @ W + b)
    loss = -np.mean(np.log(P[np.arange(n), y]))
    G = P.copy()
    G[np.arange(n), y] -= 1                       # P - one-hot  #<2>
    G /= n
    return loss, X.T @ G, G.sum(axis=0)

# Periksa gradien dengan beda hingga  #<3>
W, b = rng.normal(0, 0.1, (d, K)), np.zeros(K)
_, gW, _ = loss_dan_grad(W, b)
eps, num = 1e-6, np.zeros_like(W)
for i in range(d):
    for j in range(K):
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += eps; Wm[i, j] -= eps
        num[i, j] = (loss_dan_grad(Wp, b)[0]
                     - loss_dan_grad(Wm, b)[0]) / (2 * eps)
selisih = np.abs(gW - num).max()
print(f"Selisih gradien analitis vs numerik: {selisih:.1e}")

for langkah in range(301):
    loss, gW, gb = loss_dan_grad(W, b)
    W -= 0.5 * gW; b -= 0.5 * gb
    if langkah % 100 == 0:
        print(f"langkah {langkah:3d}  loss = {loss:.4f}")
akurasi = np.mean(softmax(X @ W + b).argmax(axis=1) == y)
print(f"Akurasi latih: {akurasi:.3f}")
