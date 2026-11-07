import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0], dtype=float)

# 1) Bobot pilihan tangan: h1 = OR, h2 = AND, y = h1 AND NOT h2  #<1>
tangga = lambda z: (z > 0).astype(float)
W1 = np.array([[1, 1], [1, 1]]); b1 = np.array([-0.5, -1.5])
w2 = np.array([1, -2]);          b2 = -0.5
H = tangga(X @ W1 + b1)
print("Lapisan tersembunyi:\n", H)
print("Keluaran (tangan)  :", tangga(H @ w2 + b2))

# 2) Bobot yang dipelajari dengan penurunan gradien  #<2>
sigmoid = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(1)
W1 = rng.normal(0, 1, (2, 2)); b1 = np.zeros(2)
w2 = rng.normal(0, 1, 2);      b2 = 0.0
laju = 1.0
for langkah in range(5001):
    H = sigmoid(X @ W1 + b1)                   # maju (forward)  #<3>
    p = sigmoid(H @ w2 + b2)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    d_out = (p - y) / len(y)                   # mundur (backward)  #<4>
    g_w2, g_b2 = H.T @ d_out, d_out.sum()
    d_H = np.outer(d_out, w2) * H * (1 - H)
    g_W1, g_b1 = X.T @ d_H, d_H.sum(axis=0)
    W1 -= laju * g_W1; b1 -= laju * g_b1       # perbarui parameter
    w2 -= laju * g_w2; b2 -= laju * g_b2
    if langkah % 1000 == 0:
        print(f"langkah {langkah:4d}  loss = {loss:.4f}")
print("Prediksi (dipelajari):", np.round(p, 3))
