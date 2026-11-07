import numpy as np

rng = np.random.default_rng(0)
# Data: 1000 titik pada cincin di bidang 2D
t = rng.uniform(0, 2 * np.pi, 1000)
X = np.c_[np.cos(t), np.sin(t)] * rng.uniform(0.8, 1.0, (1000, 1))

M = 20                          # 20 neuron dalam satu rantai
W = rng.uniform(-0.1, 0.1, (M, 2))  # posisi di ruang data
idx = np.arange(M)
T = 4000
for k in range(T):
    x = X[rng.integers(len(X))]
    pemenang = np.argmin(np.sum((W - x) ** 2, 1))  # menang  #<1>
    eta = 0.5 * (0.01 / 0.5) ** (k / T)       # laju menyusut
    lebar = 5.0 * (0.5 / 5.0) ** (k / T)      # lingkungan menyusut
    h = np.exp(-(idx - pemenang)**2 / (2 * lebar**2))  #<2>
    W += eta * h[:, None] * (x - W)           # tarik ke x

galat = np.mean(np.min(((X[:, None] - W) ** 2).sum(2), 1) ** 0.5)
sudut = np.unwrap(np.arctan2(W[:, 1], W[:, 0]))
print(f"Galat kuantisasi rata-rata: {galat:.3f}")
print(f"Jari-jari neuron: {np.linalg.norm(W, axis=1).min():.2f}"
      f" - {np.linalg.norm(W, axis=1).max():.2f}")
monoton = np.all(np.diff(sudut) > 0) or np.all(np.diff(sudut) < 0)
print(f"Neuron bertetangga tetap bertetangga di cincin: {monoton}")
