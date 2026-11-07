# Jalankan di Google Colab (PyTorch sudah terpasang).
import numpy as np
import torch

rng = np.random.default_rng(0)
X = rng.normal(size=(8, 4)); y = rng.integers(0, 3, 8)
W1, b1 = rng.normal(size=(4, 5)), rng.normal(size=5)
W2, b2 = rng.normal(size=(5, 3)), rng.normal(size=3)

# --- NumPy: backpropagation manual ----------------------------------
A = X @ W1 + b1; H = np.maximum(A, 0); Z = H @ W2 + b2
P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
G = P.copy(); G[np.arange(8), y] -= 1; G /= 8          # dL/dZ  #<1>
gW2 = H.T @ G
gA = (G @ W2.T) * (A > 0)                                # delta  #<2>
gW1 = X.T @ gA

# --- PyTorch: autograd ----------------------------------------------
t = lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)
tW1, tb1, tW2, tb2 = t(W1), t(b1), t(W2), t(b2)
tZ = torch.relu(torch.tensor(X) @ tW1 + tb1) @ tW2 + tb2
loss = torch.nn.functional.cross_entropy(tZ, torch.tensor(y))
loss.backward()                                          #<3>

for nama, a, b in [("W1", gW1, tW1.grad), ("W2", gW2, tW2.grad)]:
    d = np.abs(a - b.numpy()).max()
    print(f"Grad {nama} sama (selisih < 1e-12): {d < 1e-12}")
