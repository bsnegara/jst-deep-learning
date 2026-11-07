import numpy as np

rng = np.random.default_rng(0)
x = rng.normal(size=(5, 3))                 # 5 contoh, 3 fitur
W1, b1 = rng.normal(size=(3, 4)), rng.normal(size=4)
W2, b2 = rng.normal(size=(4, 2)), rng.normal(size=2)

dua_lapis = (x @ W1 + b1) @ W2 + b2         # tanpa aktivasi
W, b = W1 @ W2, b1 @ W2 + b2                # satu lapisan setara  #<1>
satu_lapis = x @ W + b
selisih = np.abs(dua_lapis - satu_lapis).max()
print(f"Selisih maksimum   : {selisih:.1e}")

relu = lambda z: np.maximum(z, 0)
dgn_relu = relu(x @ W1 + b1) @ W2 + b2      # sisipkan ReLU  #<2>
selisih = np.abs(dgn_relu - satu_lapis).max()
print(f"Selisih dengan ReLU: {selisih:.3f}")
