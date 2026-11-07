import numpy as np

# Hampiri f(x) = sin(x) di [-pi, pi] dengan satu lapisan
# tersembunyi ReLU. Bobot lapisan pertama acak lalu dibekukan;
# hanya bobot keluaran yang dicari (kuadrat terkecil).  #<1>
rng = np.random.default_rng(0)
x = np.linspace(-np.pi, np.pi, 400)[:, None]
f = np.sin(x).ravel()

for n_unit in [2, 4, 8, 16, 64]:
    w = rng.normal(0, 1, (1, n_unit))
    b = rng.uniform(-np.pi, np.pi, n_unit) * np.abs(w).ravel()
    H = np.maximum(x @ w + b, 0)                 # fitur ReLU  #<2>
    H = np.hstack([H, np.ones((len(x), 1))])     # kolom bias
    v, *_ = np.linalg.lstsq(H, f, rcond=None)    # bobot keluaran
    galat = np.abs(H @ v - f).max()
    print(f"{n_unit:3d} unit  galat maksimum = {galat:.4f}")
