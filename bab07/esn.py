import numpy as np

# Echo-state network: bobot rekuren acak dan TIDAK dilatih;
# hanya lapisan keluaran yang dicari dengan regresi ridge.  #<1>
rng = np.random.default_rng(0)
t = np.arange(3000)
u = np.sin(0.05 * t) * np.cos(0.013 * t)      # deret yang diramalkan
N = 200
W = rng.normal(size=(N, N))
W *= 0.9 / np.max(np.abs(np.linalg.eigvals(W)))  # rho < 1  #<2>
W_in = rng.uniform(-0.5, 0.5, N)

h, Hs = np.zeros(N), []
for x in u[:-1]:
    h = np.tanh(W @ h + W_in * x)
    Hs.append(h)
Hs, target = np.array(Hs[100:]), u[101:]      # buang masa transien
n_latih = 2000
A, y = Hs[:n_latih], target[:n_latih]
w_out = np.linalg.solve(A.T @ A + 1e-6 * np.eye(N), A.T @ y)  #<3>
pred = Hs[n_latih:] @ w_out
galat = np.sqrt(np.mean((pred - target[n_latih:]) ** 2))
print(f"RMSE ramalan satu langkah: {galat:.2e} "
      f"(simpangan baku deret: {u.std():.2f})")
