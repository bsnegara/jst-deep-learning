import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)                 # fungsi sasaran
x = rng.uniform(0, 1, 40); y = f(x) + rng.normal(0, 0.2, 40)
xu = np.linspace(0, 1, 500); yu = f(xu)

def kmeans(x, k, iterasi=50):                       # tahap 1  #<1>
    c = np.sort(rng.choice(x, k, replace=False))
    for _ in range(iterasi):
        lbl = np.abs(x[:, None] - c).argmin(1)
        c = np.array([x[lbl == j].mean() if np.any(lbl == j) else c[j]
                      for j in range(k)])
    return c

def fitur(x, c, s):                                 # lapisan RBF  #<2>
    Phi = np.exp(-(x[:, None] - c) ** 2 / (2 * s ** 2))
    return np.hstack([Phi, np.ones((len(x), 1))])   # + bias

c = kmeans(x, 10)
print("sigma    MSE latih   MSE uji")
for s in [0.01, 0.05, 0.1, 0.3, 1.0]:
    Phi = fitur(x, c, s)
    w = np.linalg.solve(Phi.T @ Phi + 1e-6 * np.eye(11),  # tahap 2  #<3>
                        Phi.T @ y)
    latih = np.mean((Phi @ w - y) ** 2)
    uji = np.mean((fitur(xu, c, s) @ w - yu) ** 2)
    print(f"{s:5.2f}   {latih:9.3f}   {uji:7.3f}")
