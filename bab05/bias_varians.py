import numpy as np

rng = np.random.default_rng(0)
f = lambda x: np.sin(2 * np.pi * x)     # fungsi sebenarnya
sigma, n, R = 0.3, 25, 500        # derau, ukuran data, ulangan
x = np.linspace(0, 1, n)
x_uji = np.linspace(0, 1, 200)

def ulangi(derajat):
    P = np.zeros((R, len(x_uji)))
    for r in range(R):                  # derau baru tiap ulangan  #<1>
        y = f(x) + rng.normal(0, sigma, n)
        koef = np.polyfit(x, y, derajat)
        P[r] = np.polyval(koef, x_uji)
    return P

print("derajat  bias^2  varians  galat uji")
for d in [1, 3, 5, 9, 12, 15]:
    P = ulangi(d)
    bias2 = np.mean((P.mean(0) - f(x_uji)) ** 2)  #<2>
    var = np.mean(P.var(0))
    galat = bias2 + var + sigma ** 2      # + derau tak tereduksi
    print(f"{d:7d} {bias2:7.4f} {var:8.4f} {galat:10.4f}")
