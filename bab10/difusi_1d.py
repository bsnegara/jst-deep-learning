import numpy as np
from mlp_kecil import MLP, Adam

rng = np.random.default_rng(0)
def data(n):                                  # sama dengan gan_1d.py
    pusat = rng.choice([-2, 2], (n, 1))
    return pusat + rng.normal(0, 0.3, (n, 1))

T = 100
beta = np.linspace(1e-3, 0.15, T)             # jadwal derau  #<1>
alfa = 1 - beta
alfa_bar = np.cumprod(alfa)
print(f"alfa_bar[T] = {alfa_bar[-1]:.1e} (hampir derau murni)")

def fitur(x, t):                  # x_t dan embedding waktu
    s = t[:, None] / T
    return np.hstack([x, np.sin(np.pi * s), np.cos(np.pi * s), s])

net, opt = MLP([4, 64, 64, 1], rng, akt="relu"), Adam(2e-3)
for langkah in range(8001):
    x0 = data(256)
    t = rng.integers(0, T, 256)
    eps = rng.normal(size=x0.shape)
    ab = alfa_bar[t][:, None]
    xt = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps    # q(x_t|x_0)  #<2>
    pred = net.maju(fitur(xt, t))
    net.mundur(2 * (pred - eps) / len(eps))          # MSE derau  #<3>
    opt.langkah(net.param_grad())
    if langkah % 4000 == 0:
        loss = np.mean((pred - eps) ** 2)
        print(f"langkah {langkah}  loss = {loss:.3f}")

x = rng.normal(size=(4000, 1))                # mulai dari derau murni
for t in reversed(range(T)):                  # langkah mundur  #<4>
    e = net.maju(fitur(x, np.full(len(x), t)))
    koef = beta[t] / np.sqrt(1 - alfa_bar[t])
    x = (x - koef * e) / np.sqrt(alfa[t])
    if t > 0:
        x += np.sqrt(beta[t]) * rng.normal(size=x.shape)
s = x[:, 0]
kiri, kanan = np.mean(abs(s + 2) < 1), np.mean(abs(s - 2) < 1)
print(f"dekat -2: {kiri:.2f}, dekat +2: {kanan:.2f}, "
      f"lainnya: {1 - kiri - kanan:.2f}")
print(f"simpangan baku tiap mode: {s[s < 0].std():.2f}, "
      f"{s[s > 0].std():.2f} (data: 0.30)")
