import numpy as np

# Turunkan E_{z~N(mu,1)}[z^2] terhadap mu. Jawaban eksak: 2*mu.
rng = np.random.default_rng(0)
mu, S, ulang = 1.5, 10, 2000
f = lambda z: z ** 2

reparam, skor = [], []
for _ in range(ulang):
    eps = rng.normal(size=S)
    z = mu + eps                              # z = mu + eps  #<1>
    reparam.append(np.mean(2 * z))            # df/dz * dz/dmu
    skor.append(np.mean(f(z) * (z - mu)))     # f * dlogp/dmu  #<2>

print(f"gradien eksak       : {2 * mu:.3f}")
for nama, g in [("reparameterisasi", reparam), ("fungsi skor", skor)]:
    print(f"{nama:17s}: rata-rata {np.mean(g):.3f}, "
          f"simpangan baku {np.std(g):.3f}")
