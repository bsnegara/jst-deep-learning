import numpy as np
from cartpole import cartpole_langkah

class MLP:                                  # Q(s, .) untuk 2 aksi
    def __init__(self, rng, H=64):
        self.p = [rng.normal(0, 0.5, (4, H)), np.zeros(H),
                  rng.normal(0, H ** -0.5, (H, 2)), np.zeros(2)]
        self.m = [np.zeros_like(q) for q in self.p]
        self.v = [np.zeros_like(q) for q in self.p]; self.t = 0
    def maju(self, S):
        W1, b1, W2, b2 = self.p
        self.S, self.A = S, S @ W1 + b1
        return np.maximum(self.A, 0) @ W2 + b2
    def mundur_adam(self, G, laju=1e-3):
        W1, b1, W2, b2 = self.p
        Hh = np.maximum(self.A, 0)
        Ga = (G @ W2.T) * (self.A > 0)
        grad = [self.S.T @ Ga, Ga.sum(0), Hh.T @ G, G.sum(0)]
        self.t += 1
        for q, g, m, v in zip(self.p, grad, self.m, self.v):
            m[:] = 0.9 * m + 0.1 * g; v[:] = 0.999 * v + 0.001 * g * g
            q -= laju * (m / (1 - 0.9 ** self.t)) / (
                np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

def uji(pilih, rng, n_ep=20):              # panjang episode rata-rata
    hasil = []
    for _ in range(n_ep):
        s, n = rng.uniform(-0.05, 0.05, 4), 0
        while n < 500:
            s, jatuh = cartpole_langkah(s, pilih(s)); n += 1
            if jatuh: break
        hasil.append(n)
    return np.mean(hasil)

rng = np.random.default_rng(0)
acak = uji(lambda s: int(rng.integers(2)), rng)
print(f"kebijakan acak      : panjang episode = {acak:5.1f}")
q = MLP(rng)
q_target = [w.copy() for w in q.p]                       #<1>
memori, s, panjang = [], rng.uniform(-0.05, 0.05, 4), 0
serakah = lambda s: int(q.maju(s[None]).argmax())
for t in range(1, 20001):
    eps = max(0.05, 1 - t / 5000)              # eksplorasi menurun
    a = int(rng.integers(2)) if rng.random() < eps else serakah(s)
    s2, jatuh = cartpole_langkah(s, a); panjang += 1
    memori = (memori + [(s, a, s2, jatuh)])[-10000:]     #<2>
    s = s2
    if jatuh or panjang == 500:
        s, panjang = rng.uniform(-0.05, 0.05, 4), 0
    if t < 500: continue
    j = rng.integers(0, len(memori), 64)       # mini-batch acak
    S, A, S2, D = (np.array(x) for x in zip(*[memori[i] for i in j]))
    W1, b1, W2, b2 = q_target
    Qn = np.maximum(S2 @ W1 + b1, 0) @ W2 + b2
    y = 1.0 + 0.99 * Qn.max(1) * (1 - D)       # target  #<3>
    Q = q.maju(S); G = np.zeros_like(Q)
    G[np.arange(64), A] = (Q[np.arange(64), A] - y) / 64
    q.mundur_adam(G)
    if t % 200 == 0: q_target = [w.copy() for w in q.p]
    if t % 4000 == 0:
        print(f"langkah {t:5d} (eps={eps:.2f}): panjang episode = "
              f"{uji(serakah, rng):5.1f}")
