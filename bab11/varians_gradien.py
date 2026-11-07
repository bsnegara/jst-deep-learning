import numpy as np
from cartpole import cartpole_langkah

sig = lambda z: 1 / (1 + np.exp(-z))
rng = np.random.default_rng(0)
theta = np.array([0.2, 1.0, 8.0, 1.5, 0.0])  # setengah jadi
gamma = 0.99

def episode():
    s, F, A, S2, D = rng.uniform(-0.05, 0.05, 4), [], [], [], []
    for _ in range(500):
        f = np.r_[s, 1.0]
        a = int(rng.random() < sig(theta @ f))
        s, jatuh = cartpole_langkah(s, a)
        F.append(f); A.append(a)
        S2.append(np.r_[s, 1.0]); D.append(jatuh)
        if jatuh: break
    return np.array(F), np.array(A), np.array(S2), np.array(D)

def phi(F):                                   # fitur kritik  #<1>
    return np.hstack([F, F[:, :4] ** 2])

def returns(n):
    G, g = np.zeros(n), 0.0
    for t in reversed(range(n)):
        g = 1.0 + gamma * g; G[t] = g
    return G

latih = [episode() for _ in range(300)]       # data untuk kritik
X = np.vstack([phi(F) for F, *_ in latih])
y = np.concatenate([returns(len(A)) for _, A, *_ in latih])
w = np.linalg.lstsq(X, y, rcond=None)[0]      # V(s) = phi(s) w  #<2>
b = y.mean()

grad = {"REINFORCE": [], "dengan baseline": [], "actor-critic": []}
for _ in range(300):                          # episode baru  #<3>
    F, A, S2, D = episode()
    skor = F * (A - sig(F @ theta))[:, None]  # grad log pi
    G = returns(len(A))
    adv = (1.0 + gamma * (phi(S2) @ w) * (1 - D)
           - phi(F) @ w)                      # galat TD  #<4>
    grad["REINFORCE"].append(skor.T @ G)
    grad["dengan baseline"].append(skor.T @ (G - b))
    grad["actor-critic"].append(skor.T @ adv)

acuan = np.array(grad["REINFORCE"])
v0, m0 = acuan.var(0).sum(), acuan.mean(0)
panjang = np.mean([len(e[1]) for e in latih])
print(f"panjang episode rata-rata: {panjang:.0f}")
print("penduga           varians relatif   cos ke arah REINFORCE")
for nama, g in grad.items():
    g = np.array(g); m = g.mean(0)
    cos = m @ m0 / np.linalg.norm(m) / np.linalg.norm(m0)
    print(f"{nama:16s}  {g.var(0).sum() / v0:15.2f}   {cos:8.2f}")
