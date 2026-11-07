import itertools
import numpy as np

N = 9
S = np.array(list(itertools.product([0, 1], repeat=N)), float)

def peluang(W, b):                     # P(s) eksak: 512 keadaan  #<1>
    E = -0.5 * np.einsum("ki,ij,kj->k", S, W, S) - S @ b
    p = np.exp(-(E - E.min())); return p / p.sum()

def momen(p):                          # <s_i s_j> dan <s_i>
    return S.T @ (p[:, None] * S), p @ S

# 1) Kemungkinan maksimum = mencocokkan momen
rng = np.random.default_rng(0)
W0 = rng.normal(0, 1, (N, N)); W0 = np.triu(W0, 1); W0 += W0.T
b0 = rng.normal(0, 0.5, N)
C_data, m_data = momen(peluang(W0, b0))   # "data" dari mesin asli
W, b = np.zeros((N, N)), np.zeros(N)
for t in range(1, 20001):
    C, m = momen(peluang(W, b))
    G = C_data - C; np.fill_diagonal(G, 0)   # positif - negatif  #<2>
    W += 0.5 * G; b += 0.5 * (m_data - m)
    if t in (10, 1000, 20000):
        print(f"iterasi {t:5d}: selisih momen = "
              f"{np.abs(G).max():.1e}, galat bobot = "
              f"{np.abs(W - W0).max():.1e}")

# 2) Gibbs sampling pada mesin dengan dua mode yang setara
print("w     P(mayoritas 1) eksak   Gibbs 2000 sapuan dari 0")
for w in [0.3, 0.6, 1.0, 1.5]:
    Wf = w * (1 - np.eye(N)); bf = np.full(N, -w * (N - 1) / 2)  #<3>
    eksak = peluang(Wf, bf) @ (S.sum(1) > N / 2)
    s, hit = np.zeros(N), 0
    for sapuan in range(2000):
        for i in range(N):             # satu sapuan: semua unit
            p1 = 1 / (1 + np.exp(-(Wf[i] @ s + bf[i])))
            s[i] = rng.random() < p1
        hit += s.sum() > N / 2
    print(f"{w:.1f}   {eksak:20.2f}   {hit / 2000:24.2f}")
