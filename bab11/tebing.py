import numpy as np

B, K = 4, 12                                   # dunia kisi 4 x 12
MULAI, TUJUAN = (3, 0), (3, 11)
GERAK = [(-1, 0), (1, 0), (0, -1), (0, 1)]     # atas bawah kiri kanan

def langkah(s, a):                             # lingkungan  #<1>
    r_, c_ = s[0] + GERAK[a][0], s[1] + GERAK[a][1]
    r_, c_ = min(max(r_, 0), B - 1), min(max(c_, 0), K - 1)
    if r_ == 3 and 1 <= c_ <= 10:               # jatuh dari tebing
        return MULAI, -100
    return (r_, c_), -1

def pilih(Q, s, eps, rng):                     # epsilon-serakah
    if rng.random() < eps:
        return int(rng.integers(4))
    return int(np.argmax(Q[s]))

def latih(metode, episode=1000, alfa=0.5, gamma=1.0, eps=0.1,
          seed=0):
    rng = np.random.default_rng(seed)
    Q, total = np.zeros((B, K, 4)), []
    for _ in range(episode):
        s, a, G = MULAI, pilih(Q, MULAI, eps, rng), 0
        while s != TUJUAN:
            s2, r = langkah(s, a)
            a2 = pilih(Q, s2, eps, rng)
            if metode == "Q-learning":       # aksi terbaik  #<2>
                target = r + gamma * Q[s2].max()
            else:                            # aksi diambil  #<3>
                target = r + gamma * Q[s2][a2]
            Q[s][a] += alfa * (target - Q[s][a])  # pembaruan TD  #<4>
            s, a, G = s2, a2, G + r
        total.append(G)
    return Q, np.array(total)

print("metode      return (100 ep. terakhir)  jalur serakah")
for metode in ["Q-learning", "SARSA"]:
    Q, G = latih(metode)
    s, jalur = MULAI, [MULAI]          # jalankan kebijakan serakah
    while s != TUJUAN and len(jalur) < 50:
        s, _ = langkah(s, int(np.argmax(Q[s])))
        jalur.append(s)
    atas = min(r for r, _ in jalur)
    print(f"{metode:10s}  {G[-100:].mean():22.1f}  "
          f"{len(jalur) - 1} langkah, sampai baris {atas}")
