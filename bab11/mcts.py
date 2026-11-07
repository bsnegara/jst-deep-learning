import math
import numpy as np

GARIS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
         (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]

def pemenang(p):
    for a, b, c in GARIS:
        if p[a] != 0 and p[a] == p[b] == p[c]:
            return p[a]
    return 0 if 0 in p else 2                  # 2 = seri

def langkah_sah(p):
    return [i for i in range(9) if p[i] == 0]

class Simpul:
    def __init__(self, papan, giliran):
        self.papan, self.giliran = papan, giliran
        self.anak, self.N, self.W = {}, 0, 0.0

def ucb(s, u, c):                              # UCT  #<1>
    return u.W / u.N + c * math.sqrt(math.log(s.N) / u.N)

def mcts(papan, giliran, rng, iterasi=300, c=1.4):
    akar = Simpul(papan, giliran)
    for _ in range(iterasi):
        s, jalur = akar, [akar]
        # 1. seleksi: turun selama simpul sudah terekspansi penuh
        while (pemenang(s.papan) == 0 and
               len(s.anak) == len(langkah_sah(s.papan))):
            s = max(s.anak.values(), key=lambda u: ucb(s, u, c))
            jalur.append(s)
        # 2. ekspansi: tambahkan satu anak baru
        if pemenang(s.papan) == 0:
            a = rng.choice([m for m in langkah_sah(s.papan)
                            if m not in s.anak])
            p = list(s.papan); p[a] = s.giliran
            s.anak[a] = Simpul(tuple(p), -s.giliran)
            s = s.anak[a]; jalur.append(s)
        # 3. simulasi: main acak sampai selesai  #<2>
        p, g = list(s.papan), s.giliran
        while pemenang(p) == 0:
            p[rng.choice(langkah_sah(p))] = g; g = -g
        hasil = pemenang(p)
        # 4. propagasi: skor bagi pemain yang melangkah ke simpul u
        for u in jalur:
            u.N += 1
            u.W += 0.5 if hasil == 2 else float(hasil == -u.giliran)
    return max(akar.anak, key=lambda a: akar.anak[a].N)  #<3>

def main(pemain_x, pemain_o, rng):
    p, g = (0,) * 9, 1
    while pemenang(p) == 0:
        f = pemain_x if g == 1 else pemain_o
        a = f(p, g, rng)
        q = list(p); q[a] = g; p, g = tuple(q), -g
    return pemenang(p)

acak = lambda p, g, rng: rng.choice(langkah_sah(p))
pakai_mcts = lambda p, g, rng: mcts(p, g, rng)
rng = np.random.default_rng(0)
for nama, x, o, n in [("acak  vs acak ", acak, acak, 200),
                      ("MCTS  vs acak ", pakai_mcts, acak, 50),
                      ("acak  vs MCTS ", acak, pakai_mcts, 50),
                      ("MCTS  vs MCTS ", pakai_mcts, pakai_mcts, 20)]:
    h = [main(x, o, rng) for _ in range(n)]
    x_, o_, d_ = h.count(1) / n, h.count(-1) / n, h.count(2) / n
    print(f"{nama} ({n:3d} laga): X menang {x_:.2f}, "
          f"O menang {o_:.2f}, seri {d_:.2f}")
