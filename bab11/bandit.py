import numpy as np

def jalankan(strategi, R=2000, T=1000, K=10, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(size=(R, K))             # rata-rata tiap lengan
    terbaik = mu.argmax(1)
    Q, N = np.zeros((R, K)), np.zeros((R, K))
    hadiah, optimal = np.zeros(T), np.zeros(T)
    baris = np.arange(R)
    for t in range(1, T + 1):
        if strategi == "serakah":
            a = Q.argmax(1)
        elif strategi == "eps-serakah":             #<1>
            acak = rng.random(R) < 0.1
            a = np.where(acak, rng.integers(0, K, R), Q.argmax(1))
        else:                                       # UCB  #<2>
            bonus = 2 * np.sqrt(np.log(t) / np.maximum(N, 1e-9))
            a = np.argmax(np.where(N == 0, np.inf, Q + bonus), 1)
        r = mu[baris, a] + rng.normal(size=R)
        N[baris, a] += 1
        Q[baris, a] += (r - Q[baris, a]) / N[baris, a]  #<3>
        hadiah[t - 1] = r.mean()
        optimal[t - 1] = (a == terbaik).mean()
    return hadiah, optimal

print("Langkah 501-1000, rata-rata atas 2000 masalah:")
for s in ["serakah", "eps-serakah", "UCB"]:
    h, o = jalankan(s)
    print(f"{s:12s} hadiah rata-rata = {h[500:].mean():.3f}, "
          f"aksi optimal = {o[500:].mean():.2f}")
