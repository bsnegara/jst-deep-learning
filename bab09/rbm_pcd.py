import contextlib, io
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    from rbm import D, sig, log_likelihood, nv, nh   # data & LL eksak

def latih(cara, k=1, epoch=20000, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, (nv, nh))
    a, b = np.zeros(nv), np.zeros(nh)
    rantai = (rng.random((100, nv)) < 0.5) * 1.0   # untuk PCD  #<1>
    catat = []
    for e in range(1, epoch + 1):
        laju = 0.05 * (1 - e / (epoch + 1))   # laju menurun  #<2>
        ph0 = sig(D @ W + b)
        v = rantai if cara == "PCD" else D        # titik awal  #<3>
        for _ in range(k):                         # k langkah Gibbs
            h = (rng.random((len(v), nh)) < sig(v @ W + b)) * 1.0
            v = (rng.random((len(v), nv)) < sig(h @ W.T + a)) * 1.0
        if cara == "PCD": rantai = v               # lanjutkan nanti
        ph1 = sig(v @ W + b)
        W += laju * (D.T @ ph0 / len(D) - v.T @ ph1 / len(v))
        a += laju * (D.mean(0) - v.mean(0))
        b += laju * (ph0.mean(0) - ph1.mean(0))
        if e % 5000 == 0: catat.append(log_likelihood(W, a, b))
    return catat

print("cara      epoch 5000   10000    15000    20000")
for nama, cara, k in [("CD-1", "CD", 1), ("CD-10", "CD", 10),
                      ("PCD", "PCD", 1)]:
    ll = latih(cara, k)
    print(f"{nama:6s}" + "".join(f"{x:9.3f}" for x in ll))
