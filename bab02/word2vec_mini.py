import numpy as np

korpus = """kucing minum susu di dapur . anjing minum air di halaman .
kucing tidur di sofa . anjing tidur di teras . kucing makan ikan .
anjing makan tulang . mobil melaju di jalan . motor melaju di jalan .
mobil parkir di garasi . motor parkir di garasi .
sopir menyetir mobil . pengendara menyetir motor .""".split() * 50   #<1>
kata = sorted(set(korpus))
idx = {k: i for i, k in enumerate(kata)}
V, d, jendela, k_neg = len(kata), 8, 2, 4
rng = np.random.default_rng(0)
W_in = rng.normal(0, 0.1, (V, d))      # embedding kata pusat
W_out = rng.normal(0, 0.1, (V, d))     # embedding kata konteks
frek = np.bincount([idx[k] for k in korpus], minlength=V) ** 0.75
frek = frek / frek.sum()               # distribusi sampel negatif
sig = lambda z: 1 / (1 + np.exp(-z))

for epoch in range(3):
    for t, w in enumerate(korpus):
        c = idx[w]
        kiri, kanan = max(0, t - jendela), t + jendela + 1
        for s in range(kiri, min(len(korpus), kanan)):
            if s == t:
                continue
            o = idx[korpus[s]]
            neg = rng.choice(V, k_neg, p=frek)  #<2>
            sasaran = np.r_[o, neg]
            label = np.r_[1.0, np.zeros(k_neg)]
            g = sig(W_out[sasaran] @ W_in[c]) - label   #<3>
            gin = g @ W_out[sasaran]
            W_out[sasaran] -= 0.05 * np.outer(g, W_in[c])
            W_in[c] -= 0.05 * gin

E = W_in / np.linalg.norm(W_in, axis=1, keepdims=True)
for q in ["kucing", "mobil", "minum"]:
    sim = E @ E[idx[q]]
    top = [kata[i] for i in np.argsort(-sim) if kata[i] != q][:3]
    print(f"{q:7s} -> {top}")
