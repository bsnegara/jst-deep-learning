import numpy as np
from atensi import atensi

def posisi_sinus(n, d):                        # pengodean posisi  #<1>
    pos, i = np.arange(n)[:, None], np.arange(0, d, 2)[None, :]
    sudut = pos / 10000 ** (i / d)
    PE = np.zeros((n, d))
    PE[:, 0::2], PE[:, 1::2] = np.sin(sudut), np.cos(sudut)
    return PE

def layer_norm(X, eps=1e-5):                    # per token
    mu, var = X.mean(-1, keepdims=True), X.var(-1, keepdims=True)
    return (X - mu) / np.sqrt(var + eps)

def multi_head(X, p, h, masker=None):
    n, d = X.shape
    Q, K, V = X @ p["Wq"], X @ p["Wk"], X @ p["Wv"]
    kepala = []
    for j in range(h):                          # tiap kepala  #<2>
        s = slice(j * d // h, (j + 1) * d // h)
        kepala.append(atensi(Q[:, s], K[:, s], V[:, s], masker)[0])
    return np.hstack(kepala) @ p["Wo"]

def blok(X, p, h=4, kausal=False):              # pre-LN  #<3>
    n = len(X)
    masker = np.tril(np.ones((n, n), bool)) if kausal else None
    X = X + multi_head(layer_norm(X), p, h, masker)   # sisa 1
    F = np.maximum(layer_norm(X) @ p["W1"] + p["b1"], 0)
    return X + F @ p["W2"] + p["b2"]                  # sisa 2

rng = np.random.default_rng(0)
n, d, dff = 6, 16, 64
p = {k: rng.normal(0, d ** -0.5, (d, d))
     for k in ["Wq", "Wk", "Wv", "Wo"]}
p.update(W1=rng.normal(0, d ** -0.5, (d, dff)), b1=np.zeros(dff),
         W2=rng.normal(0, dff ** -0.5, (dff, d)), b2=np.zeros(d))
X = rng.normal(size=(n, d))
print("Bentuk keluaran:", blok(X, p).shape)

# Masker kausal: mengubah token terakhir tidak memengaruhi token lain
X2 = X.copy(); X2[-1] = rng.normal(size=d)       # token terakhir  #<4>
for kausal in (False, True):
    beda = np.abs(blok(X, p, kausal=kausal)
                  - blok(X2, p, kausal=kausal))[:-1].max()
    print(f"kausal={str(kausal):5s}  perubahan token 0-4: {beda:.1e}")

# Tanpa pengodean posisi, urutan token tidak berarti apa-apa  #<5>
perm, PE = rng.permutation(n), posisi_sinus(n, d)
for nama, pe in [("tanpa PE", 0), ("dengan PE", PE)]:
    acak = blok(X[perm] + pe, p)              # kalimat diacak
    asli = blok(X + pe, p)[perm]              # hasil asli, diacak
    sama = np.abs(acak - asli).max() < 1e-12
    print(f"{nama:9s} hasil acak = asli diacak: {sama}")
