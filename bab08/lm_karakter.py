import numpy as np
from gpt_mini import GPT

teks = ("jaringan saraf belajar dari data. data yang baik membuat "
        "model yang baik. model belajar pola dari data, dan pola "
        "membantu model menebak kata berikutnya. atensi melihat "
        "semua kata sekaligus, dan masker menjaga agar model tidak "
        "melihat masa depan. ")
huruf = sorted(set(teks)); V = len(huruf)
kode = np.array([huruf.index(c) for c in teks])
n = 24                                       # panjang konteks
rng = np.random.default_rng(0)
gpt = GPT(V, n)

def batch(B=32):                             # potongan acak  #<1>
    i = rng.integers(0, len(kode) - n - 1, B)
    X = np.stack([kode[j:j + n] for j in i])
    return X, np.stack([kode[j + 1:j + n + 1] for j in i])

for t in range(1, 1501):
    loss = gpt.langkah(*batch())         # semua posisi sekaligus  #<2>
    if t % 500 == 0:
        print(f"langkah {t:4d}  loss/karakter = {loss:.2f}")

def bangkitkan(awal, m, suhu=0.0):          #<3>
    idx = [huruf.index(c) for c in awal]
    for _ in range(m):
        z = gpt.maju(np.array([idx[-n:]]))[0, -1]
        if suhu == 0: idx.append(int(z.argmax()))
        else:
            p = np.exp((z - z.max()) / suhu); p /= p.sum()
            idx.append(int(rng.choice(V, p=p)))
    return "".join(huruf[i] for i in idx)

print("serakah:", repr(bangkitkan("model ", 40)))
