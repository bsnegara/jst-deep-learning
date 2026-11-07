import numpy as np
from cnn_angka import CNN, data_angka, latih

X, y, Xu, yu = data_angka()
m = latih(CNN(np.random.default_rng(1)), X, y)

def gradcam(m, x, c):
    m.maju(x[None])                      # simpan aktivasi
    G = np.zeros((1, 10)); G[0, c] = 1   # turunan skor kelas c  #<1>
    G = m.p2.mundur((G @ m.V.T).reshape(1, 16, 2, 2))
    A = np.maximum(m.A2, 0)[0]           # peta fitur (16, 4, 4)
    alfa = G[0].mean(axis=(1, 2))        # bobot tiap peta  #<2>
    L = np.maximum(np.tensordot(alfa, A, 1), 0)
    return np.kron(L, np.ones((2, 2)))   # perbesar ke 8x8  #<3>

aksara = " .:-=+*#"
def baris(M, i):
    M = M / M.max()
    return "".join(aksara[int(v * 7.99)] * 2 for v in M[i])

x = Xu[3, 0]
z = m.maju(Xu[3:4])[0]
c1, c2 = np.argsort(z)[::-1][:2]         # kelas teratas dan kedua
p1 = gradcam(m, Xu[3], c1)
p2 = gradcam(m, Xu[3], c2)
print(f"label = {yu[3]}; skor tertinggi: kelas {c1}, lalu kelas {c2}")
print(f"gambar            CAM kelas {c1}       CAM kelas {c2}")
for i in range(8):
    print(f"{baris(x, i)}  {baris(p1, i)}  {baris(p2, i)}".rstrip())
