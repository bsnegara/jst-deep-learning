import random
from nilai import Nilai

# 1) Graf bercabang: f = (x*y + x)^2, x dipakai dua kali  #<1>
x, y = Nilai(2.0), Nilai(3.0)
f = (x * y + x) ** 2
f.mundur()
print(f"f = {f.data:.0f}, df/dx = {x.grad:.0f}, df/dy = {y.grad:.0f}")

# 2) Contoh backpropagation dari Bab 1  #<2>
x, t = Nilai(1.0), 1.0
w1, w2 = Nilai(0.5), Nilai(2.0)
L = 0.5 * (w2 * (w1 * x).sigmoid() - t) ** 2
L.mundur()
print(f"L = {L.data:.4f}, dL/dw1 = {w1.grad:.4f}, "
      f"dL/dw2 = {w2.grad:.4f}")

# 3) Jaringan 2-3-1 belajar XOR, hanya dengan kelas Nilai  #<3>
random.seed(0)
acak = lambda: Nilai(random.uniform(-1, 1))
W1 = [[acak() for _ in range(2)] for _ in range(3)]
b1 = [acak() for _ in range(3)]
W2, b2 = [acak() for _ in range(3)], acak()
param = [w for r in W1 for w in r] + b1 + W2 + [b2]
data = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

def maju(x):
    h = [sum((w * xi for w, xi in zip(W1[j], x)), b1[j]).sigmoid()
         for j in range(3)]
    return sum((w * hj for w, hj in zip(W2, h)), b2).sigmoid()

for langkah in range(2001):
    loss = sum((maju(x) - t) ** 2 for x, t in data)
    for p in param:
        p.grad = 0.0                         # nolkan gradien  #<4>
    loss.mundur()
    for p in param:
        p.data -= 2.0 * p.grad
    if langkah % 500 == 0:
        print(f"langkah {langkah:4d}  loss = {loss.data:.4f}")
print("Prediksi:", [round(maju(x).data, 3) for x, _ in data])
