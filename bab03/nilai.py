"""Nilai: mesin autograd skalar mini (Bab 3)."""
import math


class Nilai:
    """Bilangan skalar yang mengingat bagaimana ia dihitung."""

    def __init__(self, data, anak=(), op=""):
        self.data, self.grad = float(data), 0.0
        self._anak, self._op = anak, op
        self._mundur = lambda: None          # diisi oleh operasi  #<1>

    def __repr__(self):
        return f"Nilai(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, lain):
        lain = lain if isinstance(lain, Nilai) else Nilai(lain)
        hasil = Nilai(self.data + lain.data, (self, lain), "+")
        def _mundur():
            self.grad += hasil.grad          # turunan lokal = 1
            lain.grad += hasil.grad
        hasil._mundur = _mundur
        return hasil

    def __mul__(self, lain):
        lain = lain if isinstance(lain, Nilai) else Nilai(lain)
        hasil = Nilai(self.data * lain.data, (self, lain), "*")
        def _mundur():
            self.grad += lain.data * hasil.grad   # akumulasi  #<2>
            lain.grad += self.data * hasil.grad
        hasil._mundur = _mundur
        return hasil

    def __pow__(self, k):                    # k: bilangan biasa
        hasil = Nilai(self.data ** k, (self,), f"**{k}")
        def _mundur():
            self.grad += k * self.data ** (k - 1) * hasil.grad
        hasil._mundur = _mundur
        return hasil

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        hasil = Nilai(s, (self,), "sigmoid")
        def _mundur():
            self.grad += s * (1 - s) * hasil.grad
        hasil._mundur = _mundur
        return hasil

    def relu(self):
        hasil = Nilai(max(0.0, self.data), (self,), "relu")
        def _mundur():
            self.grad += (self.data > 0) * hasil.grad
        hasil._mundur = _mundur
        return hasil

    def __neg__(self):         return self * -1
    def __sub__(self, lain):   return self + (-lain)
    def __radd__(self, lain):  return self + lain
    def __rmul__(self, lain):  return self * lain

    def mundur(self):
        urutan, dikunjungi = [], set()
        def bangun(v):                       # urutan topologis  #<3>
            if v not in dikunjungi:
                dikunjungi.add(v)
                for a in v._anak:
                    bangun(a)
                urutan.append(v)
        bangun(self)
        self.grad = 1.0                      # dL/dL = 1  #<4>
        for v in reversed(urutan):
            v._mundur()
