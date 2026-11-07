import math


class Dual:
    """Bilangan a + b*eps dengan eps^2 = 0 (mode maju)."""
    def __init__(self, a, b=0.0):
        self.a, self.b = a, b                # nilai, turunan  #<1>
    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a + o.a, self.b + o.b)
    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.a * o.a, self.a * o.b + self.b * o.a)  #<2>
    __radd__, __rmul__ = __add__, __mul__

def sin(u):
    return Dual(math.sin(u.a), math.cos(u.a) * u.b)

f = lambda x: x * sin(x) + x * x
x = Dual(2.0, 1.0)                           # benih: dx/dx = 1  #<3>
y = f(x)
analitis = math.sin(2) + 2 * math.cos(2) + 4
print(f"f(2) = {y.a:.6f}, f'(2) mode maju = {y.b:.6f}")
print(f"f'(2) analitis      = {analitis:.6f}")
