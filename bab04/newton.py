import numpy as np

def f(p):
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2),
                     200 * (y - x ** 2)])

def hessian(p):                              # matriks turunan kedua
    x, y = p
    hxx = 2 - 400 * (y - x ** 2) + 800 * x ** 2
    return np.array([[hxx, -400 * x], [-400 * x, 200.0]])

p = np.array([-1.2, 1.0])
for t in range(1, 8):
    p = p - np.linalg.solve(hessian(p), grad(p))  # Newton  #<1>
    print(f"iterasi {t}  p = ({p[0]:+.4f}, {p[1]:+.4f})"
          f"  f = {f(p):.2e}")
