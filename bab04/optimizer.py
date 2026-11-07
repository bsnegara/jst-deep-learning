import numpy as np

def f(p):                                   # fungsi Rosenbrock  #<1>
    x, y = p
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2),
                     200 * (y - x ** 2)])

def sgd(g, s, laju=1e-3):
    return -laju * g

def momentum(g, s, laju=1e-3, beta=0.9):
    s["v"] = beta * s.get("v", 0) + g       #<2>
    return -laju * s["v"]

def adagrad(g, s, laju=0.5):
    s["r"] = s.get("r", 0) + g * g
    return -laju * g / (np.sqrt(s["r"]) + 1e-8)

def rmsprop(g, s, laju=1e-2, rho=0.9):
    s["r"] = rho * s.get("r", 0) + (1 - rho) * g * g
    return -laju * g / (np.sqrt(s["r"]) + 1e-8)

def adam(g, s, laju=0.05, b1=0.9, b2=0.999):
    t = s["t"] = s.get("t", 0) + 1
    s["m"] = b1 * s.get("m", 0) + (1 - b1) * g
    s["v"] = b2 * s.get("v", 0) + (1 - b2) * g * g
    m_hat = s["m"] / (1 - b1 ** t)          # koreksi bias  #<3>
    v_hat = s["v"] / (1 - b2 ** t)
    return -laju * m_hat / (np.sqrt(v_hat) + 1e-8)

for nama, opt in [("SGD", sgd), ("Momentum", momentum),
                  ("AdaGrad", adagrad), ("RMSProp", rmsprop),
                  ("Adam", adam)]:
    p, s = np.array([-1.2, 1.0]), {}
    for t in range(1, 3001):
        p = p + opt(grad(p), s)
        if t in (300, 3000):
            print(f"{nama:9s} t={t:4d}  f = {f(p):.2e}  "
                  f"jarak ke (1,1) = {np.linalg.norm(p - 1):.3f}")
