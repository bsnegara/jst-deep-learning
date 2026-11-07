import numpy as np

def cartpole_langkah(s, a):         # dinamika tiang di atas kereta  #<1>
    x, v, th, w = s
    F = 10.0 if a == 1 else -10.0
    c, si = np.cos(th), np.sin(th)
    tmp = (F + 0.05 * w * w * si) / 1.1
    alfa = (9.8 * si - c * tmp) / (0.5 * (4 / 3 - 0.1 * c * c / 1.1))
    acc = tmp - 0.05 * alfa * c / 1.1
    s = np.array([x + 0.02 * v, v + 0.02 * acc,
                  th + 0.02 * w, w + 0.02 * alfa])
    jatuh = abs(s[0]) > 2.4 or abs(s[2]) > 0.2095
    return s, jatuh
