import numpy as np

rng = np.random.default_rng(3)
X = np.vstack([rng.normal([1.5, 1.0], 1.0, (100, 2)),
               rng.normal([-1.5, -1.0], 1.0, (100, 2))])
y = np.hstack([np.ones(100), -np.ones(100)])      # label +-1

# Turunan loss terhadap z = w.x + b, sebagai fungsi m = y*z  #<1>
def grad_perceptron(m): return np.where(m <= 0, -1.0, 0.0)
def grad_hinge(m):      return np.where(m < 1, -1.0, 0.0)
def grad_logistik(m):   return -1.0 / (1.0 + np.exp(m))

def latih(grad_m, lam=0.0, epoch=200, laju=0.1):
    w, b = np.zeros(2), 0.0
    for _ in range(epoch):
        m = y * (X @ w + b)
        g = grad_m(m) * y                         # dL/dz  #<2>
        w -= laju * (X.T @ g / len(y) + lam * w)
        b -= laju * g.mean()
    return w, b

for nama, gm, lam in [("perceptron", grad_perceptron, 0.0),
                      ("SVM (hinge)", grad_hinge, 0.01),
                      ("logistik", grad_logistik, 0.0)]:
    w, b = latih(gm, lam)
    akurasi = np.mean(np.sign(X @ w + b) == y)
    print(f"{nama:12s} w = {np.round(w, 3)}, b = {b:+.3f}, "
          f"akurasi = {akurasi:.3f}")
