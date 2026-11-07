import numpy as np

def im2col(X, k, p):
    """(N,C,H,W) -> (N*Ho*Wo, C*k*k): satu baris per jendela."""
    N, C, H, W = X.shape
    Xp = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)))
    Ho, Wo = H + 2 * p - k + 1, W + 2 * p - k + 1
    kol = np.zeros((N, C, k, k, Ho, Wo))
    for a in range(k):                       # k*k salinan bergeser  #<1>
        for b in range(k):
            kol[:, :, a, b] = Xp[:, :, a:a + Ho, b:b + Wo]
    return kol.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, -1)

def col2im(kol, bentuk, k, p):
    """Adjoin im2col: jumlahkan kontribusi jendela yang tumpang."""
    N, C, H, W = bentuk
    Ho, Wo = H + 2 * p - k + 1, W + 2 * p - k + 1
    kol = kol.reshape(N, Ho, Wo, C, k, k).transpose(0, 3, 4, 5, 1, 2)
    Xp = np.zeros((N, C, H + 2 * p, W + 2 * p))
    for a in range(k):
        for b in range(k):
            Xp[:, :, a:a + Ho, b:b + Wo] += kol[:, :, a, b]  #<2>
    return Xp[:, :, p:p + H, p:p + W]

class Konv2D:
    def __init__(self, c_in, c_out, k, p, rng):
        self.k, self.p = k, p
        self.W = rng.normal(0, np.sqrt(2 / (c_in * k * k)),
                            (c_out, c_in, k, k))
        self.b = np.zeros(c_out)

    def maju(self, X):
        N, _, H, W = X.shape
        self.bentuk = X.shape
        self.kol = im2col(X, self.k, self.p)
        Wm = self.W.reshape(len(self.W), -1)
        Y = self.kol @ Wm.T + self.b             # satu perkalian  #<3>
        Ho = H + 2 * self.p - self.k + 1
        Wo = W + 2 * self.p - self.k + 1
        return Y.reshape(N, Ho, Wo, -1).transpose(0, 3, 1, 2)

    def mundur(self, G):
        F = len(self.W)
        G2 = G.transpose(0, 2, 3, 1).reshape(-1, F)
        self.gW = (G2.T @ self.kol).reshape(self.W.shape)  #<4>
        self.gb = G2.sum(0)
        dkol = G2 @ self.W.reshape(F, -1)
        return col2im(dkol, self.bentuk, self.k, self.p)
