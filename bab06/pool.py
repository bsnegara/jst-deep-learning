import numpy as np

class MaksPool2:
    """Max pooling 2x2 dengan stride 2 (H dan W harus genap)."""
    def maju(self, X):
        N, C, H, W = X.shape
        Xr = X.reshape(N, C, H // 2, 2, W // 2, 2)        #<1>
        Y = Xr.max(axis=(3, 5))
        self.masker = Xr == Y[:, :, :, None, :, None]     #<2>
        return Y

    def mundur(self, G):
        Gr = self.masker * G[:, :, :, None, :, None]
        N, C, h, _, w, _ = Gr.shape
        return Gr.reshape(N, C, 2 * h, 2 * w)
