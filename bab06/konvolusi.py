import numpy as np

def konv2d(X, K, stride=1, pad=0):
    """Korelasi silang 2D satu kanal ('konvolusi' versi DL)."""
    X = np.pad(X, pad)                           # padding nol  #<1>
    k = K.shape[0]
    h = (X.shape[0] - k) // stride + 1           # ukuran keluaran  #<2>
    w = (X.shape[1] - k) // stride + 1
    Y = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            jendela = X[i*stride:i*stride+k, j*stride:j*stride+k]
            Y[i, j] = np.sum(jendela * K)   # hasil kali dalam  #<3>
    return Y

if __name__ == "__main__":
    X = np.zeros((6, 6)); X[:, 3:] = 1       # tepi tegak di tengah
    K = np.array([[1, 0, -1]] * 3)           # detektor tepi tegak
    print("Keluaran (stride 1, tanpa padding):\n", konv2d(X, K))
    print("Ukuran dengan pad=1   :", konv2d(X, K, pad=1).shape)
    print("Ukuran dengan stride=2:", konv2d(X, K, stride=2).shape)
