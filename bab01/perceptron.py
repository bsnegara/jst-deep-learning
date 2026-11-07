import numpy as np

rng = np.random.default_rng(7)
# dua awan titik 2D yang terpisah secara linear  #<1>
X = np.vstack([rng.normal([2, 2], 0.6, (50, 2)),
               rng.normal([-2, -1], 0.6, (50, 2))])
y = np.hstack([np.ones(50), -np.ones(50)])       # label +1 / -1

def prediksi(X, w, b):
    return np.where(X @ w + b > 0, 1, -1)

def latih_perceptron(X, y, laju=0.1, maks_epoch=100):
    w, b = np.zeros(X.shape[1]), 0.0
    for epoch in range(1, maks_epoch + 1):
        n_salah = 0
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:     # salah klasifikasi  #<2>
                w += laju * yi * xi        # geser garis  #<3>
                b += laju * yi
                n_salah += 1
        if n_salah == 0:
            return w, b, epoch
    return w, b, None                      # tidak konvergen

w, b, epoch = latih_perceptron(X, y)
akurasi = np.mean(prediksi(X, w, b) == y)
print(f"Terpisah linear: epoch {epoch}, akurasi {akurasi:.2f}")
print(f"  w = {np.round(w, 3)}, b = {b:.3f}")

# XOR: tidak ada garis lurus yang memisahkannya  #<4>
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([-1, 1, 1, -1], dtype=float)
w, b, epoch = latih_perceptron(X_xor, y_xor, maks_epoch=1000)
akurasi = np.mean(prediksi(X_xor, w, b) == y_xor)
print(f"XOR: konvergen? {epoch is not None}, akurasi {akurasi:.2f}")
