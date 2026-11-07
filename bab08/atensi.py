import numpy as np

def softmax(S):
    E = np.exp(S - S.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True)

def atensi(Q, K, V, masker=None):
    """Atensi scaled dot-product. Q: (n,d), K: (m,d), V: (m,dv)."""
    S = Q @ K.T / np.sqrt(Q.shape[-1])        # skor kemiripan  #<1>
    if masker is not None:
        S = np.where(masker, S, -np.inf)      # posisi terlarang
    P = softmax(S)                            # bobot tiap baris = 1
    return P @ V, P                           # rata-rata berbobot

if __name__ == "__main__":
    # 1) Atensi sebagai pencarian lunak dalam "kamus"  #<2>
    K = np.array([[1., 0], [0, 1], [-1, 0], [0, -1]])   # kunci
    V = np.array([[10.], [20], [30], [40]])            # nilai
    q = np.array([[0.1, 2.0]])                         # mirip kunci 2
    o, P = atensi(q * 3, K, V)
    print("bobot :", np.round(P, 3), " keluaran:", np.round(o, 2))

    # 2) Mengapa dibagi akar d: varians hasil kali dalam  #<3>
    rng = np.random.default_rng(0)
    for d in [4, 64, 512]:
        q, k = rng.normal(size=(2000, d)), rng.normal(size=(2000, d))
        skor = np.sum(q * k, axis=1)
        S = rng.normal(size=(d, 10)).T @ rng.normal(size=(d, 10))
        maks_tanpa = softmax(S).max(-1).mean()
        maks_dengan = softmax(S / np.sqrt(d)).max(-1).mean()
        print(f"d={d:3d}  std(q.k)={skor.std():6.2f}   bobot maks: "
              f"tanpa {maks_tanpa:.2f}, dengan {maks_dengan:.2f}")
