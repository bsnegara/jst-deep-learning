import numpy as np

rng = np.random.default_rng(0)
n, d, L = 1000, 256, 20
X = rng.normal(size=(n, d))
aktivasi = {"tanh": (np.tanh, lambda a: 1 - np.tanh(a) ** 2),
            "relu": (lambda a: np.maximum(a, 0),
                     lambda a: (a > 0).astype(float))}
skema = {"kecil": lambda: 0.01,
         "Xavier": lambda: np.sqrt(1 / d),      # 2/(n_in+n_out)  #<1>
         "He": lambda: np.sqrt(2 / d)}          # 2/n_in  #<2>

def uji(nama_akt, nama_skema):
    phi, dphi = aktivasi[nama_akt]
    Ws, As, H = [], [], X
    for _ in range(L):                          # maju 20 lapisan
        W = rng.normal(0, skema[nama_skema](), (d, d))
        A = H @ W
        H = phi(A)
        Ws.append(W); As.append(A)
    G = rng.normal(size=H.shape)                # adjoin acak di atas
    for W, A in zip(reversed(Ws), reversed(As)):
        G = (G * dphi(A)) @ W.T                 # mundur  #<3>
    return H.std(), G.std()

print("aktivasi skema    std h(20)   std grad masukan")
for akt, sk in [("tanh", "kecil"), ("tanh", "Xavier"),
                ("relu", "Xavier"), ("relu", "He")]:
    h, g = uji(akt, sk)
    print(f"{akt:8s} {sk:7s} {h:10.2e}   {g:10.2e}")
