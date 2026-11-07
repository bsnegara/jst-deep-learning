import numpy as np

rng = np.random.default_rng(0)
H, T = 64, 60
A = rng.normal(size=(H, H))
A /= np.max(np.abs(np.linalg.eigvals(A)))   # jari-jari spektral 1

print("rho    k=10      k=30      k=60   (norma dh_T/dh_{T-k})")
for rho in [0.8, 1.0, 1.2]:
    W = rho * A
    h, Js = rng.normal(0, 0.5, H), []
    for t in range(T):                       # RNN tanh, masukan acak
        h = np.tanh(W @ h + rng.normal(0, 0.5, H))
        Js.append(np.diag(1 - h ** 2) @ W)   # dh_t/dh_{t-1}  #<1>
    J, norma = np.eye(H), {}
    for k in range(1, T + 1):
        J = J @ Js[T - k]                    # hasil kali Jacobian  #<2>
        norma[k] = np.linalg.norm(J, 2)
    print(f"{rho:3.1f}  {norma[10]:8.1e}  {norma[30]:8.1e}  "
          f"{norma[60]:8.1e}")
