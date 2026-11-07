import numpy as np

teks = ("data yang baik membuat model yang baik. model yang baik "
        "belajar dari data. jaringan saraf belajar pola dari data. ")
huruf = sorted(set(teks))
idx = {h: i for i, h in enumerate(huruf)}
V, H = len(huruf), 48
rng = np.random.default_rng(0)
P = {"Wxh": rng.normal(0, 0.1, (H, V)),
     "Whh": rng.normal(0, 0.1, (H, H)),
     "Why": rng.normal(0, 0.1, (V, H)),
     "bh": np.zeros(H), "by": np.zeros(V)}

def loss_dan_grad(masuk, sasaran, h0):
    hs, ps, loss = {-1: h0}, {}, 0.0
    for t, i in enumerate(masuk):                   # maju  #<1>
        x = np.zeros(V); x[i] = 1
        hs[t] = np.tanh(P["Wxh"] @ x + P["Whh"] @ hs[t - 1] + P["bh"])
        z = P["Why"] @ hs[t] + P["by"]
        ps[t] = np.exp(z - z.max()); ps[t] /= ps[t].sum()
        loss -= np.log(ps[t][sasaran[t]])
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dh_lanjut = np.zeros(H)
    for t in reversed(range(len(masuk))):          # mundur (BPTT)  #<2>
        dz = ps[t].copy(); dz[sasaran[t]] -= 1
        g["Why"] += np.outer(dz, hs[t]); g["by"] += dz
        dh = P["Why"].T @ dz + dh_lanjut           # dua sumber  #<3>
        da = dh * (1 - hs[t] ** 2)
        x = np.zeros(V); x[masuk[t]] = 1
        g["Wxh"] += np.outer(da, x); g["bh"] += da
        g["Whh"] += np.outer(da, hs[t - 1])
        dh_lanjut = P["Whh"].T @ da                # ke langkah t-1
    return loss, g, hs[len(masuk) - 1]

def sampel(h, i, n):
    keluar = []
    for _ in range(n):
        x = np.zeros(V); x[i] = 1
        h = np.tanh(P["Wxh"] @ x + P["Whh"] @ h + P["bh"])
        i = int(np.argmax(P["Why"] @ h + P["by"]))  # serakah
        keluar.append(huruf[i])
    return "".join(keluar)

if __name__ == "__main__":
    # Periksa gradien BPTT pada potongan pendek  #<4>
    m = [idx[h] for h in teks[:8]]; s = [idx[h] for h in teks[1:9]]
    _, g, _ = loss_dan_grad(m, s, np.zeros(H))
    W, (a, b) = P["Whh"], (3, 5)
    W[a, b] += 1e-6; lp = loss_dan_grad(m, s, np.zeros(H))[0]
    W[a, b] -= 2e-6; lm = loss_dan_grad(m, s, np.zeros(H))[0]
    W[a, b] += 1e-6
    num = (lp - lm) / 2e-6
    print(f"dL/dWhh[3,5]: analitis {g['Whh'][a, b]:.6f}, "
          f"numerik {num:.6f}")

    T, posisi, h = 25, 0, np.zeros(H)
    mem = {k: np.zeros_like(v) for k, v in P.items()}  # AdaGrad
    for it in range(3001):
        if posisi + T + 1 > len(teks):
            posisi, h = 0, np.zeros(H)
        m = [idx[c] for c in teks[posisi:posisi + T]]
        s = [idx[c] for c in teks[posisi + 1:posisi + T + 1]]
        loss, g, h = loss_dan_grad(m, s, h)        # truncated BPTT  #<5>
        for k in P:
            np.clip(g[k], -5, 5, out=g[k])         # clipping
            mem[k] += g[k] ** 2
            P[k] -= 0.1 * g[k] / np.sqrt(mem[k] + 1e-8)
        posisi += T
        if it % 1000 == 0:
            print(f"iterasi {it:4d}  loss/karakter = {loss / T:.3f}")
    print("Sampel:", repr("d" + sampel(np.zeros(H), idx["d"], 55)))
