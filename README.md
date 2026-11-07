# Jaringan Saraf Tiruan dan Deep Learning — Kode Pendamping

Kode dan perangkat verifikasi untuk buku
**_Jaringan Saraf Tiruan dan Deep Learning: Teori, Algoritma, dan Implementasi dari Nol_**
(Benny Sukma Negara, Teknik Informatika, UIN Sultan Syarif Kasim Riau).
Buku ini adalah pasangan teori bagi
[_Deep Learning dengan Python_](https://github.com/bsnegara/deep-learning-dengan-python).

<!-- DOI Zenodo: tambahkan lencana di sini setelah rilis pertama -->

> *Companion code for an Indonesian-language neural networks textbook: 75
> from-scratch NumPy listings (backpropagation, CNN, RNN/LSTM/GRU, a trainable
> mini-GPT, VAE, GAN, normalizing flows, diffusion, DQN, PPO) and a
> verification script that checks every printed output against the book.*

## Isi

| Folder / berkas | Isi |
|---|---|
| `bab01/` … `bab12/` | 75 listing buku sebagai skrip Python, ditambah beberapa modul pendukung. Nama berkas sama dengan yang tercetak di bawah setiap listing. |
| `harapan/` | Keluaran setiap listing persis seperti tercetak di buku. |
| `jalankan_semua.py` | Menjalankan semua skrip dan mencocokkan keluarannya dengan `harapan/`. |
| `verifikasi/PANDUAN.md` | Langkah menjalankan verifikasi di Google Colab dan Kaggle. |

Semua listing hanya membutuhkan **NumPy** (dan scikit-learn untuk satu
dataset di Bab 6), berjalan di laptop tanpa GPU, masing-masing dalam hitungan
detik sampai sekitar satu menit. Satu-satunya pengecualian adalah
`bab03/verifikasi_torch.py`, yang mencocokkan backpropagation buatan sendiri
dengan PyTorch.

## Mulai cepat

**Di komputer sendiri** (Python 3.11+):
```bash
git clone https://github.com/bsnegara/jst-deep-learning.git
cd jst-deep-learning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python jalankan_semua.py          # semua bab, sekitar 6 menit
python jalankan_semua.py 08       # satu bab saja
```

Setiap skrip dijalankan dari foldernya sendiri, karena sebagian mengimpor
modul di folder yang sama:
```bash
cd bab08
python lm_karakter.py
```

**Di Google Colab atau Kaggle:** lihat `verifikasi/PANDUAN.md`.

## Tentang status `BEDA`

Angka di buku diambil dari Kaggle (NumPy 2.1, CPU). Pemeriksaan gradien dan
hitungan pendek akan **COCOK** di mana pun. Pelatihan yang panjang peka
terhadap pustaka aljabar linear yang dipakai NumPy: selisih pembulatan sekecil
1e-16 dapat tumbuh selama ribuan langkah, sehingga digit kedua atau ketiga
dapat bergeser. Status **BEDA** pada skrip pelatihan karenanya wajar, selama
kesimpulan yang ditarik di buku tetap berlaku. Bila kesimpulannya berubah,
mohon laporkan lewat *Issues*.

## Daftar bab

| Bab | Judul | Listing |
|---|---|---|
| 1 | Neuron Buatan: dari Perceptron ke Komposisi Fungsi | 4 |
| 2 | Model Dangkal sebagai Jaringan Saraf | 7 |
| 3 | Backpropagation secara Rinci | 5 |
| 4 | Optimasi Jaringan Dalam | 6 |
| 5 | Generalisasi dan Regularisasi | 5 |
| 6 | Jaringan Saraf Konvolusional | 10 |
| 7 | Jaringan Saraf Rekuren | 9 |
| 8 | Atensi dan Transformer | 8 |
| 9 | Model Berbasis Energi: Hopfield dan RBM | 5 |
| 10 | Model Generatif Dalam | 7 |
| 11 | Deep Reinforcement Learning | 8 |
| 12 | Keterbatasan dan Arah ke Depan | 1 |

## Daftar listing

| Listing | Berkas | Judul |
|---|---|---|
| 1.1 | `bab01/perceptron.py` | Perceptron dari nol |
| 1.2 | `bab01/xor_mlp.py` | XOR: bobot pilihan tangan dan bobot hasil belajar |
| 1.3 | `bab01/kolaps_linear.py` | Lapisan linear bertumpuk runtuh menjadi satu |
| 1.4 | `bab01/aproksimasi.py` | Menghampiri sin x dengan makin banyak neuron ReLU |
| 2.1 | `bab02/regresi_linear.py` | Regresi linear: persamaan normal dan penurunan gradien |
| 2.2 | `bab02/tiga_neuron.py` | Tiga model, satu neuron, satu fungsi pelatihan |
| 2.3 | `bab02/softmax.py` | Regresi softmax dari nol, dengan pemeriksaan gradien |
| 2.4 | `bab02/rbf.py` | Jaringan RBF dua tahap dan pengaruh lebar sigma |
| 2.5 | `bab02/ae_pca.py` | Autoencoder linear menemukan subruang PCA |
| 2.6 | `bab02/faktorisasi.py` | Faktorisasi matriks dengan 30% rating teramati |
| 2.7 | `bab02/word2vec_mini.py` | Word2vec mini (skip-gram dengan negative sampling) |
| 3.1 | `bab03/dual.py` | Diferensiasi otomatis mode maju dengan bilangan dual |
| 3.2 | `bab03/nilai.py` | Kelas Nilai: mesin autograd skalar |
| 3.3 | `bab03/uji_nilai.py` | Menguji kelas Nilai |
| 3.4 | `bab03/mlp_numpy.py` | MLP berbasis lapisan dengan backpropagation matriks |
| 3.5 | `bab03/verifikasi_torch.py` | Mencocokkan gradien NumPy dengan PyTorch |
| 4.1 | `bab04/normalisasi.py` | Pengaruh standardisasi fitur pada penurunan gradien |
| 4.2 | `bab04/inisialisasi.py` | Varians sinyal melewati 20 lapisan untuk berbagai inisialisasi |
| 4.3 | `bab04/optimizer.py` | Lima optimizer pada fungsi Rosenbrock |
| 4.4 | `bab04/batchnorm.py` | BatchNorm maju dan mundur, dengan pemeriksaan gradien |
| 4.5 | `bab04/bn_latih.py` | Jaringan sigmoid enam lapisan dengan dan tanpa BatchNorm |
| 4.6 | `bab04/newton.py` | Metode Newton pada fungsi Rosenbrock |
| 5.1 | `bab05/bias_varians.py` | Mengukur bias dan varians regresi polinomial |
| 5.2 | `bab05/double_descent.py` | Double descent pada regresi fitur acak |
| 5.3 | `bab05/l1_l2.py` | Kuadrat terkecil, ridge, dan lasso dengan 3 dari 20 fitur yang berguna |
| 5.4 | `bab05/dropout_wd.py` | Weight decay dan dropout pada data dengan label berderau |
| 5.5 | `bab05/early_stopping.py` | Early stopping pada jaringan tanpa regularisasi |
| 6.1 | `bab06/konvolusi.py` | Konvolusi dua dimensi dengan perulangan |
| 6.2 | `bab06/konv_lapisan.py` | Lapisan konvolusi berbasis im2col, maju dan mundur |
| 6.3 | `bab06/cnn_mini.py` | CNN kecil lawan lapisan penuh pada garis yang berpindah tempat |
| 6.4 | `bab06/uji_konv.py` | Memeriksa gradien lapisan konvolusi |
| 6.5 | `bab06/medan_efektif.py` | Medan reseptif efektif lewat perhitungan mundur |
| 6.6 | `bab06/pool.py` | Max pooling 2×2, maju dan mundur |
| 6.7 | `bab06/cnn_angka.py` | CNN dua lapis untuk angka tulisan tangan |
| 6.8 | `bab06/geser_angka.py` | Ketahanan terhadap geseran: MLP, CNN, dan CNN dengan augmentasi |
| 6.9 | `bab06/resnet_gradien.py` | Norma gradien di lapisan pertama: polos vs. sisa |
| 6.10 | `bab06/gradcam.py` | Grad-CAM dari nol |
| 7.1 | `bab07/rnn_karakter.py` | Model bahasa karakter dengan RNN, dilatih dengan BPTT |
| 7.2 | `bab07/gradien_waktu.py` | Norma hasil kali Jacobian RNN terhadap jarak waktu |
| 7.3 | `bab07/esn.py` | Echo-state network untuk meramalkan deret waktu |
| 7.4 | `bab07/lstm.py` | Sel LSTM: maju, mundur, dan pemeriksaan gradien |
| 7.5 | `bab07/ingat_awal.py` | RNN biasa lawan LSTM pada tugas mengingat simbol pertama |
| 7.6 | `bab07/gru.py` | GRU dari nol, dengan pemeriksaan gradien dan aliran gradien |
| 7.7 | `bab07/rnn_urutan.py` | RNN banyak-ke-banyak dan pengoptimal Adam |
| 7.8 | `bab07/rnn_dua_arah.py` | RNN satu arah lawan dua arah pada tugas yang butuh masa depan |
| 7.9 | `bab07/seq2seq_balik.py` | Seq2seq dua RNN untuk membalik urutan |
| 8.1 | `bab08/atensi.py` | Atensi scaled dot-product, pencarian lunak, dan efek penskalaan |
| 8.2 | `bab08/atensi_mundur.py` | Perhitungan mundur atensi, dengan pemeriksaan gradien |
| 8.3 | `bab08/ambil_kunci.py` | Atensi belajar mencari kunci |
| 8.4 | `bab08/transformer_blok.py` | Blok Transformer dengan NumPy |
| 8.5 | `bab08/lapisan_tf.py` | Lapisan Transformer dengan perhitungan mundur |
| 8.6 | `bab08/gpt_mini.py` | GPT mini: embedding, tumpukan blok, dan kepala keluaran |
| 8.7 | `bab08/lm_karakter.py` | Model bahasa karakter dengan GPT mini |
| 8.8 | `bab08/balik_tf.py` | Membalik urutan dengan GPT mini |
| 9.1 | `bab09/hopfield.py` | Jaringan Hopfield mengingat huruf yang rusak |
| 9.2 | `bab09/kapasitas.py` | Kapasitas Hopfield klasik dan Hopfield modern |
| 9.3 | `bab09/boltzmann_penuh.py` | Mesin Boltzmann penuh: pencocokan momen dan Gibbs yang macet |
| 9.4 | `bab09/rbm.py` | RBM dengan CD-1 pada data bars and stripes |
| 9.5 | `bab09/rbm_pcd.py` | CD-1, CD-10, dan persistent CD pada bars and stripes |
| 10.1 | `bab10/reparam.py` | Varians dua penduga gradien |
| 10.2 | `bab10/vae_mini.py` | VAE mini dengan kode satu dimensi |
| 10.3 | `bab10/gan_1d.py` | Diskriminator optimal dan GAN kecil pada data dua mode |
| 10.4 | `bab10/wasserstein.py` | JSD lawan Wasserstein, dan generator yang meminimumkan W₁ |
| 10.5 | `bab10/difusi_1d.py` | Model difusi pada data dua mode |
| 10.6 | `bab10/difusi_ddim.py` | Sampling DDIM dengan jumlah langkah berbeda |
| 10.7 | `bab10/aliran_normal.py` | Normalizing flow dengan kopling afin pada data busur |
| 11.1 | `bab11/bandit.py` | Tiga strategi pada bandit 10 lengan |
| 11.2 | `bab11/tebing.py` | Q-learning dan SARSA pada dunia kisi tebing |
| 11.3 | `bab11/cartpole.py` | Dinamika tiang di atas kereta |
| 11.4 | `bab11/dqn.py` | DQN dengan experience replay dan jaringan target |
| 11.5 | `bab11/bias_max.py` | Bias penaksiran berlebih dan penduga ganda |
| 11.6 | `bab11/reinforce.py` | REINFORCE dengan baseline untuk menyeimbangkan tiang |
| 11.7 | `bab11/varians_gradien.py` | Varians tiga penduga gradien kebijakan |
| 11.8 | `bab11/mcts.py` | MCTS bermain tic-tac-toe |
| 12.1 | `bab12/som.py` | Self-organizing map satu dimensi pada data cincin |

## Lisensi

- Kode di repositori ini: **MIT License** (lihat `LICENSE`). Seluruh kode
  ditulis untuk buku ini.
- Dataset *digits* dimuat dari scikit-learn dan tunduk pada lisensinya.
- Naskah buku tidak termasuk dalam repositori ini.

## Sitasi

Bila memakai kode ini, silakan sitasi melalui berkas `CITATION.cff`
(tombol **Cite this repository** di GitHub).
