# Proyek Klasifikasi Gambar: Sampah

Proyek ini adalah implementasi *deep learning* untuk mengklasifikasikan gambar sampah ke dalam 10 kategori berbeda. Model ini dibangun menggunakan TensorFlow dan Keras dengan arsitektur *transfer learning* berbasis **MobileNetV2**.

Tujuan utamanya adalah untuk menciptakan model yang akurat dan efisien yang dapat membantu dalam proses daur ulang dan manajemen limbah.

## Daftar Isi

  - Latar Belakang
  - Dataset
  - Alur Kerja Proyek
  - Arsitektur Model
  - Hasil
  - Instalasi dan Penggunaan
  - Konversi Model

## Latar Belakang

Manajemen sampah adalah salah satu tantangan lingkungan terbesar saat ini. Proses pemilahan sampah yang efisien adalah kunci untuk meningkatkan angka daur ulang. Proyek ini bertujuan untuk mengotomatiskan proses identifikasi jenis sampah melalui gambar, yang dapat diintegrasikan ke dalam sistem pintar seperti tempat sampah otomatis atau aplikasi *mobile*.

## Dataset

Dataset yang digunakan adalah **Garbage Classification V2** yang tersedia di Kaggle. Dataset ini berisi gambar-gambar sampah yang dibagi ke dalam 10 kategori:

1.  `battery` 
2.  `biological`
3.  `cardboard` 
4.  `clothes` 
5.  `glass` 
6.  `metal` 
7.  `paper` 
8.  `plastic` 
9.  `shoes` 
10. `trash` 

Karena jumlah gambar di setiap kelas tidak seimbang, dilakukan proses *balancing* dengan augmentasi dan *downsampling* sehingga setiap kelas memiliki **2500 gambar**.

## Alur Kerja Proyek

1.  **Persiapan Data**:

      - Mengunduh dataset dari Kaggle.
      - Melakukan *balancing* data agar setiap kelas memiliki jumlah gambar yang sama (2500 gambar). Augmentasi gambar digunakan untuk kelas dengan jumlah data kurang dari target.
      - Membagi dataset menjadi data latih (70%), data validasi (15%), dan data uji (15%).

2.  **Pemodelan**:

      - Menggunakan arsitektur *transfer learning* dengan **MobileNetV2** sebagai model dasar (*base model*).
      - Membekukan sebagian besar lapisan awal MobileNetV2 dan hanya melatih ulang **30 lapisan terakhir** (*fine-tuning*).
      - Menambahkan *custom head* di atas model dasar yang terdiri dari lapisan `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, dan `Dropout` untuk klasifikasi.

3.  **Pelatihan**:

      - Model dilatih dengan *optimizer* **Adam** dan *loss function* **Categorical Crossentropy**.
      - Menggunakan *callbacks* `EarlyStopping` untuk mencegah *overfitting* dan `ReduceLROnPlateau` untuk menyesuaikan *learning rate* secara dinamis.

4.  **Evaluasi**:

      - Model dievaluasi pada data uji untuk mengukur akurasi dan performa generalisasinya.
      - Hasil evaluasi divisualisasikan menggunakan *classification report* dan *confusion matrix*.

## Arsitektur Model

Model ini menggunakan pendekatan *fine-tuning* dengan arsitektur fungsional:

  - **Base Model**: `MobileNetV2` (dengan *weights* dari ImageNet) yang lapisan atasnya (classifier) tidak disertakan.
  - **Fine-Tuning**: 30 lapisan terakhir dari `base_model` diatur agar dapat dilatih ulang.
  - **Custom Head**: Sebuah model `Sequential` ditambahkan sebagai *classifier* baru, yang terdiri dari:
      - `Conv2D(256, kernel_size=(3,3))`
      - `MaxPooling2D()`
      - `Flatten()`
      - `Dense(128, activation='relu')`
      - `Dropout(0.5)`
      - `Dense(10, activation='softmax')` (10 kelas output)

## Hasil

Setelah pelatihan, model berhasil mencapai performa yang sangat baik:

  - **Akurasi Training Tertinggi**: 99.96%
  - **Akurasi Validasi Tertinggi**: 95.87%
  - **Akurasi pada Data Uji**: **95.68%**

Grafik akurasi dan *loss* menunjukkan bahwa model belajar dengan baik dan memiliki kemampuan generalisasi yang kuat pada data yang belum pernah dilihat sebelumnya.

## Instalasi dan Penggunaan

Untuk menjalankan *notebook* ini, pastikan kamu memiliki *library* Python berikut:

```bash
pip install tensorflow split-folders seaborn pandas matplotlib
```

**Langkah-langkah untuk menjalankan:**

1.  Unduh *notebook* `Garbage_Image_Classification_DeepLearning.ipynb`.
2.  Siapkan API token Kaggle dalam file `kaggle.json` untuk mengunduh dataset.
3.  Jalankan sel-sel di dalam *notebook* secara berurutan.

## Konversi Model

Model yang telah dilatih disimpan dalam format `.h5` dan juga dikonversi ke format lain untuk keperluan *deployment*:

  - **SavedModel**: Format standar TensorFlow untuk *serving*.
  - **TensorFlow Lite (`.tflite`)**: Format yang dioptimalkan untuk perangkat *mobile* dan *embedded*.
  - **TensorFlow.js**: Format untuk menjalankan model langsung di *browser*.

File `label.txt` juga disertakan untuk memetakan indeks output prediksi ke nama kelas yang sesuai.
