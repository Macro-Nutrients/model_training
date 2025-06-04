# **Laporan Training Model Klasifikasi Gambar untuk Capstone ~ MacroNutrient**


**Catatan:** Path dataset, file output, dan libraries perlu disesuaikan jika dijalankan di luar lingkungan Kaggle/Colab.

---

# 1. **Deskripsi Dataset**
Dataset yang digunakan merupakan kumpulan **gambar makanan**, yang dikumpulkan dari beberapa sumber di Kaggle :
- https://www.kaggle.com/datasets/catarinaantelo/portuguese-meals
- https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification
- https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset

Kemudian, peneliti memilih dari **5 kelas** dari dataset tersebut, yaitu :
- **ayam_goreng**
- **burger**
- **donat**
- **kentang_goreng**
- **mie_goreng**

Setelah itu, peneliti melakukan pemilahan dan cropping pada setiap gambar pada aplikasi pengolahan gambar, memastikan gambar yang akan dilatih adalah berkualitas. Dataset final diupload pada Google Drive dan diakses melalui link berikut :
- https://drive.google.com/file/d/1WSKVHCrDwPnqFau175P5ehI-h4Glog1o/view?usp=sharing

Setiap kelas berisi gambar makanan sesuai nama kelasnya. Berdasarkan eksplorasi data, distribusi gambar pada setiap kelas **seimbang**. Jumlah total gambar dihitung menggunakan fungsi `count_images`, dan distribusi gambar per kelas divisualisasikan menggunakan grafik batang. Contoh gambar dari setiap kelas juga ditampilkan untuk memastikan kualitas dan keberagaman data.

**Jumlah gambar per kelas** (contoh, silakan sesuaikan dengan hasil output notebook Anda):
- ayam_goreng: 100 gambar
- burger: 101 gambar
- donat: 100 gambar
- kentang_goreng: 103 gambar
- mie_goreng: 96 gambar

Total gambar: **500 gambar**

---

# 2. **Augmentasi dan Normalisasi pada ImageDataGenerator**
Pada tahap preprocessing, digunakan `ImageDataGenerator` untuk melakukan augmentasi dan normalisasi data gambar:

- **Training Data:**
  - `rescale=1./255`: Normalisasi nilai piksel gambar ke rentang [0, 1].
  - `rotation_range=10`: Rotasi acak gambar hingga 10 derajat.
  - `horizontal_flip=True`: Membalik gambar secara horizontal secara acak.
  - `fill_mode='nearest'`: Mengisi piksel kosong hasil augmentasi dengan nilai piksel terdekat.

- **Validation dan Test Data:**
  - Hanya dilakukan normalisasi dengan `rescale=1./255`.

Augmentasi ini bertujuan untuk meningkatkan variasi data latih sehingga model lebih robust terhadap perubahan posisi dan orientasi objek pada gambar.

---

# 3. **Pembagian Data**
Dataset dibagi menjadi tiga bagian menggunakan `splitfolders`:
- **Training:** 80% (398 gambar)
- **Validation:** 10% (49 gambar)
- **Testing:** 10% (53 gambar)

Pembagian ini memastikan model dapat dievaluasi secara adil pada data yang tidak pernah dilihat saat pelatihan.

---

# 4. **Arsitektur Model**
Model yang digunakan berbasis **transfer learning** dengan arsitektur sebagai berikut:

1. **Base Model:**  
   - **MobileNetV2** (tanpa top layer, bobot dari ImageNet, input shape 224x224x3).
   - Semua layer pada MobileNetV2 di-freeze (tidak dilatih ulang).

2. **Custom Head:**
   - **Conv2D(32, (3,3), activation='relu')**: Layer konvolusi tambahan untuk menambah kompleksitas fitur.
   - **MaxPooling2D()**: Mengurangi dimensi spasial fitur.
   - **Dropout(0.5)**: Mengurangi overfitting dengan menghilangkan 50% neuron secara acak saat training.
   - **Flatten()**: Mengubah output 2D menjadi 1D.
   - **Dense(256, activation='relu')**: Fully connected layer dengan 256 neuron.
   - **Dense(10, activation='softmax')**: Output layer dengan 5 neuron sesuai jumlah kelas pada dataset.

3. **Kompilasi Model:**
   - Optimizer: **Adam**
   - Loss: **sparse_categorical_crossentropy**
   - Metrics: **accuracy**

4. **Training:**
   - Epochs: 28
   - Callbacks: **EarlyStopping** (patience=5), **ReduceLROnPlateau** (patience=3, factor=0.5)

---

# 5. **Evaluasi**
Tahapan evaluasi dilakukan secara menyeluruh:
- **Learning Curve:**  
  - Grafik akurasi dan loss pada data training dan validation diplot setiap epoch.
  - Titik merah pada grafik menandai epoch saat EarlyStopping terjadi.
- **Confusion Matrix:**  
  - Matriks kebingungan divisualisasikan menggunakan heatmap, memperlihatkan jumlah prediksi benar dan salah untuk tiap kelas.
- **Classification Report:**  
  - Menampilkan precision, recall, f1-score, dan support untuk setiap kelas.
  - Hasil evaluasi pada notebook menunjukkan f1-score tiap kelas sangat tinggi (bahkan 100%).
- **Akurasi Training dan Testing:**  
  - Model dievaluasi pada data training dan testing, dan nilai akurasi dicetak.
- **Analisis:**  
  - Hasil evaluasi menunjukkan model sangat baik dalam mengenali gambar pada kelima kelas makanan, tanpa adanya kelas yang mendominasi error.
  - Namun, perlu diwaspadai kemungkinan overfitting jika akurasi training dan testing terlalu tinggi tanpa error.

---

# 6. **Ekspor dan Konversi Model**
- Model diekspor ke format `.keras` dan SavedModel.
- Model dikonversi ke format TensorFlow Lite (`.tflite`) untuk deployment di perangkat mobile.
- Label kelas disimpan dalam file `label.txt`.

---

# 7. **Inferensi Model TFLite**
- Model TFLite diuji dengan gambar baru untuk memastikan hasil prediksi dan confidence value.
- Visualisasi hasil prediksi juga ditampilkan.

---

# 8. **Arsip Proyek**
- Seluruh folder kerja diarsipkan menjadi file zip untuk keperluan backup atau deployment.

---
