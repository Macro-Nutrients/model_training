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
- https://drive.google.com/file/d/1xJUvyVfqBZrm1UE9AjKral8OdWAqmCGG/view?usp=sharing

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

# 5. **Hasil Pelatihan dan Evaluasi**
## **Hasil Pelatihan**
- **Epoch Berhenti:** Pelatihan berhenti pada **epoch ke-8 (epoch 7)** karena callback **EarlyStopping** mendeteksi tidak ada peningkatan pada `val_loss` selama 5 epoch berturut-turut.
- **Akurasi dan Loss pada Epoch Akhir:**
  - **Training Accuracy:** 99.5%
  - **Validation Accuracy:** 98.2%
  - **Training Loss:** 0.012
  - **Validation Loss:** 0.045

## **Learning Curve**
Grafik akurasi dan loss menunjukkan bahwa model berhasil belajar dengan baik tanpa overfitting. Titik merah adalah saat early stop dipanggil, yaitu pada epoch ke-3 (epoch 2)

## **Confusion Matrix**
Matriks kebingungan divisualisasikan menggunakan heatmap. Berikut adalah hasil prediksi pada data testing:
- **True Positives:** Semua kelas memiliki prediksi benar yang sangat tinggi.
- **False Positives:** Tidak ada kesalahan prediksi yang signifikan.

## **Classification Report**

Berikut adalah hasil evaluasi pada data testing:

| Kelas            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| ayam_goreng      | 1.00      | 1.00   | 1.00     | 10      |
| burger           | 1.00      | 1.00   | 1.00     | 11      |
| donat            | 1.00      | 1.00   | 1.00     | 10      |
| kentang_goreng   | 0.92      | 1.00   | 0.96     | 11      |
| mie_goreng       | 1.00      | 0.91   | 0.95     | 11      |
| **Accuracy**     |           |        | **0.98** | **53**  |
| **Macro Avg**    | 0.98      | 0.98   | 0.98     | 53      |
| **Weighted Avg** | 0.98      | 0.98   | 0.98     | 53      |

Hasil ini menunjukkan bahwa model mampu mengenali gambar dengan sangat baik.

---

# 6. **Ekspor dan Konversi Model**
- Model diekspor ke format `.keras` dan SavedModel.
- Model dikonversi ke format TensorFlow Lite (`.tflite`) untuk deployment di perangkat mobile. Label kelas disimpan dalam file `label.txt`.
- Model juga dikonversi ke tfjs (Tensorflow Js) untuk deployment ke aplikasi web.

---

# 7. **Inferensi Model TFLite**
- Model TFLite diuji dengan gambar baru untuk memastikan hasil prediksi dan confidence value.
- Visualisasi hasil prediksi juga ditampilkan.

---

# 8. **Arsip Proyek**
- Seluruh folder kerja diarsipkan menjadi file zip untuk keperluan backup atau deployment.

---
