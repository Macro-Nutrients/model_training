# Laporan Proyek Machine Learning - MacroNutrient

## Project Overview
Pengembangan sistem klasifikasi gambar makanan menjadi semakin penting seiring meningkatnya kesadaran masyarakat akan kesehatan dan nutrisi. Sistem ini dapat membantu pengguna mengidentifikasi jenis makanan dan memperkirakan kandungan nutrisi secara otomatis melalui gambar.

Berdasarkan penelitian [1], penggunaan deep learning untuk klasifikasi gambar makanan telah menunjukkan akurasi yang tinggi dan dapat membantu dalam pemantauan asupan makanan. Model MobileNetV2 dipilih karena mampu mencapai akurasi tinggi pada berbagai tugas klasifikasi gambar, seperti deteksi penyakit tanaman, klasifikasi buah, dan pengenalan ekspresi wajah, dengan hasil yang seringkali lebih baik dari model lain yang lebih besar [2] [3] [4] [5].

**Referensi:**

[1] W. Min, S. Jiang, L. Liu, Y. Rui, and R. Jain, “A survey on food computing,” arXiv.org, Aug. 22, 2018. https://arxiv.org/abs/1808.07202

[2] Lu, J., Liu, X., , X., Tong, J., & Peng, J. (2023). Improved MobileNetV2 crop disease identification model for intelligent agriculture. PeerJ Computer Science, 9. https://doi.org/10.7717/peerj-cs.1595.

[3] Gulzar, Y. (2023). Fruit Image Classification Model Based on MobileNetV2 with Deep Transfer Learning Technique. Sustainability. https://doi.org/10.3390/su15031906.

[4] Yong, L., , L., Sun, D., & Du, L. (2023). Application of MobileNetV2 to waste classification. PLOS ONE, 18. https://doi.org/10.1371/journal.pone.0282336.

[5] Zhu, Q., Zhuang, H., Zhao, M., Xu, S., & Meng, R. (2024). A study on expression recognition based on improved mobilenetV2 network. Scientific Reports, 14. https://doi.org/10.1038/s41598-024-58736-x.


## Business Understanding

### Problem Statements
1. Bagaimana mengembangkan model yang dapat mengklasifikasikan 5 jenis makanan berbeda dengan akurasi tinggi?
2. Bagaimana menangani variasi visual dalam gambar makanan untuk meningkatkan robustness model?

### Goals
1. Mengembangkan model klasifikasi dengan akurasi minimal 95% untuk 5 kelas makanan
2. Meningkatkan robustness model melalui augmentasi data dan transfer learning

### Solution statements
1. Transfer Learning dengan MobileNetV2:
   - Pre-trained pada ImageNet
   - Custom head layer untuk klasifikasi 5 kelas

2. Data Augmentation:
   - Rotasi gambar
   - Horizontal flip
   - Normalisasi
   - Fill mode nearest

## Data Understanding
Dataset terdiri dari 500 gambar makanan dalam 5 kategori yang dikumpulkan dari berbagai sumber Kaggle. Distribusi data seimbang dengan rata-rata 100 gambar per kelas.

**Sumber Dataset Awal:**
- https://www.kaggle.com/datasets/catarinaantelo/portuguese-meals
- https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification
- https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset

Dataset telah melalui proses kurasi manual termasuk pemilahan dan cropping untuk memastikan kualitas gambar. Dataset final yang telah diolah dapat diakses melalui https://drive.google.com/file/d/1xJUvyVfqBZrm1UE9AjKral8OdWAqmCGG/view?usp=sharing .

**Distribusi Dataset:**
- ayam_goreng: 100 gambar
- burger: 101 gambar
- donat: 100 gambar
- kentang_goreng: 103 gambar
- mie_goreng: 96 gambar

Dataset telah melalui proses kurasi manual untuk memastikan kualitas gambar.

## Data Preparation
1. **Normalisasi**: 
   - Rescaling nilai pixel ke range [0,1]
   - Alasan: Mempercepat konvergensi model

2. **Augmentasi Data**:
   - `rotation_range=10`: Rotasi acak gambar hingga 10 derajat.
  - `horizontal_flip=True`: Membalik gambar secara horizontal secara acak.
  - `fill_mode='nearest'`: Mengisi piksel kosong hasil augmentasi dengan nilai piksel terdekat.
   - Alasan: Meningkatkan variasi data dan robustness model

3. **Train-Val-Test Split**:
   - Training: 80% (398 gambar)
   - Validation: 10% (49 gambar)
   - Testing: 10% (53 gambar)
   - Alasan: Evaluasi yang fair dan menghindari data leakage

## Modeling
Model menggunakan arsitektur transfer learning dengan MobileNetV2 sebagai base model dan custom head untuk klasifikasi.

**Arsitektur:**
1. Base Model: MobileNetV2 (frozen)
2. Custom Head:
   - **Conv2D(32, (3,3), activation='relu')**: Layer konvolusi tambahan untuk menambah kompleksitas fitur.
   - **MaxPooling2D()**: Mengurangi dimensi spasial fitur.
   - **Dropout(0.5)**: Mengurangi overfitting dengan menghilangkan 50% neuron secara acak saat training.
   - **Flatten()**: Mengubah output 2D menjadi 1D.
   - **Dense(256, activation='relu')**: Fully connected layer dengan 256 neuron.
   - **Dense(10, activation='softmax')**: Output layer dengan 5 neuron sesuai jumlah kelas pada dataset.

**Kompilasi Model:**
   - Optimizer: **Adam**
   - Loss: **sparse_categorical_crossentropy**
   - Metrics: **accuracy**

**Training:**
   - Epochs: 28
   - Callbacks: **EarlyStopping** (patience=5), **ReduceLROnPlateau** (patience=3, factor=0.5)

## Evaluation
Metrik evaluasi utama adalah accuracy, precision, recall, dan F1-score.

**Formula:**
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-score = 2 * (Precision * Recall) / (Precision + Recall)

**Hasil Pelatihan**
- **Epoch Berhenti:** Pelatihan berhenti pada **epoch ke-8 (epoch 7)** karena callback **EarlyStopping** mendeteksi tidak ada peningkatan pada `val_loss` selama 5 epoch berturut-turut.
- **Akurasi dan Loss pada Epoch Akhir:**
  - **Training Accuracy:** 99.5%
  - **Validation Accuracy:** 98.2%
  - **Training Loss:** 0.012
  - **Validation Loss:** 0.045

**Learning Curve**
Grafik akurasi dan loss menunjukkan bahwa model berhasil belajar dengan baik tanpa overfitting. Titik merah adalah saat early stop dipanggil, yaitu pada epoch ke-3 (epoch 2)

**Confusion Matrix**
Matriks kebingungan divisualisasikan menggunakan heatmap. Berikut adalah hasil prediksi pada data testing:
- **True Positives:** Semua kelas memiliki prediksi benar yang sangat tinggi.
- **False Positives:** Tidak ada kesalahan prediksi yang signifikan.

**Classification Report**

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

Classification Report menunjukkan performa tinggi untuk semua kelas dengan F1-score rata-rata 0.98, yang berarti model mampu mengenali gambar dengan sangat baik.

**Ekspor dan Konversi Model**
- Model diekspor ke format `.keras` dan SavedModel.
- Model dikonversi ke format TensorFlow Lite (`.tflite`) untuk deployment di perangkat mobile. Label kelas disimpan dalam file `label.txt`.
- Model juga dikonversi ke tfjs (Tensorflow Js) untuk deployment ke aplikasi web.

**Inferensi Model**
- Model diuji melakukan inferensi, menggunakan model tflite, dengan gambar baru untuk memastikan hasil prediksi dan confidence value.
- Hasilnya model dapat memprediksi kelas gambar yang diinput dengan benar.
