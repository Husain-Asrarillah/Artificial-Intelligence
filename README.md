# Praktikum Machine Learning: Regresi & Klasifikasi

Repositori ini berisi laporan dan kode program hasil Praktikum Mata Kuliah **Artificial Intelligence / Data Mining** yang dikerjakan menggunakan Google Colab. 

## 👤 Identitas Mahasiswa
* **Nama:** Husain Asrarillah
* **NIM:** 09020624035
* **Program Studi:** Sistem Informasi
* **Universitas:** Universitas Islam Negeri Surabaya
* **Dosen Pengampu:** Ibu Dwi Rolliawati, M.T

---

## 📂 Konten Praktikum
Repositori ini mencakup implementasi beberapa algoritma machine learning sebagai berikut:

### 1. Regresi Linear Sederhana
* **Kasus:** Prediksi jumlah bug berdasarkan jumlah baris kode (LOC).
* **Metrik:** MAE, MSE, RMSE.
* **Visualisasi:** Scatter Plot & Garis Regresi.

### 2. Regresi Linear Berganda (Multiple Regression)
* **Kasus:** Prediksi bug dengan dua variabel independen (Baris Kode & Tingkat Kompleksitas).
* **Analisis:** Membandingkan pengaruh antar variabel terhadap hasil prediksi.
* **Visualisasi:** Perbandingan Nilai Aktual vs Nilai Prediksi (10 sampel data).

### 3. Regresi Logistik
* **Kasus:** Klasifikasi kelulusan seleksi berdasarkan nilai wawancara, tes tulis, dan kehadiran.
* **Metrik:** Akurasi & Fungsi Sigmoid.
* **Visualisasi:** Confusion Matrix.

### 4. Klasifikasi K-Nearest Neighbors (KNN)
* **Kasus:** Deteksi Spam SMS menggunakan dataset dari Kaggle (`Spam_SMS.csv`).
* **Eksperimen:** Analisis sensitivitas model terhadap perubahan nilai $K$ (Optimasi dari $K=5$ ke $K=3$).
* **Preprocessing:** TF-IDF Vectorizer untuk pengolahan teks.

---

## 🛠️ Teknologi yang Digunakan
* **Bahasa:** Python
* **Library:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Platform:** Google Colab
* **Version Control:** Git & GitHub

## 🚀 Cara Menjalankan Notebook
1.  Buka file `.ipynb` di repositori ini.
2.  Klik tombol **"Open in Colab"** (jika tersedia).
3.  Pastikan dataset `Spam_SMS.csv` sudah terunggah di lingkungan kerja atau terbaca via URL Raw GitHub.
4.  Jalankan semua sel secara berurutan.

---
*Dibuat untuk memenuhi tugas praktikum mata kuliah Artificial Intelligence (H6C.4).*
