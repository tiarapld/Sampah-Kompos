# Implementasi Convolutional Neural Network untuk Klasifikasi Jenis dan Estimasi Volume Sampah Organik sebagai Pendukung Pengelolaan Kompos

Penyusun: **Tiara Putri Latifani Dianata** NIM: **20221310086** ---

## 📌 Deskripsi Proyek
Proyek ini dikembangkan untuk mengotomatisasi proses identifikasi dan pengelolaan sampah organik guna mendukung produksi pupuk kompos yang efisien. Dengan memanfaatkan teknologi **Convolutional Neural Network (CNN)**, sistem ini mampu melakukan dua fungsi utama:
1. **Klasifikasi Jenis Sampah Organik**: Mengidentifikasi kategori sampah organik (misal: sisa sayuran, buah-buahan, dedaunan, dll.).
2. **Estimasi Volume Sampah**: Memperkirakan volume sampah untuk mengukur kapasitas wadah pengomposan yang optimal.

Sistem ini diimplementasikan ke dalam bentuk aplikasi berbasis web agar mudah digunakan oleh pengelola lingkungan, komunitas peduli sampah, maupun industri kompos skala kecil.

---

## 🚀 Fitur Utama
* **Real-time Image Classification**: Mengklasifikasikan jenis sampah organik secara cepat menggunakan model CNN.
* **Volume Estimation**: Estimasi volume sampah berdasarkan dimensi citra objek.
* **Dashboard Monitoring**: Menampilkan data statistik sampah yang telah diproses untuk mendukung keputusan pengelolaan kompos.
* **Friendly User Interface**: Antarmuka web yang intuitif dan responsif.

---

## 📂 Struktur Repositori
```text
├── dataset/               # Sampel dataset citra sampah organik (jika ada)
├── models/                # File model yang sudah dilatih (.h5 / .keras / .pkl)
├── notebooks/             # Jupyter Notebook untuk proses training & evaluasi model
├── static/                # Aset statis web (CSS, JS, Gambar)
├── templates/             # File HTML untuk antarmuka web (Flask/Django)
├── app.py                 # File utama aplikasi web (Backend)

# 📍 Laporan Implementasi Sistem & Lokasi Praktek Lapangan

## 1. Identitas Peneliti dan Mitra
* **Penyusun / Peneliti:** Tiara Putri Latifani Dianata  
* **Nomor Induk Mahasiswa (NIM):** 20221310086  
* **Judul Penelitian:** Implementasi Convolutional Neural Network untuk Klasifikasi Jenis dan Estimasi Volume Sampah Organik sebagai Pendukung Pengelolaan Kompos  
* **Lokasi Praktek / Studi Kasus:** Desa [Masukkan Nama Desa], Kecamatan [Masukkan Kecamatan], Kabupaten [Masukkan Kabupaten]  
* **Mitra Lapangan:** [Masukkan nama instansi, misal: TPS3R Desa / Kelompok Tani / Karang Taruna]  

---

## 2. Latar Belakang dan Konteks Wilayah
Penelitian ini didasarkan pada permasalahan pengelolaan sampah domestik di lokasi praktek, di mana tumpukan sampah organik rumah tangga belum terdata secara kuantitatif. Kurangnya pencatatan volume harian menghambat optimalisasi pasokan bahan baku pada bak pengomposan. 

Sistem monitoring berbasis web Streamlit ini diimplementasikan untuk mendigitalisasi proses tersebut, memberikan estimasi volume riil tumpukan sampah secara instan guna mendukung efisiensi produksi pupuk organik desa.

---

## 3. Spesifikasi Teknis Alat di Lapangan
Untuk menyinkronkan perhitungan kuantitatif pada aplikasi web dengan kondisi fisik di lokasi desa praktek, digunakan parameter kalibrasi tetap sebagai berikut:
* **Dimensi Wadah Kompos Eksisting:** Panjang x Lebar x Tinggi Wadah (Kedalaman) disesuaikan melalui panel kontrol aplikasi web.
* **Instrumen Pengambilan Citra:** Kamera Smartphone / Kamera Pemantau IoT statis dengan sudut pandang tegak lurus (*Top-down View*) terhadap tumpukan sampah.
* **Konstanta Densitas Sampah Organik Desa:** $\approx 154,93 \text{ kg/m}^3$ (Merujuk standar empiris tata kota untuk konversi volume ke estimasi massa total boks).
*
├── requirements.txt       # Daftar library/dependensi Python
└── README.md              # Dokumentasi proyek
