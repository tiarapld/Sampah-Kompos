# Implementasi Convolutional Neural Network untuk Klasifikasi Jenis dan Estimasi Volume Sampah Organik sebagai Pendukung Pengelolaan Kompos

**Penyusun:** Tiara Putri Latifani Dianata  
**NPM / NIM:** 20221310086  
**Instansi:** Universitas Kebangsaan Republik Indonesia  

---

## 📌 Lokasi Kerja Praktek & Studi Kasus
* **Alamat:** Kp. Rawabebek RT.002/RW.001, Desa Bojongpicung, Kecamatan Bojongpicung, Kabupaten Cianjur.
* **Tujuan:** Mendigitalisasi monitoring data kuantitatif sampah rumah tangga guna mengoptimalkan pasokan harian bahan baku pengomposan warga/TPS3R setempat.

## ⚙️ Spesifikasi Teknis Alat & Komputasi
* **Metode Pengukuran:** Orthogonal Citra (*Top-down View*) menggunakan kamera pemantau/smartphone.
* **Konstanta Densitas Sampah:** $\approx 154,93 \text{ kg/m}^3$ (Standar empiris konversi volume m³ ke estimasi berat boks kg).
* **Framework Komputasi:** Python, Streamlit Community Cloud, TensorFlow, NumPy, Pillow.

## 📂 Struktur Repositori
```text
├── dataset/               # Citra sampah organik lapangan
├── models/                # File model latih CNN (.h5 / .keras)
├── notebooks/             # Notebook proses training model
├── static/                # Aset visual pendukung (.css)
├── templates/             # File tambahan antarmuka
├── app.py                 # Backend utama dashboard Streamlit
└── requirements.txt       # Daftar pustaka dependensi cloud
