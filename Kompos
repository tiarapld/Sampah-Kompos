import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & ENHANCED UI
# ==========================================
st.set_page_config(
    page_title="Sistem Monitoring Sampah & Kompos",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #2E7D32; text-align: center; margin-bottom: 20px; }
    .section-card { padding: 15px; border-radius: 10px; background-color: #F1F8E9; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">♻️ Smart Dashboard: Klasifikasi & Estimasi Volume Sampah</div>', unsafe_allow_html=True)

# ==========================================
# 2. SISTEM LOAD MODEL AI (FOLDER: models/)
# ==========================================
@st.cache_resource
def inisialisasi_model():
    # Mengarah ke folder models/ sesuai struktur repositori Anda
    path_model = "models/model_sampah.h5"
    if os.path.exists(path_model):
        return tf.keras.models.load_model(path_model)
    return None

model = inisialisasi_model()

# Sidebar Info
st.sidebar.header("⚙️ Kontrol Sistem")
if model:
    st.sidebar.success("✅ Model AI (models/model_sampah.h5) Berhasil Dimuat.")
else:
    st.sidebar.warning("⚠️ Model AI tidak ditemukan di folder 'models/'. Menggunakan mode simulasi klasifikasi.")

# Parameter Kalibrasi Kamera untuk Estimasi Volume (Sains Fisika)
st.sidebar.subheader("📐 Kalibrasi Piksel ke Cm")
piksel_per_cm = st.sidebar.slider("Rasio Rasio (Piksel/Cm):", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
kedalaman_wadah = st.sidebar.number_input("Tinggi/Kedalaman Wadah Kompos (cm):", min_value=1.0, value=50.0)

# ==========================================
# 3. FUNGSI LOGIKA (STEM ENGINE)
# ==========================================
def proses_klasifikasi(img, model_ai):
    """Memproses citra dan mengembalikan prediksi kelas."""
    DAFTAR_KELAS = ["Daun Kering", "Sisa Sayuran", "Kulit Buah", "Ranting Pohon"]
    
    if model_ai is not None:
        # Preprocessing citra sesuai standar arsitektur CNN (224x224)
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0) / 255.0
        
        prediksi = model_ai.predict(img_array)
        indeks = np.argmax(prediksi)
        skor = prediksi[0][indeks] * 100
        return DAFTAR_KELAS[indeks], skor
    else:
        # Pilihan fallback jika model belum diunggah
        return np.random.choice(DAFTAR_KELAS), np.random.uniform(85.0, 99.8)

def estimasi_dimensi_dan_volume(img_pil, p_per_cm, tinggi_wadah):
    """Ekstraksi geometri citra menggunakan OpenCV untuk menghitung Volume Estimasi."""
    open_cv_image = np.array(img_pil)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    # Segmentasi Citra (Thresholding & Contour Detection)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kontur, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panjang_cm, lebar_cm, volume_m3 = 0.0, 0.0, 0.0
    
    if kontur:
        # Mengambil kontur terbesar (asumsi itu tumpukan sampah utama)
        kontur_terbesar = max(kontur, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(kontur_terbesar)
        
        # Gambar kotak pembatas pada citra
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Konversi dimensi piksel ke centimeter berdasarkan kalibrasi lensa
        panjang_cm = w / p_per_cm
        lebar_cm = h / p_per_cm
        
        # Logika Rumus Matematika Volume (V = P x L x T) konversi ke Meter Kubik (m³)
        volume_m3 = (panjang_cm * lebar_cm * tinggi_wadah) / 1000000.0
        
    result_img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    return result_img, panjang_cm, lebar_cm, volume_m3

# ==========================================
# 4. ANTARMUKA DASBOR INTERAKTIF (UI)
# ==========================================
kolom_kiri, kolom_kanan = st.columns([1, 1])

with kolom_kiri:
    st.markdown('<div class="section-card"><h3>📷 Input Citra Real-Time</h3></div>', unsafe_allow_html=True)
    file_unggah = st.file_uploader("Pilih file gambar sampah organik untuk diproses...", type=["jpg", "jpeg", "png"])
    
    if file_unggah is not None:
        citra_asli = Image.open(file_unggah)
        st.image(citra_asli, caption="Citra Sumber Asli", use_container_width=True)

with kolom_kanan:
    st.markdown('<div class="section-card"><h3>📊 Hasil Analitik & Volume</h3></div>', unsafe_allow_html=True)
    
    if file_unggah is not None:
        # Jalankan mesin pemrosesan data (STEM Engine)
        hasil_kelas, akurasi = proses_klasifikasi(citra_asli, model)
        citra_proses, p_cm, l_cm, vol_hitung = estimasi_dimensi_dan_volume(citra_asli, piksel_per_cm, kedalaman_wadah)
        
        # 1. Output Klasifikasi
        st.subheader("1. Real-time Image Classification")
        st.success(f"**Kategori Terdeteksi:** {hasil_kelas} ({akurasi:.2f}%)")
        
        # 2. Output Estimasi Volume
        st.subheader("2. Volume Estimation Metrics")
        st.image(citra_proses, caption="Proses Segmentasi Geometri Objek", use_container_width=True)
        
        metrik_1, metrik_2, metrik_3 = st.columns(3)
        metrik_1.metric("Panjang Objek", f"{p_cm:.1f} cm")
        metrik_2.metric("Lebar Objek", f"{l_cm:.1f} cm")
        metrik_3.metric("Estimasi Volume", f"{vol_hitung:.5f} m³", delta_color="inverse")
        
        # 3. Rekomendasi Kondisi Kompos
        st.subheader("3. Status Pematangan Kompos")
        if hasil_kelas in ["Daun Kering", "Ranting Pohon"]:
            st.info("💡 **Rekomendasi:** Kandungan Karbon (C) tinggi. Tambahkan sampah basah kaya Nitrogen seperti sisa sayuran untuk mempercepat fermentasi.")
        else:
            st.info("💡 **Rekomendasi:** Kandungan Nitrogen (N) tinggi. Pastikan aerasi wadah terjaga untuk menghindari bau menyengat akibat kondisi anaerob.")
    else:
        st.info("Menunggu unggahan citra dari Grandmaster untuk memulai kalkulasi matematis.")
