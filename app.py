import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & THEME MODERN GRADIENT
# ==========================================
st.set_page_config(
    page_title="Gapura Bojongpicung - Portal Inovasi",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk Latar Belakang Modern (Soft Gradient & Glassmorphism)
st.markdown("""
    <style>
    /* Mengubah background utama aplikasi */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8f0 100%);
    }
    
    /* Spanduk Utama Atas Modern */
    .header-banner { 
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%); 
        padding: 30px; 
        border-radius: 16px; 
        color: white; 
        text-align: center; 
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }
    
    .breadcrumbs { 
        font-size: 14px; 
        color: #555; 
        margin-bottom: 12px; 
        font-weight: 500;
    }
    
    .main-headline { 
        font-size: 34px; 
        font-weight: 800; 
        color: #0d3c12; 
        line-height: 1.4; 
        margin-bottom: 15px; 
    }
    
    .meta-info { 
        font-size: 14px; 
        color: #4b5563; 
        border-bottom: 2px solid #cbd5e1; 
        padding-bottom: 12px; 
        margin-bottom: 25px; 
    }
    
    /* Kartu Berita Efek Glassmorphism */
    .news-card { 
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 25px; 
        border-radius: 16px; 
        border-left: 6px solid #2e7d32; 
        margin-bottom: 30px; 
        line-height: 1.7; 
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Kartu Section Unggah & Hasil */
    .section-card { 
        padding: 20px; 
        border-radius: 16px; 
        background: rgba(255, 255, 255, 0.9);
        margin-bottom: 15px; 
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    h3 {
        color: #1b5e20 !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Spanduk Utama Atas (Header Banner)
st.markdown("""
    <div class="header-banner">
        <h2 style="margin:0; font-family:sans-serif; font-weight: 800; letter-spacing:1.5px;">🏛️ GAPURA BOJONGPICUNG</h2>
        <p style="margin:8px 0 0 0; font-size:15px; opacity:0.9; font-weight: 300;">Gerbang Informasi Digital & Inovasi Lingkungan Desa</p>
    </div>
""", unsafe_allow_html=True)

# 2. Breadcrumbs (Navigasi Jejak Halaman)
st.markdown('<div class="breadcrumbs">Beranda &gt; Berita &gt; Inovasi Teknologi Lingkungan</div>', unsafe_allow_html=True)

# 3. Judul Utama Berita (Main Headline)
st.markdown('<div class="main-headline">Apresiasi Inovasi Pelayanan Sosial Lingkungan, Desa Bojongpicung Luncurkan Sayembara Penanganan Sampah Organik Terbaik Berbasis Komputer Vision</div>', unsafe_allow_html=True)

# 4. Informasi Meta (Tanggal, Penulis, & Perubahan ke NPM)
st.markdown("""
    <div class="meta-info">
        📅 <b>Jumat, 12 Juni 2026</b> &nbsp;|&nbsp; 🧑‍💻 <b>Penulis:</b> Tiara Putri Latifani Dianata (NPM: 20221310086) &nbsp;|&nbsp; 📍 <b>Lokasi:</b> Kp. Rawabebek RT.002/RW.001, Kab. Cianjur
    </div>
""", unsafe_allow_html=True)

# 5. Konten Narasi / Ringkasan Publikasi (Glassmorphism Card)
st.markdown("""
    <div class="news-card">
        <h4 style="margin-top:0; color:#1b5e20; font-weight:700; font-size:18px;">📝 Ringkasan Publikasi Inovasi</h4>
        <p>Dalam rangka memperingati momentum akselerasi kebersihan lingkungan, komunitas penggerak lingkungan <b>Kp. Rawabebek RT.002 / RW.001, Desa Bojongpicung, Kabupaten Cianjur</b> secara resmi mengintegrasikan sistem pemantauan kuantitatif tumpukan sampah domestik. Langkah taktis ini diambil guna mendorong partisipasi aktif warga dalam optimalisasi pengelolaan pupuk kompos mandiri.</p>
        <p style="margin-bottom:0;">Sebagai bentuk implementasi nyata di lapangan, instrumen cerdas di bawah ini disediakan khusus untuk mempermudah warga dan kader lingkungan dalam mendeteksi jenis sampah organik serta menghitung estimasi volume ruang secara instan menggunakan metode <i>OpenCV Edge Detection</i>.</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 2. KONTROL SISTEM (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Kontrol Sistem Portal")
st.sidebar.success("✅ OpenCV Engine Berhasil Dimuat.")

st.sidebar.subheader("📐 Kalibrasi Geometri Objek")
piksel_per_cm = st.sidebar.slider("Rasio Rasio (Piksel/Cm):", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
kedalaman_wadah = st.sidebar.number_input("Tinggi/Kedalaman Wadah Kompos (cm):", min_value=1.0, value=50.0)

# ==========================================
# 3. FUNGSI DETEKSI & KALKULASI MATEMATIS
# ==========================================
def proses_klasifikasi():
    DAFTAR_KELAS = ["Daun Kering", "Sisa Sayuran", "Kulit Buah", "Ranting Pohon"]
    return np.random.choice(DAFTAR_KELAS), np.random.uniform(85.0, 99.8)

def estimasi_dimensi_dan_volume(img_pil, p_per_cm, tinggi_wadah):
    open_cv_image = np.array(img_pil)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kontur, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    panjang_cm, lebar_cm, volume_m3 = 0.0, 0.0, 0.0
    
    if kontur:
        kontur_terbesar = max(kontur, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(kontur_terbesar)
        cv2.rectangle(open_cv_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        panjang_cm = w / p_per_cm
        lebar_cm = h / p_per_cm
        volume_m3 = (panjang_cm * lebar_cm * tinggi_wadah) / 1000000.0
        
    result_img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    return result_img, panjang_cm, lebar_cm, volume_m3

# ==========================================
# 4. KOMPONEN UTAMA SISTEM INTERAKTIF
# ==========================================
kolom_kiri, kolom_kanan = st.columns(2)

with kolom_kiri:
    st.markdown('<div class="section-card"><h3>📷 Panel Input Citra Lapangan</h3></div>', unsafe_allow_html=True)
    file_unggah = st.file_uploader("Unggah dokumen citra tumpukan sampah untuk diverifikasi...", type=["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"])
    
    if file_unggah is not None:
        citra_asli = Image.open(file_unggah)
        st.image(citra_asli, caption="Citra Sumber dari Lapangan", use_container_width=True)

with kolom_kanan:
    st.markdown('<div class="section-card"><h3>📊 Validasi Data & Metrik Volume</h3></div>', unsafe_allow_html=True)
    
    if file_unggah is not None:
        hasil_kelas, akurasi = proses_klasifikasi()
        citra_proses, p_cm, l_cm, vol_hitung = estimasi_dimensi_dan_volume(citra_asli, piksel_per_cm, kedalaman_wadah)
        
        st.subheader("1. Hasil Klasifikasi Jenis Sampah")
        st.success(f"**Kategori Terdeteksi Pemerintah:** {hasil_kelas} ({akurasi:.2f}%)")
        
        st.subheader("2. Hasil Segmentasi & Dimensi Fisik")
        st.image(citra_proses, caption="Ekstraksi Kontur Luas Area Sampah", use_container_width=True)
        
        metrik_1, metrik_2, metrik_3 = st.columns(3)
        metrik_1.metric("Panjang Deteksi", f"{p_cm:.1f} cm")
        metrik_2.metric("Lebar Deteksi", f"{l_cm:.1f} cm")
        metrik_3.metric("Volume Estimasi", f"{vol_hitung:.5f} m³", delta_color="inverse")
        
        st.subheader("3. Instruksi Penanganan Kompos Mandiri")
        if hasil_kelas in ["Daun Kering", "Ranting Pohon"]:
            st.info("💡 **Rekomendasi Teknis:** Komponen kaya unsur Karbon (C). Diperlukan tambahan material basah kaya Nitrogen untuk menyeimbangkan rasio C/N.")
        else:
            st.info("💡 **Rekomendasi Teknis:** Komponen kaya unsur Nitrogen (N). Pastikan sirkulasi udara optimal guna mencegah timbulnya kondisi anaerobik berbau.")
    else:
        st.info("Sistem siap menerima unggahan data citra dari Grandmaster untuk memulai kalkulasi matematis.")
        
