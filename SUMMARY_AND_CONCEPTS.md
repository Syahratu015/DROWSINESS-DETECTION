# 📘 Laporan Teknis: Real-time Drowsiness Detection System

Dokumen ini berisi rangkuman lengkap mengenai aspek teknis, tools, algoritma, dan alur kerja sistem.
**Catatan:** File ini bersifat _Local Only_ (masuk .gitignore) sehingga tidak akan ter-upload ke GitHub.

---

## 1. Ringkasan Eksekutif

Sistem ini adalah aplikasi **Computer Vision** berbasis Web yang bertujuan untuk mendeteksi tanda-tanda kantuk pada pengemudi secara _real-time_ menggunakan kamera. Sistem memantau pola kedipan mata (eye closure) dan memberikan peringatan visual jika mata tertutup melebihi durasi aman (2 detik).

---

## 2. Bedah Teknologi (Tech Stack)

### A. Python 3.11 (The Core)

Bahasa pemrograman utama. Dipilih karena ekosistem _Data Science_ dan _AI_ yang sangat matang.

### B. MediaPipe Face Mesh (The AI Brain)

- **Apa itu?**: Framework Machine Learning buatan Google.
- **Kenapa pakai ini?**:
  - Menggunakan model Deep Learning (Neural Networks) yang sudah dilatih (_Pre-trained_).
  - Sangat ringan (_Lightweight_), bisa jalan di CPU laptop biasa tanpa butuh GPU mahal.
  - Output: **468 Titik Landmak 3D** yang memetakan wajah dengan presisi tinggi.
- **Peran**: Mendeteksi lokasi kelopak mata kiri dan kanan dari input gambar.

### C. OpenCV (The Processor)

- **Apa itu?**: Open Source Computer Vision Library.
- **Peran**:
  - Mengakses Hardware (Webcam).
  - Memanipulasi Gambar (Flip mirror, Konversi BGR ke RGB).
  - Menggambar UI (Kotak wajah, Titik mata) di frame video sebelum ditampilkan.

### D. Streamlit (The Interface)

- **Apa itu?**: Framework Python untuk membuat Web Apps data-centric.
- **Peran**: Membuat antarmuka (GUI) modern dengan fitur:
  - Sidebar kontrol (untuk mengatur sensitivitas).
  - Real-time video streaming di browser.
  - Tampilan responsif tanpa koding HTML/CSS/JS manual.

---

## 3. Algoritma Utama: Eye Aspect Ratio (EAR)

Sistem ini tidak menggunakan "Machine Learning Klasik" (seperti SVM/KNN) untuk klasifikasi, tapi menggunakan keputusan berbasis **Geometri**.

### Rumus EAR

Rasio aspek mata dihitung untuk menentukan apakah mata sedang "terbuka" atau "tertutup".

$$
\text{EAR} = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 \times ||p_1 - p_4||}
$$

- **P1, P4**: Titik ujung horizontal mata (kiri-kanan).
- **P2, P6**: Titik vertikal mata (atas-bawah).
- **Logika**:
  - Saat mata terbuka lebar, pembilang (jarak vertikal) besar -> **EAR Tinggi. (> 0.25)**
  - Saat mata tertutup, pembilang mendekati nol -> **EAR Rendah. (< 0.20)**

### Temporal Logic (Logika Waktu)

Manusia berkedip normal sekitar 0.2 - 0.4 detik. Kita tidak boleh membunyikan alarm tiap kali orang berkedip.

- **Counter System**:
  - Jika `EAR < Threshold`: `Counter += 1`
  - Jika `EAR >= Threshold`: `Counter = 0` (Reset)
- **Trigger Alarm**:
  - Alarm hanya bunyi jika `Counter > Limit` (misal 50 frame atau ±2 detik).

---

## 4. Alur Kerja Sistem (Workflow)

1.  **Input**: Webcam mengambil 1 frame gambar (Format BGR).
2.  **Preprocessing**:
    - Gambar di-_flip_ horizontal (cermin).
    - Konversi warna BGR -> RGB (karena MediaPipe butuh RGB).
3.  **Inference**: Gambar dikirim ke MediaPipe Face Mesh -> Output: List koordinat 468 titik.
4.  **Filtering**: Program mengambil hanya indeks titik mata (Total 12 titik penting).
5.  **Calculation**: Hitung nilai EAR rata-rata (Kiri & Kanan).
6.  **Decision Making**:
    - Bandingkan EAR dengan Threshold.
    - Update Counter ngantuk.
    - Tentukan Status: `Active`, `Eyes Closing`, atau `Drowsiness Detected`.
7.  **Drawing**: OpenCV menggambar kotak UI, teks status, dan frame hitam transparan.
8.  **Output**: Streamlit merender frame final ke layar browser.

---

## 5. Struktur Direktori Proyek

- `app.py`: **Main Program** (Web App Streamlit). Jalankan file ini!
- `proj_env/`: **Virtual Environment**. Folder "Kamar Khusus" berisi instalasi library.
- `requirements.txt`: Daftar library yang dibutuhkan (resep instalasi).
- `drowsiness_main.py`: (Versi Lama) Script desktop only. Bisa dipakai untuk debug tanpa Web.
- `.gitignore`: File satpam yang mencegah folder sampah (seperti `proj_env`) ter-upload ke Git.

---

## 6. Cara Menjalankan (How to Run)

Karena kita menggunakan Virtual Environment, gunakan perintah berikut di terminal:

```powershell
.\proj_env\Scripts\python -m streamlit run app.py
```

Jika sukses, browser akan terbuka otomatis di `http://localhost:8501`.
