# 🏠 Klasifikasi Rumah Adat NTT - TFLite + Flask

Aplikasi backend berbasis Flask untuk mengklasifikasikan gambar rumah adat dari Nusa Tenggara Timur (NTT) menggunakan model deep learning TensorFlow Lite.

## 🔍 Fitur

- Deteksi 5 jenis rumah adat NTT:
  - Rumah adat Bajawa
  - Rumah adat Ende
  - Rumah adat Pulau Timur
  - Rumah adat Sumba
  - Rumah adat Waraebo
- Menggunakan model `rumah_adat_final_SGD.tflite`
- Preprocessing: resize ke 224x224 piksel, normalisasi 0–1
- Output: nama kelas, confidence, dan entropy

## 🚀 Cara Menjalankan

1. Install dependensi:

pip install -r requirements.txt

2. buat env:

python -m venv venv

3, Nyalakan env:

venv\Scripts\activate

4. Jalankan server:

python Model.py

Server berjalan di http://0.0.0.0:5000

## 📡 Endpoint API

### POST /predict

- Form field: file (gambar JPG/PNG, maks 20MB)
- Response JSON:

{
  "class": 3,
  "class_name": "Rumah adat Sumba",
  "confidence": 0.8421,
  "entropy": 0.5032
}

## 👨‍💻 Pengembang

Theofilus Dewa Arya Reinanta Putra  
Tugas akhir klasifikasi rumah adat NTT berbasis deep learning.
