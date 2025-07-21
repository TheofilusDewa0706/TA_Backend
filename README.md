# Klasifikasi Rumah Adat NTT (TensorFlow Lite + Flask)

Proyek ini merupakan aplikasi backend berbasis Flask untuk mengklasifikasikan gambar rumah adat dari Nusa Tenggara Timur (NTT) menggunakan model deep learning TensorFlow Lite.

## 📂 Isi Repositori

- `Model.py` – Skrip backend Flask yang memuat model dan API prediksi.
- `rumah_adat_final_SGD.tflite` – Model final hasil pelatihan menggunakan optimisasi SGD.
- `rumah_adat_final_97.tflite`, `rumah_adat_final2.tflite` – Versi lain dari model yang digunakan selama eksperimen.
- `requirements.txt` – Daftar dependensi Python.

## 🏠 Kelas Rumah Adat

Model ini mengenali 5 jenis rumah adat dari NTT:

1. Rumah adat Bajawa  
2. Rumah adat Ende  
3. Rumah adat Pulau Timur  
4. Rumah adat Sumba  
5. Rumah adat Waraebo

## 🚀 Cara Menjalankan

### Instal dependensi

Gunakan virtual environment dan jalankan:

```bash
pip install -r requirements.txt
python Model.py

