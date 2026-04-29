# 🌋 VolcanoAI: Intelligent Volcanic Activity Monitoring

**VolcanoAI** adalah platform analisis tingkat lanjut yang dirancang untuk memantau dan memprediksi aktivitas vulkanik. Dengan menggabungkan teknologi **Deep Learning (CNN & LSTM)** dan optimasi metaheuristik, sistem ini mampu mengenali pola seismik vulkanik yang kompleks untuk membantu mitigasi bencana gunung berapi.

---

## ✨ Fitur Utama

- **Hybrid Prediction Model**: Integrasi **CNN** untuk ekstraksi fitur spasial dari sinyal seismik dan **LSTM** untuk analisis deret waktu (*time-series*) aktivitas vulkanik.
- **Advanced Anomaly Detection**: Mendeteksi getaran tremor atau anomali seismik yang menjadi indikasi awal peningkatan status gunung berapi.
- **Optimization Architecture**: Implementasi **Ant Colony Optimization (ACO)** untuk optimalisasi parameter model dan zonasi risiko.
- **Unified Visual Interface**: Dashboard interaktif yang dikembangkan dengan integrasi `.NET/C# (VolcanoAI_App_UI)` dan Python untuk pemantauan real-time.
- **Comprehensive Dataset Management**: Manajemen data vulkanik historis dan *live stream* melalui modul `data_volcano`.

---

## 🛠️ Tech Stack

### Core Intelligence
- **Python**: Bahasa pemrograman utama untuk pengolahan data dan modeling.
- **Deep Learning**: TensorFlow/Keras (Arsitektur CNN-LSTM Hybrid).
- **Metaheuristics**: Ant Colony Optimization (ACO) untuk penyetelan model.
- **Numerical Analysis**: NumPy, Pandas, Scikit-learn.

### Interface & Apps
- **C# / .NET**: Digunakan untuk `VolcanoAI_App.sln` guna membangun aplikasi desktop yang stabil.
- **Matplotlib/Seaborn**: Untuk visualisasi statistik aktivitas vulkanik.

---

## 🗂️ Struktur Proyek

Berdasarkan struktur repositori, proyek ini terbagi menjadi beberapa bagian utama:

```text
├── VolcanoAI/            # Core logic pemrosesan sinyal vulkanik
├── VolcanoAI_App_UI/     # Source code antarmuka pengguna (C#/.NET)
├── data_volcano/         # Dataset khusus aktivitas gunung berapi
├── output/               # Hasil prediksi model dan log training
├── static/naive_bayes/   # Komponen analisis statistik tambahan
├── main.py               # Entry point utama aplikasi
└── requirements.txt      # Daftar dependensi Python
```
⚙️ Cara Menjalankan
1. Setup Backend (Python)
```
pip install -r requirements.txt
python main.py
```
3. Setup Desktop UI
Buka file ```VolcanoAI_App.sln``` menggunakan Visual Studio dan lakukan build pada project VolcanoAI_App_UI.

📊 Workflow Sistem
Data Ingestion: Data seismik mentah diambil dari folder data_volcano.

Feature Extraction: CNN memproses spektrogram sinyal untuk mengenali tipe getaran.

Temporal Analysis: LSTM menganalisis urutan kejadian untuk memprediksi probabilitas erupsi.

Optimization: ACO digunakan untuk menyempurnakan akurasi prediksi (fix ACO logs).

Visualization: Hasil akhir ditampilkan melalui Dashboard UI untuk memudahkan pengambilan keputusan.
