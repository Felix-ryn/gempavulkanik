# VolcanoAI/postprocess/cnn_csv_to_json.py                                   # Skrip konversi CSV CNN -> JSON prediksi
import pandas as pd                                                           # pandas untuk baca/olah CSV
import json                                                                   # json untuk serialisasi output
from datetime import datetime                                                 # datetime untuk timestamp UTC
from pathlib import Path                                                      # Path untuk operasi path
from pathlib import Path                                                      # (duplikat) Path lagi — redundan tapi tetap sesuai file asli
import numpy as np                                                            # numpy untuk operasi numerik
import logging                                                                # logging modul

logger = logging.getLogger("VolcanoAI.postprocess.cnn_csv_to_json")           # Logger khusus modul ini
logger.addHandler(logging.NullHandler())                                      # NullHandler agar tidak tergantung konfigurasi root logger

def deg_to_dir(d):                                                            # Konversi derajat menjadi arah kardinal 8-point
    dirs = ["N","NE","E","SE","S","SW","W","NW"]                               # Daftar arah
    return dirs[int((d + 22.5) // 45) % 8]                                     # Hitung index dari derajat dan kembalikan arah

def safe_float(x, default=0.0):                                               # Konversi aman ke float dengan fallback
    try:
        return float(x)                                                       # Return float bila berhasil
    except Exception:
        return default                                                        # Return default bila gagal

def run(csv_path: str = None, out_json: str = None, force: bool = False) -> bool:  # Fungsi utama: baca CSV -> tulis JSON
    """
    Convert cnn_predictions_latest.csv -> cnn_predictions_latest.json
    Returns True if success.
    """                                                                       # Docstring menjelaskan tujuan fungsi
    try:
        # Default paths (if caller tidak memberikan)
        if csv_path is None:
            csv_path = Path("output/cnn_results/results/cnn_predictions_latest.csv")  # Path default CSV input
        else:
            csv_path = Path(csv_path)                                          # Pastikan csv_path adalah Path object

        if out_json is None:
            out_json = Path("output/cnn_results/cnn_predictions_latest.json")  # Path default JSON output
        else:
            out_json = Path(out_json)                                          # Pastikan out_json adalah Path object

        # Safety: pastikan folder ada
        out_json.parent.mkdir(parents=True, exist_ok=True)                     # Buat folder output jika belum ada

        # ===============================
        # 1. CSV tidak ditemukan
        # ===============================
        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")                       # Log peringatan bila CSV tidak ada

            # tulis status JSON minimal agar UI tidak kosong
            payload = {
                "timestamp": datetime.utcnow().isoformat(),                    # Waktu UTC sekarang
                "next_event": None,                                            # Tidak ada event
                "status": "csv_missing"                                        # Status indikator
            }

            out_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),             # Tulis payload minimal ke file JSON
                encoding="utf-8"
            )
            return False                                                       # Kembalikan False karena tidak ada CSV


        # ===============================
        # 2. Baca CSV
        # ===============================
        df = pd.read_csv(csv_path)                                              # Baca CSV ke DataFrame

        if "Acquired_Date" in df.columns:
            df["Acquired_Date"] = pd.to_datetime(
                df["Acquired_Date"],
                errors="coerce"
            )                                                                   # Parse kolom tanggal jika ada

        df = df.sort_values("Acquired_Date") if "Acquired_Date" in df.columns else df  # Urutkan berdasarkan tanggal bila tersedia


        # ===============================
        # 3. Data belum cukup
        # ===============================
        if len(df) < 2 and not force:
            logger.warning(
                "Data CNN belum cukup (<2 rows). Writing status JSON instead of next_event."
            )                                                                   # Log bila data kurang dari 2 baris dan tidak dipaksa

            payload = {
                "timestamp": datetime.utcnow().isoformat(),                    # Timestamp UTC
                "next_event": None,                                            # next_event null
                "status": "not_enough_data",                                   # status indikator
                "rows": len(df)                                                # jumlah baris yang ada
            }

            out_json.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),             # Tulis status not_enough_data ke JSON
                encoding="utf-8"
            )
            return False                                                       # Kembalikan False karena data tidak memadai


        # Ambil dua baris terakhir
        last = df.iloc[-1]                                                        # Baris terakhir (paling baru)
        prev = df.iloc[-2]                                                        # Baris sebelumnya

        # Jika CSV punya kolom lat/lon gunakan itu; else fallback ke centroid (0)
        if all(c in df.columns for c in ["lat", "lon"]) or all(c in df.columns for c in ["Latitude", "Longitude"]):
            lat_col = "lat" if "lat" in df.columns else ("Latitude" if "Latitude" in df.columns else None)  # Pilih nama kolom lat yang ada
            lon_col = "lon" if "lon" in df.columns else ("Longitude" if "Longitude" in df.columns else None)  # Pilih nama kolom lon yang ada
            lat_prev, lon_prev = safe_float(prev.get(lat_col, np.nan)), safe_float(prev.get(lon_col, np.nan))  # Ambil prev lat/lon aman
            lat_last, lon_last = safe_float(last.get(lat_col, np.nan)), safe_float(last.get(lon_col, np.nan))  # Ambil last lat/lon aman
        else:
            # fallback: gunakan nilai statis (sebaiknya diganti dengan centroid cluster)
            # Jika kamu punya cluster centroid di DB, sebaiknya ambil dari sana
            lat_prev, lon_prev = -7.96, 112.36                                     # Default centroid prev (hardcoded)
            lat_last, lon_last = -7.97, 112.38                                     # Default centroid last (hardcoded)

        dx = lon_last - lon_prev                                                # Perubahan longitude
        dy = lat_last - lat_prev                                                # Perubahan latitude

        # hitung sudut (catatan: formula sesuai implementasimu)
        angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360                    # Hitung sudut gerak dalam derajat (0..360)
        direction = deg_to_dir(angle)                                           # Konversi derajat -> arah kardinal
        distance_km = np.sqrt(dx**2 + dy**2) * 111  # approx degree -> km        # Aproksimasi derajat ke km (~111 km/deg)

        luas = safe_float(last.get("luas_cnn", 0.0))                             # Ambil luas prediksi CNN (km2) aman
        # confidence mapping (customize)
        confidence = min(1.0, max(0.0, float(luas))) if luas >= 0 else 0.0       # Mapping awal confidence dari luas (sederhana)
        # clamp to reasonable
        confidence = round(max(0.0, min(1.0, 0.4 + confidence)), 2)              # Tambah baseline 0.4 lalu clamp & round

        output = {
            "timestamp": datetime.utcnow().isoformat(),                          # Timestamp hasil
            "next_event": {
                "lat": round(lat_last, 6),                                       # Lat dibulatkan 6 desimal
                "lon": round(lon_last, 6),                                       # Lon dibulatkan 6 desimal
                "direction_deg": round(angle, 2),                                # Arah dalam derajat, 2 desimal
                "movement": direction,                                           # Arah kardinal
                "distance_km": round(distance_km, 2),                            # Jarak dalam km, 2 desimal
                "confidence": confidence,                                        # Confidence yang dihitung
                "seismic": round(luas, 2)                                        # Ukuran seismic (luas), 2 desimal
            },
            "status": "ok",                                                       # Status sukses
            "rows": len(df)                                                       # Jumlah baris di CSV
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)                    # Tulis JSON hasil ke file

        logger.info(f"✅ CNN JSON written: {out_json}")                           # Log sukses
        return True                                                               # Kembalikan True sukses

        print("🚀 cnn_csv_to_json.py DIJALANKAN")                                  # (dead code) print post-return
        print("📄 CSV:", CSV_PATH.resolve())                                       # (dead code) debug
        print("🧾 JSON:", OUT_JSON.resolve())                                      # (dead code) debug

    except Exception as e:
        logger.exception(f"Failed to run cnn_csv_to_json: {e}")                    # Log exception lengkap
        try:
            # tulis status minimal agar UI tetap baca
            fallback = {
                "timestamp": datetime.utcnow().isoformat(),                       # Timestamp fallback
                "next_event": None,                                               # next_event null
                "status": "exception",                                            # status exception
                "error": str(e)                                                   # pesan error
            }
            out_json.write_text(json.dumps(fallback, indent=2))                   # Tulis JSON fallback agar UI tidak kosong
        except Exception:
            pass                                                                    # Jika menulis fallback juga gagal, abaikan
        return False                                                              # Return False bila ada exception

if __name__ == "__main__":
    # run with defaults if script dijalankan langsung
    run()                                                                        # Jalankan fungsi utama jika dieksekusi sebagai script
