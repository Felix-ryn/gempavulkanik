# realtime_sensor_manager.py
# -- coding: utf-8 --

import os
import glob
import logging
import pandas as pd
from typing import Optional, Dict

logger = logging.getLogger("RealtimeSensorManager")
logger.setLevel(logging.INFO)

class RealtimeSensorManager:
    """
    Manages real-time volcano and seismic sensor data.
    Auto-load MIROVA, sinkronisasi BMKG, dan injection ke pipeline.
    """
    def __init__(self, mirova_log_path: str = "output/realtime/"):
        # Path MIROVA bisa folder atau file langsung
        self.mirova_log_path = None
        if os.path.isdir(mirova_log_path):
            files = glob.glob(os.path.join(mirova_log_path, "mirova_log_*.txt"))
            if files:
                self.mirova_log_path = max(files, key=os.path.getctime)
                logger.info(f"[MIROVA] Menggunakan log terbaru: {self.mirova_log_path}")
            else:
                logger.warning(f"[MIROVA] Tidak ada file log ditemukan di folder {mirova_log_path}")
        else:
            self.mirova_log_path = mirova_log_path
            if not os.path.exists(self.mirova_log_path):
                logger.warning(f"[MIROVA] File tidak ditemukan: {self.mirova_log_path}")

        # Placeholder untuk data sinkronisasi BMKG
        self.bmkg_data: Optional[pd.DataFrame] = None

    # ----------------------------
    # Bagian MIROVA
    # ----------------------------
    def _fetch_mirova_log(self) -> Optional[pd.DataFrame]:
        if not self.mirova_log_path or not os.path.exists(self.mirova_log_path):
            logger.warning("[MIROVA] Tidak ada file log untuk dibaca.")
            return None
        try:
            df = pd.read_csv(
                self.mirova_log_path,
                sep=r"\s*\|\s*",
                engine="python"
            )
            df.columns = df.columns.str.strip()
            logger.info(f"[MIROVA] Berhasil membaca {len(df)} baris dari log.")
            return df
        except Exception as e:
            logger.error(f"[MIROVA] Gagal membaca log: {e}")
            return None

    # ----------------------------
    # Bagian BMKG (dummy / placeholder)
    # ----------------------------
    def sync_bmkg_data(self, bmkg_csv_path: str):
        """
        Sinkronisasi data BMKG. 
        Bisa diganti dengan API call BMKG asli.
        """
        if not os.path.exists(bmkg_csv_path):
            logger.warning(f"[BMKG] File tidak ditemukan: {bmkg_csv_path}")
            return
        try:
            self.bmkg_data = pd.read_csv(bmkg_csv_path)
            logger.info(f"[BMKG] Berhasil load {len(self.bmkg_data)} baris data.")
        except Exception as e:
            logger.error(f"[BMKG] Gagal membaca file: {e}")

    # ----------------------------
    # Injection / Data Aggregation
    # ----------------------------
    def get_latest_data(self) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Mengembalikan dictionary berisi data terbaru:
        - mirova
        - bmkg
        """
        mirova_df = self._fetch_mirova_log()
        return {
            "mirova": mirova_df,
            "bmkg": self.bmkg_data
        }

    def get_merged_stream(self) -> pd.DataFrame:
        """
        Mengembalikan gabungan data realtime: BMKG + MIROVA + Injected Excel.
        """
        mirova_df = self._fetch_mirova_log()
        bmkg_df = self.bmkg_data.copy() if self.bmkg_data is not None else pd.DataFrame()
    
        # Placeholder: Injected Excel (kosong dulu, bisa diganti jika ada source lain)
        inj_df = pd.DataFrame(columns=['Sumber', 'Acquired_Date', 'lat', 'lon', 'magnitude'])

        # Tambahkan kolom Sumber jika belum ada
        if mirova_df is not None and 'Sumber' not in mirova_df.columns:
            mirova_df['Sumber'] = 'MIROVA'
        if bmkg_df is not None and 'Sumber' not in bmkg_df.columns:
            bmkg_df['Sumber'] = 'BMKG'

        # Gabungkan semua data
        dfs = [df for df in [mirova_df, bmkg_df, inj_df] if df is not None and not df.empty]
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_realtime_data(self):
        """
        Mengembalikan tiga DataFrame:
        1. MIROVA raw data
        2. BMKG sinkronisasi
        3. Injected Excel (dummy placeholder)
        """
        mirova_df = self._fetch_mirova_log()
        bmkg_df = self.bmkg_data.copy() if self.bmkg_data is not None else pd.DataFrame()
        inj_df = pd.DataFrame(columns=['Sumber', 'Acquired_Date', 'lat', 'lon', 'magnitude'])
        
        return mirova_df, bmkg_df, inj_df

# ----------------------------
# Contoh pemakaian
# ----------------------------
if __name__ == "__main__":
    manager = RealtimeSensorManager("output/realtime/")
    manager.sync_bmkg_data("data/bmkg_dummy.csv")  # ganti path sesuai data asli
    data = manager.get_latest_data()

    print("MIROVA:")
    print(data["mirova"].head() if data["mirova"] is not None else "Tidak ada data")
    print("\nBMKG:")
    print(data["bmkg"].head() if data["bmkg"] is not None else "Tidak ada data")
