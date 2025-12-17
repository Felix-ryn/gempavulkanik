# VolcanoAI/processing/realtime_data_stream.py
# -- coding: utf-8 --

"""
REALTIME SENSOR MANAGER (BMKG + MIROVA + INJECTION EXCEL)
=========================================================

Modul tunggal yang menggabungkan:
  1) BMKG Realtime Sensor (JSON API TEWS)
  2) MIROVA VRP Extractor (OCR-based thermal activity)
  3) Injection Excel Manager (90-day manual injected dataset)

Output:
  - df_mirova_raw
  - df_bmkg_fe_ready
  - df_injected_fe_ready

Catatan:
  • VRP tidak dianggap event sendiri → hanya ditambahkan ke dataset sebagai
    fitur tambahan (VRP_Max).
  • Alignment VRP → tolerance = 1 jam.
  • Semuanya dinormalisasi agar kompatibel dengan FeatureEngineer.
"""

import os
import re
import cv2
import pytesseract
import requests
import logging
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# -----------------------------------------------------------
# 0. LOGGER
# -----------------------------------------------------------
logger = logging.getLogger("RealtimeSensorManager")
logger.setLevel(logging.INFO)


# ===========================================================
# 1. MIROVA SENSOR (VRP Extractor)
# ===========================================================
class MirovaRealtimeSensor:
    """
    Mengambil VRP dari log MIROVA (OCR image extraction)
    Output:
        DataFrame:
            Timestamp
            Gunung
            VRP_Max
    """

    def __init__(self, mirova_log_path: str):
        self.mirova_log_path = mirova_log_path

        # ROI default dari template MIROVA
        self.VRP_ROIS = [
            (210, 255, 60, 180), (210, 255, 230, 350), (210, 255, 400, 520),
            (210, 255, 570, 690), (210, 255, 740, 860),
            (400, 445, 60, 180), (400, 445, 230, 350), (400, 445, 400, 520),
            (400, 445, 570, 690), (400, 445, 740, 860),
        ]

    def _fetch_log(self) -> Optional[pd.DataFrame]:
        if not os.path.exists(self.mirova_log_path):
            logger.warning(f"[MIROVA] Log tidak ditemukan: {self.mirova_log_path}")
            return None
        
        try:
            df = pd.read_csv(
                self.mirova_log_path,
                sep=r"\s*\|\s*",
                engine="python"
            )
            df.columns = df.columns.str.strip()
            return df
        except Exception as e:
            logger.error(f"[MIROVA] Gagal membaca log: {e}")
            return None

    def _download_image(self, url: str):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except:
            return None

    def _extract_vrp_from_roi(self, img):
        _, _, r = cv2.split(img)
        _, thr = cv2.threshold(r, 180, 255, cv2.THRESH_BINARY)
        thr = cv2.resize(thr, None, fx=2.5, fy=2.5)
        try:
            txt = pytesseract.image_to_string(
                thr,
                config='--psm 6 -c tessedit_char_whitelist=VRP=0123456789MW'
            )
            match = re.search(r"VRP\s*=\s*(\d+)\s*MW", txt)
            if match:
                return int(match.group(1))
        except:
            return None
        return None

    def run(self) -> pd.DataFrame:
        df_log = self._fetch_log()
        if df_log is None or df_log.empty:
            return pd.DataFrame()

        results = []
        for row in df_log.itertuples():
            url = getattr(row, "URL Gambar", None)
            if not isinstance(url, str):
                continue

            img = self._download_image(url)
            vrps = []
            if img is not None:
                for (y1, y2, x1, x2) in self.VRP_ROIS:
                    roi = img[y1:y2, x1:x2]
                    val = self._extract_vrp_from_roi(roi)
                    if val is not None:
                        vrps.append(val)

            vrp_max = max(vrps) if vrps else 0

            results.append({
                "Timestamp": datetime.now(),   # fallback tidak ada timestamp di log
                "Gunung": getattr(row, "Gunung", "Unknown"),
                "VRP_Max": vrp_max
            })

        df_out = pd.DataFrame(results)
        logger.info(f"[MIROVA] VRP rows: {len(df_out)}")
        return df_out


# ===========================================================
# 2. BMKG REALTIME SENSOR
# ===========================================================
class BmkgRealtimeSensor:
    """
    Mengambil data gempa realtime dari BMKG TEWS JSON.
    Output: FE-ready event rows (tanpa VRP).
    """
    def __init__(self):
        self.url = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"

    def fetch(self) -> pd.DataFrame:
        try:
            r = requests.get(self.url, timeout=10)
            r.raise_for_status()
            data = r.json().get("Infogempa", {}).get("gempa", [])
        except:
            logger.error("[BMKG] Gagal fetch")
            return pd.DataFrame()

        rows = []
        for g in data:
            try:
                lat, lon = g["Coordinates"].split(",")
            except:
                continue

            dt_str = f"{g['Tanggal']} {g['Jam'].split(' ')[0]}"
            t = pd.to_datetime(dt_str, dayfirst=True, errors="coerce")

            rows.append({
                "Acquired_Date": t,
                "Latitude": float(lat),
                "Longitude": float(lon),
                "Magnitude": float(g.get("Magnitude", 0)),
                "Depth": float(g.get("Kedalaman", "0 km").replace(" km", "")),
                "Nama": g["Wilayah"],
                "Sumber": "BMKG",
                "VRP_Max": 0,   # default
            })

        df = pd.DataFrame(rows)
        df = df.dropna(subset=["Acquired_Date"])
        logger.info(f"[BMKG] Events: {len(df)}")
        return df


# ===========================================================
# 3. INJECTION EXCEL MANAGER (FINAL)
# ===========================================================
class InjectionExcelManager:

    def __init__(self, path="output/realtime/data_vulkanik_90hari.xlsx", max_age=90):
        self.path = Path(path)
        self.max_age = max_age

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            logger.info("[Injection] File tidak ditemukan")
            return pd.DataFrame()

        df_raw = pd.read_excel(self.path)
        if df_raw.empty:
            return pd.DataFrame()

        # Normalisasi nama kolom
        df = df_raw.copy()
        df.columns = df.columns.str.lower()

        # Parsing tanggal
        df["acquired_date"] = pd.to_datetime(
            df["tanggal"].astype(str)
                .str.replace(r"(\d{2})\.(\d{2})$", r"\1:\2", regex=True),
            dayfirst=True,
            errors="coerce"
        )

        df = df.dropna(subset=["acquired_date"])

        # Filter 90 hari
        cutoff = datetime.now() - timedelta(days=self.max_age)
        df = df.loc[df["acquired_date"] >= cutoff]

        if df.empty:
            return pd.DataFrame()

        # Build output (FE-ready)
        out = pd.DataFrame({
            "Acquired_Date": df["acquired_date"],
            "Latitude": pd.to_numeric(df.get("lintang"), errors="coerce"),
            "Longitude": pd.to_numeric(df.get("bujur"), errors="coerce"),
            "Magnitude": pd.to_numeric(df.get("magnitudo"), errors="coerce").fillna(0),
            "Depth": pd.to_numeric(df.get("kedalaman (km)"), errors="coerce").fillna(0),
            "Nama": df.get("lokasi", "Unknown").astype(str),
            "Sumber": "InjectedExcel",
            "VRP_Max": 0
        })

        logger.info(f"[Injection] Rows: {len(out)}")
        return out


# ===========================================================
# 4. ORCHESTRATOR: REALTIME SENSOR MANAGER
# ===========================================================
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

class RealtimeSensorManager:
    """
    Manajer untuk menggabungkan data sensor dari berbagai sumber:
    1. MIROVA (Thermal / VRP)
    2. BMKG (Gempa/Cuaca)
    3. Injection (Data Manual via Excel)
    """

    def __init__(self, 
                 mirova_log_path: str = "output/realtime/mirova_log_20251003-094000.txt",
                 injection_path: str = "output/realtime/data_vulkanik_90hari.xlsx"):
        
        # Simpan path config
        self.mirova_path = mirova_log_path
        self.injection_path = injection_path
        self.time_tolerance = pd.Timedelta("1h")  # Opsi C -> Toleransi merge 1 jam
        
        # Validasi path dasar untuk logging (tidak throw error di init, hanya warning)
        if not os.path.exists(self.mirova_path):
            logger.warning(f"File MIROVA tidak ditemukan di path: {self.mirova_path}")
            
        if not os.path.exists(self.injection_path):
            logger.warning(f"File Injection tidak ditemukan di path: {self.injection_path}")

        # Inisialisasi Sub-Module Sensor
        # Asumsi: Class MirovaRealtimeSensor, BmkgRealtimeSensor, InjectionExcelManager sudah ada
        self.mirova = MirovaRealtimeSensor(self.mirova_path)
        self.bmkg   = BmkgRealtimeSensor()
        self.inject = InjectionExcelManager(path=self.injection_path)

    def get_realtime_data(self):
        """
        Mengambil data dari ketiga sumber dan melakukan sinkronisasi waktu (Merge AsOf).
        Returns: Tuple(df_mirova, df_bmkg, df_inj)
        """
        logger.info("=== RealtimeSensorManager: START Fetching Data ===")

        # -----------------------------------------------------------
        # Step 1: Ambil data raw dari masing-masing sensor
        # -----------------------------------------------------------
        try:
            df_mirova = self.mirova.run()
        except Exception as e:
            logger.error(f"Gagal mengambil data Mirova: {e}")
            df_mirova = pd.DataFrame()

        try:
            df_bmkg = self.bmkg.fetch()
        except Exception as e:
            logger.error(f"Gagal mengambil data BMKG: {e}")
            df_bmkg = pd.DataFrame()

        try:
            df_inj = self.inject.load()
        except Exception as e:
            logger.error(f"Gagal mengambil data Injection: {e}")
            df_inj = pd.DataFrame()

        # -----------------------------------------------------------
        # Step 2: Sinkronisasi VRP (Mirova) ke dataset lain
        # -----------------------------------------------------------
        # Pastikan data mirova tidak kosong dan memiliki kolom yang diperlukan
        if not df_mirova.empty and "Timestamp" in df_mirova.columns:
        
        # ... (Logika Standardisasi df_mirova dan vrp_reference)
            vrp_reference = df_mirova[["Timestamp", "VRP_Max"]].copy()
            vrp_reference = vrp_reference.rename(columns={"Timestamp": "Acquired_Date"})
        
            # --- Merge ke Data BMKG ---
            if not df_bmkg.empty and "Acquired_Date" in df_bmkg.columns:
                df_bmkg["Acquired_Date"] = pd.to_datetime(df_bmkg["Acquired_Date"])
                df_bmkg = df_bmkg.sort_values("Acquired_Date")
            
                # Lakukan merge_asof dengan suffix yang jelas
                df_bmkg_merged = pd.merge_asof(
                    df_bmkg,
                    vrp_reference,
                    on="Acquired_Date",
                    direction="nearest",
                    tolerance=self.time_tolerance,
                    suffixes=('_bmkg', '_mirova') # Tambahkan suffix eksplisit
                )
            
                # [FIX KRITIS 1]: Pindahkan VRP_Max dari Mirova ke kolom VRP_Max asli BMKG
                # Kita mengambil VRP_Max_mirova (jika ada) dan menimpakan ke VRP_Max_bmkg (yang kemudian akan kita rename kembali)
                df_bmkg["VRP_Max"] = df_bmkg_merged["VRP_Max_mirova"].fillna(0) # Gunakan VRP_Max_mirova
            
                # Kita tidak menggunakan df_bmkg_merged, tetapi langsung menimpa df_bmkg
                # Logika merge_asof harusnya digunakan untuk menimpa kolom di df_bmkg
            
                # KARENA KODE ANDA MENIMPA df_bmkg, kita harus menggunakan merge_asof dengan hasil terpisah:
                df_temp = pd.merge_asof(
                    df_bmkg.drop(columns=['VRP_Max'], errors='ignore'), # Hapus VRP_Max lama agar tidak konflik
                    vrp_reference,
                    on="Acquired_Date",
                    direction="nearest",
                    tolerance=self.time_tolerance
                )
                df_bmkg = df_temp.rename(columns={"VRP_Max": "VRP_Max_Merged"}) # Kolom VRP_Max dari vrp_reference
                df_bmkg["VRP_Max"] = df_bmkg["VRP_Max_Merged"].fillna(0)
                df_bmkg = df_bmkg.drop(columns=['VRP_Max_Merged'], errors='ignore')


            # --- Merge ke Data Injection ---
            if not df_inj.empty and "Acquired_Date" in df_inj.columns:
                df_inj["Acquired_Date"] = pd.to_datetime(df_inj["Acquired_Date"])
                df_inj = df_inj.sort_values("Acquired_Date")

                # Lakukan merge untuk Injection
                df_temp = pd.merge_asof(
                    df_inj.drop(columns=['VRP_Max'], errors='ignore'), # Hapus VRP_Max lama
                    vrp_reference,
                    on="Acquired_Date",
                    direction="nearest",
                    tolerance=self.time_tolerance
                )
                df_inj = df_temp.rename(columns={"VRP_Max": "VRP_Max_Merged"})
                df_inj["VRP_Max"] = df_inj["VRP_Max_Merged"].fillna(0)
                df_inj = df_inj.drop(columns=['VRP_Max_Merged'], errors='ignore')

        # ... (Logika else tetap)

        logger.info(f"=== RealtimeSensorManager: DONE. Sizes: M({len(df_mirova)}), B({len(df_bmkg)}), I({len(df_inj)}) ===")
        return df_mirova, df_bmkg, df_inj # Return ini sudah benar, karena main.py menggabungnya

        # =======================================================
        # 5. FINAL MERGED STREAM (BMKG + INJECTION + VRP ALIGN)
        # =======================================================
        def get_merged_stream(self) -> pd.DataFrame:
            df_mirova, df_bmkg, df_inj = self.get_realtime_data()

            frames = []
            if not df_bmkg.empty:
                frames.append(df_bmkg)
            if not df_inj.empty:
                frames.append(df_inj)

            if not frames:
                return pd.DataFrame()

            df_all = pd.concat(frames, ignore_index=True)

            # Sort timestamp
            if "Acquired_Date" in df_all.columns:
                df_all = df_all.sort_values("Acquired_Date")

            # Remove duplicates (Acquired_Date + LatLon)
            if {"Acquired_Date", "Latitude", "Longitude"}.issubset(df_all.columns):
                df_all = df_all.drop_duplicates(
                    subset=["Acquired_Date", "Latitude", "Longitude"],
                    keep="last"
                )

            df_all = df_all.reset_index(drop=True)
            return df_all

