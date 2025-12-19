# VolcanoAI/processing/data_loader.py
# -- coding: utf-8 --

import os
import re
import math
import logging
import time
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from pandas import merge_asof

try:
    from ..config.config import DataLoaderConfig
except ImportError:
    pass

logger = logging.getLogger("VolcanoAI.DataLoader")
logger.addHandler(logging.NullHandler())

# ==============================================================================
# SECTION 1: DATA UTILS & BASE CLASS
# ==============================================================================

class DataGuard:
    """Utility untuk membersihkan data numerik & datetime."""

    @staticmethod
    def enforce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Paksa kolom menjadi numeric, nilai invalid → 0."""
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df
        
    @staticmethod
    def standardize_datetime_column(df: pd.DataFrame, col: str):
        """Convert kolom tanggal ke datetime dan buang baris invalid."""
        if df.empty:
            return df
        if col not in df.columns:
            logger.warning(f"[DataGuard] Kolom datetime '{col}' tidak ditemukan. Skip.")
            df[col] = pd.NaT
            return df

        df[col] = pd.to_datetime(df[col], errors='coerce')
        df.dropna(subset=[col], inplace=True)
        return df


class DataSource:
    """Loader dasar untuk membaca Excel multi-sheet."""
    def __init__(self, path: str):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        if not os.path.exists(self.path):
            logging.error(f"[DataSource] File tidak ditemukan: {self.path}")
            return False

        try:
            df_sheets = pd.read_excel(self.path, sheet_name=None)

            # Jika punya sheet 'VRP', gunakan itu
            if isinstance(df_sheets, dict) and 'VRP' in df_sheets:
                self.df = df_sheets['VRP']
            # Jika tidak ada 'VRP', ambil sheet pertama
            elif isinstance(df_sheets, dict):
                self.df = list(df_sheets.values())[0]
            else:
                self.df = df_sheets

            return True

        except Exception as e:
            logging.error(f"[DataSource] Gagal membaca file Excel: {e}")
            return False

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.df


# ==============================================================================
# SECTION 2: SUMBER DATA GEMPA & VRP
# ==============================================================================

class EarthquakeSource:
    """Loader dataset gempa BMKG lokal."""
    def __init__(self, path: str):
        self.source = DataSource(path)

    def load_and_clean(self) -> pd.DataFrame:
        if not self.source.load():
            return pd.DataFrame()

        df = self.source.get_dataframe()
        
        # ... (Standarisasi nama kolom)

        df = DataGuard.standardize_datetime_column(df, "Acquired_Date")

        # [FIX KRITIS]: TIDAK MENGISI NaN DENGAN 0 di tahap awal untuk Magnitudo dan Kedalaman.
        # Biarkan NaN (missing values) ditangani oleh Smart Imputer di Feature Engineer.
        # Hanya bersihkan Latitude/Longitude yang kritis untuk DBSCAN/GeoMath.

        # 1. Pastikan Lintang/Bujur Numerik dan drop NaN setelah ini
        df = DataGuard.enforce_numeric(df, [
            "EQ_Lintang", "EQ_Bujur"
        ])
        
        # 2. Magnitudo dan Kedalaman hanya diconvert ke float, JANGAN diisi 0.0
        for col in ["Magnitudo", "Kedalaman (km)"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # [FIX] Drop NaN hanya untuk koordinat dan tanggal
        df.dropna(subset=["Acquired_Date", "EQ_Lintang", "EQ_Bujur"], inplace=True)

        return df.copy()


class VolcanoSource:
    """Loader MIROVA / VRP / MSI (jika tersedia)."""
    def __init__(self, path: str):
        self.source = DataSource(path)

    def load_and_clean(self) -> pd.DataFrame:
        if not self.source.load():
            return pd.DataFrame()

        df = self.source.get_dataframe()

        # Standarisasi nama kolom
        df.rename(columns={
            "Tanggal": "VRP_Date",
            "Gunung": "Nama"
        }, inplace=True)

        # FIX: Aman jika VRP_Date tidak ada
        if "VRP_Date" in df.columns:
            df = DataGuard.standardize_datetime_column(df, "VRP_Date")
        else:
            logger.warning("[VolcanoSource] Kolom VRP_Date tidak ditemukan. Mengisi dengan NaT.")
            df["VRP_Date"] = pd.NaT

        # Ambil kolom thermal: MSI_summit & MSI_total bila ada
        msi_cols = []
        if "MSI_summit (W)" in df.columns:
            msi_cols.append("MSI_summit (W)")
        if "MSI_total (W)" in df.columns:
            msi_cols.append("MSI_total (W)")

        # Ambil VRP maksimal
        vrp_cols = [c for c in df.columns if "VRP" in c or "W)" in c]
        if vrp_cols:
            df["VRP_Max"] = df[vrp_cols].max(axis=1).fillna(0.0)
        else:
            df["VRP_Max"] = 0.0

        df = DataGuard.enforce_numeric(df, ["VRP_Max"] + msi_cols)

        base_cols = ["VRP_Date", "VRP_Max", "Nama"]

        df = df[base_cols + msi_cols].copy()

        return df


# ==============================================================================
# SECTION 3: DataLoader ORKESTRATOR
# ==============================================================================

class DataLoader:

    TARGET_BOUNDING_BOX = {
        "lat_min": -9.0,
        "lat_max": -6.5,
        "lon_min": 111.0,
        "lon_max": 116.0,
    }

    def __init__(self, config: DataLoaderConfig):
        self.cfg = config
        self.cache_path = self.cfg.merged_output_path.replace(".xlsx", ".pkl")

    # ---------------------------------------------------------
    # PIPELINE UTAMA
    # ---------------------------------------------------------

    def run(self) -> pd.DataFrame:

        # -----------------------------
        # 1. LOAD GEMPA
        # -----------------------------
        df_eq = EarthquakeSource(self.cfg.earthquake_data_path).load_and_clean()

        if df_eq.empty:
            logging.critical("[DataLoader] Dataset gempa kosong!")
            return pd.DataFrame()

        df_eq = self._filter_spatial(df_eq)
        df_eq.sort_values("Acquired_Date", inplace=True)

        # -----------------------------
        # 2. LOAD VRP / MSI
        # -----------------------------
        df_vol = VolcanoSource(self.cfg.volcano_data_path).load_and_clean()

        if "VRP_Date" in df_vol.columns:
            df_vol.sort_values("VRP_Date", inplace=True)
        else:
            logger.warning("[DataLoader] VRP_Date tidak ditemukan → skip sorting VRP.")

        # -----------------------------
        # 3. MERGE GEMPA + VRP
        # -----------------------------
        if not df_vol.empty and "VRP_Date" in df_vol.columns:
            df_merged = self._merge_datasets_temporal(df_eq, df_vol)
        else:
            logger.warning("[DataLoader] Data VRP kosong atau invalid → lanjut tanpa VRP.")
            df_merged = df_eq.copy()
            df_merged["VRP_Max"] = 0.0

        # -----------------------------
        # 4. CLEANUP & FORMAT OUTPUT
        # -----------------------------
        df_final = self._cleanup_final(df_merged)

        # FIX: Untuk Excel agar tidak #####
        df_final["Acquired_Date"] = pd.to_datetime(df_final["Acquired_Date"], errors="coerce")
        df_final["Acquired_Date"] = df_final["Acquired_Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # Simpan ke Excel + Pickle
        self._save_output(df_final)

        return df_final

    def load_last_n_days(self) -> pd.DataFrame:
        """
        Mengambil data terakhir N hari berdasarkan konfigurasi
        DataLoaderConfig.hybrid_window_days.
        """
        df = self.run()  # existing loader (Excel / DB / merged)
        if df is None or df.empty:
            return pd.DataFrame()

        if 'Acquired_Date' not in df.columns:
            raise ValueError("Kolom 'Acquired_Date' tidak ditemukan pada dataset")

        df = df.copy()
        df['Acquired_Date'] = pd.to_datetime(df['Acquired_Date'], errors='coerce')

        cutoff = datetime.utcnow() - timedelta(days=self.cfg.hybrid_window_days)
        df_recent = df[df['Acquired_Date'] >= cutoff]

        return df_recent.reset_index(drop=True)

    # ---------------------------------------------------------
    # HELPER FUNCTIONS
    # ---------------------------------------------------------

    def _filter_spatial(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data berdasarkan bounding box area Jawa Timur."""
        b = self.TARGET_BOUNDING_BOX
        mask = (
            (df["EQ_Lintang"] >= b["lat_min"]) &
            (df["EQ_Lintang"] <= b["lat_max"]) &
            (df["EQ_Bujur"] >= b["lon_min"]) &
            (df["EQ_Bujur"] <= b["lon_max"])
        )
        return df[mask].copy()

    def _merge_datasets_temporal(self, df_eq: pd.DataFrame, df_vol: pd.DataFrame) -> pd.DataFrame:
        """Merge gempa & VRP berdasarkan nama gunung + waktu terdekat."""

        # FIX: Jika VRP tidak valid → skip merge
        if df_vol.empty or "VRP_Date" not in df_vol.columns:
            logger.warning("[DataLoader] VRP tidak valid → merge dilewati.")
            df_eq["VRP_Max"] = 0.0
            return df_eq

        df = merge_asof(
            df_eq,
            df_vol,
            left_on="Acquired_Date",
            right_on="VRP_Date",
            by="Nama",
            direction="nearest",
            tolerance=pd.Timedelta(days=self.cfg.date_tolerance_days)
        )

        df.drop(columns=["VRP_Date"], inplace=True, errors="ignore")

        if "VRP_Max" not in df.columns:
            df["VRP_Max"] = 0.0

        df["VRP_Max"] = df["VRP_Max"].fillna(0.0)

        return df

    def _cleanup_final(self, df: pd.DataFrame) -> pd.DataFrame:
        # [FIX]: Hanya enforces numeric untuk kolom Geo, Magnitude/Depth 
        # sudah dihandle dan dibiarkan NaN di langkah sebelumnya.
        df = DataGuard.enforce_numeric(df, [
            "EQ_Lintang", "EQ_Bujur"
        ]) 

        # Pastikan kolom VRP_Max diisi 0 jika NaN
        if "VRP_Max" in df.columns:
            df["VRP_Max"] = df["VRP_Max"].fillna(0.0)

        df["Nama"] = df["Nama"].astype(str).str.strip().str.title()
        df.sort_values("Acquired_Date", inplace=True)

        return df

    def _save_output(self, df: pd.DataFrame):
        """Simpan Excel + Pickle."""
        outdir = os.path.dirname(self.cfg.merged_output_path)
        os.makedirs(outdir, exist_ok=True)

        try:
            df.to_excel(self.cfg.merged_output_path, index=False)
            df.to_pickle(self.cache_path)
        except Exception as e:
            logging.error(f"[DataLoader] Gagal menyimpan file output: {e}")
