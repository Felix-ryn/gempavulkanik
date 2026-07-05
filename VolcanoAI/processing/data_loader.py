# VolcanoAI/processing/data_loader.py
# -- coding: utf-8 --

import os
import logging
from typing import Optional, List
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

try:
    from ..config.config import DataLoaderConfig
except Exception:
    DataLoaderConfig = None

logger = logging.getLogger("VolcanoAI.DataLoader")
logger.addHandler(logging.NullHandler())

# ==========================================================
# DATA GUARD & SOURCE
# ==========================================================
class DataGuard:
    @staticmethod
    def enforce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

class DataSource:
    def __init__(self, path: str):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        if not self.path or not os.path.exists(self.path):
            return False
        try:
            if self.path.lower().endswith((".xls", ".xlsx")):
                xls = pd.ExcelFile(self.path)
                self.df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            else:
                self.df = pd.read_csv(self.path, delimiter=None, engine='python')
            return True
        except Exception as e:
            logger.error(f"[DataSource] Gagal membaca {self.path}: {e}")
            return False

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.df

# ==========================================================
# EARTHQUAKE SOURCE
# ==========================================================
class EarthquakeSource:
    def __init__(self, main_path: str, extra_path: Optional[str] = None):
        self.main_path = main_path
        self.extra_path = extra_path

    def _load_single(self, path: str) -> pd.DataFrame:
        src = DataSource(path)
        if not src.load() or src.get_dataframe() is None or src.get_dataframe().empty:
            return pd.DataFrame()

        df = src.get_dataframe()

        # Standarisasi Nama Kolom
        df = df.rename(columns={
            "Tanggal": "Acquired_Date", "Waktu": "Acquired_Date",
            "Lintang": "EQ_Lintang", "Latitude": "EQ_Lintang",
            "Bujur": "EQ_Bujur", "Longitude": "EQ_Bujur",
            "Lokasi": "Nama"
        })

        # [FIX KRITIS 1]: Parsing Waktu Cerdas (Tahan Banting)
        if "Acquired_Date" in df.columns:
            # Pastikan jika dia pandas datetime, dibiarkan saja. Jika string, perbaiki.
            df['Acquired_Date'] = pd.to_datetime(
                df['Acquired_Date'].astype(str).str.replace(r'(\d{2})\.(\d{2})\.(\d{2})', r'\1:\2:\3', regex=True),
                errors='coerce'
            )

        # [FIX KRITIS 2]: Parsing Angka Indonesia (Koma ke Titik)
        for c in ["EQ_Lintang", "EQ_Bujur", "Magnitudo", "Kedalaman (km)", "Depth"]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.replace(",", ".", regex=False)
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "Nama" not in df.columns:
            df["Nama"] = "Unknown"
        df["Nama"] = df["Nama"].astype(str).str.replace("Gunung", "", regex=False).str.split(",").str[0].str.strip().str.title()

        df = df.dropna(subset=["Acquired_Date"]).reset_index(drop=True)
        return df

    def load_and_clean(self) -> pd.DataFrame:
        df_main = self._load_single(self.main_path)
        
        df_extra = pd.DataFrame()
        if self.extra_path and os.path.exists(self.extra_path):
            df_extra = self._load_single(self.extra_path)

        if not df_extra.empty:
            df_all = pd.concat([df_main, df_extra], ignore_index=True, sort=False)
        else:
            df_all = df_main.copy()

        # [FIX KRITIS 3]: JANGAN BUANG DATA BARU! Gunakan keep='last'
        if not df_all.empty:
            df_all = df_all.drop_duplicates(
                subset=[c for c in ["Acquired_Date", "EQ_Lintang", "EQ_Bujur"] if c in df_all.columns],
                keep="last"
            )
            df_all = df_all.sort_values("Acquired_Date").reset_index(drop=True)
            
        return df_all

# ==========================================================
# DATA LOADER ORCHESTRATOR
# ==========================================================
class DataLoader:
    # Memperlebar Bounding Box agar tidak memblokir data Ijen/Raung/Semeru
    TARGET_BOUNDING_BOX = {
        "lat_min": -15.0, "lat_max": 0.0,   
        "lon_min": 100.0, "lon_max": 130.0,  
    }

    def __init__(self, config: DataLoaderConfig):
        self.cfg = config
        self.base_dir = Path(__file__).resolve().parents[2]

        def resolve(p):
            return str(self.base_dir / p) if p and not os.path.isabs(p) else p

        self.earthquake_main_path = resolve(self.cfg.earthquake_data_path)
        self.earthquake_extra_path = resolve(getattr(self.cfg, "earthquake_extra_path", None))
        
        # Cari otomatis Data 15 Hari.xlsx
        if not self.earthquake_extra_path:
            auto_path = self.base_dir / "data" / "Data 15 Hari.xlsx"
            if auto_path.exists():
                self.earthquake_extra_path = str(auto_path)

        self.cache_path = resolve(self.cfg.merged_output_path).replace(".xlsx", ".pkl")

    def run(self) -> pd.DataFrame:
        # [FIX KRITIS 4]: Selalu hancurkan cache!
        if os.path.exists(self.cache_path):
            try: os.remove(self.cache_path)
            except: pass
                
        # [SUDAH SAYA PERBAIKI DI SINI, SEBELUMNYA SALAH KETIK self.extra_path]
        df_eq = EarthquakeSource(self.earthquake_main_path, self.earthquake_extra_path).load_and_clean()

        if df_eq.empty:
            return pd.DataFrame()

        df_eq = self._filter_spatial(df_eq)
        
        # [FIX KRITIS 5]: MATIKAN FILTER CUTOFF AGAR DATA 2026 LOLOS
        # window_days = getattr(self.cfg, "window_days", None)
        # if window_days is not None and window_days > 0:
        #     cutoff = datetime.utcnow() - timedelta(days=window_days)
        #     df_eq = df_eq[df_eq["Acquired_Date"] >= cutoff]

        if "Nama" in df_eq.columns and "Acquired_Date" in df_eq.columns:
            df_eq = df_eq.sort_values(["Nama", "Acquired_Date"])

        df_eq["VRP_Max"] = 0.0
        self._save(df_eq)
        
        # Validasi Visual di Terminal
        if len(df_eq) > 0:
            print(f"\n[VALIDASI DATA LOADER] Tanggal Data Terakhir yang Lolos: {df_eq['Acquired_Date'].max()}\n")
            
        return df_eq.reset_index(drop=True)

    def _filter_spatial(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EQ_Lintang" not in df.columns or "EQ_Bujur" not in df.columns:
            return df
        b = self.TARGET_BOUNDING_BOX
        mask = ((df["EQ_Lintang"] >= b["lat_min"]) & (df["EQ_Lintang"] <= b["lat_max"]) &
                (df["EQ_Bujur"] >= b["lon_min"]) & (df["EQ_Bujur"] <= b["lon_max"]))
        return df[mask].copy()

    def _save(self, df: pd.DataFrame):
        outdir = os.path.dirname(self.cfg.merged_output_path)
        os.makedirs(outdir, exist_ok=True)
        df.to_excel(self.cfg.merged_output_path, index=False)
        df.to_pickle(self.cache_path)