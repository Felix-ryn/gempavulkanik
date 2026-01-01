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
# DATA GUARD
# ==========================================================

class DataGuard:
    @staticmethod
    def enforce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df


# ==========================================================
# BASIC DATA SOURCE
# ==========================================================

class DataSource:
    def __init__(self, path: str):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        if not self.path or not os.path.exists(self.path):
            logger.error(f"[DataSource] File tidak ditemukan: {self.path}")
            return False

        try:
            if self.path.lower().endswith((".xls", ".xlsx")):
                xls = pd.ExcelFile(self.path)
                logger.info(f"[DataSource] Sheets in {self.path}: {xls.sheet_names}")
                self.df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
            else:
                self.df = pd.read_csv(self.path)

            if self.df.empty:
                logger.warning(f"[DataSource] File kosong: {self.path}")
            return True

        except Exception as e:
            logger.error(f"[DataSource] Gagal membaca {self.path}: {e}")
            return False

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.df


# ==========================================================
# EARTHQUAKE SOURCE (MAIN + EXTRA)
# ==========================================================

class EarthquakeSource:
    """
    MAIN = Volcanic_Earthquake_Data.xlsx
    EXTRA = Data 15 Hari.xlsx (HANYA TAMBAHAN)
    """

    def __init__(self, main_path: str, extra_path: Optional[str] = None):
        self.main_path = main_path
        self.extra_path = extra_path

    def _load_single(self, path: str) -> pd.DataFrame:
        src = DataSource(path)
        if not src.load():
            return pd.DataFrame()

        df = src.get_dataframe()
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "Tanggal": "Acquired_Date",
            "Waktu": "Acquired_Date",
            "Lintang": "EQ_Lintang",
            "Bujur": "EQ_Bujur",
            "Lokasi": "Nama"
        })

        if "Acquired_Date" in df.columns:
            df["Acquired_Date"] = pd.to_datetime(df["Acquired_Date"], errors="coerce", dayfirst=True)

        for c in ["EQ_Lintang", "EQ_Bujur", "Magnitudo", "Kedalaman (km)"]:
            if c in df.columns:
                df[c] = (
                    df[c].astype(str)
                    .str.replace(",", ".", regex=False)
                )
                df[c] = pd.to_numeric(df[c], errors="coerce")

        if "Nama" not in df.columns:
            df["Nama"] = "Unknown"

        df["Nama"] = (
            df["Nama"].astype(str)
            .str.replace("Gunung", "", regex=False)
            .str.split(",").str[0]
            .str.strip().str.title()
        )

        df = df.dropna(subset=["Acquired_Date"])
        return df.reset_index(drop=True)

    def load_and_clean(self) -> pd.DataFrame:
        # ========= MAIN (WAJIB) =========
        df_main = self._load_single(self.main_path)
        if df_main.empty:
            logger.critical("[EarthquakeSource] DATA UTAMA KOSONG!")
            return pd.DataFrame()

        # ========= EXTRA (OPTIONAL) =========
        df_extra = pd.DataFrame()
        if self.extra_path and os.path.exists(self.extra_path):
            df_extra = self._load_single(self.extra_path)

        logger.info(f"[EarthquakeSource] Rows main={len(df_main)}, extra={len(df_extra)}")
        print(f"[DataLoader] Rows main={len(df_main)}, extra={len(df_extra)}")

        # ========= GABUNG (MAIN TETAP DOMINAN) =========
        if not df_extra.empty:
            df_all = pd.concat([df_main, df_extra], ignore_index=True, sort=False)
        else:
            df_all = df_main.copy()

        # ========= CLEAN =========
        df_all = df_all.drop_duplicates(
            subset=[c for c in ["Acquired_Date", "EQ_Lintang", "EQ_Bujur", "Nama"] if c in df_all.columns]
        )

        df_all = df_all.sort_values("Acquired_Date").reset_index(drop=True)
        return df_all


# ==========================================================
# DATA LOADER ORCHESTRATOR
# ==========================================================

class DataLoader:
    TARGET_BOUNDING_BOX = {
        "lat_min": -9.0,
        "lat_max": -6.5,
        "lon_min": 111.0,
        "lon_max": 116.0,
    }

    def __init__(self, config: DataLoaderConfig):
        self.cfg = config
        self.base_dir = Path(__file__).resolve().parents[2]

        def resolve(p):
            if not p:
                return None
            if os.path.isabs(p):
                return p
            return str(self.base_dir / p)

        self.earthquake_main_path = resolve(self.cfg.earthquake_data_path)
        self.earthquake_extra_path = resolve(getattr(self.cfg, "earthquake_extra_path", None))
        self.cache_path = resolve(self.cfg.merged_output_path).replace(".xlsx", ".pkl")

    def run(self) -> pd.DataFrame:
        df_eq = EarthquakeSource(
            self.earthquake_main_path,
            extra_path=self.earthquake_extra_path
        ).load_and_clean()

        if df_eq.empty:
            logger.critical("[DataLoader] Dataset kosong setelah load.")
            return pd.DataFrame()

        df_eq = self._filter_spatial(df_eq)

        if "Nama" in df_eq.columns and "Acquired_Date" in df_eq.columns:
            df_eq = df_eq.sort_values(["Nama", "Acquired_Date"])

        df_eq["VRP_Max"] = 0.0
        self._save(df_eq)
        return df_eq.reset_index(drop=True)

    def _filter_spatial(self, df: pd.DataFrame) -> pd.DataFrame:
        if "EQ_Lintang" not in df.columns or "EQ_Bujur" not in df.columns:
            return df

        b = self.TARGET_BOUNDING_BOX
        mask = (
            (df["EQ_Lintang"] >= b["lat_min"]) &
            (df["EQ_Lintang"] <= b["lat_max"]) &
            (df["EQ_Bujur"] >= b["lon_min"]) &
            (df["EQ_Bujur"] <= b["lon_max"])
        )
        logger.info(f"[DataLoader] Spatial filter: {mask.sum()}/{len(df)} kept")
        return df[mask].copy()

    def _save(self, df: pd.DataFrame):
        outdir = os.path.dirname(self.cfg.merged_output_path)
        os.makedirs(outdir, exist_ok=True)
        df.to_excel(self.cfg.merged_output_path, index=False)
        df.to_pickle(self.cache_path)
        logger.info(f"[DataLoader] Output saved -> {self.cfg.merged_output_path}")
