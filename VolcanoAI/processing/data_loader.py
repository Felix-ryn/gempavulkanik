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
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df

    @staticmethod
    def standardize_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Convert kolom tanggal ke datetime dan buang baris invalid."""
        if df.empty:
            return df

        if col not in df.columns:
            logger.warning(
                f"[DataGuard] Kolom datetime '{col}' tidak ditemukan. Skip."
            )
            df[col] = pd.NaT
            return df

        df[col] = pd.to_datetime(df[col], errors="coerce")
        df.dropna(subset=[col], inplace=True)
        return df


class DataSource:
    """Loader dasar untuk membaca Excel multi-sheet."""

    def __init__(self, path: str):
        self.path = path
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> bool:
        if not os.path.exists(self.path):
            logger.error(f"[DataSource] File tidak ditemukan: {self.path}")
            return False

        try:
            df_sheets = pd.read_excel(self.path, sheet_name=None)

            if isinstance(df_sheets, dict) and "VRP" in df_sheets:
                self.df = df_sheets["VRP"]
            elif isinstance(df_sheets, dict):
                self.df = list(df_sheets.values())[0]
            else:
                self.df = df_sheets

            return True

        except Exception as e:
            logger.error(f"[DataSource] Gagal membaca file Excel: {e}")
            return False

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.df


# ==============================================================================
# SECTION 2: SUMBER DATA GEMPA & VRP
# ==============================================================================

class EarthquakeSource:
    """Loader dataset gempa + tambahan 15 hari."""

    def __init__(self, main_path: str, extra_path: Optional[str] = None):
        self.main_source = DataSource(main_path)
        self.extra_path = extra_path

    def _load_single(self, path: str) -> pd.DataFrame:
        src = DataSource(path)
        if not src.load():
            return pd.DataFrame()

        df = src.get_dataframe()

        df.rename(
            columns={
                "Tanggal": "Acquired_Date",
                "Lintang": "EQ_Lintang",
                "Bujur": "EQ_Bujur",
                "Magnitudo": "Magnitudo",
                "Kedalaman (km)": "Kedalaman (km)",
                "Lokasi": "Nama",
            },
            inplace=True,
        )

        for col in ["EQ_Lintang", "EQ_Bujur", "Magnitudo", "Kedalaman (km)"]:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )

        date_candidates = [
            "Acquired_Date",
            "Tanggal",
            "Waktu",
            "Datetime",
            "Date",
            "time",
            "origin_time",
        ]

        found_date_col = None
        for c in date_candidates:
            if c in df.columns:
                found_date_col = c
                break

        if found_date_col is None:
            raise ValueError(
                f"Tidak ditemukan kolom tanggal. Kolom tersedia: {list(df.columns)}"
            )

        df["Acquired_Date"] = pd.to_datetime(
            df[found_date_col], dayfirst=True, errors="coerce"
        )

        df["Nama"] = (
            df["Nama"]
            .astype(str)
            .str.replace("Gunung", "", regex=False)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.title()
        )

        df.dropna(
            subset=["Acquired_Date", "EQ_Lintang", "EQ_Bujur"], inplace=True
        )
        return df

    def load_and_clean(self) -> pd.DataFrame:
        if not self.main_source.load():
            return pd.DataFrame()

        df_main = self.main_source.get_dataframe()
        df_main = DataGuard.standardize_datetime_column(
            df_main, "Acquired_Date"
        )

        if "Nama" not in df_main.columns:
            df_main["Nama"] = "Unknown"

        df_main["Nama"] = (
            df_main["Nama"]
            .astype(str)
            .str.replace("Gunung", "", regex=False)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.title()
        )

        df_extra = (
            self._load_single(self.extra_path)
            if self.extra_path and os.path.exists(self.extra_path)
            else pd.DataFrame()
        )

        df_all = pd.concat([df_main, df_extra], ignore_index=True)
        df_all.sort_values("Acquired_Date", inplace=True)

        return df_all


class VolcanoSource:
    """Loader MIROVA / VRP / MSI (jika tersedia)."""

    def __init__(self, path: str):
        self.source = DataSource(path)

    def load_and_clean(self) -> pd.DataFrame:
        if not self.source.load():
            return pd.DataFrame()

        df = self.source.get_dataframe()

        df.rename(
            columns={
                "Tanggal": "VRP_Date",
                "Gunung": "Nama",
            },
            inplace=True,
        )

        if "VRP_Date" in df.columns:
            df = DataGuard.standardize_datetime_column(df, "VRP_Date")
        else:
            logger.warning(
                "[VolcanoSource] Kolom VRP_Date tidak ditemukan. Mengisi NaT."
            )
            df["VRP_Date"] = pd.NaT

        msi_cols = []
        if "MSI_summit (W)" in df.columns:
            msi_cols.append("MSI_summit (W)")
        if "MSI_total (W)" in df.columns:
            msi_cols.append("MSI_total (W)")

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
# SECTION 3: DATA LOADER ORKESTRATOR
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

    def run(self) -> pd.DataFrame:

        df_eq = EarthquakeSource(
            self.cfg.earthquake_data_path,
            extra_path=(
                "C:/Users/USER/Downloads/Earthquake_Volcan/"
                "Earthquake_Volcano/data/Data 15 Hari.xlsx"
            ),
        ).load_and_clean()

        if df_eq.empty:
            logger.critical("[DataLoader] Dataset gempa kosong!")
            return pd.DataFrame()

        df_eq = self._filter_spatial(df_eq)
        df_eq.sort_values(["Nama", "Acquired_Date"], inplace=True)

        df_vol = pd.DataFrame()

        if self.cfg.volcanic_data_path and os.path.exists(
            self.cfg.volcanic_data_path
        ):
            df_vol = VolcanoSource(
                self.cfg.volcanic_data_path
            ).load_and_clean()

            if not df_vol.empty and "VRP_Date" in df_vol.columns:
                df_vol.sort_values(
                    ["Nama", "VRP_Date"], inplace=True
                )
        else:
            logger.info("[DataLoader] Data VRP tidak disediakan.")

        if not df_vol.empty and "VRP_Date" in df_vol.columns:
            df_merged = self._merge_datasets_temporal(df_eq, df_vol)
        else:
            df_merged = df_eq.copy()
            df_merged["VRP_Max"] = 0.0

        df_final = self._cleanup_final(df_merged)

        df_final["Acquired_Date"] = pd.to_datetime(
            df_final["Acquired_Date"], errors="coerce"
        )
        df_final["Acquired_Date"] = df_final[
            "Acquired_Date"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")

        self._save_output(df_final)
        return df_final

    def load_last_n_days(self) -> pd.DataFrame:
        df = self.run()
        if df.empty:
            return pd.DataFrame()

        df["Acquired_Date"] = pd.to_datetime(
            df["Acquired_Date"], errors="coerce"
        )
        cutoff = datetime.utcnow() - timedelta(
            days=self.cfg.hybrid_window_days
        )
        return df[df["Acquired_Date"] >= cutoff].reset_index(drop=True)

    def _filter_spatial(self, df: pd.DataFrame) -> pd.DataFrame:
        b = self.TARGET_BOUNDING_BOX
        mask = (
            (df["EQ_Lintang"] >= b["lat_min"])
            & (df["EQ_Lintang"] <= b["lat_max"])
            & (df["EQ_Bujur"] >= b["lon_min"])
            & (df["EQ_Bujur"] <= b["lon_max"])
        )
        return df[mask].copy()

    def _merge_datasets_temporal(
        self, df_eq: pd.DataFrame, df_vol: pd.DataFrame
    ) -> pd.DataFrame:
        # ===============================
        # 1. Pastikan kolom waktu DATETIME
        # ===============================
        df_eq["Acquired_Date"] = pd.to_datetime(
            df_eq.get("Acquired_Date"), errors="coerce"
        )
        df_vol["VRP_Date"] = pd.to_datetime(
            df_vol.get("VRP_Date"), errors="coerce"
        )

        # ===============================
        # 2. Drop baris tanpa waktu
        # ===============================
        df_eq = df_eq.dropna(subset=["Acquired_Date"])
        df_vol = df_vol.dropna(subset=["VRP_Date"])

        # ===============================
        # 3. SORT WAJIB (INI KUNCI ERROR KAMU)
        # ===============================
        df_eq = df_eq.sort_values("Acquired_Date").reset_index(drop=True)
        df_vol = df_vol.sort_values("VRP_Date").reset_index(drop=True)

        # ===============================
        # 4. merge_asof (AMAN)
        # ===============================
        df = pd.merge_asof(
            df_eq,
            df_vol,
            left_on="Acquired_Date",
            right_on="VRP_Date",
            by="Nama",
            direction="nearest",
            tolerance=pd.Timedelta(days=self.cfg.date_tolerance_days),
        )

        # ===============================
        # 5. Cleanup
        # ===============================
        df.drop(columns=["VRP_Date"], inplace=True, errors="ignore")

        if "VRP_Max" not in df.columns:
            df["VRP_Max"] = 0.0
        else:
            df["VRP_Max"] = df["VRP_Max"].fillna(0.0)

        return df

    def _cleanup_final(self, df: pd.DataFrame) -> pd.DataFrame:
        df = DataGuard.enforce_numeric(
            df, ["EQ_Lintang", "EQ_Bujur"]
        )

        if "VRP_Max" in df.columns:
            df["VRP_Max"] = df["VRP_Max"].fillna(0.0)

        df["Nama"] = df["Nama"].astype(str).str.strip().str.title()
        df.sort_values("Acquired_Date", inplace=True)
        return df

    def _save_output(self, df: pd.DataFrame):
        outdir = os.path.dirname(self.cfg.merged_output_path)
        os.makedirs(outdir, exist_ok=True)

        try:
            df.to_excel(self.cfg.merged_output_path, index=False)
            df.to_pickle(self.cache_path)
        except Exception as e:
            logger.error(
                f"[DataLoader] Gagal menyimpan file output: {e}"
            )
