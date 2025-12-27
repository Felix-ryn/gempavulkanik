# VolcanoAI/processing/data_loader.py                                                  # File loader data utama
# -- coding: utf-8 --                                                                 # Encoding UTF-8

import os                                                                              # modul operasi file/direktori
import re                                                                              # modul regex
import math                                                                            # modul matematika
import logging                                                                         # modul logging
import time                                                                            # modul time
from typing import Optional, Tuple, Dict, Any, List                                   # type hints
from pathlib import Path                                                               # Path object modern
from datetime import timedelta, datetime                                              # timedelta & datetime untuk waktu

import pandas as pd                                                                   # pandas untuk manipulasi tabel
import numpy as np                                                                    # numpy untuk operasi numerik
from pandas import merge_asof                                                          # merge_asof untuk join temporal

try:
    from ..config.config import DataLoaderConfig                                       # import konfigurasi jika tersedia
except ImportError:
    pass                                                                              # lanjut tanpa config jika tidak ada

logger = logging.getLogger("VolcanoAI.DataLoader")                                    # logger khusus DataLoader
logger.addHandler(logging.NullHandler())                                              # tambahkan NullHandler default

# ==============================================================================
# SECTION 1: DATA UTILS & BASE CLASS
# ==============================================================================

class DataGuard:                                                                       # Utility untuk pembersihan data
    """Utility untuk membersihkan data numerik & datetime."""                        # docstring singkat

    @staticmethod
    def enforce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:            # paksa kolom menjadi numeric
        """Paksa kolom menjadi numeric, nilai invalid → 0."""                          # docstring
        for col in cols:                                                               # loop setiap kolom
            if col in df.columns:                                                      # jika kolom ada
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)         # konversi ke numeric dan isi NA dengan 0
        return df                                                                      # kembalikan df

    @staticmethod
    def standardize_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:       # normalisasi kolom datetime
        """Convert kolom tanggal ke datetime dan buang baris invalid."""               # docstring
        if df.empty:                                                                   # jika df kosong
            return df                                                                  # kembalikan df kosong

        if col not in df.columns:                                                      # jika kolom tidak ada
            logger.warning(                                                            # log warning
                f"[DataGuard] Kolom datetime '{col}' tidak ditemukan. Skip."
            )
            df[col] = pd.NaT                                                           # buat kolom dengan NaT
            return df                                                                  # kembalikan df

        df[col] = pd.to_datetime(df[col], errors="coerce")                            # parse kolom menjadi datetime (coerce invalid)
        df.dropna(subset=[col], inplace=True)                                          # buang baris dengan tanggal invalid
        return df                                                                      # kembalikan df


class DataSource:                                                                     # Loader dasar untuk sumber Excel
    """Loader dasar untuk membaca Excel multi-sheet."""                               # docstring

    def __init__(self, path: str):
        self.path = path                                                              # simpan path file
        self.df: Optional[pd.DataFrame] = None                                        # inisialisasi df None

    def load(self) -> bool:
        if not os.path.exists(self.path):                                             # cek file ada
            logger.error(f"[DataSource] File tidak ditemukan: {self.path}")           # log error
            return False                                                              # return False jika tidak ada

        try:
            df_sheets = pd.read_excel(self.path, sheet_name=None)                     # baca semua sheet Excel

            if isinstance(df_sheets, dict) and "VRP" in df_sheets:                    # jika sheet VRP ada
                self.df = df_sheets["VRP"]                                            # pakai sheet VRP
            elif isinstance(df_sheets, dict):
                self.df = list(df_sheets.values())[0]                                 # pakai sheet pertama
            else:
                self.df = df_sheets                                                     # jika bentuk lain, langsung pakai

            return True                                                               # load sukses

        except Exception as e:
            logger.error(f"[DataSource] Gagal membaca file Excel: {e}")                # log error baca Excel
            return False                                                              # load gagal

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        return self.df                                                                 # kembalikan DataFrame yang dimuat


# ==============================================================================
# SECTION 2: SUMBER DATA GEMPA & VRP
# ==============================================================================

class EarthquakeSource:                                                               # Loader dataset gempa + tambahan
    """Loader dataset gempa + tambahan 15 hari."""                                   # docstring

    def __init__(self, main_path: str, extra_path: Optional[str] = None):
        self.main_source = DataSource(main_path)                                      # inisialisasi DataSource utama
        self.extra_path = extra_path                                                  # path data ekstra jika ada

    def _load_single(self, path: str) -> pd.DataFrame:
        src = DataSource(path)                                                        # buat DataSource untuk path
        if not src.load():                                                            # coba load
            return pd.DataFrame()                                                     # jika gagal, kembalikan df kosong

        df = src.get_dataframe()                                                      # ambil DataFrame

        df.rename(                                                                    # rename kolom ke standar internal
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
            if col in df.columns:                                                     # jika kolom ada
                df[col] = (                                                           # bersihkan angka dengan koma jadi titik lalu float
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )

        date_candidates = [                                                           # daftar kandidat nama kolom tanggal
            "Acquired_Date",
            "Tanggal",
            "Waktu",
            "Datetime",
            "Date",
            "time",
            "origin_time",
        ]

        found_date_col = None                                                          # inisialisasi penanda kolom tanggal ditemukan
        for c in date_candidates:
            if c in df.columns:
                found_date_col = c                                                     # set kolom tanggal yang ditemukan
                break

        if found_date_col is None:
            raise ValueError(                                                           # jika tidak ada kolom tanggal, raise error
                f"Tidak ditemukan kolom tanggal. Kolom tersedia: {list(df.columns)}"
            )

        df["Acquired_Date"] = pd.to_datetime(                                          # parse kolom tanggal yang ditemukan
            df[found_date_col], dayfirst=True, errors="coerce"
        )

        df["Nama"] = (                                                                  # bersihkan kolom Nama (hapus kata 'Gunung', ambil before comma, title case)
            df["Nama"]
            .astype(str)
            .str.replace("Gunung", "", regex=False)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.title()
        )

        df.dropna(                                                                      # buang baris tanpa acquired_date atau koordinat
            subset=["Acquired_Date", "EQ_Lintang", "EQ_Bujur"], inplace=True
        )
        return df                                                                       # kembalikan df bersih

    def load_and_clean(self) -> pd.DataFrame:
        if not self.main_source.load():                                                 # load sumber utama
            return pd.DataFrame()                                                       # jika gagal, kembalikan df kosong

        df_main = self.main_source.get_dataframe()                                      # ambil df utama
        df_main = DataGuard.standardize_datetime_column(                                # standardisasi kolom tanggal utama
            df_main, "Acquired_Date"
        )

        if "Nama" not in df_main.columns:
            df_main["Nama"] = "Unknown"                                                 # isi Nama jika tidak ada

        df_main["Nama"] = (                                                              # bersihkan kolom Nama seperti di _load_single
            df_main["Nama"]
            .astype(str)
            .str.replace("Gunung", "", regex=False)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.title()
        )

        df_extra = (                                                                     # load extra jika path diberikan dan file ada
            self._load_single(self.extra_path)
            if self.extra_path and os.path.exists(self.extra_path)
            else pd.DataFrame()
        )

        df_all = pd.concat([df_main, df_extra], ignore_index=True)                       # gabungkan main + extra
        df_all.sort_values("Acquired_Date", inplace=True)                                # urutkan berdasarkan tanggal

        return df_all                                                                    # kembalikan gabungan


class VolcanoSource:                                                                  # Loader MIROVA/VRP/MSI jika tersedia
    """Loader MIROVA / VRP / MSI (jika tersedia)."""                                 # docstring

    def __init__(self, path: str):
        self.source = DataSource(path)                                                 # inisialisasi DataSource

    def load_and_clean(self) -> pd.DataFrame:
        if not self.source.load():                                                     # load sumber
            return pd.DataFrame()                                                      # return kosong jika gagal

        df = self.source.get_dataframe()                                               # ambil df

        df.rename(                                                                     # rename kolom ke standar VRP
            columns={
                "Tanggal": "VRP_Date",
                "Gunung": "Nama",
            },
            inplace=True,
        )

        if "VRP_Date" in df.columns:
            df = DataGuard.standardize_datetime_column(df, "VRP_Date")                 # standardisasi VRP_Date jika ada
        else:
            logger.warning(                                                            # log jika kolom VRP_Date tidak ada
                "[VolcanoSource] Kolom VRP_Date tidak ditemukan. Mengisi NaT."
            )
            df["VRP_Date"] = pd.NaT                                                   # isi VRP_Date dengan NaT

        msi_cols = []                                                                  # kumpulan kolom MSI yang ada
        if "MSI_summit (W)" in df.columns:
            msi_cols.append("MSI_summit (W)")                                          # tambahkan kolom MSI summit jika ada
        if "MSI_total (W)" in df.columns:
            msi_cols.append("MSI_total (W)")                                           # tambahkan kolom MSI total jika ada

        vrp_cols = [c for c in df.columns if "VRP" in c or "W)" in c]                  # cari kolom yang berkaitan VRP/W
        if vrp_cols:
            df["VRP_Max"] = df[vrp_cols].max(axis=1).fillna(0.0)                       # ambil max VRP dari kolom terkait
        else:
            df["VRP_Max"] = 0.0                                                        # default 0 jika tidak ada

        df = DataGuard.enforce_numeric(df, ["VRP_Max"] + msi_cols)                     # paksa numeric pada VRP dan MSI cols

        base_cols = ["VRP_Date", "VRP_Max", "Nama"]                                     # kolom dasar output
        df = df[base_cols + msi_cols].copy()                                            # seleksi kolom final

        return df                                                                        # kembalikan df VRP

# ==============================================================================
# SECTION 3: DATA LOADER ORKESTRATOR
# ==============================================================================

class DataLoader:                                                                     # Orkestrator utama untuk load & merge data

    TARGET_BOUNDING_BOX = {                                                           # batas wilayah target (lat/lon)
        "lat_min": -9.0,
        "lat_max": -6.5,
        "lon_min": 111.0,
        "lon_max": 116.0,
    }

    def __init__(self, config: DataLoaderConfig):
        self.cfg = config                                                             # simpan config
        self.cache_path = self.cfg.merged_output_path.replace(".xlsx", ".pkl")        # path cache pickle dari output xlsx

    def run(self) -> pd.DataFrame:

        df_eq = EarthquakeSource(                                                     # inisialisasi EarthquakeSource dengan extra path hardcoded
            self.cfg.earthquake_data_path,
            extra_path=(
                "C:/Users/USER/Downloads/Earthquake_Volcan/"
                "Earthquake_Volcano/data/Data 15 Hari.xlsx"
            ),
        ).load_and_clean()                                                             # load dan bersihkan data gempa

        if df_eq.empty:
            logger.critical("[DataLoader] Dataset gempa kosong!")                       # log kritikal jika dataset gempa kosong
            return pd.DataFrame()                                                       # kembalikan df kosong

        df_eq = self._filter_spatial(df_eq)                                            # filter berdasarkan bounding box
        df_eq.sort_values(["Nama", "Acquired_Date"], inplace=True)                     # urutkan berdasarkan nama & tanggal

        df_vol = pd.DataFrame()                                                         # inisialisasi df volcanic kosong

        if self.cfg.volcanic_data_path and os.path.exists(
            self.cfg.volcanic_data_path
        ):
            df_vol = VolcanoSource(
                self.cfg.volcanic_data_path
            ).load_and_clean()                                                         # load data volcanic jika tersedia

            if not df_vol.empty and "VRP_Date" in df_vol.columns:
                df_vol.sort_values(
                    ["Nama", "VRP_Date"], inplace=True
                )                                                                      # urutkan df_vol jika ada data
        else:
            logger.info("[DataLoader] Data VRP tidak disediakan.")                       # log info jika tidak ada VRP

        if not df_vol.empty and "VRP_Date" in df_vol.columns:
            df_merged = self._merge_datasets_temporal(df_eq, df_vol)                    # merge temporal jika VRP ada
        else:
            df_merged = df_eq.copy()                                                    # salin df_eq
            df_merged["VRP_Max"] = 0.0                                                  # isi default VRP_Max = 0

        df_final = self._cleanup_final(df_merged)                                      # final cleanup

        df_final["Acquired_Date"] = pd.to_datetime(
            df_final["Acquired_Date"], errors="coerce"
        )                                                                              # pastikan datetime

        df_final["Acquired_Date"] = df_final[
            "Acquired_Date"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")                                              # format Acquired_Date menjadi string

        self._save_output(df_final)                                                     # simpan hasil ke file
        return df_final                                                                 # return df final

    def load_last_n_days(self) -> pd.DataFrame:
        df = self.run()                                                                 # run full loader
        if df.empty:
            return pd.DataFrame()                                                       # jika kosong, return kosong

        df["Acquired_Date"] = pd.to_datetime(
            df["Acquired_Date"], errors="coerce"
        )                                                                              # parse kembali Acquired_Date ke datetime
        cutoff = datetime.utcnow() - timedelta(
            days=self.cfg.hybrid_window_days
        )                                                                              # cutoff berdasarkan hybrid_window_days
        return df[df["Acquired_Date"] >= cutoff].reset_index(drop=True)                 # filter berdasarkan cutoff dan reset index

    def _filter_spatial(self, df: pd.DataFrame) -> pd.DataFrame:
        b = self.TARGET_BOUNDING_BOX                                                       # ambil bounding box
        mask = (
            (df["EQ_Lintang"] >= b["lat_min"])
            & (df["EQ_Lintang"] <= b["lat_max"])
            & (df["EQ_Bujur"] >= b["lon_min"])
            & (df["EQ_Bujur"] <= b["lon_max"])
        )                                                                              # buat mask boolean untuk spatial filter
        return df[mask].copy()                                                          # kembalikan df yang difilter

    def _merge_datasets_temporal(
        self, df_eq: pd.DataFrame, df_vol: pd.DataFrame
    ) -> pd.DataFrame:
        # ===============================
        # 1. Pastikan kolom waktu DATETIME
        # ===============================
        df_eq["Acquired_Date"] = pd.to_datetime(
            df_eq.get("Acquired_Date"), errors="coerce"
        )                                                                              # parse Acquired_Date menjadi datetime
        df_vol["VRP_Date"] = pd.to_datetime(
            df_vol.get("VRP_Date"), errors="coerce"
        )                                                                              # parse VRP_Date menjadi datetime

        # ===============================
        # 2. Drop baris tanpa waktu
        # ===============================
        df_eq = df_eq.dropna(subset=["Acquired_Date"])                                  # buang baris tanpa Acquired_Date
        df_vol = df_vol.dropna(subset=["VRP_Date"])                                     # buang baris tanpa VRP_Date

        # ===============================
        # 3. SORT WAJIB (INI KUNCI ERROR KAMU)
        # ===============================
        df_eq = df_eq.sort_values("Acquired_Date").reset_index(drop=True)               # wajib sort sebelum merge_asof
        df_vol = df_vol.sort_values("VRP_Date").reset_index(drop=True)                  # wajib sort sebelum merge_asof

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
        )                                                                              # merge temporal terdekat per Nama dengan tolerance

        # ===============================
        # 5. Cleanup
        # ===============================
        df.drop(columns=["VRP_Date"], inplace=True, errors="ignore")                    # buang kolom VRP_Date setelah merge

        if "VRP_Max" not in df.columns:
            df["VRP_Max"] = 0.0                                                         # isi default jika tidak ada
        else:
            df["VRP_Max"] = df["VRP_Max"].fillna(0.0)                                   # isi NaN VRP_Max dengan 0

        return df                                                                       # kembalikan hasil merge

    def _cleanup_final(self, df: pd.DataFrame) -> pd.DataFrame:
        df = DataGuard.enforce_numeric(
            df, ["EQ_Lintang", "EQ_Bujur"]
        )                                                                              # pastikan koordinat numeric

        if "VRP_Max" in df.columns:
            df["VRP_Max"] = df["VRP_Max"].fillna(0.0)                                   # isi NaN VRP_Max dengan 0

        df["Nama"] = df["Nama"].astype(str).str.strip().str.title()                     # normalisasi Nama (strip & title)
        df.sort_values("Acquired_Date", inplace=True)                                   # urutkan berdasarkan tanggal
        return df                                                                       # kembalikan df bersih

    def _save_output(self, df: pd.DataFrame):
        outdir = os.path.dirname(self.cfg.merged_output_path)                           # direktori output dari config
        os.makedirs(outdir, exist_ok=True)                                              # buat direktori jika belum ada

        try:
            df.to_excel(self.cfg.merged_output_path, index=False)                       # simpan ke xlsx
            df.to_pickle(self.cache_path)                                               # simpan cache pickle
        except Exception as e:
            logger.error(
                f"[DataLoader] Gagal menyimpan file output: {e}"                       # log error jika gagal simpan
            )
