# VolcanoAI/processing/realtime_buffer_manager.py
# -- coding: utf-8 --

import os
import logging
import pandas as pd
from datetime import datetime, timedelta


class RealtimeBufferManager:
    """
    Buffer 90 hari untuk:
    - raw_realtime   (BMKG / MIROVA / API lain)
    - raw_injection  (Excel suntikan 90 hari)
    - processed      (hasil FE untuk inference)
    
    Buffer ini digunakan oleh:
    - run_realtime_inference()
    - start_monitoring_loop()
    """

    def __init__(self, buffer_days: int = 90):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.buffer_days = buffer_days

        self.raw_realtime = pd.DataFrame()
        self.raw_injection = pd.DataFrame()
        self.processed = pd.DataFrame()

    # ======================================================================
    # PRIVATE UTIL
    # ======================================================================
    def _filter_last_n_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame agar hanya 90 hari terakhir."""
        if df.empty:
            return df

        try:
            df["Acquired_Date"] = pd.to_datetime(df["Acquired_Date"], errors="coerce")
            df = df.dropna(subset=["Acquired_Date"])
        except:
            return pd.DataFrame()

        limit = datetime.now() - timedelta(days=self.buffer_days)
        df = df[df["Acquired_Date"] >= limit]

        return df.reset_index(drop=True)

    # ======================================================================
    # RAW — REALTIME
    # ======================================================================
    def append_raw_realtime(self, df_new: pd.DataFrame):
        if df_new is None or df_new.empty:
            return

        required = ["Acquired_Date", "Latitude", "Longitude"]
        for col in required:
            if col not in df_new.columns:
                self.logger.error(f"[raw_realtime] Missing column: {col}")
                return

        self.raw_realtime = pd.concat([self.raw_realtime, df_new], ignore_index=True)
        self.raw_realtime = self._filter_last_n_days(self.raw_realtime)

    # ======================================================================
    # RAW — INJECTION (from Excel)
    # ======================================================================
    def append_raw_injection(self, df_inj: pd.DataFrame):
        if df_inj is None or df_inj.empty:
            return

        self.raw_injection = pd.concat([self.raw_injection, df_inj], ignore_index=True)
        self.raw_injection = self._filter_last_n_days(self.raw_injection)

    # ======================================================================
    # GET MERGED RAW
    # ======================================================================
    def get_merged_raw(self) -> pd.DataFrame:
        """
        Menggabungkan data:
        - raw_realtime
        - raw_injection
        
        Melakukan standardisasi tanggal dan membersihkan duplikasi berdasarkan (Waktu + Lokasi).
        """
        # [FIX]: Cek defensive programming agar method ini aman jika input None
        rt_df = self.raw_realtime if self.raw_realtime is not None else pd.DataFrame()
        inj_df = self.raw_injection if self.raw_injection is not None else pd.DataFrame()

        # 1. PRE-FILTERING (Optimasi performa: kurangi data sebelum merge)
        df_rt = self._filter_last_n_days(rt_df)
        df_inj = self._filter_last_n_days(inj_df)

        # 2. CONCATENATION
        if df_rt.empty and df_inj.empty:
            return pd.DataFrame()

        df = pd.concat([df_rt, df_inj], ignore_index=True)

        # 3. DATA CLEANING
        if "Acquired_Date" not in df.columns:
            # Log error bisa ditambahkan di sini jika perlu
            return pd.DataFrame()

        # Konversi ke datetime & hapus NaT (Not a Time)
        df["Acquired_Date"] = pd.to_datetime(df["Acquired_Date"], errors="coerce")
        df = df.dropna(subset=["Acquired_Date"])

        # 4. SORTING & DEDUPLICATION
        # [FIX] Urutkan berdasarkan waktu agar urutan data konsisten
        df = df.sort_values("Acquired_Date")

        # Definisikan kolom kunci untuk mengidentifikasi duplikat
        # Minimal harus ada Waktu dan Koordinat agar dianggap unik
        dedup_keys = ["Acquired_Date", "Latitude", "Longitude"]
        available_keys = [col for col in dedup_keys if col in df.columns]

        # Syarat: Semua kolom kunci harus ada. Jika data injection tidak punya lat/lon, logika ini perlu disesuaikan.
        # Saat ini mengikuti logika: hanya drop jika ketiganya ada.
        if len(available_keys) == 3:
            # keep="last" -> Jika ada duplikat, ambil data yang paling 'bawah' (seringkali data injection/terbaru)
            df = df.drop_duplicates(subset=available_keys, keep="last")

        # 5. RETURN RESULT
        # Reset index agar rapi (0, 1, 2...)
        return df.reset_index(drop=True)

    # ======================================================================
    # PROCESSED BUFFER
    # ======================================================================
    def append_processed(self, df_proc: pd.DataFrame):
        if df_proc is None or df_proc.empty:
            return

        if "Acquired_Date" not in df_proc.columns:
            self.logger.error("[processed] Missing Acquired_Date column")
            return

        self.processed = pd.concat([self.processed, df_proc], ignore_index=True)
        self.processed = self._filter_last_n_days(self.processed)

    # ======================================================================
    # GET SUBSET — LAST N DAYS (FOR HYBRID TRAINING)
    # ======================================================================
    def get_last_n_days(self, n_days: int = 15, source: str = "processed") -> pd.DataFrame:
        """
        Mengambil subset data n hari terakhir dari buffer.
    
        source:
        - "processed" (default) → hasil FE (dipakai CNN/LSTM)
        - "raw" → raw_realtime + raw_injection
        """
        if source == "processed":
            df = self.processed
        elif source == "raw":
            df = self.get_merged_raw()
        else:
            self.logger.warning(f"Unknown source '{source}'")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        if "Acquired_Date" not in df.columns:
            self.logger.error("Acquired_Date tidak ditemukan di buffer")
            return pd.DataFrame()

        df = df.copy()
        df["Acquired_Date"] = pd.to_datetime(df["Acquired_Date"], errors="coerce")
        df = df.dropna(subset=["Acquired_Date"])

        cutoff = datetime.utcnow() - timedelta(days=n_days)
        return df[df["Acquired_Date"] >= cutoff].reset_index(drop=True)

    # ======================================================================
    # EXPORT ALL BUFFERS
    # ======================================================================
    def export_all(self, base_dir="output/realtime"):
        """Dipanggil oleh main.py untuk menyimpan buffer ke CSV."""

        os.makedirs(base_dir, exist_ok=True)

        try:
            self.raw_realtime.to_csv(os.path.join(base_dir, "raw_realtime.csv"), index=False)
            self.raw_injection.to_csv(os.path.join(base_dir, "raw_injection.csv"), index=False)
            self.processed.to_csv(os.path.join(base_dir, "processed.csv"), index=False)
        except Exception as e:
            self.logger.error(f"[Buffer Export Error] {e}")

