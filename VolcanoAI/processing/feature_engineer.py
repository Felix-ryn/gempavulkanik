# VolcanoAI/processing/feature_engineer.py
# -- coding: utf-8 --

"""
VOLCANO AI - FEATURE ENGINEERING ENGINE (V6.0 TITAN)
====================================================
Modul ini adalah 'Laboratorium Data' yang mengubah data mentah menjadi sinyal cerdas.
Menggunakan pendekatan Micro-Modular untuk menangani Fisika, Waktu, dan Statistik.
"""

import os
import logging
import pickle
import warnings
import functools
import time
import math
from typing import Tuple, Optional, Dict, List, Any, Union
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Advanced ML Libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import NearestNeighbors

# Config Import
try:
    from ..config.config import FeatureEngineeringConfig, AcoEngineConfig, TYPE_KEYWORDS
except ImportError:
    pass

# Setup Logger
logger = logging.getLogger("VolcanoAI.FeatureEngineer")
logger.addHandler(logging.NullHandler())

# =============================================================================
# SECTION 0: UTILITIES & TELEMETRY
# =============================================================================

def engineer_telemetry(func):
    """Decorator untuk mencatat performa setiap sub-modul engineering."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise e
        finally:
            t1 = time.perf_counter()
            duration = t1 - t0
            if duration > 0.1: # Log hanya jika signifikan
                logger.debug(f"[FE Telemetry] {func.__name__} executed in {duration:.4f}s")
    return wrapper

class DataGuard:
    """
    Penjaga integritas data tingkat rendah.
    """
    @staticmethod
    def enforce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Memaksa kolom menjadi numerik, mengisi error dengan 0."""
        for col in cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(0.0)
        return df

    @staticmethod
    def sanitize_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Memastikan kolom tanggal valid dan terurut."""
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            df = df.dropna(subset=[col])
            df = df.sort_values(col).reset_index(drop=True)
        return df

# =============================================================================
# SECTION 1: PREPROCESSOR STATE CONTAINER
# =============================================================================

class FeaturePreprocessor:
    """
    Menyimpan seluruh 'otak' (state) dari feature engineering.
    """
    def __init__(self):
        self.scalers: Dict[int, Any] = {}
        self.global_scaler: Optional[Any] = None
        self.imputer: Optional[Any] = None
        self.type_classifier: Optional[RandomForestClassifier] = None
        self.cluster_centroids: Dict[int, Tuple[float, float]] = {}
        
        # [NEW] Menyimpan median data training sebagai fallback yang lebih baik dari 0
        self.imputation_fallback_median: Dict[str, float] = {} 
        self.version = "6.0.0"

    def save(self, path: str) -> None:
        try:
            directory = os.path.dirname(path)
            if directory: os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Preprocessor state saved to: {path}")
        except Exception as e:
            logger.error(f"Failed to save preprocessor: {e}")

    @staticmethod
    def load(path: str) -> 'FeaturePreprocessor':
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            logger.info(f"Preprocessor state loaded from: {path}")
            return obj
        except Exception as e:
            logger.critical(f"Corrupt preprocessor file: {e}")
            raise

# =============================================================================
# SECTION 2: PHYSICS KERNEL (GEOPHYSICS CALCULATION)
# =============================================================================

class PhysicsKernel:
    """Modul perhitungan fisika gempa bumi."""
    def __init__(self, config: FeatureEngineeringConfig):
        self.cfg = config
        self.rad_params = getattr(self.cfg, "radius_estimation_defaults", {"c0":0, "c1":12, "c2":80, "d_min":1.0})
        self.depth_params = getattr(self.cfg, "depth_factor_defaults", {"beta_v":0.3, "d_ref_v":30})
        self.ring_params = getattr(self.cfg, "ring_multipliers_defaults", {"m1":1.0, "m2":1.5, "m3":2.0})

    @engineer_telemetry
    def compute_energy_and_radius(self, df: pd.DataFrame) -> pd.DataFrame:
        mag = df["Magnitudo"]
        df['seismic_energy_log10'] = 1.5 * mag + 4.8

        depth = df["Kedalaman (km)"].clip(lower=self.rad_params.get("d_min", 1.0))
        c0 = self.rad_params.get("c0", 0.0)
        c1 = self.rad_params.get("c1", 12.0)
        c2 = self.rad_params.get("c2", 80.0)
        
        r_base = c0 + (c1 * mag) + (c2 / depth)
        
        is_v = (df["isVulkanik"] == 1)
        
        beta_v = self.depth_params.get("beta_v", 0.3)
        d_ref_v = self.depth_params.get("d_ref_v", 30.0)
        beta_t = self.depth_params.get("beta_t", 0.5)
        d_ref_t = self.depth_params.get("d_ref_t", 40.0)
        
        f_depth = np.where(
            is_v,
            1.0 + beta_v * ((1.0 / depth) - (1.0 / d_ref_v)),
            1.0 + beta_t * ((1.0 / depth) - (1.0 / d_ref_t))
        )
        
        r_adjusted = r_base * f_depth
        m1 = self.ring_params.get("m1", 0.25)
        m2 = self.ring_params.get("m2", 0.55)
        m3 = self.ring_params.get("m3", 1.00)
        
        df["R1_final"] = (r_adjusted * m1).clip(lower=0.0)
        df["R2_final"] = (r_adjusted * m2).clip(lower=0.0)
        df["R3_final"] = (r_adjusted * m3).clip(lower=0.0)
        
        df["AreaTerdampak_km2"] = np.pi * (df["R3_final"] ** 2)
        
        return df

# =============================================================================
# SECTION 3: SMART IMPUTER BRAIN (Tidak ada perubahan di sini)
# =============================================================================

import numpy as np
import pandas as pd
import logging
from typing import List, Optional

# [WAJIB] enable_iterative_imputer harus di-import sebelum IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Setup Logger jika belum ada
logger = logging.getLogger(__name__)

class SmartImputerBrain:
    """
    Multivariate Imputation Strategy.
    Menggunakan IterativeImputer dengan estimator RandomForestRegressor (Non-linear)
    untuk menangani data yang hilang secara cerdas. 
    Jika gagal, fallback ke SimpleImputer (Median).
    """
    
    def __init__(self, strategy: str = 'iterative', n_estimators: int = 10, random_state: int = 42):
        self.strategy = strategy
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        
    def fit(self, df: pd.DataFrame, cols: Optional[List[str]] = None):
        """
        Melatih imputer berdasarkan kolom yang ditentukan.
        Returns: Trained Sklearn Imputer Object.
        """
        # 1. Pilih data yang akan digunakan
        if cols:
            # Pastikan hanya mengambil kolom yang ada di DF
            valid_cols = [c for c in cols if c in df.columns]
            X = df[valid_cols].values
        else:
            X = df.values

        # 2. Cek apakah X kosong atau hanya berisi NaN sebelum fit
        if X.size == 0 or np.all(pd.isna(X)):
            logger.warning("[SmartImputerBrain] Data kosong total. Menggunakan Dummy SimpleImputer(strategy='constant', fill_value=0).")
            self.model = SimpleImputer(strategy='constant', fill_value=0)
            # Buat data dummy shape (1, n_features) agar fit berhasil
            dummy_X = np.zeros((1, X.shape[1])) if X.ndim > 1 else np.zeros((1, 1))
            self.model.fit(dummy_X)
            return self.model

        # 3. Definisikan Strategi Imputasi
        if self.strategy == 'iterative':
            # Menggunakan RandomForest untuk menangkap hubungan non-linear antar fitur
            rf_estimator = RandomForestRegressor(
                n_estimators=self.n_estimators, 
                n_jobs=-1,  # Gunakan semua core CPU
                random_state=self.random_state
            )
            
            self.model = IterativeImputer(
                estimator=rf_estimator, 
                max_iter=10, 
                random_state=self.random_state,
                initial_strategy='median', # Mulai dengan median sebelum iterasi
                imputation_order='ascending'
            )
        else:
            self.model = SimpleImputer(strategy='median')

        # 4. Eksekusi Fit dengan Safety Net
        try:
            # logger.info(f"Fitting Imputer dengan strategi: {self.strategy}")
            self.model.fit(X)
            
        except Exception as e:
            logger.error(f"[SmartImputerBrain] Fit GAGAL menggunakan {self.strategy}: {e}.")
            logger.info("--> Melakukan Fallback ke SimpleImputer (Median).")
            
            # Fallback Logic: Selalu pastikan model terisi sesuatu yang valid
            self.model = SimpleImputer(strategy='median')
            
            # Coba fit fallback. Jika data benar-benar rusak (all NaN), isi dengan 0
            try:
                self.model.fit(X)
            except Exception as e_fallback:
                logger.error(f"[SmartImputerBrain] Fallback Median juga gagal: {e_fallback}. Force Constant=0.")
                self.model = SimpleImputer(strategy='constant', fill_value=0)
                # Fit dengan dummy zeros agar tidak crash saat transform
                dummy_X = np.zeros_like(X)
                self.model.fit(dummy_X)

        return self.model

# =============================================================================
# SECTION 4: CLUSTER & OUTLIER ENGINE (Tidak ada perubahan di sini)
# =============================================================================

class SpatialClusterEngine:
    """Mesin Clustering Spasial (DBSCAN) untuk memisahkan data per gunung."""
    def __init__(self, eps: float, min_samples: int):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit_predict(self, coords: np.ndarray) -> np.ndarray:
        coords_rad = np.radians(coords)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='haversine')
        return db.fit_predict(coords_rad)

# =============================================================================
# SECTION 5: MAIN FEATURE ENGINEER CLASS
# =============================================================================

class FeatureEngineer:
    """
    Fasad Utama (The Main Facade).
    Mengorkestrasi seluruh sub-modul.
    """
    def __init__(self, fe_config: FeatureEngineeringConfig, aco_config: AcoEngineConfig):
        self.fe_config = fe_config
        self.aco_config = aco_config
        self.preprocessor: Optional[FeaturePreprocessor] = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Sub-Engines Initialization
        self.physics_kernel = PhysicsKernel(fe_config)
        self.imputer_brain = SmartImputerBrain(strategy='iterative')
        self.cluster_engine = SpatialClusterEngine(
            eps=getattr(self.aco_config, "dbscan_eps", 0.1),
            min_samples=getattr(self.aco_config, "dbscan_min_samples", 3)
        )

    # =========================================================================
    # PUBLIC API (Wajib ada untuk Kompatibilitas LSTM)
    # =========================================================================

    @engineer_telemetry
    def basic_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Membersihkan data dasar (Tanggal, Numerik, String) + normalisasi nama kolom."""
        if df is None or df.empty:
            return pd.DataFrame()

        df_clean = df.copy()

        # 🟢 1. Normalisasi nama kolom
        rename_map = {
            "Lintang": "EQ_Lintang", "Latitude": "EQ_Lintang", "Lat": "EQ_Lintang",
            "Bujur": "EQ_Bujur", "Longitude": "EQ_Bujur", "Lon": "EQ_Bujur",
            "Kedalaman_km": "Kedalaman (km)", "Depth": "Kedalaman (km)"
        }

        for old, new in rename_map.items():
            if old in df_clean.columns and new not in df_clean.columns:
                df_clean.rename(columns={old: new}, inplace=True)

        # 🟢 2. Pastikan kolom wajib SELALU ADA
        required_cols = ["EQ_Lintang", "EQ_Bujur", "Magnitudo", "Kedalaman (km)", "VRP_Max"]
        df_clean = DataGuard.enforce_numeric(df_clean, required_cols)
        for col in required_cols:
            if col not in df_clean.columns:
                df_clean[col] = 0.0  # fallback aman
        
        # [CATATAN]: PheromoneScore TIDAK boleh dipaksa 0.0 di sini karena akan diimpute nanti.
        if 'PheromoneScore' in df_clean.columns:
            df_clean['PheromoneScore'] = pd.to_numeric(df_clean['PheromoneScore'], errors='coerce')
        if 'Pheromone_Score' in df_clean.columns:
            df_clean['Pheromone_Score'] = pd.to_numeric(df_clean['Pheromone_Score'], errors='coerce')

        # 🟢 3. Date Sanitization
        df_clean = DataGuard.sanitize_dates(df_clean, 'Acquired_Date')

        # 🟢 4. Numeric Sanitization
        df_clean = DataGuard.enforce_numeric(df_clean, required_cols)

        # 🟢 5. Text Sanitization
        df_clean["Keterangan"] = df_clean.get("Keterangan", "Unknown").fillna("Unknown").astype(str)

        return df_clean


    @engineer_telemetry
    def add_spatio_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menambahkan fitur waktu siklis dan memanggil kernel fisika."""
        if df.empty: return df
        df_out = df.copy()
        
        # 1. Cyclical Time
        if 'Acquired_Date' in df_out.columns:
            ts = df_out['Acquired_Date']
            df_out['tfe_doy_sin'] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
            df_out['tfe_doy_cos'] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
            df_out['tfe_weekday_sin'] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            
        # 2. Smart Classification placeholder
        if 'isVulkanik' not in df_out.columns: df_out['isVulkanik'] = 0
        
        # 3. Intensity Feature
        if 'Magnitudo' in df_out.columns:
             vrp = df_out['VRP_Max'] if 'VRP_Max' in df_out.columns else 0.0
             df_out['Intensity'] = vrp + (df_out['Magnitudo'] * 10.0)
             
        return df_out

    @engineer_telemetry
    def add_lag_and_rolling(self, df: pd.DataFrame, lags=[1, 3], windows=[3, 7]) -> pd.DataFrame:
        """Membuat fitur Time-Series (Lag & Rolling Window)."""
        if df.empty: return df
        df_out = df.copy()
        
        # [FIX] Tambahkan PheromoneScore ke targets jika ada.
        targets = ['Magnitudo', 'VRP_Max', 'R3_final', 'seismic_energy_log10']
        if 'PheromoneScore' in df_out.columns:
            targets.append('PheromoneScore')
            
        valid_cols = [c for c in targets if c in df_out.columns]
        
        if 'Acquired_Date' in df_out.columns:
            df_out = df_out.sort_values('Acquired_Date')

        for col in valid_cols:
            for lag in lags:
                # [FIX]: Menggunakan fillna(0.0) di sini aman karena ini adalah fitu lag/roll.
                df_out[f"{col}_lag{lag}"] = df_out[col].shift(lag).fillna(0.0)
            for w in windows:
                roll = df_out[col].rolling(window=w, min_periods=1)
                df_out[f"{col}_roll_mean{w}"] = roll.mean().fillna(0.0)
                df_out[f"{col}_roll_std{w}"] = roll.std().fillna(0.0)
        
        return df_out.reset_index(drop=True)

    # =========================================================================
    # PRIVATE GRANULAR STEPS (Logika Inti)
    # =========================================================================
   

    def run(self, df: pd.DataFrame, is_training: bool = True, preprocessor: Optional[FeaturePreprocessor] = None) -> Tuple[pd.DataFrame, FeaturePreprocessor]:
        """
        Pipeline Utama. Menjalankan semua langkah secara berurutan.
        [FIX]: Menambahkan is_training ke signature dan memperbaiki logika ACO input.
        """
        print(df.head())
        print(df.columns)
        print(df.dtypes)
        print(df.isna().sum())
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be DataFrame")
        if df.empty:
            return df, (preprocessor if preprocessor else FeaturePreprocessor())

        # Setup Preprocessor
        if is_training:
            self.logger.info(">>> FE TRAIN: Initializing new preprocessor state.")
            self.preprocessor = FeaturePreprocessor()
        else:
            if preprocessor is None:
                 self.logger.error("Preprocessor state is missing in inference mode!")
                 return df, FeaturePreprocessor()
            self.logger.info(">>> FE INFERENCE: Using existing preprocessor state.")
            self.preprocessor = preprocessor
            if not hasattr(self.preprocessor, 'scalers'):
                self.preprocessor.scalers = {}

        # [FIX] Inisialisasi df_proc di sini, memastikan scope lokal.
        df_proc = df.copy() 

        # =============================================================
        # 🔥 FIX TERPENTING — STANDARDISASI KOLOM UNTUK ACO ENGINE
        # =============================================================
        # Menjamin kolom 'Original' ada untuk ACO/GA
        if "Magnitudo" in df_proc.columns and "Magnitudo_Original" not in df_proc.columns:
            df_proc["Magnitudo_Original"] = df_proc["Magnitudo"]

        if "Kedalaman (km)" in df_proc.columns and "Kedalaman_Original" not in df_proc.columns:
            df_proc["Kedalaman_Original"] = df_proc["Kedalaman (km)"]

        # --- PIPELINE STEPS ---

        # 1. Basic Cleaning (Memastikan kolom wajib ada, TAPI TIDAK mengisi PheromoneScore)
        df_proc = self.basic_cleanup(df_proc)

        # 2. Smart Classification (Rule-based + ML)
        df_proc = self._exec_smart_classification(df_proc, is_training)

        # 3. Physics Kernel (Energy + Radius)
        df_proc = self.physics_kernel.compute_energy_and_radius(df_proc)

        # 4. Spatio-Temporal (Waktu siklis)
        df_proc = self.add_spatio_temporal_features(df_proc)

        # 5. Spatial Clustering (Juga menghitung time_since_last_event_days)
        df_proc = self._exec_spatial_clustering(df_proc, is_training)

        # 6. Smart Imputation (Mengisi time_since_last_event_days, VRP_Max, OLI, MSI, DAN PheromoneScore)
        df_proc = self._exec_smart_imputation(df_proc, is_training)

        # 7. Adaptive Scaling per Cluster
        df_proc = self._exec_adaptive_scaling(df_proc, is_training)

        # 8. Target Labeling (Impact Class)
        df_proc = self._exec_target_labeling(df_proc)

        # 9. Lag & Rolling Features
        # [CATATAN]: PheromoneScore sudah diimpute di step 6, jadi aman untuk Lag/Roll.
        df_proc = self.add_lag_and_rolling(df_proc)

        # 10. Save Preprocessor State
        if is_training:
            save_path = getattr(self.fe_config, 'preprocessor_output_path', 'output/feature_preprocessor.pkl')
            if save_path:
                self.preprocessor.save(save_path)

        self.logger.info(f"Feature Engineering Complete. Shape: {df_proc.shape}")

        return df_proc, self.preprocessor


    # =========================================================================
    # PRIVATE GRANULAR STEPS (Lanjutan)
    # =========================================================================

    def _exec_smart_classification(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """Klasifikasi tipe gempa dengan Machine Learning sebagai backup aturan teks."""
        
        v_keys = ["vulkanik", "letusan", "abu"]
        t_keys = ["tektonik", "jauh", "lokal"]
        try:
            from ..config.config import TYPE_KEYWORDS as TK
            v_keys = TK.get("vulkanik", v_keys)
            t_keys = TK.get("tektonik", t_keys)
        except: pass
        
        v_pat = "|".join(v_keys)
        t_pat = "|".join(t_keys)
        
        txt = df["Keterangan"].astype(str)
        is_v = txt.str.contains(v_pat, case=False).astype(int)
        is_t = txt.str.contains(t_pat, case=False).astype(int)
        
        df["isVulkanik"] = is_v
        df["isTektonik"] = is_t
        
        # 2. Machine Learning Fallback (Random Forest)
        ml_feats = df[['EQ_Lintang', 'EQ_Bujur', 'Kedalaman (km)', 'Magnitudo']].fillna(0)
        unk_mask = (is_v == 0) & (is_t == 0)
        
        if is_training:
            y_train = is_v # Target: 1=Vulkanik
            clf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
            # Hanya fit pada data yang sudah terlabeli dengan baik
            labeled_mask = (is_v == 1) | (is_t == 1)
            clf.fit(ml_feats.loc[labeled_mask], y_train.loc[labeled_mask])
            self.preprocessor.type_classifier = clf
            self.logger.info("Smart Classifier Trained.")
            df.loc[unk_mask, "isTektonik"] = 1 # Default unkown to Tektonik during training if logic fails
        else:
            if unk_mask.any() and self.preprocessor.type_classifier:
                preds = self.preprocessor.type_classifier.predict(ml_feats.loc[unk_mask])
                df.loc[unk_mask, "isVulkanik"] = preds
                df.loc[unk_mask, "isTektonik"] = 1 - preds
                self.logger.info(f"Smart Inference classified {len(preds)} unknown events.")
            else:
                df.loc[unk_mask, "isTektonik"] = 1 
                
        return df

    def _exec_spatial_clustering(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        coords = df[["EQ_Lintang", "EQ_Bujur"]].dropna().values
        if len(coords) == 0:
            df["cluster_id"] = -1
            return df

        # [FIX LOGIC]: Gunakan data yang sudah terurut untuk hitung time_since
        if 'Acquired_Date' in df.columns:
            df = df.sort_values("Acquired_Date").reset_index(drop=True)
            
        # 1. Clustering / Assignment
        if is_training:
            coords_rad = np.radians(coords)
            db = DBSCAN(
                eps=self.aco_config.dbscan_eps,
                min_samples=self.aco_config.dbscan_min_samples,
                metric='haversine'
            ).fit(coords_rad)
            
            valid_idx = df[["EQ_Lintang", "EQ_Bujur"]].dropna().index
            df.loc[valid_idx, "cluster_id"] = db.labels_
            
            # Store Centroids
            for cid in set(db.labels_):
                if cid == -1: continue
                mask = (df["cluster_id"] == cid)
                pts = df.loc[mask, ["EQ_Lintang", "EQ_Bujur"]].values
                centroid = pts.mean(axis=0)
                self.preprocessor.cluster_centroids[int(cid)] = tuple(centroid)
        else:
            # Assign to nearest centroid (Inference Mode)
            centroids = self.preprocessor.cluster_centroids
            if not centroids:
                df["cluster_id"] = -1
                return df
            
            ids = []
            for pt in coords:
                best_c = -1
                min_dist = float('inf')
                for cid, ctr in centroids.items():
                    # Haversine distance (approx) menggunakan Euclidean pada koordinat biasa
                    d = np.linalg.norm(pt - np.array(ctr)) 
                    if d < min_dist:
                        min_dist = d
                        best_c = cid
                # Gunakan threshold sederhana 1.0 derajat (untuk mengabaikan data yang sangat jauh)
                if min_dist > 1.0: 
                    ids.append(-1)
                else: 
                    ids.append(best_c)
            
            valid_idx = df[["EQ_Lintang", "EQ_Bujur"]].dropna().index
            df.loc[valid_idx, "cluster_id"] = ids
            
        # 2. Time Since Last Event (berbasis cluster ID)
        df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
        grp = "cluster_id"
        
        # Hitung diff (ini akan menghasilkan NaN di baris pertama setiap cluster,
        # yang kemudian akan di-impute di _exec_smart_imputation)
        df['time_since_last_event_days'] = df.groupby(grp)['Acquired_Date'].diff().dt.total_seconds() / 86400.0
        
        # JANGAN mengisi .fillna(0.0) di sini, biarkan Smart Imputer menanganinya.
        
        return df

    def _exec_smart_imputation(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Smart Imputation Logic.
        
        Mekanisme:
        1. Training: 
           - Hitung Median sebagai cadangan (fallback).
           - Coba Iterative Imputer (Smart).
           - Jika gagal, gunakan Simple Imputer.
        2. Inference:
           - Coba Transform dengan imputer terlatih.
           - Jika gagal (atau kolom baru), gunakan Fallback Median dari Training.
           - Pastikan tidak ada NaN tersisa (isi 0 sebagai langkah terakhir).
        """
        from sklearn.impute import SimpleImputer
        import numpy as np

        # Load konfigurasi kolom target
        cols = getattr(self.fe_config, "imputation_columns",
                       ['time_since_last_event_days', 'VRP_Max', 'OLI_total (W)', 'MSI_total (W)', 'PheromoneScore'])

        # 1. Initialization: Pastikan semua kolom target ada di DF
        for col in cols:
            if col not in df.columns:
                # Inisialisasi NaN agar dikenali sebagai missing values oleh Imputer
                df[col] = np.nan
        
        # Buat subset data dan bersihkan dari Infinity
        sub_df = df[cols].replace([np.inf, -np.inf], np.nan)
        res = None

        # ---------------------------------------------------------
        # 2. TRAINING PHASE
        # ---------------------------------------------------------
        if is_training:
            # [NEW] Hitung dan Simpan Median DULU untuk fallback yang konsisten
            # skipna=True memastikan kita dapat nilai tengah yang valid dari data yang ada
            median_series = sub_df.median(skipna=True, numeric_only=True)
            self.preprocessor.imputation_fallback_median = median_series.to_dict()

            # Handle Critical Case: Kolom yang 100% NaN saat training
            # (Imputer brain akan crash jika inputnya full NaN)
            full_null_cols = [c for c in cols if sub_df[c].isnull().all()]
            if full_null_cols:
                self.logger.warning(f"Imputer Warning: Kolom berikut kosong total saat training -> diisi 0.0: {full_null_cols}")
                for col in full_null_cols:
                    sub_df[col] = 0.0

            try:
                # [CORE LOGIC] Fit Imputer Cerdas (Random Forest/Iterative)
                # self.imputer_brain harus sudah terinisialisasi (misal di __init__)
                self.preprocessor.imputer = self.imputer_brain.fit(sub_df, cols)
                res = self.preprocessor.imputer.transform(sub_df)

            except Exception as e:
                self.logger.error(f"IterativeImputer Fit Gagal ({e}) -> Fallback SimpleImputer(median).")
                
                # Fallback Strategy: SimpleImputer
                fallback_imputer = SimpleImputer(strategy='median')
                res = fallback_imputer.fit_transform(sub_df)
                self.preprocessor.imputer = fallback_imputer

        # ---------------------------------------------------------
        # 3. INFERENCE PHASE
        # ---------------------------------------------------------
        else:
            if hasattr(self.preprocessor, 'imputer') and self.preprocessor.imputer is not None:
                try:
                    res = self.preprocessor.imputer.transform(sub_df)
                except Exception as e:
                    self.logger.error(f"Imputer Mismatch saat Inference ({e}) -> Menggunakan Saved Median.")
                    
                    # [FIX KRITIS]: Gunakan median fallback yang tersimpan dari Training
                    # Ini mencegah bias statistik dari data live yang sedikit
                    saved_medians = getattr(self.preprocessor, 'imputation_fallback_median', {})
                    
                    # Prioritas 1: Fill dengan saved median
                    fallback_df = sub_df.fillna(saved_medians)
                    
                    # Prioritas 2: Jika saved median tidak ada, fill dengan median data saat ini
                    fallback_df = fallback_df.fillna(fallback_df.median(numeric_only=True))
                    
                    # Prioritas 3: Terakhir fill dengan 0
                    res = fallback_df.fillna(0.0).values
            else:
                self.logger.warning("Imputer State tidak ditemukan saat Inference. Menggunakan Fallback Median/0.")
                res = sub_df.fillna(sub_df.median(numeric_only=True)).fillna(0).values

        # ---------------------------------------------------------
        # 4. SAFETY FINALIZATION
        # ---------------------------------------------------------
        # Safety net: jika res masih mengandung NaN (kasus ekstrem), ganti ke 0
        if isinstance(res, np.ndarray):
             res[~np.isfinite(res)] = 0.0

        # Assign hasil kembali ke DataFrame utama
        df[cols] = res

        # [FIX KRITIS PheromoneScore]
        if 'PheromoneScore' in df.columns:
            # 1. Bersihkan sisa infinity/NaN
            df['PheromoneScore'] = df['PheromoneScore'].replace([np.inf, -np.inf], np.nan).fillna(1e-4)
            
            # 2. Clamping: Paksa nilai minimal 0.0001
            # Logic: Pheromone tidak boleh 0 mutlak untuk perhitungan logaritmik atau probability pembagi
            mask_too_small = df['PheromoneScore'] < 1e-4
            if mask_too_small.any():
                df.loc[mask_too_small, 'PheromoneScore'] = 1e-4

            # 3. Sync ke nama kolom lain jika diperlukan
            df['Pheromone_Score'] = df['PheromoneScore']

        return df


    def _exec_adaptive_scaling(self, df: pd.DataFrame, is_training: bool) -> pd.DataFrame:
        """
        Adaptive Robust Scaling Logic.
        - Membersihkan data dari nilai Infinity/NaN dengan Median.
        - Menggunakan Global Scaler sebagai base.
        - Jika ada Cluster, gunakan Scaler spesifik per cluster.
        """
        from sklearn.preprocessing import RobustScaler
        import numpy as np
        
        # Load config kolom yang perlu discale
        scale_cols = getattr(self.fe_config, "scaling_features",
                             ['Magnitudo', 'Kedalaman (km)', 'seismic_energy_log10', 'time_since_last_event_days'])

        # Validasi kolom yang ada di DataFrame
        valid_cols = [c for c in scale_cols if c in df.columns]
        if not valid_cols:
            return df

        # Pastikan kolom cluster ada, default ke -1 (Global/Noise)
        if 'cluster_id' not in df.columns:
            df['cluster_id'] = -1

        # ---------------------------------------------------------
        # 1. PRE-CLEANING (FIX KRITIS)
        # ---------------------------------------------------------
        # Menggunakan df_temp agar tidak mengotori data asli sebelum scaling selesai
        df_temp = df.copy() 
        
        for col in valid_cols:
            # Langkah 1: Ganti Infinite (inf/-inf) menjadi NaN
            df_temp[col] = df_temp[col].replace([np.inf, -np.inf], np.nan)
            
            # Langkah 2: Hitung median dari data yang valid
            median_val = df_temp[col].median(skipna=True)
            
            # Langkah 3: Isi NaN dengan median (atau 0.0 jika kolom kosong total)
            fill_val = median_val if np.isfinite(median_val) else 0.0
            df_temp[col] = df_temp[col].fillna(fill_val)

        # ---------------------------------------------------------
        # 2. GLOBAL SCALER HANDLING
        # ---------------------------------------------------------
        # Jika Training: Fit scaler global baru menggunakan data bersih
        if is_training:
            self.preprocessor.global_scaler = RobustScaler().fit(df_temp[valid_cols])
            
        # Jika Inference tapi Scaler hilang: Re-fit sesaat agar kode tidak crash (Fail-safe)
        if not hasattr(self.preprocessor, 'global_scaler') or self.preprocessor.global_scaler is None:
            self.preprocessor.global_scaler = RobustScaler().fit(df_temp[valid_cols])

        # ---------------------------------------------------------
        # 3. PER-CLUSTER SCALING LOGIC
        # ---------------------------------------------------------
        # Loop unik cluster ID
        unique_clusters = df_temp["cluster_id"].unique()
        
        for cid in unique_clusters:
            mask = df_temp["cluster_id"] == cid
            
            # Ambil slice data (subset) untuk cluster ini
            data_subset = df_temp.loc[mask, valid_cols]

            if data_subset.empty:
                continue

            # Tentukan Scaler yang dipakai
            scaler = None
            
            # --- TRAINING ---
            if is_training and cid != -1:
                # Fit Scaler baru spesifik untuk cluster ini
                try:
                    scaler = RobustScaler().fit(data_subset)
                    self.preprocessor.scalers[cid] = scaler
                except Exception:
                    # Fallback ke global jika data terlalu sedikit untuk fit
                    scaler = self.preprocessor.global_scaler
            
            # --- INFERENCE / NOISE CLUSTER ---
            else:
                # Ambil dari dictionary scalers yang sudah ditrain, atau gunakan global
                scaler = self.preprocessor.scalers.get(cid, self.preprocessor.global_scaler)

            # Eksekusi Transformasi
            if scaler:
                try:
                    # Transform mengembalikan numpy array
                    scaled_values = scaler.transform(data_subset)
                except Exception as e:
                    # Fallback panic: kembalikan nilai asli
                    self.logger.warning(f"Scaling failed for cluster {cid}: {e}")
                    scaled_values = data_subset.values
            else:
                scaled_values = data_subset.values

            # Assign hasil kembali ke DataFrame utama
            # Membuat kolom baru dengan suffix "_scaled"
            for i, col in enumerate(valid_cols):
                col_name_scaled = f"{col}_scaled"
                # Menggunakan loc assignment yang aman
                df.loc[mask, col_name_scaled] = scaled_values[:, i]

        return df


    def _exec_target_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label dampak (impact_level) menggunakan magnitude + kedalaman + radius.
        [FIX]: Logika score ditingkatkan sensitivitasnya dan menggunakan PheromoneScore
        """

        mag = df["Magnitudo"]
        depth = df["Kedalaman (km)"].clip(lower=1)
        radius = df.get("R3_final", pd.Series([0] * len(df))) 
        pheromone = df.get("PheromoneScore", pd.Series([0] * len(df))) # Menggunakan PheromoneScore yang sudah diimpute/dihitung
        
        # Threshold bawaan dari config
        rules = getattr(self.fe_config, "impact_thresholds", {})
        mag_parah = rules.get("parah_mag", 6.5)
        mag_sedang = rules.get("sedang_mag_min", 4.0)

        # [REVISED SCORE FORMULA] Formula yang lebih sensitif (M x 2.0) + Kontribusi Pheromone
        score_base = (mag * 2.0) + (radius / 15) - (depth * 0.05)
        
        # Tambahkan kontribusi PheromoneScore yang sudah diimpute/dihitung
        score = score_base + (pheromone * 2.0) 

        df["impact_level"] = np.select(
            [
                (mag >= mag_parah) | (score >= 10), 
                (mag >= mag_sedang) | (score >= 6),
            ],
            ["Parah", "Sedang"],
            default="Ringan",
        )

        return df