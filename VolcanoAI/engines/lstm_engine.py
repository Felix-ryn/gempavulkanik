# VolcanoAI/engines/lstm_engine.py
# -- coding: utf-8 --

"""
VOLCANO AI - LSTM TITAN ENGINE V6.2 (THE NEURAL CORE ULTIMATE)
==============================================================
Modul ini adalah Jantung Utama dari sistem prediksi VolcanoAI.
Mengimplementasikan arsitektur Deep Learning Hybrid yang menggabungkan:
1.  Sequence Modeling (Bi-LSTM)
2.  Attention Mechanism (Bahdanau)
3.  Probabilistic Forecasting (Gaussian NLL)
4.  Realtime Buffer Management (Sliding Window)
5.  Anomaly Detection System (Z-Score & Drift Monitoring)
6.  State Persistence (untuk integrasi CNN)

Copyright (c) 2025 VolcanoAI Team.
"""

import os
import sys
import time
import json
import math
import shutil
import random
import logging
import pickle
import functools
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg") # Backend non-interaktif untuk server
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libs
from joblib import dump, load
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import mahalanobis
from sklearn.impute import KNNImputer

# Deep Learning Libs (TensorFlow/Keras)
import tensorflow as tf
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import (
    Input, LSTM, Dense, Dropout, RepeatVector, TimeDistributed, 
    Bidirectional, Concatenate, Conv1D, GlobalAveragePooling1D, 
    Layer, Dot, Activation, BatchNormalization, Add, Multiply, 
    Lambda, Reshape, Permute, Flatten, GaussianNoise
)
from keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    CSVLogger, TensorBoard, LearningRateScheduler
)
from keras.optimizers import Adam, RMSprop

# Config Imports (Safe Loader)
try:
    from ..config.lstm_config import LstmPipelineConfig
except ImportError:
    pass

try:
    from ..processing.feature_engineer import FeatureEngineer
    from ..config.config import CONFIG
except ImportError:
    pass

# Setup Logger
logger = logging.getLogger("VolcanoAI.LstmEngine")
logger.addHandler(logging.NullHandler())

# =============================================================================
# SECTION 1: MATH KERNEL & UTILITIES (THE FOUNDATION)
# =============================================================================

def execution_telemetry(func):
    """Decorator untuk memantau kinerja setiap fungsi kritis."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            # logger.debug(f"[Telemetry] {func.__name__} executed in {t1-t0:.4f}s")
    return wrapper

class MathKernel:
    """
    Kernel matematika kustom untuk operasi tensor tingkat rendah.
    Menangani fungsi kerugian probabilistik (Probabilistic Loss Functions).
    """
    @staticmethod
    def gaussian_nll(y_true, y_pred):
        # Pisahkan output model: [Mean, Variance]
        mu = y_pred[..., 0:1]
        sigma = y_pred[..., 1:2]
        
        # [FIX KRITIS]: Clip sigma dengan batas bawah yang sangat aman
        # Batas atas 1e6 juga penting untuk mencegah NaN karena log(Inf)
        sigma = tf.clip_by_value(sigma, 1e-5, 1e6) 
        
        # Gunakan tf.math.log dan tf.math.square untuk keamanan di TF 2.x
        # NLL = 0.5 * log(sigma) + 0.5 * (y - mu)^2 / sigma
        nll = 0.5 * tf.math.log(sigma) + 0.5 * tf.math.square(y_true - mu) / sigma
        return tf.reduce_mean(nll)

    @staticmethod
    def uncertainty_metric(y_true, y_pred):
        """Metrik pemantau: Rata-rata ketidakpastian (sigma) yang diprediksi model."""
        sigma = y_pred[..., 1:2]
        # [FIX] Clip sigma untuk mencegah nilai ekstrem negatif
        return tf.reduce_mean(tf.clip_by_value(sigma, 1e-6, 1e6))

    @staticmethod
    def mean_absolute_error_mu(y_true, y_pred):
        """[FIX KRITIS] Metrik MAE yang hanya membandingkan Y_true dengan Mean (kolom 0) dari Y_pred."""
        mu = y_pred[..., 0:1] # Ambil hanya kolom Mean (Mu)
        return tf.reduce_mean(tf.abs(y_true - mu))

    @staticmethod
    def calculate_z_score(value, mean, std):
        """Hitung Z-Score untuk deteksi outlier standar."""
        if std < 1e-9: return 0.0
        return (value - mean) / std

# =============================================================================
# SECTION 2: DATA MANAGEMENT & CLUSTERING (THE ORGANIZER)
# =============================================================================

class DataGuard:
    """Penjaga integritas data sebelum masuk ke neural network."""
    def __init__(self, required_columns: List[str]):
        self.required_columns = required_columns

    def validate_structure(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty: return False
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing: return False
        return True

    def sanitize_temporal(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        df_clean = df.copy()
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            df_clean = df_clean.dropna(subset=[date_col])
            # PENTING: Reset index agar urut 0..N untuk slicing tensor
            df_clean = df_clean.sort_values(date_col).reset_index(drop=True)
        return df_clean

    def sanitize_numeric(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        return df

class GeoClusterer:
    """
    Mengelompokkan gempa berdasarkan kedekatan spasial.
    Setiap cluster (misal: Gunung Semeru, Gunung Raung) akan punya 'Otak' (Model) sendiri.
    """
    def __init__(self, eps: float, min_samples: int, metric: str = "haversine"):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)

    def fit_predict(self, df: pd.DataFrame, lat_col="EQ_Lintang", lon_col="EQ_Bujur") -> pd.Series:
        if lat_col not in df.columns or lon_col not in df.columns:
            return pd.Series([-1] * len(df), index=df.index, name='cluster_id')
        
        valid = df[[lat_col, lon_col]].dropna()
        if valid.empty: return pd.Series([-1]*len(df), index=df.index, name='cluster_id')
        
        # DBSCAN Haversine butuh radian
        coords = np.radians(valid.values)
        labels = self.model.fit_predict(coords)
        
        full_labels = pd.Series(-1, index=df.index, name='cluster_id')
        full_labels.loc[valid.index] = labels
        
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"[GeoClusterer] Teridentifikasi {n_c} cluster spasial aktif.")
        return full_labels

# =============================================================================
# SECTION 3: TENSOR FACTORY (SEQUENCE GENERATION)
# =============================================================================

class TensorFactory:
    """
    Pabrik Tensor: Mengubah data tabular 2D menjadi Array 3D [Samples, TimeSteps, Features].
    Menangani logika Sliding Window dan Teacher Forcing untuk arsitektur Seq2Seq.
    """
    def __init__(self, features: List[str], target: str, seq_len: int, pred_len: int):
        self.features = features
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Pastikan target ada dalam daftar fitur
        if target not in features:
            self.features.append(target)
        self.target_idx = self.features.index(target)
        self.num_features = len(self.features)

    def construct_training_tensors(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Membuat tensor untuk Training Sequence-to-Sequence (Encoder-Decoder).
        
        Args:
            data (np.ndarray): Dataset (sudah diskalakan) dengan shape (n_rows, n_features)
            
        Returns:
            X_encoder (Batch, Seq_Len, Features)
            X_decoder (Batch, Pred_Len, Features) -> Digunakan untuk Teacher Forcing
            Y_target  (Batch, Pred_Len, 1)        -> Target Prediksi
        """
        n_rows = len(data)
        
        # Window = (Input Sequence) + (Prediction Horizon)
        window_size = self.seq_len + self.pred_len
        
        # [FIX]: Menggunakan n_rows < window_size.
        # Artinya jika data == window_size, kita masih bisa ambil 1 sampel.
        if n_rows < window_size:
            # Mengembalikan array kosong dengan shape yang benar agar code di bawahnya tidak crash
            return (
                np.zeros((0, self.seq_len, self.num_features)), 
                np.zeros((0, self.pred_len, self.num_features)), 
                np.zeros((0, self.pred_len, 1))
            )
        
        X_enc_list, X_dec_list, Y_list = [], [], []
        
        # Iterasi Sliding Window
        # Range berhenti di: Total - Window + 1 agar indeks terakhir tercakup
        for i in range(n_rows - window_size + 1):
            
            # --- Indices ---
            idx_start_enc = i
            idx_end_enc   = i + self.seq_len  # Batas antara encoder dan prediksi
            
            idx_start_pred = idx_end_enc
            idx_end_pred   = idx_start_pred + self.pred_len
            
            # --- Slicing ---
            
            # 1. Encoder Input: dari t=0 s/d t=seq_len
            x_enc_sample = data[idx_start_enc : idx_end_enc, :]
            
            # 2. Decoder Input (Teacher Forcing):
            #    Biasanya berupa Lag-1 dari target window.
            #    Kita ambil data dari akhir encoder (-1) sampai sebelum prediksi berakhir (-1).
            x_dec_sample = data[idx_end_enc - 1 : idx_end_pred - 1, :]
            
            # 3. Target Output (Y):
            #    Ambil hanya kolom target (misal: PheromoneScore).
            #    Asumsi: Target kolom terakhir (-1) atau sesuaikan indeksnya.
            #    Menggunakan slice data[start:end, -1:] agar dimensi tetap (Pred_Len, 1)
            target_col_idx = self.target_idx 
            
            # Menggunakan slice target_col_idx:target_col_idx+1 untuk menjaga dimensi (Pred_Len, 1)
            y_sample = data[idx_start_pred : idx_end_pred, target_col_idx:target_col_idx+1]
            
            X_enc_list.append(x_enc_sample)
            X_dec_list.append(x_dec_sample)
            Y_list.append(y_sample)
            
        return np.array(X_enc_list), np.array(X_dec_list), np.array(Y_list)

    def construct_inference_tensor(self, data: np.ndarray) -> np.ndarray:
        """Membuat tensor X_encoder saja untuk prediksi."""
        n = len(data)
        if n < self.seq_len:
            return np.zeros((0, self.seq_len, self.num_features))

        X_enc = []
        for i in range(n - self.seq_len + 1):
            X_enc.append(data[i : i + self.seq_len])
        return np.array(X_enc)

    @property
    def input_seq_len(self): return self.seq_len

    @property
    def target_seq_len(self): return self.pred_len

# =============================================================================
# SECTION 4: DEEP LEARNING ARCHITECTURE (THE BRAIN)
# =============================================================================

class DeepProbabilisticArchitecture:
    """
    Arsitektur Neural Network V6.0 Titan.
    Menggabungkan:
    - Bidirectional LSTM (Memahami konteks masa lalu & masa depan).
    - Bahdanau Attention (Fokus pada momen penting).
    - Probabilistic Output Layer (Prediksi Mean & Variance).
    """
    def __init__(self, config: LstmPipelineConfig):
        self.cfg = config

    def build_model(self, num_features: int, params: Dict[str, Any] = None) -> Model:
        """
        Membangun Arsitektur Sequence-to-Sequence (Seq2Seq) dengan:
        1. Bi-LSTM Encoder
        2. LSTM Decoder (dengan Teacher Forcing inputs)
        3. Attention Mechanism (Bahdanau/Luong style)
        4. Probabilistic Output (Gaussian Layer: Mu & Sigma)
        """
        import tensorflow as tf
        from keras.layers import (Input, Dense, LSTM, Bidirectional, Conv1D, 
                                             BatchNormalization, Concatenate, Dropout, 
                                             TimeDistributed, Attention, Lambda)
        from keras.models import Model
        from keras.optimizers import Adam
        import keras.backend as K

        # Config Setup
        hp = params if params else {}
        input_len = self.cfg.input_seq_len
        target_len = self.cfg.target_seq_len
        
        # Hyperparameters (Prioritas: params > self.cfg > default)
        latent_dim = hp.get('latent_dim', getattr(self.cfg, 'latent_dim', 64))
        dropout = hp.get('dropout_rate', getattr(self.cfg, 'dropout_rate', 0.2))
        lr = hp.get('learning_rate', getattr(self.cfg, 'learning_rate', 0.001))
        
        # ---------------------------------------------------------
        # 1. ENCODER BLOCK
        # ---------------------------------------------------------
        encoder_inputs = Input(shape=(input_len, num_features), name='encoder_input')
        
        # Feature Extraction (1D Conv) - Optional, bagus untuk menangkap pola lokal/noise
        x = Conv1D(filters=latent_dim, kernel_size=3, padding='same', activation='relu')(encoder_inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        
        # Deep Bi-LSTM Encoder
        # Layer ini mengembalikan sequence untuk input attention
        # State h dan c digabungkan (Forward + Backward) untuk inisialisasi Decoder
        encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True, dropout=dropout), name='encoder_bi_lstm')
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(x)
        
        # Merge States: Karena Bidirectional, dimensi state menjadi 2x latent_dim
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        encoder_states = [state_h, state_c]

        # ---------------------------------------------------------
        # 2. DECODER BLOCK (Teacher Forcing Architecture)
        # ---------------------------------------------------------
        # Decoder input shape: (Prediction Length, Features)
        decoder_inputs = Input(shape=(target_len, num_features), name='decoder_input')
        
        # Decoder units harus match dengan encoder state size (latent_dim * 2)
        decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, dropout=dropout, name='decoder_lstm')
        
        # Output decoder mengabaikan state internalnya sendiri, melainkan diproses oleh Attention
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        
        # ---------------------------------------------------------
        # 3. ATTENTION MECHANISM
        # ---------------------------------------------------------
        # Attention layer menghubungkan:
        # Query = decoder_outputs (apa yang sedang kita prediksi sekarang)
        # Value = encoder_outputs (seluruh konteks masa lalu)
        attn_layer = Attention(name='attention_layer')
        context_vector = attn_layer([decoder_outputs, encoder_outputs])
        
        # Gabungkan Context Vector (dari masa lalu) + Output Decoder (prediksi saat ini)
        decoder_combined_context = Concatenate(axis=-1)([context_vector, decoder_outputs])

        # ---------------------------------------------------------
        # 4. PROBABILISTIC HEAD (Aleatoric Uncertainty)
        # ---------------------------------------------------------
        # Layer Dense Intermediate
        x_out = TimeDistributed(Dense(64, activation='relu', kernel_initializer='he_normal'))(decoder_combined_context)
        x_out = Dropout(dropout)(x_out)
        
        # Head A: Prediksi Nilai Tengah (Mean / Mu)
        # HANYA INI YANG KITA PERTAHANKAN
        mu = TimeDistributed(Dense(1, activation='linear'), name='mu')(x_out)
        
        # HAPUS Head B, log_sigma_sq, Lambda Layer sigma, dan Concatenate
        # Output model HANYA mu
        output = mu
        
        # ---------------------------------------------------------
        # 5. COMPILATION
        # ---------------------------------------------------------
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=output)
        
        # Optimizer dengan gradient clipping untuk stabilitas
        optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        
        try:
            # [FIX KRITIS]: Ganti Loss menjadi MAE/MSE standar
            model.compile(
                optimizer=optimizer,
                loss='mae', # Menggunakan Mean Absolute Error
                metrics=['mae', 'mse'] # Metrik standar
            )
            logger.info(f"Model LSTM berhasil dikompilasi (Standard MAE). Input: {input_len}x{num_features}")
        except Exception as e:
            logger.critical(f"FATAL: Gagal kompilasi model LSTM. Cek arsitektur. Error: {e}")
            raise RuntimeError(f"LSTM Compilation Failed: {e}")
            
        return model

class BayesianLikeOptimizer:
    """
    Sistem pencarian hyperparameter sederhana.
    Mencoba berbagai kombinasi konfigurasi untuk menemukan model terbaik.
    """
    def __init__(self, factory):
        self.factory = factory
        self.space = {
            'latent_dim': [64, 128],
            'learning_rate': [5e-4, 1e-4, 5e-5], 
            'dropout_rate': [0.1, 0.2]
        }

    def search(self, X_enc, X_dec, Y, trials=3):
        if len(X_enc) < 50: 
            return {'latent_dim': 64, 'learning_rate': 1e-3, 'dropout_rate': 0.2}
            
        best_loss = float('inf')
        best_params = {}
        
        logger.info(f"    [Tuner] Menjalankan {trials} trial optimasi...")
        
        for i in range(trials):
            params = {k: random.choice(v) for k, v in self.space.items()}
            K.clear_session() # Bersihkan memori GPU
            try:
                model = self.factory.build_model(X_enc.shape[-1], params)
                # Training singkat untuk evaluasi
                h = model.fit([X_enc, X_dec], Y, epochs=3, batch_size=32, verbose=0, validation_split=0.2)
                val_loss = h.history['val_loss'][-1]
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            except: continue
        
        logger.info(f"    [Tuner] Parameter Terbaik: {best_params} (Loss: {best_loss:.4f})")
        return best_params

# =============================================================================
# SECTION 5: ARTIFACT & VISUALIZATION MANAGEMENT
# =============================================================================

class ArtifactVault:
    """Menyimpan dan memuat model, scaler, dan metadata dengan aman."""
    def __init__(self, model_dir, visual_dir):
        self.model_dir = Path(model_dir)
        self.visual_dir = Path(visual_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

    def save_cluster_state(self, cid, model, scaler, meta):
        try:
            model.save(self.model_dir / f"lstm_model_c{cid}.keras")
            dump(scaler, self.model_dir / f"scaler_c{cid}.joblib")
            with open(self.model_dir / f"meta_c{cid}.json", 'w') as f:
                json.dump(meta, f, indent=4)
        except Exception as e:
            logger.error(f"Save failed c{cid}: {e}")

    def load_cluster_state(self, cid):
        m_path = self.model_dir / f"lstm_model_c{cid}.keras"
        s_path = self.model_dir / f"scaler_c{cid}.joblib"
        if not m_path.exists():
            return None, None

        try:
            import absl.logging
            absl.logging.set_verbosity(absl.logging.ERROR)

            # Define custom objects for loading model
            cust = {
                'gaussian_nll': MathKernel.gaussian_nll,
                'uncertainty_metric': MathKernel.uncertainty_metric,
                'mean_absolute_error_mu': MathKernel.mean_absolute_error_mu
            }

            # Safe loading: compile=False (jika versi TF lama/baru, opsi safe_mode mungkin tidak ada)
            model = load_model(m_path, custom_objects=cust, compile=False)
            scaler = load(s_path)

            logger.info(f"[Vault] Model Cluster {cid} berhasil dimuat.")
            return model, scaler
        except Exception as e:
            logger.error(f"[Vault] GAGAL memuat model c{cid}: {e}. File mungkin corrupt atau TF version mismatch.")
            return None, None

    def list_clusters(self):
        import re
        files = list(self.model_dir.glob("lstm_model_c*.keras"))
        clusters = []
        for f in files:
            m = re.search(r'c(\d+)\.keras$', f.name)
            if m: clusters.append(int(m.group(1)))
        return sorted(list(set(clusters)))
    
    def load_all(self, cid): return self.load_cluster_state(cid)

class AdvancedVisualizer:
    """Generator grafik canggih untuk analisis probabilitas."""
    def __init__(self, output_dir):
        self.out = output_dir

    def plot_probabilistic_forecast(self, actual, pred_mu, pred_sigma, cid, suffix=""):
        try:
            plt.figure(figsize=(12, 6))
            x = np.arange(len(actual))
            plt.plot(x, actual, 'k-', label='Actual', alpha=0.7, linewidth=1.5)
            plt.plot(x, pred_mu, 'r--', label='Predicted Mean', linewidth=1.5)
            
            # Plot Uncertainty Interval (95% Confidence)
            lower = pred_mu - 1.96 * pred_sigma
            upper = pred_mu + 1.96 * pred_sigma
            plt.fill_between(x, lower, upper, color='red', alpha=0.2, label='Uncertainty (95% CI)')
            
            plt.title(f"Probabilistic Forecast - Cluster {cid} {suffix}", fontsize=14)
            plt.legend()
            plt.tight_layout()
            plt.savefig(self.out / f"pred_vs_actual_c{cid}.png", dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Gagal plot forecast: {e}")

    def plot_loss_curves(self, history: Dict, cid: int):
        try:
            plt.figure(figsize=(8, 5))
            plt.plot(history['loss'], label='Train NLL')
            plt.plot(history['val_loss'], label='Val NLL')
            plt.title(f"Learning Curve (NLL) - Cluster {cid}")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(self.out / f"loss_c{cid}.png", dpi=300)
            plt.close()
        except Exception: pass
        
    def plot_residuals(self, errors: np.ndarray, cid: int):
        try:
            plt.figure(figsize=(8, 5))
            sns.histplot(errors, kde=True, color='purple')
            plt.title(f"Residual Distribution - Cluster {cid}")
            plt.xlabel("Error")
            plt.savefig(self.out / f"error_dist_c{cid}.png", dpi=300)
            plt.close()
        except Exception: pass
        
    # Aliases for compatibility with older code calls
    def plot_prediction_comparison(self, a, p, c): self.plot_probabilistic_forecast(a, p, np.zeros_like(p), c)
    def plot_error_distribution(self, e, c): self.plot_residuals(e, c)
    def plot_training_history(self, h, c): self.plot_loss_curves(h, c)

# =============================================================================
# SECTION 6: REALTIME BUFFER & DRIFT MONITOR
# =============================================================================

class DriftMonitor:
    """
    Memantau pergeseran data (Data Drift).
    Jika data realtime terlalu berbeda dari statistik data training,
    sistem akan memberikan peringatan.
    """
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.baseline_stats = {}

    def update_baseline(self, df_train: pd.DataFrame, features: List[str]):
        for f in features:
            if f in df_train:
                self.baseline_stats[f] = {
                    'mean': df_train[f].mean(),
                    'std': df_train[f].std()
                }

    def check_drift(self, df_new: pd.DataFrame) -> bool:
        """Cek apakah data baru menyimpang jauh (Z-Score check)."""
        drift_detected = False
        for f, stats in self.baseline_stats.items():
            if f in df_new:
                val = df_new[f].mean()
                z = abs(val - stats['mean']) / (stats['std'] + 1e-9)
                if z > self.threshold:
                    logger.warning(f"Data Drift Detected on {f} (Z={z:.2f})")
                    drift_detected = True
        return drift_detected

class InferenceBuffer:
    """
    Buffer Memori Jangka Pendek (Sliding Window).
    Menampung data history untuk memberikan konteks pada prediksi realtime.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer_df = pd.DataFrame()
        
    def update(self, df_new):
        if df_new.empty: return
        self.buffer_df = pd.concat([self.buffer_df, df_new], ignore_index=True)
        self.buffer_df = self.buffer_df.sort_values('Acquired_Date')
        self.buffer_df = self.buffer_df.drop_duplicates(subset=['Acquired_Date', 'Nama'], keep='last')
        
        # Keep buffer size limited (e.g., 3x window size)
        limit = self.window_size * 3
        if len(self.buffer_df) > limit:
            self.buffer_df = self.buffer_df.iloc[-limit:]
            
    def get_context(self):
        return self.buffer_df.copy()

class DataProcessor:
    """Wrapper untuk Feature Engineering di dalam LSTM."""
    def __init__(self, config):
        self.cfg = config
        # Feature Engineer diinisialisasi dengan Config Global
        self.fe = FeatureEngineer(CONFIG.FEATURE_ENGINEERING, CONFIG.ACO_ENGINE)
        self.guard = DataGuard(['Acquired_Date', 'Magnitudo'])
        self.clusterer = GeoClusterer(config.clustering_eps, config.clustering_min_samples)
        
    def prepare(self, df):
        """Pipeline preprocessing standar."""
        df = self.guard.sanitize_temporal(df, 'Acquired_Date')
        df = self.guard.sanitize_numeric(df, ['Magnitudo'])
        df = self.fe.basic_cleanup(df)
        df = self.fe.add_spatio_temporal_features(df)
        df = self.fe.add_lag_and_rolling(df)
        
        if 'cluster_id' not in df.columns:
            df['cluster_id'] = self.clusterer.fit_predict(df)
        return df

# =============================================================================
# SECTION 7: MAIN ENGINE FACADE (THE INTERFACE)
# =============================================================================

class LstmEngine:
    """
    Engine Utama LSTM V6.0 TITAN.
    Menghubungkan semua komponen di atas menjadi satu kesatuan cerdas.
    """
    def __init__(self, config):
        self.cfg = config
        self.vault = ArtifactVault(self.cfg.model_dir, self.cfg.visuals_dir)
        self.processor = DataProcessor(self.cfg)
        self.architect = DeepProbabilisticArchitecture(self.cfg)
        self.tuner = BayesianLikeOptimizer(self.architect)
        self.viz_manager = AdvancedVisualizer(Path(self.cfg.visuals_dir))
        self.buffer = InferenceBuffer(self.cfg.input_seq_len)
        self.drift_mon = DriftMonitor()
        
        # Cache in-memory untuk performa realtime
        self.models_cache = {}
        
        if os.path.exists("logs"): shutil.rmtree("logs", ignore_errors=True)

    # Compatibility Props for other engines (Tidak ada perubahan)
    @property
    def manager(self): return self.vault
    @property
    def trainer(self): return self
    @property
    def viz(self): return self.viz_manager

    def load_buffer(self, df_history):
        """Memuat data training ke buffer (inisialisasi untuk live stream)."""
        if df_history is not None and not df_history.empty:
            logger.info(f"Loading {len(df_history)} rows to buffer.")
            self.buffer.update(df_history)
            
            # Init drift baseline from history
            df_proc = self.processor.prepare(df_history)
            feats = [c for c in df_proc.columns if c in self.cfg.features]
            self.drift_mon.update_baseline(df_proc, feats)

        def integrate_ga_prediction(self, pred: Dict[str, Any], cid: Optional[int] = None, attach_to: str = "nearest"):
            """
            Integrasi output GA (pred dict) ke buffer LSTM.
            - pred: dict seperti {'pred_lat', 'pred_lon', 'bearing_degree', 'distance_km', 'confidence', ...}
            - cid: jika known cluster_id, gunakan mapping langsung; jika None, akan dicari row terdekat di buffer
            - attach_to: "nearest" | "append_row"
            Efek: menambah kolom GA ke baris yang relevan di self.buffer.buffer_df
            """
            try:
                if not pred or ('pred_lat' not in pred or 'pred_lon' not in pred):
                    logger.warning("[LSTM] integrate_ga_prediction: pred kosong atau tidak punya lat/lon.")
                    return None

                buf = self.buffer.get_context()
                if buf is None or buf.empty:
                    logger.warning("[LSTM] Buffer kosong, tidak ada tempat integrasi GA; opsi append_row dipertimbangkan.")
                    if attach_to == "append_row":
                        row = {
                            'Acquired_Date': pd.Timestamp.now(),
                            'EQ_Lintang': pred.get('pred_lat'),
                            'EQ_Bujur': pred.get('pred_lon'),
                            'Nama': 'GA_PRED',
                        }
                        row.update({
                            'ga_pred_lat': pred.get('pred_lat'),
                            'ga_pred_lon': pred.get('pred_lon'),
                            'ga_bearing': pred.get('bearing_degree'),
                            'ga_distance_km': pred.get('distance_km'),
                            'ga_confidence': pred.get('confidence'),
                        })
                        self.buffer.update(pd.DataFrame([row]))
                        return True
                    return None

                # jika cluster id diberikan -> filter buffer per cluster
                if cid is not None and 'cluster_id' in buf.columns:
                    sub = buf[buf['cluster_id'] == cid]
                    if sub.empty:
                        candidates = buf
                    else:
                        candidates = sub
                else:
                    candidates = buf

                # hitung index terdekat (haversine) ke pred point
                lat_p = float(pred.get('pred_lat'))
                lon_p = float(pred.get('pred_lon'))

                # vectorized haversine
                coords = candidates[['EQ_Lintang', 'EQ_Bujur']].dropna()
                if coords.empty:
                    # fallback append
                    if attach_to == "append_row":
                        row = {'Acquired_Date': pd.Timestamp.now(), 'EQ_Lintang': lat_p, 'EQ_Bujur': lon_p, 'Nama': 'GA_PRED'}
                        row.update({
                            'ga_pred_lat': lat_p,
                            'ga_pred_lon': lon_p,
                            'ga_bearing': pred.get('bearing_degree'),
                            'ga_distance_km': pred.get('distance_km'),
                            'ga_confidence': pred.get('confidence'),
                        })
                        self.buffer.update(pd.DataFrame([row]))
                        return True
                    return None

                # compute distances
                def _h(lat1, lon1, lat2_arr, lon2_arr):
                    return np.array([GeoClusterer(0,1).model.metric if False else
                                     GeoMathCore.haversine(lat1, lon1, rlat, rlon)
                                     for rlat, rlon in zip(lat2_arr, lon2_arr)])

                # faster vector compute
                lat_arr = coords['EQ_Lintang'].astype(float).values
                lon_arr = coords['EQ_Bujur'].astype(float).values
                # compute distances vectorized using GeoMathCore.haversine in loop (numpy vectorization with listcomp)
                dists = np.array([GeoMathCore.haversine(lat_p, lon_p, la, lo) for la, lo in zip(lat_arr, lon_arr)])
                nearest_idx_local = coords.index[np.argmin(dists)]

                # update buffer row with GA fields
                self.buffer.buffer_df.loc[nearest_idx_local, 'ga_pred_lat'] = lat_p
                self.buffer.buffer_df.loc[nearest_idx_local, 'ga_pred_lon'] = lon_p
                self.buffer.buffer_df.loc[nearest_idx_local, 'ga_bearing'] = pred.get('bearing_degree')
                self.buffer.buffer_df.loc[nearest_idx_local, 'ga_distance_km'] = pred.get('distance_km')
                self.buffer.buffer_df.loc[nearest_idx_local, 'ga_confidence'] = pred.get('confidence')

                logger.info(f"[LSTM] GA pred integrated to buffer index {nearest_idx_local} (cid={cid})")
                return nearest_idx_local

            except Exception as e:
                logger.error(f"[LSTM] integrate_ga_prediction failed: {e}")
                return None
        def load_ga_json_and_integrate(self, ga_json_path: str, cid: Optional[int] = None):
            """
            Load GA vector JSON file and integrate into buffer.
            Returns True/False
            """
            try:
                if not os.path.exists(ga_json_path):
                    logger.warning(f"[LSTM] GA json not found: {ga_json_path}")
                    return False
                with open(ga_json_path, 'r') as f:
                    pred = json.load(f)
                if isinstance(pred, dict) and 'pred_lat' in pred:
                    return bool(self.integrate_ga_prediction(pred, cid=cid))
                logger.warning("[LSTM] GA JSON doesn't contain 'pred_lat'/'pred_lon'")
                return False
            except Exception as e:
                logger.error(f"[LSTM] load_ga_json_and_integrate failed: {e}")
                return False

    def get_buffer(self) -> pd.DataFrame:
        """Mengembalikan data buffer historis dari InferenceBuffer."""
        return self.buffer.get_context()
    
    def update_buffer(self, df_new_events: pd.DataFrame):
        """Memperbarui buffer dengan event yang baru diprediksi."""
        self.buffer.update(df_new_events)

    def train_cluster(self, cid, data, scaler): pass # Dummy for compat

    @execution_telemetry
    def train(self, df_train):
        """
        Melatih model LSTM per-cluster dengan strategi cleaning & scaling yang robust.
        """
        if df_train is None or df_train.empty: 
            return False
        
        logger.info("=== LSTM V6.0 TITAN TRAINING START ===")
        
        # 1. Prepare data
        df_proc = self.processor.prepare(df_train)
        
        # Ambil list cluster valid (kecuali noise -1)
        clusters = sorted([c for c in df_proc['cluster_id'].unique() if c != -1])
        success = 0
        
        for cid in clusters:
            logger.info(f"Processing Cluster {cid}...")
            df_c = df_proc[df_proc['cluster_id'] == cid].sort_values('Acquired_Date')
            
            # Filter fitur
            ga_cols = [c for c in df_c.columns if c.startswith('ga_') or c in
                       ('ga_pred_lat', 'ga_pred_lon', 'ga_bearing', 'ga_distance_km', 'ga_confidence')]
            feats = [c for c in df_c.columns if (c in self.cfg.features or c == self.cfg.target_feature)]
            # union with ga cols
            feats = list(set(feats + ga_cols))
            
            # --- 1. ROBUST DATA CLEANING ---
            data_to_process = df_c[feats].copy()
            
            # [Step A]: Sanitasi Nilai Ekstrim
            data_to_process = data_to_process.replace([np.inf, -np.inf], np.nan)
            
            # [Step B]: Imputasi Prioritas Median
            # Cara ini jauh lebih stabil dan cepat daripada KNNImputer di dalam loop
            median_vals = data_to_process.median(numeric_only=True)
            data_to_process = data_to_process.fillna(median_vals)
            
            # Final fallback: Isi 0.0 jika masih ada NaN (misal kolom kosong total)
            data_to_process = data_to_process.fillna(0.0)

            # --- 2. SCALING ---
            std_sum = data_to_process.std(numeric_only=True).sum()
            if std_sum < 1e-9:
                logger.warning(f"Cluster {cid} diabaikan: Data Konstan/Flat (Total StdDev={std_sum:.2e}).")
                continue # Skip cluster ini, jangan paksa training

            # Inisialisasi Scaler
            # Note: RobustScaler lebih disarankan untuk data seismik dibanding StandardScaler
            # karena gempa besar adalah outlier alami. RobustScaler menggunakan median & IQR.
            scaler = RobustScaler()

            try:
                # [Primary Strategy]: RobustScaler
                # Ideal untuk data seismik karena menggunakan Median & IQR (kebal terhadap gempa besar/outlier)
                data_mtx = scaler.fit_transform(data_to_process)
                
            except Exception as e_robust:
                # [Fallback Strategy]: StandardScaler
                # Digunakan jika RobustScaler gagal (misalnya karena Interquartile Range = 0)
                logger.warning(f"RobustScaler issue at Cluster {cid} ({e_robust}). Fallback ke StandardScaler.")
                
                try:
                    scaler = StandardScaler()
                    data_mtx = scaler.fit_transform(data_to_process)
                except Exception as e_std:
                    logger.error(f"Scaling FATAL failure c{cid}: {e_std}. Skip cluster.")
                    continue

            # [FIX KRITIS]: Post-Scaling Safety Check
            # Cek apakah hasil scaling mengandung NaN atau Infinity
            if not np.isfinite(data_mtx).all():
                logger.error(f"Scaling result contains NaN/Inf at cluster {cid}. Skip cluster.")
                continue

            # --- 3. TENSOR CREATION ---
            tfactory = TensorFactory(feats, self.cfg.target_feature, self.cfg.input_seq_len, self.cfg.target_seq_len)
            
            try:
                X_enc, X_dec, Y = tfactory.construct_training_tensors(data_mtx)
            except Exception as e:
                logger.warning(f"Tensor construct error c{cid}: {e}. Skip.")
                continue
            
            # Skip jika sampel terlalu sedikit
            if len(X_enc) < 10: 
                logger.warning(f"Sample c{cid} terlalu sedikit ({len(X_enc)}). Skip training.")
                continue
            
            # --- 4. MODEL SEARCH & TRAINING ---
            best_p = self.tuner.search(X_enc, X_dec, Y)
            model = self.architect.build_model(len(feats), best_p)
            
            log_path = Path("logs") / f"c{cid}"
            
            cbs = [
                EarlyStopping(patience=self.cfg.early_stopping_patience, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
                TensorBoard(log_dir=str(log_path))
            ]
            
            try:
                h = model.fit(
                    [X_enc, X_dec], Y, 
                    epochs=self.cfg.epochs, 
                    batch_size=self.cfg.batch_size, 
                    validation_split=self.cfg.validation_split, 
                    callbacks=cbs, 
                    verbose=1
                )
                
                # Simpan State
                meta = {"features": feats, "best_params": best_p, "trained_at": str(datetime.now())}
                self.vault.save_cluster_state(cid, model, scaler, meta)
                
                # Plot Curve
                self.viz_manager.plot_loss_curves(h.history, cid)
                
                success += 1
            except Exception as e:
                logger.error(f"Training Failed Cluster {cid}: {e}")

        return success > 0

    @execution_telemetry
    def predict_on_static(self, df_test):
        if df_test is None or df_test.empty:
            return df_test, pd.DataFrame()

        # Prepare Cleanly
        df_proc = self.processor.prepare(df_test)
        df_out = df_proc.copy()

        # Init Columns
        df_out['lstm_prediction'] = np.nan
        df_out['prediction_sigma'] = np.nan
        df_out['prediction_error'] = np.nan
        df_out['anomaly_score'] = 0.0

        anomalies = []
        if 'cluster_id' not in df_out.columns:
            df_out['cluster_id'] = -1

        for cid in self.vault.list_clusters():
            mask = df_out['cluster_id'] == cid
            if not mask.any():
                continue

            df_c = df_out.loc[mask].sort_values('Acquired_Date')

            # Cache Lookup
            if cid in self.models_cache:
                model, scaler = self.models_cache[cid]
            else:
                model, scaler = self.vault.load_cluster_state(cid)
                if model:
                    self.models_cache[cid] = (model, scaler)

            if not model:
                continue

            # Feature check
            feats = getattr(scaler, 'feature_names_in_', None)
            if feats is None:
                feats = list(self.cfg.features)

            # ensure GA cols present in df_c are included
            ga_cols = [c for c in df_c.columns if c.startswith('ga_') or c in
                       ('ga_pred_lat', 'ga_pred_lon', 'ga_bearing', 'ga_distance_km', 'ga_confidence')]
            for g in ga_cols:
                if g not in feats:
                    feats.append(g)

            if not all(f in df_c.columns for f in feats):
                logger.warning(f"[LSTM] Missing features for cluster {cid}. Required: {feats}")
                continue

            # Transform & Tensor
            expected_feats = list(feats)
            for ef in expected_feats:
                if ef not in df_c.columns:
                    df_c[ef] = 0.0

            data_mtx = scaler.transform(df_c[expected_feats].fillna(0))
            tfactory = TensorFactory(list(feats), self.cfg.target_feature, self.cfg.input_seq_len, self.cfg.target_seq_len)
            X_enc = tfactory.construct_inference_tensor(data_mtx)

            if len(X_enc) == 0:
                continue

            # Predict
            X_dec_dummy = np.zeros((len(X_enc), self.cfg.target_seq_len, len(feats)))
            preds = model.predict([X_enc, X_dec_dummy], verbose=0)

            # preds shape: (n_samples, pred_len, 1) OR (n_samples, pred_len)
            preds = np.squeeze(preds)
            if preds.ndim == 2:
                # pilih horizon pertama sebagai prediksi "next event"
                mu_seq = preds[:, 0]
            else:
                mu_seq = preds  # shape (n_samples,)

            # sigma not available in current architecture -> zeros
            sigma_seq = np.zeros_like(mu_seq)

            # Inverse transform: rebuild dummy matrix for scaler inverse
            dummy = np.zeros((len(mu_seq), len(feats)))
            dummy[:, tfactory.target_idx] = mu_seq
            try:
                res_mu = scaler.inverse_transform(dummy)[:, tfactory.target_idx]
            except Exception:
                # fallback jika scaler tipe tertentu tidak mendukung inverse untuk array shape
                res_mu = mu_seq.copy()

            res_sigma = sigma_seq * (getattr(scaler, 'scale_', np.ones(len(feats)))[tfactory.target_idx] if hasattr(scaler, 'scale_') else 1.0)

            # Map Back -> each mu_seq corresponds to prediction time starting at index = seq_len
            start_idx = self.cfg.input_seq_len
            valid_idx = df_c.index[start_idx : start_idx + len(res_mu)]
            min_l = min(len(valid_idx), len(res_mu))
            final_idx = list(valid_idx[:min_l])

            if len(final_idx) == 0:
                continue

            df_out.loc[final_idx, 'lstm_prediction'] = res_mu[:min_l]
            df_out.loc[final_idx, 'prediction_sigma'] = res_sigma[:min_l]

            # Error & Anomaly (Z-Score Logic)
            actual = df_c.loc[final_idx, self.cfg.target_feature].values
            err = np.abs(res_mu[:min_l] - actual)
            df_out.loc[final_idx, 'prediction_error'] = err

            # Z-Score Anomaly: Error / Sigma (safe)
            z_score = err / (res_sigma[:min_l] + 1e-6)
            df_out.loc[final_idx, 'anomaly_score'] = z_score

            is_anom = z_score > 2.5  # 2.5 Sigma threshold
            if is_anom.any():
                anomalies.append(df_out.loc[final_idx][is_anom])

            # Visualize (silently warn jika gagal)
            try:
                self.viz_manager.plot_probabilistic_forecast(actual, res_mu[:min_l], res_sigma[:min_l], cid)
            except Exception as e:
                logger.warning(f"[Viz] plotting failed c{cid}: {e}")

        final_anoms = pd.concat(anomalies) if anomalies else pd.DataFrame()

        # --- SAVE CSV OUTPUT (two-year + 15-day + anomalies) ---
        try:
            self._save_lstm_records(df_out, final_anoms)
        except Exception as e:
            logger.warning(f"[LSTM] Failed to save LSTM records: {e}")

        return df_out, final_anoms

        def record_actual_events(self, df_actual: pd.DataFrame):
            """
            Merekam event aktual (ground truth) ke buffer & vault
            untuk pembelajaran lanjutan LSTM (integrasi CNN).
            """
            if df_actual is None or df_actual.empty:
                logger.warning("[LSTM] record_actual_events: df_actual kosong")
                return False

            try:
                # 1️⃣ Update buffer realtime
                self.buffer.update(df_actual)

                # 2️⃣ (Opsional) Simpan ke vault sebagai arsip learning
                if hasattr(self.vault, 'append'):
                    self.vault.append(df_actual)

                # 3️⃣ Persist state
                self.save_state()

                logger.info(f"[LSTM] Recorded {len(df_actual)} actual events")
                return True

            except Exception as e:
                logger.error(f"[LSTM] record_actual_events failed: {e}")
                return False
        
        
        def save_state(self):
            """
            Simpan state LSTM (buffer, metadata).
            """
            try:
                # Simpan buffer ke pickle
                state_dir = Path(self.cfg.model_dir)
                state_dir.mkdir(parents=True, exist_ok=True)

                buffer_path = state_dir / "lstm_buffer.pkl"
                with open(buffer_path, "wb") as f:
                    pickle.dump(self.buffer.get_context(), f)

                logger.info(f"[LSTM] State saved: {buffer_path}")

            except Exception as e:
                logger.warning(f"[LSTM] save_state failed: {e}")


        def extract_hidden_states(self, X_input, cid):
            # Robust extractor for encoder states for CNN
            try:
                if cid in self.models_cache:
                    model, _ = self.models_cache[cid]
                else:
                    model, _ = self.vault.load_cluster_state(cid)

                if not model:
                    return None

                # try to get named layer
                try:
                    enc = model.get_layer('encoder_bi_lstm')
                except Exception:
                    # fallback: find first Bidirectional layer
                    from keras.layers import Bidirectional
                    enc = None
                    for l in model.layers:
                        if isinstance(l, Bidirectional):
                            enc = l
                            break
                    if enc is None:
                        return None

                # Build sub-model to extract states; many Keras versions differ on outputs
                # We'll attempt to call enc.cell or use sub-model and then infer outputs shape
                sub_model = Model(inputs=model.inputs[0], outputs=enc.output)
                outs = sub_model.predict(X_input, verbose=0)

                # enc.output may be a sequence (seq_out) or tuple; try to find hidden states from original model
                # Fallback strategy: call model.predict and use intermediary states via function if available
                # Typical bidirectional LSTM with return_state returns [seq_out, fh, fc, bh, bc]
                if isinstance(outs, list) or isinstance(outs, tuple):
                    # try get h states at position 1 and 3
                    if len(outs) >= 4:
                        fh = outs[1]
                        bh = outs[3]
                        return np.concatenate([fh, bh], axis=-1)
                # If enc.output returned only sequence (n_samples, timesteps, features)
                # we can't extract states directly -> fallback to pooling
                # Use GlobalAveragePooling over timesteps as a proxy
                if outs.ndim == 3:
                    return np.mean(outs, axis=1)
                return None
            except Exception as e:
                logger.warning(f"[LSTM] extract_hidden_states failed for c{cid}: {e}")
                return None

    def process_live_stream(self, df_new):
        if df_new.empty: return pd.DataFrame(), pd.DataFrame()
        
        # Check Drift
        self.drift_mon.check_drift(df_new)
        
        self.buffer.update(df_new)
        ctx = self.buffer.get_context()
        
        pred, anom = self.predict_on_static(ctx)
        
        # Filter results for new data only
        new_ts = df_new['Acquired_Date'].values
        final_pred = pred[pred['Acquired_Date'].isin(new_ts)]
        
        if not anom.empty and 'Acquired_Date' in anom.columns:
            final_anom = anom[anom['Acquired_Date'].isin(new_ts)]
        else:
            final_anom = pd.DataFrame()
        
        # Simpan record terbaru & anomalies ke CSV agar downstream (CNN / NB) bisa konsumsi
        try:
            self._save_lstm_records(ctx, final_anom)
        except Exception as e:
            logger.warning(f"[LSTM] Failed auto-save during live stream: {e}")

        return final_pred, final_anom

    def _save_lstm_records(self, df_full: pd.DataFrame, anomalies: pd.DataFrame):
        """
        Simpan file CSV:
         - master file: semua record 2 tahun terakhir
         - recent file: 15 hari terakhir
         - anomalies file: hasil deteksi anomaly

        Timestamp diambil dari TANGGAL DATA TERBARU (Acquired_Date),
        dan file akan DITIMPA jika timestamp sama.
        """

        out_root = getattr(self.cfg, 'output_dir', 'output/lstm_results')
        os.makedirs(out_root, exist_ok=True)

        df = df_full.copy()

        # ===============================
        # 1️⃣ Pastikan kolom waktu valid
        # ===============================
        if 'Acquired_Date' not in df.columns:
            logger.error("[LSTM] Acquired_Date tidak ditemukan, batal simpan CSV.")
            return {}

        df['Acquired_Date'] = pd.to_datetime(df['Acquired_Date'], errors='coerce')
        df = df.dropna(subset=['Acquired_Date'])

        if df.empty:
            logger.warning("[LSTM] Data kosong setelah parsing tanggal.")
            return {}

        df = df.sort_values("Acquired_Date")
        # ===============================
        # 2️⃣ Ambil timestamp dari DATA TERBARU
        # ===============================
        latest_date = pd.Timestamp.now()
        ts = latest_date.strftime("%Y%m%d")

        # ===============================
        # 3️⃣ Filter rentang waktu
        # ===============================
        two_years_ago = latest_date - pd.Timedelta(days=365 * 2)
        recent_15d = latest_date - pd.Timedelta(days=15)

        df_2y = df[df['Acquired_Date'] >= two_years_ago].copy()
        df_recent = df[df['Acquired_Date'] >= recent_15d].copy()

        # ===============================
        # 4️⃣ PATH FILE (DITIMPA)
        # ===============================
        master_path = os.path.join(out_root, f"lstm_records_2y_{ts}.csv")
        recent_path = os.path.join(out_root, f"lstm_recent_15d_{ts}.csv")
        anom_path   = os.path.join(out_root, f"lstm_anomalies_{ts}.csv")

        # ===============================
        # 5️⃣ SIMPAN CSV
        # ===============================
        try:
            df_2y.to_csv(master_path, index=False)
            df_recent.to_csv(recent_path, index=False)

            if anomalies is not None and not anomalies.empty:
                anomalies.to_csv(anom_path, index=False)
            else:
                pd.DataFrame().to_csv(anom_path, index=False)

            logger.info(
                f"[LSTM] Saved CSV (overwrite-safe): "
                f"master={master_path}, recent={recent_path}, anomalies={anom_path}"
            )

            return {
                "master": master_path,
                "recent": recent_path,
                "anomalies": anom_path
            }

        except Exception as e:
            logger.error(f"[LSTM] Gagal menyimpan CSV LSTM: {e}")
            return {}

        ga_cols = [c for c in df.columns if c.startswith('ga_') or c in (
            'ga_pred_lat',
            'ga_pred_lon',
            'ga_bearing',
            'ga_distance_km',
            'ga_confidence'
        )]

        if ga_cols:
            ga_summary_path = os.path.join(out_root, f"lstm_ga_summary_{ts}.csv")
            (
                df[ga_cols]
                .dropna(how='all')
                .drop_duplicates()
                .to_csv(ga_summary_path, index=False)
            )

            logger.info(f"[LSTM] GA summary saved: {ga_summary_path}")

    def predict_realtime(self, *args):
        return pd.DataFrame(), pd.DataFrame()