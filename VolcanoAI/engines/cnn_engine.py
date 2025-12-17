# VolcanoAI/engines/cnn_engine.py
# -- coding: utf-8 --

import os
import logging
import time
import random
import shutil
import pickle
import functools
from typing import Union, Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Concatenate,
    Conv2DTranspose, Dropout, BatchNormalization,
    Dense, Reshape, Activation, Multiply, Add
)
from keras.optimizers import Adam
from keras.utils import Sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
from scipy.ndimage import rotate

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger("VolcanoAI_CNN")
logger.addHandler(logging.NullHandler())

# =============================================================================
# SECTION 1: SPATIAL DATA GENERATION (TABULAR TO IMAGE)
# =============================================================================

class CnnMathKernel:
    """Class penampung untuk Custom Loss dan Metrics CNN."""
    
    # Dice Coefficient Metric
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        # Implementasi Anda yang sudah ada
        y_true_f = tf.cast(y_true, tf.float32)
        y_pred_f = tf.cast(y_pred, tf.float32)
        y_true_f = tf.reshape(y_true_f, [-1])
        y_pred_f = tf.reshape(y_pred_f, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    # Hybrid Loss (Dice + Binary Crossentropy)
    @staticmethod
    def hybrid_dice_bce_loss(y_true, y_pred):
        # Asumsi: Ini adalah implementasi Anda untuk menggabungkan loss
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        dice_loss = 1 - CnnMathKernel.dice_coefficient(y_true, y_pred)
        # Bobot custom: Misal 50/50
        return 0.5 * bce + 0.5 * dice_loss

class SpatialDataGenerator:
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        self.grid_size = int(self.cfg.get("grid_size", 64))
        self.domain_km = float(self.cfg.get("domain_km", 200.0))
        
        self.km_per_pixel = float(self.domain_km) / float(self.grid_size)
        self.input_channels = int(self.cfg.get("input_channels", 3))
        self.radius_columns = self.cfg.get("radius_columns", ["R1_final", "R2_final", "R3_final"])

    # =======================================================
    # METHOD UTAMA: KONTROL PEMBUATAN HEATMAP
    # =======================================================

    def create_input_mask(self, row: pd.Series) -> np.ndarray:
        radii = []
        for i in range(self.input_channels):
            col = self.radius_columns[i] if i < len(self.radius_columns) else None
            val = float(row[col]) if (col and col in row and pd.notna(row[col])) else 0.0
            radii.append(val)
        channels = [self._create_gaussian_heatmap(r) for r in radii]
        while len(channels) < self.input_channels:
            channels.append(np.zeros((self.grid_size, self.grid_size), dtype=np.float32))
        return np.stack(channels, axis=-1)

    # [NEW/PINDAHKAN]: Metode untuk membuat target mask
    def create_target_mask(self, row: pd.Series) -> np.ndarray:
        target_col = 'AreaTerdampak_km2'
        if target_col in row and pd.notna(row[target_col]) and float(row[target_col]) > 0:
            area = float(row[target_col])
            radius_km = np.sqrt(area / np.pi)
            mask = self._create_gaussian_heatmap(radius_km)
            return mask.reshape(self.grid_size, self.grid_size, 1).astype(np.float32)

    # [NEW/PINDAHKAN]: Metode untuk membuat heatmap (private helper)
    def _create_gaussian_heatmap(self, radius_km: float) -> np.ndarray:
        gs = self.grid_size
        if radius_km is None or radius_km <= 0:
            return np.zeros((gs, gs), dtype=np.float32)
        radius_px = radius_km / (self.km_per_pixel + 1e-12)
        center = (gs - 1) / 2.0
        x, y = np.ogrid[:gs, :gs]
        sigma = max(radius_px / 3.0, 1.0)
        dist_sq = (x - center + 0.5) ** 2 + (y - center + 0.5) ** 2
        heatmap = np.exp(-dist_sq / (2.0 * (sigma ** 2) + 1e-9))
        return heatmap.astype(np.float32)

class CnnDataGenerator(Sequence):
    def __init__(self, spatial_data, temporal_data, targets, config):
        self.spatial = spatial_data
        self.temporal = temporal_data
        self.targets = targets
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        self.batch_size = int(self.cfg.get("batch_size", 8))
        self.shuffle = bool(self.cfg.get("shuffle", True))
        self.augment = bool(self.cfg.get("use_augmentation", True)) and (self.targets is not None)
        self.indexes = np.arange(len(self.spatial))

    def __len__(self):
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _augment(self, img, mask):
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        if np.random.rand() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        if np.random.rand() > 0.5:
            angle = np.random.randint(-25, 25)
            img = rotate(img, angle, reshape=False, order=1)
            mask = rotate(mask, angle, reshape=False, order=1)
        return img, mask

    def __getitem__(self, idx):
        inds = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        
        batch_spatial = self.spatial[inds]
        batch_temporal = self.temporal[inds]
        
        inputs = {
            'spatial_input': batch_spatial,
            'temporal_input': batch_temporal
        }

        if self.targets is None:
            return inputs

        batch_y = self.targets[inds]
        
        if self.augment:
            aug_spatial = []
            aug_y = []
            for i in range(len(batch_spatial)):
                s, y = self._augment(batch_spatial[i], batch_y[i])
                aug_spatial.append(s)
                aug_y.append(y)
            inputs['spatial_input'] = np.array(aug_spatial)
            batch_y = np.array(aug_y)

        return inputs, batch_y

# =============================================================================
# SECTION 2: HYBRID ARCHITECTURE FACTORY
# =============================================================================

class UnetFactory:
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        self.grid_size = int(self.cfg.get("grid_size", 64))
        self.input_channels = int(self.cfg.get("input_channels", 3))

    def _conv_block(self, x, filters, name, dropout=0.0):
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f"{name}_c1")(x)
        x = BatchNormalization(name=f"{name}_bn1")(x)
        if dropout > 0: x = Dropout(dropout)(x)
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f"{name}_c2")(x)
        x = BatchNormalization(name=f"{name}_bn2")(x)
        return x

    def build_model(self, temporal_dim: int, params: Dict[str, Any] = None) -> Model:
        """
        Membangun Spatiotemporal U-Net:
        - Input 1 (Spatial): Gambar grid (H, W, C)
        - Input 2 (Temporal): Vektor data cuaca/sensor (1D)
        - Output: Binary Mask (Segmentasi Dampak)
        """
        # Asumsi: Keras Layers, Model, dan Optimizer sudah diimpor di bagian atas modul.
        # Asumsi: CnnMathKernel diimpor atau tersedia di scope modul.
    
        # Setup Parameter
        p = params if params else {}
        base_filters = p.get('base_filters', 16)
        dropout = p.get('dropout', 0.1)
        lr = p.get('learning_rate', 0.001)

        # Menghapus import Keras layers (asumsi sudah diimpor secara global)
        # from keras.layers import (Input, Conv2D, MaxPooling2D, Conv2DTranspose,  
        #                          Dense, Reshape, Concatenate, Activation)
        # from keras.models import Model
        # from keras.optimizers import Adam
    
        # Keras Layers harus diakses secara langsung jika diimpor di atas.
    
        # ---------------------------
        # 1. SPATIAL ENCODER (U-NET DOWNSTREAM)
        # ---------------------------
        # Asumsi self.input_channels tersedia di class scope.
        spatial_in = Input(shape=(self.grid_size, self.grid_size, self.input_channels), name='spatial_input')
    
        # Level 1
        c1 = self._conv_block(spatial_in, base_filters, 'enc1', dropout)
        p1 = MaxPooling2D((2, 2))(c1)
    
        # Level 2
        c2 = self._conv_block(p1, base_filters*2, 'enc2', dropout)
        p2 = MaxPooling2D((2, 2))(c2)
    
        # Level 3
        c3 = self._conv_block(p2, base_filters*4, 'enc3', dropout)
        p3 = MaxPooling2D((2, 2))(c3)
    
        # Bottleneck Spatial
        bn = self._conv_block(p3, base_filters*8, 'bottleneck', dropout)

        # ---------------------------
        # 2. TEMPORAL FEATURE FUSION (INJECTION)
        # ---------------------------
        temporal_in = Input(shape=(temporal_dim,), name='temporal_input')
    
        # Menghitung dimensi bottleneck 
        target_h = self.grid_size // 8
        target_w = self.grid_size // 8
    
        # Transformasi vektor 1D ke dimensi Spatial Bottleneck
        t = Dense(target_h * target_w * base_filters, activation='relu')(temporal_in)
        t = Reshape((target_h, target_w, base_filters))(t)
        t = Conv2D(base_filters*8, (1, 1), activation='relu', padding='same')(t)
    
        # Gabungkan Spatial Features + Temporal Features di bottleneck
        bn_fused = Concatenate(name='temporal_spatial_fusion')([bn, t])
    
        # [FIX]: Ensure consistency with dropout parameter
        bn_final = self._conv_block(bn_fused, base_filters*8, 'fusion_block', dropout)

        # ---------------------------
        # 3. SPATIAL DECODER (UPSAMPLING) - Skip Connections
        # ---------------------------
        # Level 3 Up
        u1 = Conv2DTranspose(base_filters*4, (2, 2), strides=(2, 2), padding='same')(bn_final)
        u1 = Concatenate()([u1, c3]) # Skip connection
        c4 = self._conv_block(u1, base_filters*4, 'dec1', dropout)

        # Level 2 Up
        u2 = Conv2DTranspose(base_filters*2, (2, 2), strides=(2, 2), padding='same')(c4)
        u2 = Concatenate()([u2, c2]) # Skip connection
        c5 = self._conv_block(u2, base_filters*2, 'dec2', dropout)

        # Level 1 Up
        u3 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c5)
        u3 = Concatenate()([u3, c1]) # Skip connection
        c6 = self._conv_block(u3, base_filters, 'dec3', dropout)

        # ---------------------------
        # 4. OUTPUT HEAD
        # ---------------------------
        # Sigmoid untuk output biner (Mask 0 atau 1)
        output = Conv2D(1, (1, 1), activation='sigmoid', name='output_mask')(c6)

        # Compile Model
        model = Model(inputs=[spatial_in, temporal_in], outputs=output)
    
        # [FIX]: Menggunakan loss dan metrics yang disediakan di CnnMathKernel
        model.compile(
            optimizer=Adam(learning_rate=lr), 
            # Asumsi CnnMathKernel dan metodenya dapat diakses di scope ini
            loss=CnnMathKernel.hybrid_dice_bce_loss,
            metrics=[CnnMathKernel.dice_coefficient, 'accuracy']
        )
    
        return model

# =============================================================================
# SECTION 3: TUNER & DUAL VISUALIZER
# =============================================================================

class CnnTuner:
    def __init__(self, factory: UnetFactory, trials=2):
        self.factory = factory
        self.trials = trials
        self.grid = {
            'base_filters': [16, 32],
            'learning_rate': [0.001, 0.0005],
            'dropout': [0.1, 0.2]
        }

    def search(self, gen_train, gen_val, temporal_dim) -> Dict[str, Any]:
        best_loss = float('inf')
        best_params = {'base_filters': 16, 'learning_rate': 0.001, 'dropout': 0.1}
        
        logger.info(f"    [CNN Tuner] Memulai {self.trials} trial...")
        
        for _ in range(self.trials):
            params = {k: random.choice(v) for k, v in self.grid.items()}
            K.clear_session()
            
            try:
                model = self.factory.build_model(temporal_dim, params)
                hist = model.fit(gen_train, validation_data=gen_val, epochs=3, verbose=0)
                val_loss = hist.history['val_loss'][-1]
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            except: continue
        
        logger.info(f"    [CNN Tuner] Terbaik: {best_params} (Loss: {best_loss:.4f})")
        return best_params

class CnnVisualizer:
    """
    Generator visualisasi ganda: Baseline vs AI Prediction.
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_dual_inference_view(self, cid, idx, input_img, pred_mask, timestamp_str):
        """
        Menyimpan gambar perbandingan side-by-side.
        Left: Input Fisika (Statis)
        Right: Prediksi AI (Realtime/Dinami)
        """
        try:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            
            # Left: Physics Baseline (Combined Channels)
            # Input img shape: (H, W, C) -> Sum to (H, W) for heatmap visualization
            baseline_heatmap = np.sum(input_img, axis=-1)
            axs[0].imshow(baseline_heatmap, cmap='hot')
            axs[0].set_title("1. Visual Statis (Physics Baseline)\nInput Data Mentah")
            axs[0].axis('off')
            
            # Right: AI Prediction
            axs[1].imshow(pred_mask.squeeze(), cmap='jet')
            axs[1].set_title("2. Visual Realtime (AI Prediction)\nHybrid Context-Aware")
            axs[1].axis('off')
            
            plt.suptitle(f"Cluster {cid} | Event: {timestamp_str}", fontsize=12)
            plt.tight_layout()
            
            filename = f"dual_view_c{cid}_{timestamp_str.replace(':','-').replace(' ','_')}.png"
            plt.savefig(self.output_dir / filename, dpi=150)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Dual visualisasi gagal: {e}")

    def save_sample_prediction(self, cid, idx, input_img, true_mask, pred_mask):
        """Versi Training dengan Ground Truth."""
        try:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(np.sum(input_img, axis=-1), cmap='hot'); axs[0].set_title("Input")
            if true_mask is not None:
                axs[1].imshow(true_mask.squeeze(), cmap='gray'); axs[1].set_title("Truth")
            else: axs[1].axis('off')
            axs[2].imshow(pred_mask.squeeze(), cmap='jet'); axs[2].set_title("Prediction")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"c{cid}_train_sample_{idx}.png")
            plt.close()
        except: pass

# =============================================================================
# SECTION 4: MAIN CNN ENGINE FACADE
# =============================================================================

class CnnEngine:
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        self.model_dir = Path(self.cfg.get("model_dir", "output/cnn/models"))
        self.visual_dir = Path(self.cfg.get("output_dir", "output/cnn")) / "visuals"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.visual_dir.mkdir(parents=True, exist_ok=True)

        self.spatial_gen = SpatialDataGenerator(self.cfg)
        self.unet_factory = UnetFactory(self.cfg)
        self.tuner = CnnTuner(self.unet_factory)
        self.visualizer = CnnVisualizer(self.visual_dir)
        self.models = {} 

    def _extract_lstm_features(self, df_cluster: pd.DataFrame, lstm_engine, cid: int) -> Optional[Tuple[np.ndarray, pd.Index]]:
        try:
            lstm_cfg = getattr(lstm_engine, 'cfg', None)
            if not lstm_cfg: return None
            seq_len = lstm_cfg.input_seq_len
            
            comps = None
            if hasattr(lstm_engine, 'manager') and hasattr(lstm_engine.manager, 'load_all'):
                comps = lstm_engine.manager.load_all(cid)
            elif hasattr(lstm_engine, 'vault') and hasattr(lstm_engine.vault, 'load_cluster_state'):
                comps = lstm_engine.vault.load_cluster_state(cid)
            
            if not comps or comps[0] is None: return None
            model, scaler = comps
            
            feature_cols = lstm_cfg.features
            if hasattr(scaler, 'feature_names_in_'):
                feature_cols = list(scaler.feature_names_in_)
            
            missing = [c for c in feature_cols if c not in df_cluster.columns]
            if missing: return None
            
            df_sorted = df_cluster.sort_values("Acquired_Date")
            original_indices = df_sorted.index 
            
            data_vals = df_sorted[feature_cols].fillna(0.0).astype(float).values
            if len(data_vals) < seq_len: return None
            
            scaled_vals = scaler.transform(data_vals)
            
            X_enc_list = []
            valid_indices = []
            for i in range(len(scaled_vals) - seq_len + 1):
                X_enc_list.append(scaled_vals[i : i+seq_len])
                valid_indices.append(original_indices[i + seq_len - 1])
                
            X_enc = np.array(X_enc_list)
            
            if hasattr(lstm_engine, 'extract_hidden_states'):
                states = lstm_engine.extract_hidden_states(X_enc, cid)
                if states is not None:
                    return states, pd.Index(valid_indices)
            
            return None
        except Exception as e:
            logger.error(f"Gagal ekstrak LSTM c{cid}: {e}")
            return None

    def train(self, df_train: pd.DataFrame, lstm_engine) -> bool:
        if df_train.empty: return False
        logger.info("=== MEMULAI CNN TRAINING PIPELINE V5.0 (TITAN) ===")
        
        unique_clusters = sorted([c for c in df_train['cluster_id'].unique() if c != -1])
        success_count = 0
        
        for cid in unique_clusters:
            logger.info(f"\n>>> Training CNN Cluster {cid}")
            df_c = df_train[df_train['cluster_id'] == cid]
            
            res = self._extract_lstm_features(df_c, lstm_engine, cid)
            if not res: 
                logger.warning(f"Skip c{cid}: Gagal get LSTM features (mungkin data kurang).")
                continue
            lstm_feats, valid_idx = res
            
            df_aligned = df_c.loc[valid_idx]
            spatial_data = np.array([self.spatial_gen.create_input_mask(r) for _, r in df_aligned.iterrows()])
            target_data = np.array([self.spatial_gen.create_target_mask(r) for _, r in df_aligned.iterrows()])
            
            if len(spatial_data) < 5: continue

            split = int(0.8 * len(spatial_data))
            gen_train = CnnDataGenerator(spatial_data[:split], lstm_feats[:split], target_data[:split], self.cfg)
            gen_val = CnnDataGenerator(spatial_data[split:], lstm_feats[split:], target_data[split:], self.cfg)
            
            best_params = self.tuner.search(gen_train, gen_val, lstm_feats.shape[-1])
            model = self.unet_factory.build_model(lstm_feats.shape[-1], best_params)
            
            ckpt_path = self.model_dir / f"cnn_model_c{cid}.keras"
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint(str(ckpt_path), save_best_only=True, verbose=0)
            ]
            
            model.fit(gen_train, validation_data=gen_val, epochs=int(self.cfg.get('epochs', 20)), callbacks=callbacks, verbose=1)
            self.models[cid] = model
            success_count += 1
            
            try:
                val_sample = gen_val[0]
                pred = model.predict(val_sample[0], verbose=0)
                self.visualizer.save_sample_prediction(cid, 0, val_sample[0]['spatial_input'][0], val_sample[1][0], pred[0])
            except: pass

        return success_count > 0

    def predict(self, df_predict: pd.DataFrame, lstm_engine) -> pd.DataFrame:
        df_out = df_predict.copy()
        df_out['luas_cnn'] = 0.0
        
        unique_clusters = sorted([c for c in df_out['cluster_id'].unique() if c != -1])
        
        for cid in unique_clusters:
            model = self.models.get(cid)
            if not model:
                try:
                    path = self.model_dir / f"cnn_model_c{cid}.keras"
                    if path.exists():
                        model = load_model(path, compile=False)
                        self.models[cid] = model
                except: pass
            if not model: continue
            
            df_c = df_out[df_out['cluster_id'] == cid]
            res = self._extract_lstm_features(df_c, lstm_engine, cid)
            if not res: continue
            lstm_feats, valid_idx = res
            
            if len(lstm_feats) == 0: continue
            
            df_aligned = df_out.loc[valid_idx]
            spatial_data = np.array([self.spatial_gen.create_input_mask(r) for _, r in df_aligned.iterrows()])
            
            # Predict
            gen = CnnDataGenerator(spatial_data, lstm_feats, None, self.cfg)
            preds = model.predict(gen, verbose=0)
            
            # [FEATURE BARU] Save Dual Visualization untuk baris terakhir (Realtime Context)
            try:
                last_idx_in_batch = -1 # Ambil prediksi paling akhir dari batch ini
                last_img = spatial_data[last_idx_in_batch]
                last_pred = preds[last_idx_in_batch]
                
                # Ambil timestamp asli
                timestamp = str(df_aligned.iloc[last_idx_in_batch]['Acquired_Date'])
                
                self.visualizer.save_dual_inference_view(cid, last_idx_in_batch, last_img, last_pred, timestamp)
            except Exception as e:
                pass # Jangan biarkan error visualisasi menghentikan pipeline

            # Hitung Luas
            km2_px = self.spatial_gen.km_per_pixel ** 2
            areas = np.sum(preds > 0.5, axis=(1, 2, 3)) * km2_px
            
            # Map Back
            min_len = min(len(valid_idx), len(areas))
            df_out.loc[valid_idx[:min_len], 'luas_cnn'] = areas[:min_len]
            
        return df_out