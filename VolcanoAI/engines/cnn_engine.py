# VolcanoAI/engines/cnn_engine.py
# -- coding: utf-8 --

import os # operasi filesystem dan environment
import logging # logging untuk pesan runtime
import time # utilitas waktu (sleep, timestamp)
import random # fungsi acak untuk tuner/augmentasi
import shutil # operasi file/dir tingkat tinggi (move, copy, remove)
import pickle # serialisasi objek Python
import functools # utilitas fungsi (wraps, partial)
from typing import Union, Dict, Any, List, Tuple, Optional # hint typing
from pathlib import Path # path objek modern
from datetime import datetime # tanggal/waktu

import numpy as np # array numerik
import pandas as pd # manipulasi DataFrame

import tensorflow as tf # tensorflow core
from keras.models import Model, load_model # model API dan load helper
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Concatenate,
    Conv2DTranspose, Dropout, BatchNormalization,
    Dense, Reshape, Activation, Multiply, Add
) # layer-layer Keras yang dipakai
from keras.optimizers import Adam # optimizer Adam
from keras.utils import Sequence # base class untuk data generator
from keras.callbacks import EarlyStopping, ModelCheckpoint # callbacks training
import keras.backend as K # backend utilities (mean, etc.)
from scipy.ndimage import rotate # fungsi rotasi gambar untuk augmentasi

import matplotlib # plotting backend config
matplotlib.use('Agg') # pakai backend non-interaktif untuk server
import matplotlib.pyplot as plt # plotting

logger = logging.getLogger("VolcanoAI_CNN") # buat logger khusus modul
logger.addHandler(logging.NullHandler()) # default handler kosong agar tidak spam

# =============================================================================
# SECTION 1: SPATIAL DATA GENERATION (TABULAR TO IMAGE)
# =============================================================================

class CnnMathKernel:
    """Class penampung untuk Custom Loss dan Metrics CNN."""
    
    # Dice Coefficient Metric
    @staticmethod
    def dice_coefficient(y_true, y_pred, smooth=1e-6): # metrik dice
    # Implementasi Anda yang sudah ada
        y_true_f = tf.cast(y_true, tf.float32) # cast target ke float32
        y_pred_f = tf.cast(y_pred, tf.float32) # cast prediksi ke float32
        y_true_f = tf.reshape(y_true_f, [-1]) # flatten tensor target
        y_pred_f = tf.reshape(y_pred_f, [-1]) # flatten tensor prediksi
        intersection = tf.reduce_sum(y_true_f * y_pred_f) # irisan sum
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth) # dice formula

    # Hybrid Loss (Dice + Binary Crossentropy)
    @staticmethod
    def hybrid_dice_bce_loss(y_true, y_pred): # gabungan BCE + Dice loss
        # Asumsi: Ini adalah implementasi Anda untuk menggabungkan loss
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred) # BCE per-pixel
        dice_loss = 1 - CnnMathKernel.dice_coefficient(y_true, y_pred) # Dice loss
        # Bobot custom: Misal 50/50
        return 0.5 * bce + 0.5 * dice_loss # gabungkan keduanya

class SpatialDataGenerator: # buat input spatial dari tabel
    def __init__(self, config: Any): # init dengan config dict/obj
        self.cfg = config.__dict__ if not isinstance(config, dict) else config # unify config ke dict
        self.grid_size = int(self.cfg.get("grid_size", 64)) # ukuran grid (pixel)
        self.domain_km = float(self.cfg.get("domain_km", 200.0)) # domain size dalam km
        self.km_per_pixel = float(self.domain_km) / float(self.grid_size) # konversi km->pixel
        self.input_channels = int(self.cfg.get("input_channels", 3)) # jumlah channel input
        self.radius_columns = self.cfg.get("radius_columns", ["R1_final", "R2_final", "R3_final"]) # kolom radius

    # =======================================================
    # METHOD UTAMA: KONTROL PEMBUATAN HEATMAP
    # =======================================================

    def create_input_mask(self, row: pd.Series) -> np.ndarray: # buat input gambar dari baris data
        radii = [] # list radius per channel
        for i in range(self.input_channels): # loop tiap channel
            col = self.radius_columns[i] if i < len(self.radius_columns) else None # ambil kolom jika ada
            val = float(row[col]) if (col and col in row and pd.notna(row[col])) else 0.0 # ambil nilai atau 0
            radii.append(val) # tambahkan
            channels = [self._create_gaussian_heatmap(r) for r in radii] # buat heatmap tiap radius
        while len(channels) < self.input_channels: # jika kurang channel, pad dengan zeros
            channels.append(np.zeros((self.grid_size, self.grid_size), dtype=np.float32)) # pad
        return np.stack(channels, axis=-1) # gabungkan menjadi (H,W,C)

    # [NEW/PINDAHKAN]: Metode untuk membuat target mask
    def create_target_mask(self, row: pd.Series) -> np.ndarray: # buat mask target (ground truth)
        """
        Target mask dibuat dari radius GA (pseudo ground truth).
        """ # docstring
        gs = self.grid_size # grid size lokal


        radius = row.get("ga_distance_km", np.nan) # ambil jarak GA
        if pd.notna(radius) and float(radius) > 0: # jika valid
            return self._create_gaussian_heatmap(float(radius)).reshape(gs, gs, 1) # buat heatmap dan reshape


        return np.zeros((gs, gs, 1), dtype=np.float32) # kalau tidak, zero mask

    # Helper: buat heatmap gaussian dari radius (pixel)
    def _create_gaussian_heatmap(self, radius_km: float) -> np.ndarray: # helper heatmap
        gs = self.grid_size # grid size
        if radius_km is None or radius_km <= 0: # jika radius tidak valid
            return np.zeros((gs, gs), dtype=np.float32) # return zero map
        radius_px = radius_km / (self.km_per_pixel + 1e-12) # konversi km->pixel aman
        center = (gs - 1) / 2.0 # pusat grid
        x, y = np.ogrid[:gs, :gs] # grid koordinat
        sigma = max(radius_px / 3.0, 1.0) # sigma untuk gaussian
        dist_sq = (x - center + 0.5) ** 2 + (y - center + 0.5) ** 2 # jarak kuadrat ke center
        heatmap = np.exp(-dist_sq / (2.0 * (sigma ** 2) + 1e-9)) # gaussian formula
        return heatmap.astype(np.float32) # pastikan tipe float32

    def create_delta_mask(self, r_now: pd.Series, r_next: pd.Series) -> np.ndarray: # mask target untuk event selanjutnya
        """
        Buat mask target yang merepresentasikan area terkait event berikutnya (r_next).
        Dipakai saat training: input = kondisi sekarang (r_now), target = lokasi/luas next (r_next).
        """ # docstring
        gs = self.grid_size # ukuran grid
        if r_next is None: # jika tidak ada next
            return np.zeros((gs, gs, 1), dtype=np.float32) # zero mask


        d = r_next.get("ga_distance_km", np.nan) # ambil jarak next
        # Jika tidak ada jarak/invalid -> zero mask
        if pd.isna(d) or float(d) <= 0:
            return np.zeros((gs, gs, 1), dtype=np.float32)


        return self._create_gaussian_heatmap(float(d)).reshape(gs, gs, 1) # buat dan reshape

class CnnDataGenerator(Sequence):
    def __init__(self, spatial_data, temporal_data, targets, config):
        """
        targets: either None OR tuple (mask_array, vec_array)
        - mask_array shape (N, H, W, 1)
        - vec_array shape (N, 3) -> [sin(angle_rad), cos(angle_rad), distance_km]
        temporal_data: numpy array shaped (N, temporal_dim) (e.g. LSTM hidden states)
        """ # docstring
        self.spatial = np.asarray(spatial_data) # spatial inputs ndarray
        self.temporal = np.asarray(temporal_data) # temporal features ndarray
        self.targets = targets # targets raw
        self.cfg = config.__dict__ if not isinstance(config, dict) else config # unify config
        self.batch_size = int(self.cfg.get("batch_size", 8)) # batch size
        self.shuffle = bool(self.cfg.get("shuffle", True)) # shuffle flag
        self.augment = bool(self.cfg.get("use_augmentation", True)) and (self.targets is not None) # augmentation if targets present
        self.indexes = np.arange(len(self.spatial)) # index array
        # if targets provided as tuple, unpack
        if self.targets is not None and isinstance(self.targets, (list, tuple)) and len(self.targets) == 2: # tuple (mask,vec)
            self.mask_targets = np.asarray(self.targets[0]) # mask targets ndarray
            self.vec_targets = np.asarray(self.targets[1]) # vector targets ndarray
        elif self.targets is None:
            self.mask_targets = None # no mask targets
            self.vec_targets = None # no vec targets
        else:
        # backward compat: single-array target assumed mask
            self.mask_targets = np.asarray(self.targets) # assume targets is mask array
            self.vec_targets = None # no vector

    def __len__(self): # jumlah batch per epoch
        return int(np.ceil(len(self.indexes) / self.batch_size)) # ceil division

    def on_epoch_end(self): #dipanggil riap epoch selesai
        if self.shuffle:
            np.random.shuffle(self.indexes) #acak indeks

    def _augment(self, img, mask, vec): #argumentasi fungsi inernal
        """
        Apply horizontal/vertical flips and rotation.
        If vec provided (sin,cos,dist), update vec accordingly for rotation.
        """
        rot_angle = 0 # default tidak rotasi
        if np.random.rand() > 0.5:
            img = np.fliplr(img) # flip horizontal image
            mask = np.fliplr(mask) # flip mask
            if vec is not None:
            # flipping horizontally: angle -> 180 - angle
                angle_rad = np.arctan2(vec[0], vec[1]) # vec stored as sin,cos -> arctan2(y,x) but here param order chosen
                angle_deg = (np.degrees(angle_rad) + 360) % 360 # normalisasi 0-360
                angle_deg = (180.0 - angle_deg) % 360 # transform horizontal flip
                rad = np.radians(angle_deg) # balik ke rad
                vec[0] = np.sin(rad); vec[1] = np.cos(rad) # update sin,cos
        if np.random.rand() > 0.5:
            img = np.flipud(img) # flip vertical
            mask = np.flipud(mask) # flip vertical mask
            if vec is not None:
            # flipping vertically: angle -> -angle
                angle_rad = np.arctan2(vec[0], vec[1]) # current angle rad
                angle_deg = (np.degrees(angle_rad) + 360) % 360 # to deg
                angle_deg = (-angle_deg) % 360 # negate vertical
                rad = np.radians(angle_deg) # back to rad
                vec[0] = np.sin(rad); vec[1] = np.cos(rad) # update sin,cos
        if np.random.rand() > 0.5:
            rot_angle = np.random.randint(-25, 25) # random rotation angle
            img = rotate(img, rot_angle, reshape=False, order=1) # rotate image
            mask = rotate(mask, rot_angle, reshape=False, order=1) # rotate mask
            if vec is not None:
            # rotate vector angle by rot_angle
                angle_rad = np.arctan2(vec[0], vec[1]) # current angle
                angle_deg = (np.degrees(angle_rad) + 360) % 360 # to deg
                angle_deg = (angle_deg + rot_angle) % 360 # add rotation
                rad = np.radians(angle_deg) # back to rad
                vec[0] = np.sin(rad); vec[1] = np.cos(rad) # update sin,cos
        return img, mask, vec # kembalikan hasil augmentasi

    def __getitem__(self, idx): # ambil satu batch
        inds = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size] # indeks batch
        batch_spatial = self.spatial[inds] # spatial batch
        batch_temporal = self.temporal[inds] if len(self.temporal) > 0 else np.zeros((len(inds), 1)) # temporal batch atau zeros
        inputs = {
        'spatial_input': batch_spatial, # key spatial
        'temporal_input': batch_temporal # key temporal
        }


        if self.mask_targets is None: # jika tidak ada target mask -> inference-mode generator
            return inputs # hanya input dict


        batch_y_mask = self.mask_targets[inds] # mask target batch
        batch_y_vec = self.vec_targets[inds] if self.vec_targets is not None else None # vec target batch
        if self.augment: # jika augment aktif
            aug_spatial = [] # list penampung
            aug_mask = [] # list mask augment
            aug_vec = [] # list vector augment
            for i in range(len(batch_spatial)):
                s = batch_spatial[i].copy() # salin spatial
                m = batch_y_mask[i].copy() # salin mask
                v = batch_y_vec[i].copy() if batch_y_vec is not None else None # salin vec jika ada
                s2, m2, v2 = self._augment(s, m, v) # augment
                aug_spatial.append(s2) # append hasil
                aug_mask.append(m2) # append mask
                if v2 is not None:
                    aug_vec.append(v2) # append vec jika ada
            inputs['spatial_input'] = np.array(aug_spatial) # set input spatial ke augmented
            batch_y_mask = np.array(aug_mask) # update batch_y_mask
            if batch_y_vec is not None:
                batch_y_vec = np.array(aug_vec) # update vector


        # return (inputs, [mask, vec]) to match multi-output model
        if batch_y_vec is not None:
            return inputs, [batch_y_mask, batch_y_vec] # multi-output format
        else:
            return inputs, batch_y_mask # single-output format

# =============================================================================
# SECTION 2: HYBRID ARCHITECTURE FACTORY
# =============================================================================

class UnetFactory: # factory untuk membangun model UNet hybrid
    def __init__(self, config: Any): # init config
        self.cfg = config.__dict__ if not isinstance(config, dict) else config # unify config
        self.grid_size = int(self.cfg.get("grid_size", 64)) # grid size
        self.input_channels = int(self.cfg.get("input_channels", 3)) # input channels


    def _conv_block(self, x, filters, name, dropout=0.0): # block konvolusi berulang
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f"{name}_c1")(x) # conv1
        x = BatchNormalization(name=f"{name}_bn1")(x) # batchnorm
        if dropout > 0: x = Dropout(dropout)(x) # optional dropout
        x = Conv2D(filters, (3, 3), activation='relu', padding='same', name=f"{name}_c2")(x) # conv2
        x = BatchNormalization(name=f"{name}_bn2")(x) # batchnorm
        return x # return fitur

    def build_model(self, temporal_dim: int, params: Dict[str, Any] = None) -> Model:
        """
        Build multi-head UNet:
         - output[0] = segmentation mask (H,W,1) with sigmoid
         - output[1] = vector head (sin(angle), cos(angle), distance_km) as regression
        Loss = w_seg * hybrid_dice_bce_loss + w_vec * mse(vector)
        """
        p = params if params else {} # params dict
        base_filters = p.get('base_filters', 16) # base filter count
        dropout = p.get('dropout', 0.1) # dropout
        lr = p.get('learning_rate', 0.001) # learning rate
        w_seg = p.get('weight_seg', 1.0) # weight segmentation loss
        w_vec = p.get('weight_vec', 1.0) # weight vector loss

        spatial_in = Input(shape=(self.grid_size, self.grid_size, self.input_channels), name='spatial_input') # spatial input
        # encoder...
        c1 = self._conv_block(spatial_in, base_filters, 'enc1', dropout) # encoder level1
        p1 = MaxPooling2D((2, 2))(c1) # pool1
        c2 = self._conv_block(p1, base_filters*2, 'enc2', dropout) # encoder level2
        p2 = MaxPooling2D((2, 2))(c2) # pool2
        c3 = self._conv_block(p2, base_filters*4, 'enc3', dropout) # encoder level3
        p3 = MaxPooling2D((2, 2))(c3) # pool3
        bn = self._conv_block(p3, base_filters*8, 'bottleneck', dropout) # bottleneck

        temporal_in = Input(shape=(temporal_dim,), name='temporal_input') # temporal features input
        target_h = self.grid_size // 8 # target height after downsampling
        target_w = self.grid_size // 8 # target width after downsampling
        t = Dense(target_h * target_w * base_filters, activation='relu')(temporal_in) # dense expand
        t = Reshape((target_h, target_w, base_filters))(t) # reshape ke bentuk spatial kecil
        t = Conv2D(base_filters*8, (1, 1), activation='relu', padding='same')(t) # conv1x1 untuk matching channel

        bn_fused = Concatenate(name='temporal_spatial_fusion')([bn, t]) # gabungkan bottleneck + temporal
        bn_final = self._conv_block(bn_fused, base_filters*8, 'fusion_block', dropout) # conv setelah fusion

        # decoder
        u1 = Conv2DTranspose(base_filters*4, (2, 2), strides=(2, 2), padding='same')(bn_final) # upsample1
        u1 = Concatenate()([u1, c3]) # skip connection
        c4 = self._conv_block(u1, base_filters*4, 'dec1', dropout) # decoder block1


        u2 = Conv2DTranspose(base_filters*2, (2, 2), strides=(2, 2), padding='same')(c4) # upsample2
        u2 = Concatenate()([u2, c2]) # skip
        c5 = self._conv_block(u2, base_filters*2, 'dec2', dropout) # decoder block2


        u3 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c5) # upsample3
        u3 = Concatenate()([u3, c1]) # skip
        c6 = self._conv_block(u3, base_filters, 'dec3', dropout) # decoder block3

        # segmentation head
        mask_out = Conv2D(1, (1, 1), activation='sigmoid', name='output_mask')(c6)

        # vector head: global pooling -> dense -> [sin, cos, distance]
        gp = K.mean(bn_final, axis=[1,2])  # global average pooling of bottleneck fused (avoid extra import)
        v = Dense(128, activation='relu')(gp)
        v = Dropout(dropout)(v)
        # outputs: sin(angle), cos(angle), distance_km
        vec_out = Dense(3, activation='linear', name='output_vector')(v)

        model = Model(inputs=[spatial_in, temporal_in], outputs=[mask_out, vec_out])

        # custom multi-loss function
        def multi_loss(y_true, y_pred):
            # y_true and y_pred for segmentation handled by Keras; here we create wrapper below
            return None

        # compile: segmentation uses hybrid_dice_bce_loss, vector uses mse
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss={'output_mask': CnnMathKernel.hybrid_dice_bce_loss, 'output_vector': 'mse'},
            loss_weights={'output_mask': w_seg, 'output_vector': w_vec},
            metrics={'output_mask': CnnMathKernel.dice_coefficient}
        )
        return model

# =============================================================================
# SECTION 3: TUNER & DUAL VISUALIZER
# =============================================================================

class CnnTuner: # simple hyperparameter tuner
    def __init__(self, factory: UnetFactory, trials=2):
        self.factory = factory # factory reference
        self.trials = trials # jumlah trial
        self.grid = {
            'base_filters': [16, 32], # opsi base filters
            'learning_rate': [0.001, 0.0005], # opsi lr
            'dropout': [0.1, 0.2] # opsi dropout
    }

    def search(self, gen_train, gen_val, temporal_dim) -> Dict[str, Any]: # search terbaik dari grid random
        best_loss = float('inf') # init best loss
        best_params = {'base_filters': 16, 'learning_rate': 0.001, 'dropout': 0.1} # default best
        
        logger.info(f" [CNN Tuner] Memulai {self.trials} trial...") # log start
        
        for _ in range(self.trials): # loop trial
            params = {k: random.choice(v) for k, v in self.grid.items()} # random pick
            K.clear_session() # clear session to avoid memory grow
            
            try:
                model = self.factory.build_model(temporal_dim, params) # build model
                hist = model.fit(gen_train, validation_data=gen_val, epochs=3, verbose=0) # quick fit
                val_loss = hist.history['val_loss'][-1] # ambil val loss terakhir
        
                if val_loss < best_loss: # update best
                    best_loss = val_loss
                    best_params = params
            except: continue # jika error skip trial
        
        logger.info(f"    [CNN Tuner] Terbaik: {best_params} (Loss: {best_loss:.4f})")
        return best_params

class CnnVisualizer:
    """
    Generator visualisasi ganda: Baseline vs AI Prediction.
    """
    def __init__(self, output_dir): # init output dir
        self.output_dir = Path(output_dir) # Path object
        self.output_dir.mkdir(parents=True, exist_ok=True) # pastikan ada

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
            baseline_heatmap = np.sum(input_img, axis=-1) # gabungkan channel menjadi heatmap
            axs[0].imshow(baseline_heatmap, cmap='hot') # tampilkan heatmap
            axs[0].set_title("1. Visual Statis (Physics Baseline)\nInput Data Mentah") # title
            axs[0].axis('off') # sembunyikan axis
            
            # Right: AI Prediction
            axs[1].imshow(pred_mask.squeeze(), cmap='jet') # tampilkan prediksi mask
            axs[1].set_title("2. Visual Realtime (AI Prediction)\nHybrid Context-Aware") # title
            axs[1].axis('off') # sembunyikan axis
            
            plt.suptitle(f"Cluster {cid} | Event: {timestamp_str}", fontsize=12) # suptitle
            plt.tight_layout() # rapiin layout
            
            filename = f"dual_view_c{cid}_{timestamp_str.replace(':','-').replace(' ','_')}.png" # buat nama file aman
            plt.savefig(self.output_dir / filename, dpi=150) # simpan
            plt.close() # tutup figure
            
        except Exception as e:
            logger.warning(f"Dual visualisasi gagal: {e}")

    def save_sample_prediction(self, cid, idx, input_img, true_mask, pred_mask): # simpan contoh prediksi saat training
        """Versi Training dengan Ground Truth.""" # docstring singkat
        try:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4)) # buat 1x3 subplot
            axs[0].imshow(np.sum(input_img, axis=-1), cmap='hot'); axs[0].set_title("Input") # input
            if true_mask is not None:
                axs[1].imshow(true_mask.squeeze(), cmap='gray'); axs[1].set_title("Truth") # true mask
            else: axs[1].axis('off') # hide if no truth
            axs[2].imshow(pred_mask.squeeze(), cmap='jet'); axs[2].set_title("Prediction") # prediksi
            plt.tight_layout() # rapikan
            plt.savefig(self.output_dir / f"c{cid}_train_sample_{idx}.png") # simpan
            plt.close() # tutup
        except: pass # jangan crash jika gagal

# =============================================================================
# SECTION 4: MAIN CNN ENGINE FACADE
# =============================================================================

class CnnEngine:
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config

        # =====================
        # DEFAULT CONFIG SAFETY 
        # =====================
        self.cfg.setdefault('grid_size', 64) # default grid_size jika belum ada
        self.cfg.setdefault('input_channels', 3) # default channels
        self.cfg.setdefault('batch_size', 8) # default batch size
        self.cfg.setdefault('use_augmentation', True) # default augment
        self.cfg.setdefault('epochs', 20) # default epochs
        self.cfg.setdefault('domain_km', 200.0) # default domain km

        self.model_dir = Path(self.cfg.get("model_dir", "output/cnn/models")) # lokasi simpan model
        self.visual_dir = Path(self.cfg.get("output_dir", "output/cnn")) / "visuals" # lokasi visual
        self.model_dir.mkdir(parents=True, exist_ok=True) # pastikan ada folder model
        self.visual_dir.mkdir(parents=True, exist_ok=True) # pastikan ada folder visuals

        self.spatial_gen = SpatialDataGenerator(self.cfg) # spatial generator instance
        self.unet_factory = UnetFactory(self.cfg) # model factory
        self.tuner = CnnTuner(self.unet_factory) # tuner instance
        self.visualizer = CnnVisualizer(self.visual_dir) # visualizer instance
        self.models = {} # dict untuk menyimpan model per cluster

    def _extract_lstm_features(self, df_cluster: pd.DataFrame, lstm_engine, cid: int) -> Optional[Tuple[np.ndarray, pd.Index]]: # ekstrak fitur LSTM per cluster
        try:
            lstm_cfg = getattr(lstm_engine, 'cfg', None) # ambil config LSTM
            if not lstm_cfg: return None # jika ga ada, return None
            seq_len = lstm_cfg.input_seq_len # panjang sequence yang dibutuhkan
        
            comps = None # placeholder komponen
            if hasattr(lstm_engine, 'manager') and hasattr(lstm_engine.manager, 'load_all'):
                comps = lstm_engine.manager.load_all(cid) # load manager based
            elif hasattr(lstm_engine, 'vault') and hasattr(lstm_engine.vault, 'load_cluster_state'):
                comps = lstm_engine.vault.load_cluster_state(cid) # load vault based
        
                if not comps or comps[0] is None: return None # jika kosong return
                model, scaler = comps # unpack
        
                feature_cols = lstm_cfg.features # columns fitur expect
                if hasattr(scaler, 'feature_names_in_'):
                    feature_cols = list(scaler.feature_names_in_) # ambil nama fitur dari scaler jika ada
        
                missing = [c for c in feature_cols if c not in df_cluster.columns] # cek kolom hilang
                return None # jika ada missing, return None
        
                df_sorted = df_cluster.sort_values("Acquired_Date") # urutkan berdasarkan tanggal akuisisi
                original_indices = df_sorted.index # simpan indeks asli
        
                data_vals = df_sorted[feature_cols].fillna(0.0).astype(float).values # ambil nilai fitur sebagai numpy
                if len(data_vals) < seq_len: return None # kalau kurang sequence, skip
        
                scaled_vals = scaler.transform(data_vals) # transform dengan scaler
        
                X_enc_list = [] # kumpulan sequence
                valid_indices = [] # indeks yang valid
                for i in range(len(scaled_vals) - seq_len + 1): # buat sliding windows
                    X_enc_list.append(scaled_vals[i : i+seq_len]) # append window
                    valid_indices.append(original_indices[i + seq_len - 1]) # indeks target
                X_enc = np.array(X_enc_list) # hasil array
        
                if hasattr(lstm_engine, 'extract_hidden_states'):
                    states = lstm_engine.extract_hidden_states(X_enc, cid) # coba ekstrak hidden states
                    if states is not None:
                        return states, pd.Index(valid_indices) # return states dan indeks
                    
                return None # default None jika tidak bisa
        except Exception as e:
            logger.error(f"Gagal ekstrak LSTM c{cid}: {e}") # log error
            return None # return None on error
    
    def train(self, df_train: pd.DataFrame, lstm_engine) -> bool: # main training loop CNN
        if df_train.empty: return False # tidak ada data -> false
        logger.info("=== MEMULAI CNN TRAINING PIPELINE V5.0 (TITAN) ===") # log start
        
        unique_clusters = sorted([c for c in df_train['cluster_id'].unique() if c != -1]) # cluster unik kecuali -1
        success_count = 0 # counter sukses
        
        for cid in unique_clusters: # loop tiap cluster
            logger.info(f"\n>>> Training CNN Cluster {cid}") # log cluster
            df_c = df_train[df_train['cluster_id'] == cid] # pilih data cluster
        
            res = self._extract_lstm_features(df_c, lstm_engine, cid) # ekstrak fitur LSTM
            if not res:
                logger.warning(f"Skip c{cid}: Gagal get LSTM features (mungkin data kurang).") # warn jika gagal
                continue # lanjut cluster berikutnya
            lstm_feats, valid_idx = res # unpack
        
            df_aligned = df_c.loc[valid_idx].sort_values("Acquired_Date").reset_index(drop=True) # align dan reset idx


            spatial_list, mask_list, vec_list = [], [], [] # penampung data

            for i in range(len(df_aligned) - 1): # ?? sampai N-1
                r_now = df_aligned.iloc[i] # current row
                r_next = df_aligned.iloc[i + 1] # next row

                # spatial input (current state)
                spatial_list.append(self.spatial_gen.create_input_mask(r_now)) # buat input mask
                mask_list.append(
                    self.spatial_gen.create_delta_mask(r_now, r_next)
                ) # buat target mask next

                # vector target = GA NEXT EVENT
                angle = r_next.get('ga_angle_deg', np.nan) # ambil angle
                dist = r_next.get('ga_distance_km', np.nan) # ambil distance

                if pd.isna(angle) or pd.isna(dist):
                    vec_list.append([0.0, 1.0, 0.0]) # fallback vector jika missing
                else:
                    rad = np.radians(float(angle)) # convert deg->rad
                    vec_list.append([np.sin(rad), np.cos(rad), float(dist)]) # simpan sin,cos,dist
                
            lstm_feats = lstm_feats[:-1] # align length dengan spatial/targets (buang last)

            spatial_data = np.array(spatial_list) # spatial ndarray
            target_masks = np.array(mask_list) # masks ndarray
            target_vecs = np.array(vec_list) # vec ndarray

            # optionally filter out rows where mask is all zeros and also vector is zero?
            # but keep as-is for now

            if len(spatial_data) < 5:  # Validasi jumlah data minimum untuk training CNN
                logger.warning(f"Skip c{cid}: too few samples after alignment.")  # Logging data tidak mencukupi
                continue  # Lewati cluster ini

            split = int(0.8 * len(spatial_data))  # Menentukan titik split 80% train, 20% validasi

            gen_train = CnnDataGenerator(  # Generator data training CNN
                spatial_data[:split],  # Data spasial untuk training
                lstm_feats[:split],  # Fitur temporal LSTM untuk training
                (target_masks[:split], target_vecs[:split]),  # Target mask & vektor training
                self.cfg  # Konfigurasi CNN
            )

            gen_val = CnnDataGenerator(  # Generator data validasi CNN
                spatial_data[split:],  # Data spasial untuk validasi
                lstm_feats[split:],  # Fitur temporal LSTM untuk validasi
                (target_masks[split:], target_vecs[split:]),  # Target mask & vektor validasi
                self.cfg  # Konfigurasi CNN
            )
            
            # =========================
            # BUILD + TRAIN CNN
            # =========================
            temporal_dim = lstm_feats.shape[1]  # Mengambil dimensi temporal (jumlah timestep) dari fitur LSTM

            params = self.tuner.search(gen_train, gen_val, temporal_dim)  # Mencari hyperparameter terbaik CNN
            model = self.unet_factory.build_model(temporal_dim, params)  # Membangun model CNN (U-Net) dengan parameter terbaik

            callbacks = [  # Daftar callback selama proses training
                EarlyStopping(patience=5, restore_best_weights=True),  # Hentikan training jika tidak ada perbaikan validasi
                ModelCheckpoint(
                    filepath=self.model_dir / f"cnn_model_c{cid}.keras",  # Lokasi penyimpanan model per cluster
                    save_best_only=True  # Simpan hanya model dengan performa terbaik
                )
            ]

            model.fit(  # Proses training model CNN
                gen_train,  # Data generator training
                validation_data=gen_val,  # Data generator validasi
                epochs=self.cfg['epochs'],  # Jumlah epoch dari konfigurasi
                callbacks=callbacks,  # Callback training (early stop & checkpoint)
                verbose=1  # Menampilkan progress training
            )

            # =========================
            # REGISTER MODEL
            # =========================
            self.models[cid] = model
            success_count += 1
    

        def train_from_scratch(  # Fungsi untuk melatih ulang CNN dari awal
            self,  # Referensi instance class
            df_train: pd.DataFrame,  # DataFrame dataset training
            lstm_engine,  # Engine LSTM pendukung fitur temporal
            epochs: Optional[int] = None  # Jumlah epoch opsional (override konfigurasi)
        ) -> bool:  # Mengembalikan status keberhasilan training
            """
            Retrain CNN dari dataset training baru (misal 70% dari 15 hari terakhir).  # Deskripsi fungsi
            """
            if df_train.empty:  # Cek apakah dataset training kosong
                logger.warning("[CNN] train_from_scratch: df_train kosong")  # Logging peringatan
                return False  # Hentikan proses training

            prev_epochs = self.cfg.get('epochs', 20)  # Simpan jumlah epoch lama dari konfigurasi

            if epochs is not None:  # Jika epoch baru diberikan
                self.cfg['epochs'] = epochs  # Override konfigurasi epoch sementara

            try:
                logger.info("[CNN] Retraining CNN from scratch...")  # Logging awal proses retraining
                self.train(df_train, lstm_engine)  # Panggil fungsi training utama CNN
                return True  # Training berhasil
            finally:
                self.cfg['epochs'] = prev_epochs  # Kembalikan konfigurasi epoch ke nilai semula


    def predict(self, df_predict: pd.DataFrame, lstm_engine) -> pd.DataFrame:  # Fungsi utama prediksi CNN per cluster
        df_out = df_predict.copy()  # Menyalin DataFrame input agar data asli tidak berubah
        df_out['luas_cnn'] = 0.0  # Inisialisasi kolom luas area hasil CNN
        df_out['cnn_confidence'] = 0.0  # Inisialisasi kolom confidence CNN

        unique_clusters = sorted([c for c in df_out['cluster_id'].unique() if c != -1])  # Ambil cluster valid (bukan noise)

        for cid in unique_clusters:  # Iterasi tiap cluster gempa
            # --- try load model if not in memory
            model = self.models.get(cid)  # Ambil model CNN dari cache memori
            if model is None:  # Jika model belum pernah dimuat
                try:
                    path = self.model_dir / f"cnn_model_c{cid}.keras"  # Path file model CNN cluster
                    if path.exists():  # Jika file model tersedia di disk
                        model = load_model(path, compile=False)  # Load model tanpa kompilasi ulang
                        self.models[cid] = model  # Simpan model ke cache
                        logger.info(f"[CNN] Loaded model for cluster {cid} from disk.")  # Logging sukses load model
                    else:
                        logger.warning(f"[CNN] No model file for cluster {cid}, will fallback.")  # Logging model tidak tersedia
                except Exception as e:
                    logger.warning(f"[CNN] Failed to load model for c{cid}: {e}")  # Logging error saat load model

            # get rows for this cluster
            df_c = df_out[df_out['cluster_id'] == cid]

            # try extract LSTM features (may return None)
            res = self._extract_lstm_features(df_c, lstm_engine, cid)
            if not res:
                # fallback: use AreaTerdampak_km2 and low confidence
                logger.warning(f"[CNN] Skip c{cid}: no LSTM features — applying fallback area.")
                fallback_area = df_c.get('AreaTerdampak_km2', pd.Series(0.0, index=df_c.index)).values
                df_out.loc[df_c.index, 'luas_cnn'] = fallback_area
                df_out.loc[df_c.index, 'cnn_confidence'] = 0.25
                continue

            lstm_feats, valid_idx = res
            if lstm_feats is None or len(lstm_feats) == 0 or len(valid_idx) == 0:
                logger.warning(f"[CNN] Skip c{cid}: empty LSTM features/indices — applying fallback.")
                fallback_area = df_c.get('AreaTerdampak_km2', pd.Series(0.0, index=df_c.index)).values
                df_out.loc[df_c.index, 'luas_cnn'] = fallback_area
                df_out.loc[df_c.index, 'cnn_confidence'] = 0.25
                continue

            # align df rows (valid_idx are original df indices)
            df_aligned = df_out.loc[valid_idx]
            # build spatial inputs
            spatial_list = [self.spatial_gen.create_input_mask(row) for _, row in df_aligned.iterrows()]
            if len(spatial_list) == 0:
                logger.warning(f"[CNN] Skip c{cid}: no spatial data for aligned rows — fallback.")
                fallback_area = df_aligned.get('AreaTerdampak_km2', pd.Series(0.0, index=df_aligned.index)).values
                df_out.loc[df_aligned.index, 'luas_cnn'] = fallback_area
                df_out.loc[df_aligned.index, 'cnn_confidence'] = 0.25
                continue

            spatial_data = np.array(spatial_list)

            # GUARD: if model missing or spatial_data all zeros -> fallback
            if model is None or np.all(spatial_data == 0):
                logger.warning(f"[CNN] Cluster {cid} invalid (no model or empty spatial) → fallback area.")
                fallback_area = df_aligned.get('AreaTerdampak_km2', pd.Series(0.0, index=df_aligned.index)).values
                df_out.loc[df_aligned.index, 'luas_cnn'] = fallback_area
                df_out.loc[df_aligned.index, 'cnn_confidence'] = 0.25
                continue

            # create generator and predict
            gen = CnnDataGenerator(spatial_data, lstm_feats, None, self.cfg)
            try:
                preds_out = model.predict(gen, verbose=0)
            except Exception as e:
                logger.warning(f"[CNN] Prediction failed for c{cid}: {e} — fallback applied.")
                fallback_area = df_aligned.get('AreaTerdampak_km2', pd.Series(0.0, index=df_aligned.index)).values
                df_out.loc[df_aligned.index, 'luas_cnn'] = fallback_area
                df_out.loc[df_aligned.index, 'cnn_confidence'] = 0.25
                continue

            # unpack outputs
            if isinstance(preds_out, (list, tuple)) and len(preds_out) >= 2:
                preds_mask = preds_out[0]
                preds_vec = preds_out[1]
            else:
                preds_mask = preds_out
                preds_vec = np.zeros((preds_mask.shape[0], 3), dtype=float)

            # compute confidence (mean activation)
            try:
                confidence = np.mean(preds_mask, axis=(1, 2, 3))
            except Exception:
                # if mask dims unexpected
                confidence = np.mean(preds_mask.reshape(preds_mask.shape[0], -1), axis=1)

            # compute areas from thresholded mask
            pixel_area_km2 = float(self.spatial_gen.km_per_pixel) ** 2
            thr = float(self.cfg.get("cnn_area_threshold", 0.3))

            # ensure preds_mask is numeric float ndarray
            preds_mask = preds_mask.astype(float)
            binary_mask = preds_mask >= thr
            pixel_count = np.sum(binary_mask, axis=tuple(range(1, preds_mask.ndim)))  # sum over H,W,(C)
            areas = pixel_count * pixel_area_km2

            # soft fallback: if all pixel_count == 0 then use soft sum
            if np.all(pixel_count == 0):
                areas = np.sum(preds_mask, axis=tuple(range(1, preds_mask.ndim))) * pixel_area_km2

            # HARD FALLBACK: if still all zero (model produced near-zero), use engineered AreaTerdampak_km2
            min_len = min(len(areas), len(valid_idx))  # Menentukan panjang minimum agar aman saat indexing
            if min_len == 0:  # Jika tidak ada area prediksi atau indeks valid
                logger.warning(f"[CNN] No predicted areas for c{cid} — applying fallback areas.")  # Logging kondisi fallback
                fallback_area = df_aligned.get('AreaTerdampak_km2', pd.Series(0.0, index=df_aligned.index)).values  # Ambil area hasil rekayasa fitur
                df_out.loc[df_aligned.index, 'luas_cnn'] = fallback_area  # Mengisi luas CNN dengan nilai fallback
                df_out.loc[df_aligned.index, 'cnn_confidence'] = 0.25  # Memberi confidence rendah (fallback)
                continue  # Lanjut ke cluster berikutnya

            if np.all(areas[:min_len] == 0):  # Jika semua prediksi area CNN bernilai nol
                logger.info(f"[CNN] Areas zero for c{cid} → using AreaTerdampak_km2 fallback.")  # Logging fallback area
                fallback_area = df_aligned.get('AreaTerdampak_km2', pd.Series(0.0, index=df_aligned.index)).values[:min_len]  # Ambil area fallback sesuai panjang
                areas_to_write = fallback_area  # Gunakan area fallback sebagai output
                confidence_to_write = np.full((min_len,), 0.25, dtype=float)  # Confidence rendah karena fallback
            else:
                areas_to_write = areas[:min_len]  # Gunakan hasil prediksi area CNN
                confidence_to_write = confidence[:min_len]  # Gunakan confidence asli CNN

            # convert vector outputs
            sin_vals = preds_vec[:min_len, 0]  # Komponen sinus arah dari output CNN
            cos_vals = preds_vec[:min_len, 1]  # Komponen cosinus arah dari output CNN
            dist_vals = preds_vec[:min_len, 2]  # Jarak prediksi (km) dari output CNN
            angles_rad = np.arctan2(sin_vals, cos_vals)  # Konversi vektor (sin, cos) ke sudut radian
            angles_deg = (np.degrees(angles_rad) + 360) % 360  # Normalisasi sudut ke rentang 0–360 derajat

            # map back to original df indices (valid_idx contains original indices)
            idxs_to_write = list(valid_idx[:min_len])  # Ambil indeks asli DataFrame untuk penulisan hasil
            df_out.loc[idxs_to_write, 'luas_cnn'] = areas_to_write  # Menulis luas area CNN ke DataFrame output
            df_out.loc[idxs_to_write, 'cnn_confidence'] = confidence_to_write  # Menulis confidence CNN
            df_out.loc[idxs_to_write, 'cnn_angle_deg'] = angles_deg  # Menulis sudut arah prediksi CNN
            df_out.loc[idxs_to_write, 'cnn_distance_km'] = dist_vals  # Menulis jarak prediksi CNN

        return df_out  # Mengembalikan DataFrame akhir berisi hasil CNN

 

    # =========================================================
    # CNN → EXPORT SIMPLE MAP (LAT/LON POINT)
    # =========================================================
    def export_cnn_prediction_map(self, json_path):  # Fungsi untuk membuat peta prediksi CNN dari file JSON
        import json, folium  # Import modul JSON dan Folium untuk peta
        from pathlib import Path  # Import Path untuk manajemen path file

        with open(json_path) as f:  # Membuka file JSON hasil prediksi CNN
            j = json.load(f)  # Membaca dan mengonversi JSON menjadi dictionary

        ne = j.get("next_event", {})  # Mengambil data event berikutnya dari JSON
        lat, lon = ne.get("lat"), ne.get("lon")  # Mengambil koordinat latitude dan longitude

        if lat is None or lon is None:  # Validasi jika koordinat tidak tersedia
            logger.warning("[CNN MAP] lat/lon kosong")  # Logging peringatan koordinat kosong
            return None  # Menghentikan proses dan mengembalikan None

        m = folium.Map(location=[lat, lon], zoom_start=9)  # Membuat peta Folium berpusat di lokasi prediksi

        folium.Marker(  # Membuat marker pada lokasi prediksi CNN
            [lat, lon],  # Koordinat marker
            popup="CNN Prediction",  # Teks popup marker
            icon=folium.Icon(color="purple", icon="info-sign")  # Ikon marker berwarna ungu
        ).add_to(m)  # Menambahkan marker ke peta

        out = Path(json_path).parent  # Mengambil direktori tempat file JSON berada
        map_path = out / "cnn_prediction_map.html"  # Menentukan path file HTML output peta
        m.save(map_path)  # Menyimpan peta ke file HTML

        logger.info(f"[CNN MAP] Saved → {map_path}")  # Logging bahwa peta berhasil disimpan
        return map_path  # Mengembalikan path file peta




    def evaluate_predictions(  # Fungsi untuk mengevaluasi kebenaran prediksi CNN
            self,  # Referensi instance class
            df_out: pd.DataFrame,  # DataFrame hasil prediksi CNN dan data GA
            thresholds: Dict[str, float]  # Batas toleransi error (jarak & sudut)
        ) -> pd.DataFrame:  # Mengembalikan DataFrame hasil evaluasi
            """
            Menilai apakah prediksi CNN benar atau tidak.  # Penjelasan fungsi
            thresholds contoh:  # Contoh parameter threshold
            {
                'dist_km': 10.0,  # Maksimum error jarak (km)
                'angle_deg': 30.0  # Maksimum error sudut (derajat)
            }
            """
            df = df_out.copy()  # Menyalin DataFrame agar data asli tidak berubah
            df['cnn_correct'] = False  # Inisialisasi status kebenaran prediksi

            def angle_diff(a, b):  # Fungsi bantu menghitung selisih sudut melingkar
                return abs((a - b + 180) % 360 - 180)  # Selisih sudut minimum (0–180°)

            if {'cnn_distance_km', 'ga_distance_km'}.issubset(df.columns):  # Cek kolom jarak tersedia
                df['dist_err'] = (df['cnn_distance_km'] - df['ga_distance_km']).abs()  # Hitung error jarak absolut

            if {'cnn_angle_deg', 'ga_angle_deg'}.issubset(df.columns):  # Cek kolom sudut tersedia
                df['angle_err'] = df.apply(  # Hitung error sudut per baris
                    lambda r: angle_diff(r['cnn_angle_deg'], r['ga_angle_deg']),  # Selisih sudut CNN vs GA
                    axis=1  # Operasi dilakukan per baris
                )

            cond = pd.Series(True, index=df.index)  # Inisialisasi kondisi benar untuk semua data

            if 'dist_km' in thresholds and 'dist_err' in df.columns:  # Jika threshold jarak tersedia
                cond &= df['dist_err'] <= thresholds['dist_km']  # Validasi berdasarkan error jarak

            if 'angle_deg' in thresholds and 'angle_err' in df.columns:  # Jika threshold sudut tersedia
                cond &= df['angle_err'] <= thresholds['angle_deg']  # Validasi berdasarkan error sudut

            df['cnn_correct'] = cond  # Menentukan apakah prediksi CNN dianggap benar
            return df  # Mengembalikan DataFrame hasil evaluasi
