# VolcanoAI/engines/cnn_engine.py
# -- coding: utf-8 --

"""
VOLCANO AI - CNN ENGINE (TITANIUM EDITION - LITE)
=================================================
Modul ini mengimplementasikan "Simple CNN" (Neural Network)

Input Nodes (5):
1. ACO Area (Current)
2. ACO Pusat/Risk (Current)
3. ACO Area (Previous/Lag-1)
4. ACO Pusat/Risk (Previous/Lag-1)
5. LSTM Prediction (Anomaly/Output)

Output Nodes (2):
1. Arah (Bearing/Angle)
2. Jarak (Distance/Sudut)
"""

import os
import logging
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

import matplotlib
matplotlib.use('Agg') # Backend non-interaktif
import matplotlib.pyplot as plt

# Setup Logger
logger = logging.getLogger("VolcanoAI_CNN")
logger.addHandler(logging.NullHandler())

# =============================================================================
# SECTION 1: DATA PREPARATION (TABULAR FEATURE EXTRACTOR)
# =============================================================================

class TabularFeatureExtractor:
    """
    Menyiapkan data tabular 5-Node Input sesuai spesifikasi client.
    Menggantikan SpatialDataGenerator (Image).
    """
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        # Normalisasi sederhana agar NN lebih cepat konvergen
        self.norm_area = 1000.0  # Pembagi untuk area km2
        self.norm_dist = 100.0   # Pembagi untuk jarak km

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        [FIXED] Menyiapkan data training CNN.
        Target (Y) DIHITUNG SECARA MATEMATIS dari pergerakan Lat/Lon aktual.
        """
        if df.empty:
            return np.array([]), np.array([])

        # 1. Urutkan data berdasarkan waktu
        df = df.copy().sort_values('Acquired_Date').reset_index(drop=True)
        
        # --- FEATURE ENGINEERING (INPUT X - 5 NODES) ---
        df['aco_area_prev'] = df['aco_area_km2'].shift(1).fillna(0.0)
        df['aco_center_scalar'] = df['aco_center_lat'].fillna(0.0) 
        df['aco_center_prev'] = df['aco_center_scalar'].shift(1).fillna(0.0)
        
        if 'lstm_prediction' not in df.columns:
            df['lstm_prediction'] = df.get('PheromoneScore', 0.0)

        feature_cols = [
            'aco_area_km2',       # Node 1
            'aco_center_scalar',  # Node 2
            'aco_area_prev',      # Node 3
            'aco_center_prev',    # Node 4
            'lstm_prediction'     # Node 5
        ]
        
        X = df[feature_cols].fillna(0.0).values
        
        # Scaling Input
        X[:, 0] /= self.norm_area 
        X[:, 2] /= self.norm_area 

        # --- TARGET CALCULATION (OUTPUT Y - 2 NODES) ---
        lat1 = df['EQ_Lintang'].values[:-1]
        lon1 = df['EQ_Bujur'].values[:-1]
        lat2 = df['EQ_Lintang'].values[1:]
        lon2 = df['EQ_Bujur'].values[1:]
        
        # Rumus Bearing
        def calculate_bearing_vec(lat1, lon1, lat2, lon2):
            lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
            dlon_rad = np.radians(lon2 - lon1)
            y = np.sin(dlon_rad) * np.cos(lat2_rad)
            x = np.cos(lat1_rad) * np.sin(lat2_rad) - \
                np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
            bearing = np.degrees(np.arctan2(y, x))
            return (bearing + 360) % 360

        # Rumus Haversine
        def calculate_distance_vec(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        target_angles = calculate_bearing_vec(lat1, lon1, lat2, lon2)
        target_dists = calculate_distance_vec(lat1, lon1, lat2, lon2)
        
        # Alignment
        X_final = X[:-1]
        Y_final = np.column_stack((target_angles, target_dists))
        
        valid_mask = ~np.isnan(Y_final).any(axis=1)
        X_final = X_final[valid_mask]
        Y_final = Y_final[valid_mask]

        # Normalisasi Target
        Y_final[:, 0] = Y_final[:, 0] / 360.0       
        Y_final[:, 1] = Y_final[:, 1] / self.norm_dist 

        return X_final, Y_final

    def denormalize_output(self, y_pred: np.ndarray) -> np.ndarray:
        """
        [MISSING FUNCTION RESTORED]
        Mengembalikan output prediksi dari skala 0-1 ke skala asli (Derajat & Km).
        """
        y_real = np.zeros_like(y_pred)
        # Kolom 0 adalah Angle (dikali 360)
        y_real[:, 0] = y_pred[:, 0] * 360.0       
        # Kolom 1 adalah Distance (dikali 100)
        y_real[:, 1] = y_pred[:, 1] * self.norm_dist 
        return y_real

# =============================================================================
# SECTION 2: NEURAL NETWORK ARCHITECTURE (SIMPLE NN)
# =============================================================================

class SimpleNNFactory:
    """
    Membangun Neural Network sederhana sesuai request:
    Input (5) -> Hidden (3 Layers) -> Output (2: Arah, Jarak)
    """
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config

    def build_model(self, params: Dict[str, Any] = None) -> Model:
        p = params if params else {}
        units_1 = p.get('units_1', 64)
        units_2 = p.get('units_2', 32)
        units_3 = p.get('units_3', 16)
        dropout = p.get('dropout', 0.1)
        lr = p.get('learning_rate', 0.001)

        # Definisi Model Sequential (Simple Stack)
        model = Sequential(name="Simple_CNN_VolcanoAI")
        
        # Input Layer (5 Nodes)
        model.add(Input(shape=(5,), name="Input_5_Nodes"))
        
        # Hidden Layer 1
        model.add(Dense(units_1, name="Hidden_Layer_1"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if dropout > 0: model.add(Dropout(dropout))
        
        # Hidden Layer 2
        model.add(Dense(units_2, name="Hidden_Layer_2"))
        model.add(Activation('relu'))
        
        # Hidden Layer 3
        model.add(Dense(units_3, name="Hidden_Layer_3"))
        model.add(Activation('relu'))
        
        # Output Layer (2 Nodes: Arah & Jarak)
        # Menggunakan aktivasi linear (regresi) atau sigmoid jika strict 0-1
        # Di sini linear lebih fleksibel untuk regresi jarak
        model.add(Dense(2, activation='linear', name="Output_2_Nodes"))

        # Kompilasi
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='mse', # Mean Squared Error cocok untuk regresi vektor
            metrics=['mae']
        )
        
        return model

# =============================================================================
# SECTION 3: TUNER & ENGINE
# =============================================================================

class NNTuner:
    """Tuner sederhana untuk mencari konfigurasi jumlah neuron optimal."""
    def __init__(self, factory: SimpleNNFactory, trials=3):
        self.factory = factory
        self.trials = trials
        self.grid = {
            'units_1': [32, 64, 128],
            'units_2': [16, 32, 64],
            'units_3': [8, 16, 32],
            'learning_rate': [0.001, 0.005]
        }

    def search(self, X_train, Y_train, X_val, Y_val) -> Dict[str, Any]:
        best_loss = float('inf')
        best_params = {'units_1': 64, 'units_2': 32, 'units_3': 16, 'learning_rate': 0.001}
        
        logger.info(f" [NN Tuner] Memulai {self.trials} trial optimasi...")
        
        for i in range(self.trials):
            params = {k: random.choice(v) for k, v in self.grid.items()}
            K.clear_session()
            try:
                model = self.factory.build_model(params)
                # Training singkat
                hist = model.fit(
                    X_train, Y_train, 
                    validation_data=(X_val, Y_val), 
                    epochs=5, batch_size=16, verbose=0
                )
                val_loss = hist.history['val_loss'][-1]
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = params
            except Exception:
                continue
        
        logger.info(f" [NN Tuner] Params terbaik: {best_params}")
        return best_params

class CnnEngine:
    """
    Engine Utama (Rebranded 'Simple CNN').
    Mengelola training dan prediksi vektor gempa.
    """
    def __init__(self, config: Any):
        self.cfg = config.__dict__ if not isinstance(config, dict) else config
        
        # Default Configs
        self.cfg.setdefault('epochs', 50)
        self.cfg.setdefault('batch_size', 16)
        
        self.model_dir = Path(self.cfg.get("model_dir", "output/cnn/models"))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = TabularFeatureExtractor(self.cfg)
        self.factory = SimpleNNFactory(self.cfg)
        self.tuner = NNTuner(self.factory)
        
        self.models = {} # Cache model per cluster

    def train(self, df_train: pd.DataFrame, lstm_engine=None) -> bool:
        """
        Melatih model NN per cluster.
        Note: lstm_engine parameter disimpan untuk kompatibilitas, 
        tapi data diambil langsung dari kolom df_train['lstm_prediction'].
        """
        if df_train.empty: return False
        
        logger.info("=== START TRAINING SIMPLE NN (5-INPUT / 2-OUTPUT) ===")
        
        unique_clusters = sorted([c for c in df_train['cluster_id'].unique() if c != -1])
        success_count = 0
        
        for cid in unique_clusters:
            logger.info(f">>> Training Cluster {cid}")
            df_c = df_train[df_train['cluster_id'] == cid]
            
            # 1. Prepare Data
            X, Y = self.extractor.prepare_dataset(df_c)
            
            if len(X) < 10:
                logger.warning(f"Skip c{cid}: Data kurang ({len(X)} sampel).")
                continue
                
            # Split Data
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            Y_train, Y_val = Y[:split], Y[split:]
            
            # 2. Tune & Build
            params = self.tuner.search(X_train, Y_train, X_val, Y_val)
            model = self.factory.build_model(params)
            
            # 3. Train
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=self.model_dir / f"cnn_model_c{cid}.keras",
                    save_best_only=True
                )
            ]
            
            hist = model.fit(
                X_train, Y_train,
                validation_data=(X_val, Y_val),
                epochs=self.cfg['epochs'],
                batch_size=self.cfg['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # 4. Save Cache
            self.models[cid] = model
            success_count += 1
            
            # Log metrics
            loss = hist.history['loss'][-1]
            logger.info(f"Cluster {cid} Trained. Final Loss: {loss:.4f}")

        return success_count > 0

    def train_from_scratch(self, df_train: pd.DataFrame, lstm_engine, epochs: int = None) -> bool:
        # Wrapper kompatibilitas
        if epochs: self.cfg['epochs'] = epochs
        return self.train(df_train, lstm_engine)

    def predict(self, df_predict: pd.DataFrame, lstm_engine=None) -> pd.DataFrame:
        """
        Melakukan prediksi Arah dan Jarak menggunakan model NN yang sudah dilatih.
        """
        df_out = df_predict.copy()
        
        # Init Columns
        for col in ['cnn_angle_deg', 'cnn_distance_km', 'cnn_confidence', 'cnn_cardinal']:
            df_out[col] = np.nan
        
        unique_clusters = sorted([c for c in df_out['cluster_id'].unique() if c != -1])
        
        for cid in unique_clusters:
            # Load Model
            model = self.models.get(cid)
            if not model:
                try:
                    path = self.model_dir / f"cnn_model_c{cid}.keras"
                    if path.exists():
                        model = load_model(path, compile=False)
                        self.models[cid] = model
                except Exception: pass
            
            if not model:
                logger.warning(f"No model for c{cid}, skipping prediction.")
                continue
            
            # Filter Data Cluster
            mask = df_out['cluster_id'] == cid
            df_c = df_out[mask]
            
            if df_c.empty: continue

            # Prepare Input (Tanpa Target shift, karena kita mau prediksi row ini)
            # Kita gunakan logika extractor tapi manual untuk X saja
            
            # Feature Engineering on the fly
            df_c_proc = df_c.copy().sort_values('Acquired_Date')
            df_c_proc['aco_area_prev'] = df_c_proc['aco_area_km2'].shift(1).fillna(0.0)
            df_c_proc['aco_center_scalar'] = df_c_proc['aco_center_lat'].fillna(0.0)
            df_c_proc['aco_center_prev'] = df_c_proc['aco_center_scalar'].shift(1).fillna(0.0)
            
            if 'lstm_prediction' not in df_c_proc.columns:
                df_c_proc['lstm_prediction'] = df_c_proc.get('anomaly_score', 0.0)

            cols = [
                'aco_area_km2', 'aco_center_scalar', 
                'aco_area_prev', 'aco_center_prev', 
                'lstm_prediction'
            ]
            X = df_c_proc[cols].fillna(0.0).values
            
            # Apply same normalization
            X[:, 0] /= self.extractor.norm_area
            X[:, 2] /= self.extractor.norm_area
            
            # Predict
            try:
                preds_norm = model.predict(X, verbose=0)
                preds_real = self.extractor.denormalize_output(preds_norm)
                
                # Assign Results
                # Node 1: Arah (Angle)
                angles = np.abs(preds_real[:, 0]) % 360.0 
                # Node 2: Jarak (Distance)
                dists = np.abs(preds_real[:, 1])

                # Map back to original dataframe
                # Note: preds_real urut berdasarkan sort Acquired_Date
                idx_aligned = df_c_proc.index
                
                df_out.loc[idx_aligned, 'cnn_angle_deg'] = angles
                df_out.loc[idx_aligned, 'cnn_distance_km'] = dists
                
                # Confidence sederhana (Dummy based on prediction stability or magnitude)
                # Di sini kita set default 0.8 karena NN deterministik
                df_out.loc[idx_aligned, 'cnn_confidence'] = 0.85
                
                # Cardinal Direction
                df_out.loc[idx_aligned, 'cnn_cardinal'] = [self._get_cardinal(a) for a in angles]
                
            except Exception as e:
                logger.error(f"Prediction error c{cid}: {e}")

        return df_out

    def _get_cardinal(self, angle):
        """Helper untuk konversi sudut ke arah mata angin."""
        dirs = ["Utara", "Timur Laut", "Timur", "Tenggara", "Selatan", "Barat Daya", "Barat", "Barat Laut"]
        ix = round(angle / (360. / len(dirs)))
        return dirs[ix % len(dirs)]

    # =========================================================
    # EXPORT MAP HELPER (Tetap Dipertahankan)
    # =========================================================
    def export_cnn_prediction_map(self, json_path):
        import json, folium
        from pathlib import Path
        
        try:
            with open(json_path) as f:
                j = json.load(f)
            
            ne = j.get("next_event", {})
            lat, lon = ne.get("lat"), ne.get("lon")
            angle = ne.get("direction_deg", 0)
            dist = ne.get("distance_km", 0)
            
            if lat is None or lon is None: return None
            
            m = folium.Map(location=[lat, lon], zoom_start=10)
            
            # Marker Pusat (Origin)
            folium.Marker(
                [lat, lon], 
                popup=f"Origin<br>Prediksi Arah: {angle:.1f}°<br>Jarak: {dist:.1f}km",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(m)
            
            # Garis Prediksi (Visualisasi Vector)
            import math
            # Simple approximation: 1 deg lat ~= 111km
            dy = (dist * math.cos(math.radians(angle))) / 111.0
            dx = (dist * math.sin(math.radians(angle))) / (111.0 * math.cos(math.radians(lat)))
            
            end_lat = lat + dy
            end_lon = lon + dx
            
            folium.PolyLine([[lat, lon], [end_lat, end_lon]], color="blue", weight=3, opacity=0.8).add_to(m)
            folium.Marker([end_lat, end_lon], popup="Predicted Location", icon=folium.Icon(color="blue")).add_to(m)

            out = Path(json_path).parent
            map_path = out / "cnn_prediction_map.html"
            m.save(map_path)
            return map_path
        except Exception as e:
            logger.warning(f"Map export failed: {e}")
            return None

    def evaluate_predictions(self, df_out: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
        """Evaluasi akurasi prediksi (Compare CNN vs GA/Ground Truth)."""
        df = df_out.copy()
        df['cnn_correct'] = False
        
        if 'cnn_distance_km' in df and 'ga_distance_km' in df:
            df['dist_err'] = (df['cnn_distance_km'] - df['ga_distance_km']).abs()
            
        if 'cnn_angle_deg' in df and 'ga_bearing_deg' in df:
            df['angle_err'] = (df['cnn_angle_deg'] - df['ga_bearing_deg']).abs() % 360
            # Normalize angle diff (shortest path)
            df['angle_err'] = df['angle_err'].apply(lambda x: 360-x if x>180 else x)

        cond = pd.Series(True, index=df.index)
        if 'dist_km' in thresholds and 'dist_err' in df:
            cond &= df['dist_err'] <= thresholds['dist_km']
        if 'angle_deg' in thresholds and 'angle_err' in df:
            cond &= df['angle_err'] <= thresholds['angle_deg']
            
        df['cnn_correct'] = cond
        return df
