#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VolcanoAI - Activation Function Visualization Script
=================================================================
Script ini memvisualisasikan fungsi aktivasi ReLU dari data empiris.
Jalur (path) dinamis secara penuh. Akan selalu tersimpan di:
[ROOT_PROJECT]/output/cnn_results/plots/
tanpa terpengaruh nama user komputer.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras.models import load_model

# 1. DETEKSI ROOT PROJECT SECARA DINAMIS (BEBAS HARDCODE)
# Skrip ini ada di: VolcanoAI/visualization/
# Kita naik 3 level ke atas untuk mendapatkan folder root (gempavulkanik)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Daftarkan base_dir ke environment path agar import lokal berhasil
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from VolcanoAI.engines.cnn_engine import CnnEngine
except ImportError:
    print("[Error] Berkas cnn_engine.py tidak ditemukan. Path tidak valid.")
    sys.exit(1)


def generate_relu_activation_plot(cluster_id=0):
    """
    Fungsi utama untuk membuat grafik empiris aktivasi CNN.
    """
    config_dict = {
        'norm_area': 1000.0,
        'norm_dist': 100.0,
    }
    
    # 2. DEFINISI PATH DINAMIS (Selalu mengarah ke dalam folder output proyek)
    output_cnn_dir = BASE_DIR / "output" / "cnn_results"
    
    model_path = output_cnn_dir / "models" / f"cnn_model_c{cluster_id}.keras"
    csv_data_path = output_cnn_dir / "results" / "cnn_predictions_latest.csv"
    save_dir = output_cnn_dir / "plots"
    
    # Validasi awal
    if not csv_data_path.exists():
        print(f"[Plot CNN] Batal: Data CSV tidak ditemukan di -> {csv_data_path}")
        return None
    if not model_path.exists():
        print(f"[Plot CNN] Batal: Model Keras tidak ditemukan di -> {model_path}")
        return None

    # 3. LOAD DATA & EKSTRAKSI AKTIVASI
    df = pd.read_csv(csv_data_path)
    feature_cols = [
        'aco_center_scalar', 'aco_area_km2', 
        'aco_center_prev', 'aco_area_prev', 'lstm_prediction'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X_matrix = df[feature_cols].fillna(0.0).values.astype(float)
    
    # Normalisasi
    if X_matrix.shape[0] > 0:
        X_matrix[:, 1] /= config_dict['norm_area']
        X_matrix[:, 3] /= config_dict['norm_area']

    model = load_model(str(model_path), compile=False)
    engine = CnnEngine(config_dict)
    
    z_list = []
    a_list = []
    max_samples = min(1500, len(X_matrix))
    
    for i in range(max_samples):
        activations = engine.manual_forward_pass(model, X_matrix[i], verbose=False)
        if 'Hidden_1' in activations:
            z_list.extend(activations['Hidden_1']['z'].flatten())
            a_list.extend(activations['Hidden_1']['a'].flatten())

    if not z_list:
        return None

    # 4. GENERASI GRAFIK PROFESIONAL
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    
    ax.scatter(z_list, a_list, color='#1E40AF', alpha=0.4, s=20, 
               edgecolors='none', label=f'Sebaran Aktivasi Node (Cluster {cluster_id})')
    
    z_ref = np.linspace(min(z_list) - 1, max(z_list) + 1, 500)
    a_ref = np.maximum(0, z_ref)
    ax.plot(z_ref, a_ref, color='#DC2626', linestyle='--', linewidth=1.5, label='Fungsi Ideal ReLU: max(0, z)')
    
    ax.axhline(0, color='#4B5563', linewidth=1.2)
    ax.axvline(0, color='#4B5563', linewidth=1.2)
    
    ax.set_title('Visualisasi Empiris Fungsi Aktivasi ReLU pada CNN', fontsize=12, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel('Nilai Pre-Activation ($z = W^T x + b$)', fontsize=11, labelpad=8, color='#374151')
    ax.set_ylabel('Nilai Post-Activation ($a = \max(0, z)$)', fontsize=11, labelpad=8, color='#374151')
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=10)
    plt.tight_layout()
    
    # 5. SIMPAN GAMBAR KE FOLDER OUTPUT
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "cnn_relu_activation_empirical.png"
    
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(save_path)

if __name__ == "__main__":
    # Eksekusi langsung jika dijalankan manual
    print("[Info] Membuat grafik aktivasi CNN secara mandiri...")
    res = generate_relu_activation_plot(0)
    if res:
        print(f"[Sukses] Grafik disimpan di:\n{res}")