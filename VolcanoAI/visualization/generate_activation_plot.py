#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VolcanoAI - Activation Function Visualization Script
=================================================================
Script ini memvisualisasikan fungsi aktivasi ReLU (Hidden Layer) 
dan Linear (Output Layer) dari data empiris hasil inferensi CNN.

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
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Daftarkan base_dir ke environment path agar import lokal berhasil
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from VolcanoAI.engines.cnn_engine import CnnEngine
except ImportError:
    print("[Error] Berkas cnn_engine.py tidak ditemukan. Path tidak valid.")
    sys.exit(1)


def _render_and_save_plot(z_list, a_list, title, xlabel, ylabel, ideal_label, ideal_func, dot_color, save_path):
    """
    Fungsi internal (DRY) untuk merender grafik dan menyimpannya.
    """
    if not z_list or not a_list:
        return

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    
    # Plot Sebaran Titik Asli
    ax.scatter(z_list, a_list, color=dot_color, alpha=0.4, s=20, 
               edgecolors='none', label='Sebaran Aktivasi Node (Empiris)')
    
    # Plot Garis Referensi Ideal
    z_ref = np.linspace(min(z_list) - 1, max(z_list) + 1, 500)
    a_ref = ideal_func(z_ref)
    ax.plot(z_ref, a_ref, color='#DC2626', linestyle='--', linewidth=1.5, label=ideal_label)
    
    # Garis kuadran 0
    ax.axhline(0, color='#4B5563', linewidth=1.2)
    ax.axvline(0, color='#4B5563', linewidth=1.2)
    
    # Formatting
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15, color='#111827')
    ax.set_xlabel(xlabel, fontsize=11, labelpad=8, color='#374151')
    ax.set_ylabel(ylabel, fontsize=11, labelpad=8, color='#374151')
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=10)
    
    plt.tight_layout()
    
    # Simpan Gambar
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()


def generate_relu_activation_plot(cluster_id=0):
    """
    Fungsi utama untuk mengekstrak data dari model dan membuat 
    grafik aktivasi untuk ReLU (Hidden) dan Linear (Output).
    
    (Nama fungsi tetap 'generate_relu_activation_plot' agar kompatibel 
    dengan panggilan di main.py)
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
    if not csv_data_path.exists() or not model_path.exists():
        print("[Plot CNN] Batal: Model atau Data CSV tidak ditemukan.")
        return None

    # 3. LOAD DATA
    df = pd.read_csv(csv_data_path)
    feature_cols = [
        'aco_center_scalar', 'aco_area_km2', 
        'aco_center_prev', 'aco_area_prev', 'lstm_prediction'
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X_matrix = df[feature_cols].fillna(0.0).values.astype(float)
    
    # Normalisasi Input
    if X_matrix.shape[0] > 0:
        X_matrix[:, 1] /= config_dict['norm_area']
        X_matrix[:, 3] /= config_dict['norm_area']

    model = load_model(str(model_path), compile=False)
    engine = CnnEngine(config_dict)
    
    # Wadah untuk Data ReLU dan Linear
    z_relu, a_relu = [], []
    z_linear, a_linear = [], []
    
    max_samples = min(1500, len(X_matrix))
    
    # 4. EKSTRAKSI DATA DALAM SEKALI JALAN (FORWARD PASS)
    for i in range(max_samples):
        activations = engine.manual_forward_pass(model, X_matrix[i], verbose=False)
        
        # Ekstrak ReLU dari Hidden Layer 1
        if 'Hidden_1' in activations:
            z_relu.extend(activations['Hidden_1']['z'].flatten())
            a_relu.extend(activations['Hidden_1']['a'].flatten())
            
        # Ekstrak Linear dari Output Layer
        if 'Output_2_Nodes' in activations:
            z_linear.extend(activations['Output_2_Nodes']['z'].flatten())
            a_linear.extend(activations['Output_2_Nodes']['a'].flatten())

    if not z_relu or not z_linear:
        return None

    # 5. GENERASI KEDUA GRAFIK
    path_relu = save_dir / "cnn_relu_activation_empirical.png"
    path_linear = save_dir / "cnn_linear_activation_empirical.png"
    
    # -- Plot ReLU --
    _render_and_save_plot(
        z_list=z_relu, a_list=a_relu,
        title=f'Visualisasi Empiris Fungsi Aktivasi ReLU (Cluster {cluster_id})',
        xlabel='Nilai Pre-Activation ($z = W^T x + b$)',
        ylabel='Nilai Post-Activation ($a = \max(0, z)$)',
        ideal_label='Fungsi Ideal ReLU: a = max(0, z)',
        ideal_func=lambda z: np.maximum(0, z),
        dot_color='#1E40AF', # Biru untuk ReLU
        save_path=path_relu
    )

    # -- Plot Linear --
    _render_and_save_plot(
        z_list=z_linear, a_list=a_linear,
        title=f'Visualisasi Empiris Fungsi Aktivasi Linear Output (Cluster {cluster_id})',
        xlabel='Nilai Pre-Activation ($z = W^T x + b$)',
        ylabel='Nilai Post-Activation ($a = z$)',
        ideal_label='Fungsi Ideal Linear: a = z',
        ideal_func=lambda z: z, # Linear adalah a = z
        dot_color='#059669', # Hijau Zamrud untuk Linear
        save_path=path_linear
    )
    
    # Mengembalikan string gabungan agar tercatat rapi di log main.py
    return f"ReLU.png & Linear.png di {save_dir}"

if __name__ == "__main__":
    # Eksekusi langsung jika dijalankan manual
    print("[Info] Membuat grafik aktivasi CNN (ReLU & Linear) secara mandiri...")
    res = generate_relu_activation_plot(0)
    if res:
        print(f"[Sukses] Gambar telah berhasil dibuat: {res}")