#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VolcanoAI - Activation Function Visualization Script (Fixed Path)
=================================================================
Script ini telah disesuaikan dengan path absolut model cnn_model_c0.keras
untuk memvisualisasikan fungsi aktivasi ReLU berdasarkan data riil.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Setup sys.path agar bisa mendeteksi folder engines secara otomatis
base_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(base_dir, 'engines'))

try:
    from cnn_engine import CnnEngine
except ImportError:
    print("[Error] Berkas cnn_engine.py tidak ditemukan.")
    print("Pastikan script ini berada di dalam folder VolcanoAI/visualization")
    sys.exit(1)


def main():
    # -------------------------------------------------------------------------
    # 1. KONFIGURASI PATH ABSOLUT 
    # -------------------------------------------------------------------------
    config_dict = {
        'norm_area': 1000.0,
        'norm_dist': 100.0,
        'output_dir': r"gempavulkanik\output\cnn_results"
    }
    
    # Path absolut langsung ke file model dan data sesuai struktur direktori Anda
    model_path = r"gempavulkanik\output\cnn_results\models\cnn_model_c0.keras"
    csv_data_path = r"gempavulkanik\output\cnn_results\results\cnn_predictions_latest.csv"
    
    # Validasi keberadaan berkas sebelum eksekusi
    if not os.path.exists(csv_data_path):
        print(f"[Error] Berkas CSV tidak ditemukan di: {csv_data_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"[Error] Berkas model tidak ditemukan di: {model_path}")
        return

    print("[Info] Jalur berkas valid. Memulai ekstraksi data aktivasi CNN untuk Cluster 0...")

    # -------------------------------------------------------------------------
    # 2. LOAD DATA DAN PRE-PROCESSING
    # -------------------------------------------------------------------------
    df = pd.read_csv(csv_data_path)
    
    feature_cols = [
        'aco_center_scalar', 
        'aco_area_km2',       
        'aco_center_prev',    
        'aco_area_prev',      
        'lstm_prediction'     
    ]
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    X_matrix = df[feature_cols].fillna(0.0).values.astype(float)
    
    # Normalisasi fitur area (Index 1 dan 3)
    if X_matrix.shape[0] > 0:
        X_matrix[:, 1] /= config_dict['norm_area']
        X_matrix[:, 3] /= config_dict['norm_area']

    # -------------------------------------------------------------------------
    # 3. LOAD MODEL & EKSTRAKSI AKTIVASI (FORWARD PASS)
    # -------------------------------------------------------------------------
    model = load_model(model_path, compile=False)
    engine = CnnEngine(config_dict)
    
    z_list = []
    a_list = []
    
    # Batasi sampel data yang diplot (1500 baris) agar grafik tetap bersih
    max_samples = min(1500, len(X_matrix))
    
    for i in range(max_samples):
        activations = engine.manual_forward_pass(model, X_matrix[i], verbose=False)
        
        if 'Hidden_1' in activations:
            z_values = activations['Hidden_1']['z'].flatten()
            a_values = activations['Hidden_1']['a'].flatten()
            
            z_list.extend(z_values)
            a_list.extend(a_values)

    # -------------------------------------------------------------------------
    # 4. PROSES GENERASI GRAFIK PROFESIONAL
    # -------------------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    
    # Plot sebaran data empiris (Scatter Plot)
    ax.scatter(
        z_list, a_list, 
        color='#1E40AF',          
        alpha=0.4,                
        s=20,                     
        edgecolors='none', 
        label='Sebaran Aktivasi Node (Data Aktual Cluster 0)'
    )
    
    # Membuat garis referensi ideal fungsi ReLU sebagai pembanding
    z_ref = np.linspace(min(z_list) - 1, max(z_list) + 1, 500)
    a_ref = np.maximum(0, z_ref)
    ax.plot(z_ref, a_ref, color='#DC2626', linestyle='--', linewidth=1.5, label='Fungsi Ideal ReLU: max(0, z)')
    
    ax.axhline(0, color='#4B5563', linewidth=1.2)
    ax.axvline(0, color='#4B5563', linewidth=1.2)
    
    ax.set_title(
        'Visualisasi Empiris Fungsi Aktivasi ReLU', 
        fontsize=12, fontweight='bold', pad=15, color='#111827'
    )
    ax.set_xlabel('Nilai Pre-Activation ($z = W^T x + b$)', fontsize=11, labelpad=8, color='#374151')
    ax.set_ylabel('Nilai Post-Activation ($a = \max(0, z)$)', fontsize=11, labelpad=8, color='#374151')
    
    ax.legend(loc='upper left', frameon=True, facecolor='#FFFFFF', edgecolor='#E5E7EB', fontsize=10)
    plt.tight_layout()
    
    # Tempat penyimpanan grafik plot hasil
    output_directory = os.path.join(config_dict['output_dir'], 'plots')
    os.makedirs(output_directory, exist_ok=True)
    save_path = os.path.join(output_directory, 'cnn_relu_activation_empirical.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n[Sukses] Grafik aktivasi CNN profesional berhasil disimpan pada:\n-> {save_path}")


if __name__ == "__main__":
    main()