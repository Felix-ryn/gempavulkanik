# VolcanoAI/pipelines/naive_bayes_main.py
# -- coding: utf-8 --

import logging
import pandas as pd
import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Setup Project Root untuk Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from VolcanoAI.config.config import CONFIG
from VolcanoAI.engines.naive_bayes_engine import NaiveBayesEngine
from VolcanoAI.utils.setup_logging import setup_logging
from VolcanoAI.processing.feature_engineer import FeatureEngineer
from VolcanoAI.processing.data_loader import DataLoader

def run_naive_bayes_pipeline():
    # Setup Logging khusus Pipeline ini
    setup_logging(CONFIG.NAIVE_BAYES_ENGINE.output_dir)
    logger = logging.getLogger("NAIVE_BAYES_PIPELINE")

    logger.info("=== [Pipeline Standalone] Memulai Naive Bayes ===")

    # -------------------------------------------------------------
    # 1. LOAD DATA (Cache Pickle -> Excel -> Raw Data Loader)
    # -------------------------------------------------------------
    merged_path = CONFIG.DATA_LOADER.merged_output_path
    cache_path = merged_path.replace(".xlsx", ".pkl")
    
    df = pd.DataFrame()

    # Prioritas 1: Cache Pickle (Cepat)
    if os.path.exists(cache_path):
        try:
            df = pd.read_pickle(cache_path)
            logger.info(f"Data dimuat dari Cache Pickle: {cache_path}")
        except Exception as e:
            logger.warning(f"Gagal memuat pickle ({e}). Mencoba Excel...")
    
    # Prioritas 2: Merged Excel
    if df.empty and os.path.exists(merged_path):
        try:
            df = pd.read_excel(merged_path)
            logger.info(f"Data dimuat dari Excel Merged: {merged_path}")
        except Exception as e:
            logger.warning(f"Gagal memuat Excel ({e}). Jalankan DataLoader...")

    # Prioritas 3: Raw DataLoader (Jika file belum ada)
    if df.empty:
        logger.warning("Data cache tidak ditemukan. Menjalankan DataLoader dari awal.")
        loader = DataLoader(CONFIG.DATA_LOADER)
        df = loader.run()

    if df.empty:
        logger.error("[CRITICAL] Data kosong setelah semua metode loading. Pipeline berhenti.")
        return

    # -------------------------------------------------------------
    # 2. PREPARATION & FEATURE ENGINEERING CHECK
    # -------------------------------------------------------------
    target_col = CONFIG.NAIVE_BAYES_ENGINE.target_column
    required_feats = CONFIG.NAIVE_BAYES_ENGINE.features
    
    # [FIX KRITIS]: Cek apakah Target Labeling dan Fitur Inti sudah ada.
    # Jika standalone run, kemungkinan R3_final (ACO result) atau label impact_level belum ada.
    fe_columns_needed = ['R3_final', 'PheromoneScore', target_col]
    
    needs_fe = any(col not in df.columns for col in fe_columns_needed)

    if needs_fe:
        logger.info("[Standalone Fix] Menjalankan Feature Engineering untuk mendapatkan target Label & R3.")
        fe = FeatureEngineer(CONFIG.FEATURE_ENGINEERING, CONFIG.ACO_ENGINE)
        # FE run biasanya return (df, other_info). Ambil df-nya.
        df, _ = fe.run(df) 
    
    # Fallback Fitur Eksternal (Hybrid Scenario)
    # Jika NB membutuhkan input dari model DL (misal: 'lstm_embedding'), 
    # namun skrip ini jalan sendiri, kita isi dengan 0.0 agar NB tetap bisa jalan.
    missing_features = [col for col in required_feats if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Fitur berikut tidak ditemukan (mungkin output DL engine lain?): {missing_features}")
        logger.warning("--> Mengisi dengan 0.0 untuk Standalone Testing.")
        for col in missing_features:
            df[col] = 0.0

    # Safety: Pastikan target benar-benar ada
    if target_col not in df.columns:
        logger.error(f"[CRITICAL] Target column '{target_col}' tidak ditemukan bahkan setelah FE. Cek Config.")
        return

    # Drop NaN di target agar training bersih
    df = df.dropna(subset=[target_col])

    # -------------------------------------------------------------
    # 3. TRAINING & EVALUATION
    # -------------------------------------------------------------
    # Split Data
    try:
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target_col])
    except ValueError:
        # Fallback jika stratify gagal (kelas terlalu sedikit)
        logger.warning("Stratify gagal (data terlalu sedikit per kelas?). Menggunakan split acak.")
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)

    logger.info(f"Shape: Train {df_train.shape}, Test {df_test.shape}")

    # Init & Train Engine
    engine = NaiveBayesEngine(CONFIG.NAIVE_BAYES_ENGINE)
    
    logger.info("Training Naive Bayes Model...")
    engine.train(df_train)

    logger.info("Melakukan Evaluasi...")
    # Evaluate return df result dan dictionary metrics
    df_out, metrics = engine.evaluate(df_test)
    
    logger.info(f"Metrics Naive Bayes: {metrics}")

    # -------------------------------------------------------------
    # 4. SAVE OUTPUT
    # -------------------------------------------------------------
    out_dir = CONFIG.NAIVE_BAYES_ENGINE.output_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, "naive_bayes_standalone_predictions.csv")
    df_out.to_csv(out_path, index=False)
    
    logger.info(f"=== [Pipeline NB Selesai] Hasil disimpan di: {out_path} ===")

if __name__ == "__main__":
    run_naive_bayes_pipeline()