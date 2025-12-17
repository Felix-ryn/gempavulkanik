# VolcanoAI/pipelines/cnn_main.py
# -- coding: utf-8 --

import logging
import pandas as pd
import sys
import os

# Pastikan root project ada di path agar bisa import module
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from VolcanoAI.config.config import CONFIG
from VolcanoAI.engines.cnn_engine import CnnEngine
from VolcanoAI.engines.lstm_engine import LstmEngine
from VolcanoAI.utils.setup_logging import setup_logging

def run_cnn_pipeline():
    setup_logging(CONFIG.CNN_ENGINE.output_dir)
    logger = logging.getLogger("CNN_PIPELINE")

    logger.info("=== [Pipeline Standalone] Memulai CNN ===")
    
    # Inisialisasi kedua engine karena CNN butuh LSTM untuk ekstraksi fitur
    lstm_engine = LstmEngine(CONFIG.LSTM_ENGINE)
    cnn_engine = CnnEngine(CONFIG.CNN_ENGINE)

    logger.info("Memuat Data via LSTM Loader...")
    # Menggunakan mekanisme ingest yang sama dengan main.py
    df = lstm_engine.ingest_and_prepare_static() if hasattr(lstm_engine, 'ingest_and_prepare_static') else None
    
    # Fallback jika metode ingest tidak ada di LstmEngine V3.0 (karena dipindah ke DataLoader)
    if df is None:
        from VolcanoAI.processing.data_loader import DataLoader
        loader = DataLoader(CONFIG.DATA_LOADER)
        df = loader.run()

    if df.empty:
        logger.error("Data kosong.")
        return

    # Feature Engineering (Penting agar kolom radius R1, R2, R3 ada)
    # Kita gunakan feature_engineer milik lstm_engine yang sudah terinisialisasi
    df = lstm_engine.feature_engineer.run(df)[0]

    logger.info("Training CNN (Hybrid)...")
    cnn_engine.train(df, lstm_engine)

    logger.info("Evaluasi CNN...")
    df_out = cnn_engine.predict(df, lstm_engine)
    
    out_path = os.path.join(CONFIG.CNN_ENGINE.output_dir, "cnn_standalone_predictions.csv")
    df_out.to_csv(out_path, index=False)
    logger.info(f"Selesai. Hasil tersimpan di {out_path}")

if __name__ == "__main__":
    run_cnn_pipeline()