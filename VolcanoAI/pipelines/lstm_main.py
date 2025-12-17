# VolcanoAI/pipelines/lstm_main.py
# -- coding: utf-8 --

import logging
import pandas as pd
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from VolcanoAI.config.config import CONFIG
from VolcanoAI.engines.lstm_engine import LstmEngine
from VolcanoAI.utils.setup_logging import setup_logging

def run_lstm_pipeline():
    setup_logging(CONFIG.LSTM_ENGINE.output_dir)
    logger = logging.getLogger("LSTM_PIPELINE")

    logger.info("=== [Pipeline Standalone] Memulai LSTM ===")
    
    engine = LstmEngine(CONFIG.LSTM_ENGINE)
    
    # Load data
    from VolcanoAI.processing.data_loader import DataLoader
    loader = DataLoader(CONFIG.DATA_LOADER)
    df = loader.run()
    
    if df.empty:
        logger.error("Data kosong.")
        return

    logger.info("Training LSTM...")
    # Train method di V3.0 sudah handle feature engineering internal
    engine.train(df)

    logger.info("Evaluasi LSTM...")
    df_out, anomalies = engine.predict_on_static(df)
    
    out_path = os.path.join(CONFIG.LSTM_ENGINE.output_dir, "lstm_standalone_predictions.csv")
    df_out.to_csv(out_path, index=False)
    
    if not anomalies.empty:
        anom_path = os.path.join(CONFIG.LSTM_ENGINE.output_dir, "lstm_standalone_anomalies.csv")
        anomalies.to_csv(anom_path, index=False)
        
    logger.info(f"Selesai. Hasil tersimpan di {out_path}")

if __name__ == "__main__":
    run_lstm_pipeline()