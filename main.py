# main.py (Versi V5.2 - Excel Storage Edition + ACO Activated First - FIXED FINAL)
# -- coding: utf-8 --

# Di awal main.py
import warnings
# Mengabaikan FutureWarning Pandas dan RuntimeWarning NumPy yang umum
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", message="X does not have valid feature names")
# Non-aktifkan pesan diagnostik internal TF yang sudah usang
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import os
import sys
import time
import json
import logging
import argparse
import platform
import traceback
import functools
from datetime import datetime
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==============================================================================
# 1. SYSTEM INITIALIZATION & PATH SETUP
# ==============================================================================

def setup_project_path():
    """Memastikan root project terdaftar di sys.path."""
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        project_root = os.path.abspath('.')
    
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parent_dir = os.path.dirname(project_root)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

setup_project_path()

# ==============================================================================
# 2. MODULE IMPORTS (SAFE LOADER)
# ==============================================================================

SYSTEM_READY = False

try:
    from VolcanoAI.config.config import CONFIG, ProjectConfig
    from VolcanoAI.utils.setup_logging import setup_logging
    
    from VolcanoAI.processing.data_loader import DataLoader
    from VolcanoAI.processing.feature_engineer import FeatureEngineer, FeaturePreprocessor
    
    # Realtime manager baru (BMKG + Mirova + Injection Excel)
    from VolcanoAI.processing.realtime_data_stream import RealtimeSensorManager
    from VolcanoAI.processing.realtime_buffer_manager import RealtimeBufferManager

    from VolcanoAI.engines.aco_engine import DynamicAcoEngine
    from VolcanoAI.engines.ga_engine import GaEngine
    from VolcanoAI.engines.lstm_engine import LstmEngine
    from VolcanoAI.engines.cnn_engine import CnnEngine
    from VolcanoAI.engines.naive_bayes_engine import NaiveBayesEngine
    
    from VolcanoAI.reporting.comprehensive_reporter import ComprehensiveReporter, GraphVisualizer 

    SYSTEM_READY = True

except ImportError as e:
    print(f"\n[CRITICAL IMPORT ERROR] {e}")
    print("Sistem tidak dapat dijalankan karena modul hilang.\n")


# ==============================================================================
# 3. GLOBAL CONSTANTS & HELPER METHODS
# ==============================================================================

SYSTEM_VERSION = "5.2.0-TITAN-EXCEL"

def pipeline_guard(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger("PipelineGuard") if SYSTEM_READY else None
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if logger:
                logger.critical(f"CRASH di {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
            else:
                print(f"CRASH di {func.__name__}: {str(e)}")
            raise e
    return wrapper


class SystemHealthMonitor:
    @staticmethod
    def check_resources():
        if not SYSTEM_READY:
            return
        logger = logging.getLogger("SystemMonitor")
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"RAM Usage: {mem.percent}%")
        except ImportError:
            pass


class PipelineStateManager:
    def __init__(self, output_dir: str):
        self.state_file = os.path.join(output_dir, "pipeline_state.json")

    def update_stage(self, stage_name: str, status: str):
        state = self.load_state()
        state[stage_name] = {"status": status, "timestamp": datetime.now().isoformat()}
        try:
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=4)
        except Exception:
            pass

    def load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


# ======================================================================
# 4. SAFE LOAD / SAVE CSV (GLOBAL — WAJIB DI LUAR CLASS)
# ======================================================================

def safe_load_csv(path: str):
    try:
        if os.path.exists(path):
            return pd.read_csv(path, parse_dates=["Acquired_Date"])
    except Exception as e:
        logging.error(f"[SAFE LOAD ERROR] {path} → {e}")
    return pd.DataFrame()


def safe_save_csv(path: str, df: pd.DataFrame):
    try:
        df.to_csv(path, index=False)
        logging.info(f"[BUFFER SAVED] {path}")
    except Exception as e:
        logging.error(f"[SAFE SAVE ERROR] {path} → {e}")


# ==============================================================================
# 5. MAIN PIPELINE CLASS
# ==============================================================================

class VolcanoAiPipeline:
    def __init__(self, config):
        if not SYSTEM_READY:
            return

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.state_mgr = PipelineStateManager(self.config.OUTPUT.directory)

        self.df_train = None
        self.df_test = None
        self.feature_preprocessor = None

        self.trained_aco_engine = None
        self.trained_ga_engine = None
        self.trained_lstm_engine = None
        self.trained_cnn_engine = None
        self.trained_nb_engine = None

        self._init_subsystems()
        self._init_paths()

        self.logger.info(f"Pipeline VolcanoAI {SYSTEM_VERSION} initialized successfully.")

    # ----------------------------------------------------------------------
    # INIT SYSTEM
    # ----------------------------------------------------------------------

    def _init_subsystems(self):
        self.data_loader = DataLoader(self.config.DATA_LOADER)
        self.feature_engineer = FeatureEngineer(self.config.FEATURE_ENGINEERING, self.config.ACO_ENGINE)

        self.aco_engine = DynamicAcoEngine(self.config.ACO_ENGINE)
        self.ga_engine = GaEngine(self.config.GA_ENGINE)
        self.lstm_engine = LstmEngine(self.config.LSTM_ENGINE)
        self.cnn_engine = CnnEngine(self.config.CNN_ENGINE)
        self.nb_engine = NaiveBayesEngine(self.config.NAIVE_BAYES_ENGINE)

        self.reporter = ComprehensiveReporter(self.config)

        # Realtime Sensor Manager (BMKG + MIROVA + Injection Excel)
        # Sesuaikan path log MIROVA kalau perlu
        self.sensor_manager = RealtimeSensorManager(
            mirova_log_path="output/realtime/mirova_log.txt"
        )

    def _init_paths(self):
        self.data_cache_path = self.config.DATA_LOADER.merged_output_path.replace(".xlsx", ".pkl")
        self.preprocessor_cache_path = self.config.FEATURE_ENGINEERING.preprocessor_output_path


    # ----------------------------------------------------------------------
    # PHASE 1 — LOAD & SPLIT DATA
    # ----------------------------------------------------------------------

    @pipeline_guard
    def _step_load_and_split_data(self) -> bool:
        self.logger.info("\n========== PHASE 1: DATA LOADING ==========")
        SystemHealthMonitor.check_resources()
        self.state_mgr.update_stage("DataLoading", "Running")

        df_full = None
        cache_exists = os.path.exists(self.data_cache_path)

        if self.config.PIPELINE.run_data_loading:
            if not cache_exists:
                df_full = self.data_loader.run()
                if df_full is None or df_full.empty:
                    return False
                df_full.to_pickle(self.data_cache_path)
            else:
                df_full = pd.read_pickle(self.data_cache_path)

        if df_full is None or df_full.empty:
            return False

        self.df_train, self.df_test = train_test_split(
            df_full,
            test_size=self.config.DATA_SPLIT.test_size,
            random_state=self.config.DATA_SPLIT.random_state,
        )

        self.state_mgr.update_stage("DataLoading", "Success")
        return True


    # ----------------------------------------------------------------------
    # PHASE 2 — FEATURE ENGINEERING
    # ----------------------------------------------------------------------

    @pipeline_guard
    def _step_feature_engineering(self, is_training: bool) -> bool:
        self.logger.info("\n========== PHASE 2: FEATURE ENGINEERING ==========")
        self.state_mgr.update_stage("FeatureEngineering", "Running")

        if is_training:
            self.df_train, self.feature_preprocessor = self.feature_engineer.run(self.df_train)

            if self.df_test is not None:
                self.df_test, _ = self.feature_engineer.run(self.df_test, preprocessor=self.feature_preprocessor)

        else:
            if self.feature_preprocessor is None:
                if os.path.exists(self.preprocessor_cache_path):
                    self.feature_preprocessor = FeaturePreprocessor.load(self.preprocessor_cache_path)
                else:
                    return False

            if self.df_test is not None:
                self.df_test, _ = self.feature_engineer.run(self.df_test, preprocessor=self.feature_preprocessor)
            if self.df_train is not None:
                self.df_train, _ = self.feature_engineer.run(self.df_train, preprocessor=self.feature_preprocessor)

        self.state_mgr.update_stage("FeatureEngineering", "Success")
        return True


    # ----------------------------------------------------------------------
    # PHASE 3 — TRAINING FLOW (FINAL VERSION)
    # ----------------------------------------------------------------------

    @pipeline_guard
    def _run_training_flow(self):
        self.logger.info("\n========== PHASE 3: MODEL TRAINING ==========")

        df_processed = self.df_train.copy()

        # 1️⃣ ACO selalu dijalankan ulang — tidak memakai cache
        df_processed, _ = self.aco_engine.run(df_processed)
        self.trained_aco_engine = self.aco_engine

        # 2️⃣ GA selalu run
        best_path, graphs_result = self.ga_engine.run(df_processed)
        self.trained_ga_engine = self.ga_engine

        # 3️⃣ LSTM
        self.lstm_engine.train(df_processed)
        self.trained_lstm_engine = self.lstm_engine

        df_processed, _ = self.lstm_engine.predict_on_static(df_processed)

        # 4️⃣ CNN
        self.cnn_engine.train(df_processed, self.lstm_engine)
        self.trained_cnn_engine = self.cnn_engine

        df_processed = self.cnn_engine.predict(df_processed, self.lstm_engine)

        # =========================
        # CNN ERROR CHECK (TRAINING)
        # =========================
        if hasattr(self.cnn_engine, "evaluate_error"):
            cnn_error = self.cnn_engine.evaluate_error(df_processed)
            self.logger.info(f"[CNN] Training error: {cnn_error:.4f}")

            if cnn_error > self.config.CNN_ENGINE.max_error_threshold:
                self.logger.warning("[CNN] Error tinggi, retraining CNN...")
                self.cnn_engine.train(df_processed, self.lstm_engine)


        from datetime import datetime
        from pathlib import Path

        out_dir = Path(self.config.OUTPUT.directory) / "cnn_results" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)

        export_cols = [
            'cluster_id',
            'Acquired_Date',
            'luas_cnn',
            'cnn_angle_deg',
            'cnn_distance_km'
        ]
        export_cols = [c for c in export_cols if c in df_processed.columns]

        # ===============================
        # 1️⃣ FILE TERKINI (UNTUK HTML)
        # ===============================
        latest_path = out_dir / "cnn_predictions_latest.csv"
        df_processed[export_cols].to_csv(latest_path, index=False)

        self.logger.info(f"✅ CNN latest overwritten: {latest_path}")

        # setelah menulis latest_path
        try:
            from VolcanoAI.postprocess.cnn_csv_to_json import run as cnn_postprocess_run
            cnn_postprocess_run(csv_path=str(latest_path), out_json=str(Path(self.config.OUTPUT.directory) / "cnn_results" / "cnn_predictions_latest.json"))
            self.logger.info("✅ CNN JSON updated for client")
        except Exception as e:
            self.logger.error(f"❌ CNN postprocess failed: {e}")

        # ===============================
        # 2️⃣ FILE ARSIP (HISTORI)
        # ===============================
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = out_dir / f"cnn_predictions_{ts}.csv"
        df_processed[export_cols].to_csv(archive_path, index=False)
        self.logger.info(f" CNN archive saved: {archive_path}")

                # 5️⃣ Naive Bayes
        self.nb_engine.train(df_processed)
        self.trained_nb_engine = self.nb_engine

        self.state_mgr.update_stage("Training", "Success")


    # ----------------------------------------------------------------------
    # PHASE 4 — EVALUATION
    # ----------------------------------------------------------------------

    @pipeline_guard
    def _run_evaluation_flow(self):
        self.logger.info("\n========== PHASE 4: MODEL EVALUATION ==========")

        df_eval = self.df_test.copy()

        df_eval, _ = DynamicAcoEngine(self.config.ACO_ENGINE).run(df_eval)

        lstm = self.trained_lstm_engine
        cnn = self.trained_cnn_engine
        nb = self.trained_nb_engine

        df_eval, anomalies = lstm.predict_on_static(df_eval)
        df_eval = cnn.predict(df_eval, lstm)
        df_final, metrics = nb.evaluate(df_eval)

        self.state_mgr.update_stage("Evaluation", "Success")
        return df_final, metrics, anomalies


    # ----------------------------------------------------------------------
    # PHASE X — REALTIME INFERENCE (BUFFER + INJECTION + FE)
    # ----------------------------------------------------------------------

    @pipeline_guard
    def run_realtime_inference(self):
        self.logger.info("\n========== REALTIME INFERENCE MODE ==========")

        buffer = RealtimeBufferManager(buffer_days=90)

        # Load buffer lama (kalau ada)
        buffer.raw_realtime = safe_load_csv("output/realtime/raw_realtime.csv")
        buffer.raw_injection = safe_load_csv("output/realtime/raw_injection.csv")
        buffer.processed = safe_load_csv("output/realtime/processed.csv")

        # Ambil data realtime dari sensor manager (BMKG + Mirova + Injection)
        try:
            # [FIX 1]: Mengganti get_realtime_data() dengan get_merged_stream()
            df_raw_new_stream = self.sensor_manager.get_merged_stream() 
        except Exception as e:
            self.logger.error(f"Realtime fetch error: {e}")
            df_raw_new_stream = pd.DataFrame() # Mengganti 3 df output menjadi 1

        # Pisahkan BMKG dan Injection sebelum append (agar data mentah tersimpan terpisah)
        # Asumsi: df_raw_new_stream mengandung kedua sumber, dan kita perlu memisahkannya 
        df_bmkg = df_raw_new_stream[df_raw_new_stream['Sumber'] == 'BMKG'].copy()
        df_inj = df_raw_new_stream[df_raw_new_stream['Sumber'] == 'InjectedExcel'].copy()

        # Simpan BMKG ke buffer realtime
        if df_bmkg is not None and not df_bmkg.empty:
            buffer.append_raw_realtime(df_bmkg)

        # Simpan data suntik ke buffer injection
        if df_inj is not None and not df_inj.empty:
            buffer.append_raw_injection(df_inj)

        # df_raw sekarang adalah gabungan dari data mentah baru dan data mentah historis
        df_raw = buffer.get_merged_raw()
        if df_raw.empty:
            self.logger.warning("Tidak ada data inference.")
            return

        # FE untuk realtime (pakai preprocessor hasil training)
        df_processed, _ = self.feature_engineer.run(
            df_raw,
            is_training=False,
            preprocessor=self.feature_preprocessor,
        )

        buffer.append_processed(df_processed)

        # Simpan balik buffer ke disk
        safe_save_csv("output/realtime/raw_realtime.csv", buffer.raw_realtime)
        safe_save_csv("output/realtime/raw_injection.csv", buffer.raw_injection)
        safe_save_csv("output/realtime/processed.csv", buffer.processed)

        self.logger.info("Realtime inference complete.")


    # ----------------------------------------------------------------------
    # PHASE 5 — LIVE MONITORING LOOP
    # ----------------------------------------------------------------------

    def start_monitoring_loop(self):
        self.logger.info("\n========== PHASE 5: LIVE MONITORING ==========")

        if self.df_train is not None:
            self.lstm_engine.load_buffer(self.df_train)

        interval = self.config.REALTIME.check_interval_seconds

        try:
            while True:
                # 1. AMBIL SEMUA DATA MENTAH BARU (BMKG + VRP + INJECT)
                # Dapatkan data mentah dari semua sumber (sudah di-VRP-merge di StreamManager)
                df_mirova_raw, df_bmkg_synced, df_inj_synced = self.sensor_manager.get_realtime_data()
                
                # Gabungkan hanya BMKG dan INJECTION (karena Mirova sudah di-merge)
                frames = []
                if not df_bmkg_synced.empty: frames.append(df_bmkg_synced)
                if not df_inj_synced.empty: frames.append(df_inj_synced)
                
                if not frames:
                    time.sleep(interval)
                    continue
                
                df_raw_new_stream = pd.concat(frames, ignore_index=True)
                
                # ----------------------------------------------------
                # 1. GABUNGKAN DENGAN HISTORY UNTUK KONTEKS FE (LAG/ROLLING)
                df_hist = self.lstm_engine.get_buffer()
                df_combined_for_fe = pd.concat([df_hist, df_raw_new_stream], ignore_index=True)
                
                # 2. FEATURE ENGINEERING (pada seluruh combined data)
                df_proc, _ = self.feature_engineer.run(
                    df_combined_for_fe, is_training=False, preprocessor=self.feature_preprocessor
                )
                
                # 3. FILTER HANYA EVENT BARU (TIDAK ADA LAG) & LOKASI
                
                # Filter A: Ambil baris terbaru yang memiliki Acquired_Date > dari data history terlama
                last_hist_date = df_hist['Acquired_Date'].max() if not df_hist.empty and 'Acquired_Date' in df_hist.columns else pd.to_datetime('1900-01-01')
                # Kita harus mencari baris yang memiliki Acquired_Date yang lebih besar dari history TERAKHIR
                df_target_raw = df_proc[df_proc['Acquired_Date'] > last_hist_date].copy()
                
                if df_target_raw.empty:
                    self.logger.info("Tidak ada gempa baru teridentifikasi (atau sudah ada di buffer).")
                    time.sleep(interval)
                    continue
                    
                # Filter B: Geografis (Hanya Cluster Aktif)
                valid_clusters = self.lstm_engine.vault.list_clusters()
                df_target = df_target_raw[df_target_raw['cluster_id'].isin(valid_clusters)]
                
                if df_target.empty:
                    self.logger.info("Gempa baru terdeteksi, tetapi di luar area fokus yang dilatih. Skip prediksi.")
                    time.sleep(interval)
                    continue

                self.logger.info(f"🚨 GEMPA BARU TERDETEKSI: {len(df_target)}")

                # 4. PREDICION FLOW 
                df_pred, anomalies = self.lstm_engine.process_live_stream(df_target)
                df_pred = self.cnn_engine.predict(df_pred, self.lstm_engine)
                df_pred, _ = self.nb_engine.evaluate(df_pred)

                # =========================
                # SEMI-HYBRID FEEDBACK LOOP
                # =========================

                if 'actual_event' in df_pred.columns:
                    df_actual = df_pred[df_pred['actual_event'] == 1].copy()

                    if not df_actual.empty:
                        self.logger.info(f"[FEEDBACK] Actual events detected: {len(df_actual)}")

                        # 1️⃣ RECORD ACTUAL KE LSTM (INI WAJIB ADA)
                        self.lstm_engine.record_actual_events(df_actual)

                        # 2️⃣ EVALUASI ERROR CNN
                        if hasattr(self.cnn_engine, "evaluate_error"):
                            cnn_error = self.cnn_engine.evaluate_error(df_actual)
                            self.logger.info(f"[CNN] Live error: {cnn_error:.4f}")

                            # 3️⃣ RETRAIN JIKA SALAH
                            if cnn_error > self.config.CNN_ENGINE.max_error_threshold:
                                self.logger.warning("[CNN] Retraining due to live mismatch...")
                                self.cnn_engine.train(
                                    self.lstm_engine.get_buffer(),
                                    self.lstm_engine
                                )

               # =====================================
                # 5. UPDATE BUFFER (CONFIRMED ONLY)
                # =====================================

                # Update buffer hanya event yang cukup valid
                confirmed_cols = ['actual_event', 'confidence']

                if all(c in df_pred.columns for c in confirmed_cols):
                    df_confirmed = df_pred[df_pred['confidence'] >= 0.7].copy()
                    self.logger.info(
                        f"[BUFFER] Confirmed events: {len(df_confirmed)} / {len(df_pred)}"
                    )
                else:
                    # Fallback: jika belum ada mekanisme confidence
                    self.logger.warning(
                        "[BUFFER] Kolom confidence belum tersedia → fallback update semua (TEMP)"
                    )
                    df_confirmed = df_pred.copy()

                # Update buffer LSTM dengan data terkonfirmasi
                if not df_confirmed.empty:
                    self.lstm_engine.update_buffer(df_confirmed)

                # Persist buffer
                df_current_buffer = self.lstm_engine.get_buffer()
                safe_save_csv("output/realtime/processed.csv", df_current_buffer)

                time.sleep(interval)


        except KeyboardInterrupt:
            self.logger.info("Monitoring dihentikan. Melakukan penyimpanan akhir...")
            
            # Simpan buffer akhir saat dihentikan
            df_final_buffer = self.lstm_engine.get_buffer()
            safe_save_csv("output/realtime/processed.csv", df_final_buffer)
            
            self.logger.info("Penyimpanan buffer live selesai.")


    # ----------------------------------------------------------------------
    # MAIN PIPELINE RUN WRAPPER
    # ----------------------------------------------------------------------

    def run(self):
        start_ts = time.time()
        self.logger.info("Starting Pipeline Execution...")

        try:
            if self.config.PIPELINE.run_data_loading:
                if not self._step_load_and_split_data():
                    return

            if self.config.PIPELINE.run_feature_engineering:
                if not self._step_feature_engineering(is_training=True):
                    return

            if self.config.PIPELINE.run_model_training:
                self._run_training_flow()

            if self.config.PIPELINE.run_model_evaluation:
                df_final, metrics, anomalies = self._run_evaluation_flow()
                if self.config.PIPELINE.run_reporting:
                    self.reporter.run(df_final, metrics, anomalies)

            # NEW: Realtime inference pipeline (opsional, tergantung config)
            if hasattr(self.config.REALTIME, "enable_realtime_inference") and \
               self.config.REALTIME.enable_realtime_inference:
                self.run_realtime_inference()

            if self.config.REALTIME.enable_monitoring:
                self.start_monitoring_loop()

        except Exception as e:
            self.logger.critical(f"PIPELINE FAILED: {e}", exc_info=True)
        finally:
            end_ts = time.time()
            self.logger.info(f"System shutdown. Uptime: {end_ts - start_ts:.2f}s")


# ==============================================================================
# CLI ARGUMENT SETUP
# ==============================================================================

def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VolcanoAI Titan System")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--no-ga", action="store_true")
    parser.add_argument("--no-aco", action="store_true")
    parser.add_argument("--no-reporting", action="store_true")
    return parser


def configure_pipeline_from_args(args: argparse.Namespace, config: ProjectConfig):
    if args.skip_training:
        config.PIPELINE.run_model_training = False
    if args.eval_only:
        config.PIPELINE.run_model_training = False
        config.PIPELINE.run_data_loading = False
        config.PIPELINE.run_feature_engineering = False
    if args.no_ga:
        config.PIPELINE.run_ga_engine = False
    if args.no_aco:
        config.PIPELINE.run_aco_engine = False
    if args.no_reporting:
        config.PIPELINE.run_reporting = False


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    if not SYSTEM_READY:
        print("System tidak siap. Ada modul yang hilang.")
        return

    os.makedirs(CONFIG.OUTPUT.directory, exist_ok=True)
    setup_logging(CONFIG.OUTPUT.directory)

    logging.info("=" * 80)
    logging.info("  VOLCANOAI SYSTEM STARTUP  ".center(80))
    logging.info("=" * 80)

    configure_pipeline_from_args(args, CONFIG)

    pipeline = VolcanoAiPipeline(CONFIG)
    pipeline.run()


if __name__ == "__main__":
    main()