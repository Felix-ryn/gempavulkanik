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
    from VolcanoAI.processing.realtime_sensor_manager import RealtimeSensorManager
    from VolcanoAI.processing.realtime_buffer_manager import RealtimeBufferManager

    from VolcanoAI.engines.aco_engine import DynamicAcoEngine
    from VolcanoAI.engines.ga_engine import GaEngine
    from VolcanoAI.engines.lstm_engine import LstmEngine
    from VolcanoAI.engines.cnn_engine import CnnEngine
    from VolcanoAI.engines.naive_bayes_engine import NaiveBayesEngine
    from VolcanoAI.engines.cnn_map_generator import CNNMapGenerator
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
            mirova_log_path="output/realtime/"
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
                from VolcanoAI.processing.preprocess_eq import preprocess_earthquake_data
                df_full = preprocess_earthquake_data(df_full)
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
        # 1️⃣ FILE TERKINI (UNTUK TML)
        # ===============================
        latest_path = out_dir / "cnn_predictions_latest.csv"
        df_processed[export_cols].to_csv(latest_path, index=False)

        self.logger.info(f"✅ CNN latest overwritten: {latest_path}")

        # ===============================
        # 3️⃣ GENERATE CNN MAP (FOLIUM)
        # ===============================
        try:
            cnn_json_path = Path(self.config.OUTPUT.directory) / "cnn_results" / "cnn_predictions_latest.json"
            cnn_output_dir = Path(self.config.OUTPUT.directory) / "cnn_results"

            if cnn_json_path.exists():
                cnn_map_gen = CNNMapGenerator(output_dir=cnn_output_dir)
                map_path = cnn_map_gen.generate(json_path=cnn_json_path)

                if map_path:
                    self.logger.info(f"🗺️ CNN map generated: {map_path}")
                else:
                    self.logger.warning("CNN map not generated (next_event missing)")
            else:
                self.logger.warning(f"CNN JSON not found: {cnn_json_path}")

        except Exception as e:
            self.logger.exception(f"Failed to generate CNN map: {e}")

        
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

            if getattr(self.config.PIPELINE, "run_model_evaluation", True):
                df_final, metrics, anomalies = self._run_evaluation_flow()
            else:
                df_final, metrics, anomalies = None, {}, None


            # 🔥 BARU REPORTING
            if self.config.PIPELINE.run_reporting and df_final is not None:
                from pathlib import Path
                outdir = Path(self.config.OUTPUT.directory)

                normalized = {}

                aco_json = outdir / "aco_results" / "aco_to_ga.json"
                if aco_json.exists():
                    try:
                        j = json.loads(aco_json.read_text(encoding="utf-8"))
                        lat = j.get("center_lat")
                        lon = j.get("center_lon")
                        if lat is not None and lon is not None:
                            normalized["aco_center"] = f"{lat}, {lon}"
                        normalized["aco_area"] = j.get("impact_area_km2")
                        normalized["aco_map"] = str(outdir / "aco_results" / "aco_impact_zones.html")
                    except Exception:
                        pass

                    # 2) GA map
                    ga_map = outdir / "ga_results" / "ga_path_map.html"
                    if ga_map.exists():
                        normalized["ga_map"] = str(ga_map)

                    # 3) LSTM CSV files (look for common filenames)
                    lstm_dir = outdir / "lstm_results"
                    if lstm_dir.exists():
                        for f in ["lstm_records_2y_20241230.csv", "master.csv"]:
                            p = lstm_dir / f
                            if p.exists():
                                normalized["lstm_master_csv"] = str(p)
                                break
                        for f in ["lstm_recent_15d_20241230.csv", "recent.csv"]:
                            p = lstm_dir / f
                            if p.exists():
                                normalized["lstm_recent_csv"] = str(p)
                                break
                        for f in ["lstm_anomalies_20241230.csv", "anomalies.csv"]:
                            p = lstm_dir / f
                            if p.exists():
                                normalized["lstm_anomalies_csv"] = str(p)
                                break

                    # 4) CNN outputs
                    cnn_latest = outdir / "cnn_results" / "results" / "cnn_predictions_latest.csv"
                    if cnn_latest.exists():
                        normalized["cnn_pred_csv"] = str(cnn_latest)
                    cnn_json = outdir / "cnn_results" / "cnn_predictions_latest.json"
                    if cnn_json.exists():
                        normalized["cnn_pred_json"] = str(cnn_json)

                    # 5) NaiveBayes outputs (support multiple candidate dirs & files)
                    nb_candidates = [
                                outdir / "naive_bayes",           # legacy
                                outdir / "naive_bayes_results",
                                outdir / "naive_bayes_outputs"
                    ]
                    nb_found = None
                    for d in nb_candidates:
                        if d.exists():
                            nb_found = d
                            break

                    if nb_found:
                        p1 = nb_found / "confusion_matrix.png"
                        p2 = nb_found / "roc_curves.png"
                        p_csv = nb_found / "naive_bayes_predictions.csv"
                        p_json = nb_found / "naive_bayes_metrics.json"
                        if p1.exists(): normalized["nb_confusion_png"] = str(p1)
                        if p2.exists(): normalized["nb_roc_png"] = str(p2)
                        if p_csv.exists(): normalized["nb_pred_csv"] = str(p_csv)
                        if p_json.exists(): normalized["nb_metrics_json"] = str(p_json)


                    # 6) combine: normalized keys override only when not present in original metrics
                    final_metrics = {}
                    if isinstance(metrics, dict):
                        final_metrics.update(metrics)
                    # set defaults where missing
                    for k, v in normalized.items():
                        final_metrics.setdefault(k, v)

                    # optional: log keys for debug
                    self.logger.info(f"Reporter metrics keys being passed: {list(final_metrics.keys())}")

                    # main.py sebelum self.reporter.run()
                    print("=== COLUMNS df_final ===")
                    print(df_final.columns)
                    print("=== LAST ROW df_final ===")
                    print(df_final.tail(1))

                    # 7) run reporter
                    self.reporter.run(df_final, final_metrics, anomalies)



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

# === START: Flask dashboard integration (paste di akhir main.py) ===
import threading
import webbrowser
from pathlib import Path
from flask import Flask, send_from_directory, Response
from jinja2 import Template

# konfigurasi default
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
# lokasi template file kamu (sudah kamu sebutkan)
TEMPLATE_PATH = Path("VolcanoAI/reporting/templates/monitor_live_template.html")
PROJECT_ROOT = Path.cwd()

def _file_url_for(path: Path) -> str:
    """
    Convert file-system path to URL served by /files/... route.
    keeps path relative to project root.
    """
    try:
        rel = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except Exception:
        rel = path
    return f"/files/{rel.as_posix()}"

def build_dashboard_context(output_dir: Path) -> dict:
    """
    Baca file-file output dan isi context dict untuk menggantikan placeholder template.
    Gunakan safe defaults bila file tidak ada.
    """
    ctx = {
        "TIMESTAMP": datetime.now().isoformat(sep=' '),
        "ACO_IMPACT_CENTER": "N/A",
        "ACO_IMPACT_AREA": "N/A",
        "ACO_MAP": "#",
        "GA_MAP": "#",
        "GA_PRED_LAT": "N/A",
        "GA_PRED_LON": "N/A",
        "GA_BEARING": "N/A",
        "GA_DISTANCE": "N/A",
        "GA_CONFIDENCE": "N/A",
        "LATEST_ROW_HTML": "<em>No data yet</em>",
        "LSTM_MASTER_CSV": "#",
        "LSTM_MASTER_FILENAME": "N/A",
        "LSTM_RECENT_CSV": "#",
        "LSTM_RECENT_FILENAME": "N/A",
        "LSTM_ANOMALIES_CSV": "#",
        "LSTM_ANOMALIES_FILENAME": "N/A",
        "CNN_PRED_CSV": "#",
        "CNN_PRED_JSON": "#",
        "CNN_IMAGE_LIST_HTML": "",
        "NB_METRICS_HTML": "<em>Not available</em>",
        "NB_CONFUSION_PNG": "#",
        "NB_ROC_PNG": "#",
        "NB_REPORT_STR": "N/A"
    }

    out = output_dir

    # ACO json
    aco_json = out / "aco_results" / "aco_to_ga.json"
    if aco_json.exists():
        try:
            j = json.loads(aco_json.read_text(encoding="utf-8"))
            lat = j.get("center_lat")
            lon = j.get("center_lon")
            if lat is not None and lon is not None:
                ctx["ACO_IMPACT_CENTER"] = f"{lat}, {lon}"
            ctx["ACO_IMPACT_AREA"] = j.get("impact_area_km2", ctx["ACO_IMPACT_AREA"])
            aco_map_file = out / "aco_results" / "aco_impact_zones.html"
            if aco_map_file.exists():
                ctx["ACO_MAP"] = _file_url_for(aco_map_file)
        except Exception:
            pass

    # GA map (path json produced earlier)
    ga_map = out / "ga_results" / "ga_path_map.html"
    if ga_map.exists():
        ctx["GA_MAP"] = _file_url_for(ga_map)

    # try to read GA predicted fields (if aco_to_ga.json contains next_event)
    try:
        if aco_json.exists():
            j = json.loads(aco_json.read_text(encoding="utf-8"))
            nxt = j.get("next_event") or {}
            ctx["GA_PRED_LAT"] = nxt.get("lat", ctx["GA_PRED_LAT"])
            ctx["GA_PRED_LON"] = nxt.get("lon", ctx["GA_PRED_LON"])
            ctx["GA_BEARING"] = nxt.get("direction_deg", ctx["GA_BEARING"])
            ctx["GA_DISTANCE"] = nxt.get("distance_km", ctx["GA_DISTANCE"])
            ctx["GA_CONFIDENCE"] = nxt.get("confidence", ctx["GA_CONFIDENCE"])
    except Exception:
        pass

    # LSTM CSVs
    lstm_dir = out / "lstm_results"
    if lstm_dir.exists():
        for f in ["lstm_records_2y_20241230.csv", "master.csv"]:
            p = lstm_dir / f
            if p.exists():
                ctx["LSTM_MASTER_CSV"] = _file_url_for(p)
                ctx["LSTM_MASTER_FILENAME"] = p.name
                break
        for f in ["lstm_recent_15d_20241230.csv", "recent.csv"]:
            p = lstm_dir / f
            if p.exists():
                ctx["LSTM_RECENT_CSV"] = _file_url_for(p)
                ctx["LSTM_RECENT_FILENAME"] = p.name
                break
        for f in ["lstm_anomalies_20241230.csv", "anomalies.csv"]:
            p = lstm_dir / f
            if p.exists():
                ctx["LSTM_ANOMALIES_CSV"] = _file_url_for(p)
                ctx["LSTM_ANOMALIES_FILENAME"] = p.name
                break

    # CNN predictions latest CSV / JSON and latest row for LATEST_ROW_HTML
    cnn_latest = out / "cnn_results" / "results" / "cnn_predictions_latest.csv"
    if cnn_latest.exists():
        ctx["CNN_PRED_CSV"] = _file_url_for(cnn_latest)
        try:
            df = pd.read_csv(cnn_latest, parse_dates=["Acquired_Date"])
            if not df.empty:
                last = df.tail(1)
                # render a small HTML table with last row
                ctx["LATEST_ROW_HTML"] = last.to_html(index=False, classes="table", border=0)
        except Exception:
            pass

    cnn_json = out / "cnn_results" / "cnn_predictions_latest.json"
    if cnn_json.exists():
        ctx["CNN_PRED_JSON"] = _file_url_for(cnn_json)
    
    cnn_img = out / "cnn_results" / "cnn_prediction_map.png"

    if cnn_img.exists():
        ctx["CNN_IMAGE_LIST_HTML"] = f"""
            <img src="{_file_url_for(cnn_img)}"
                 class="plot"
                 alt="CNN Prediction Map">
        """
    else:
        ctx["CNN_IMAGE_LIST_HTML"] = "<p class='muted'></p>"

    # CNN MAP (HTML)
    cnn_map = out / "cnn_results" / "cnn_prediction_map.html"
    if cnn_map.exists():
        ctx["CNN_MAP"] = _file_url_for(cnn_map)
    else:
        ctx["CNN_MAP"] = "#"

    # NaiveBayes outputs (support multiple candidate dirs & files)
    nb_candidates = [
        out / "naive_bayes",           # legacy
        out / "naive_bayes_results",   # earlier assistant default
        out / "naive_bayes_outputs"    # any other naming variant
    ]

    nb_found = None
    for cand in nb_candidates:
        if cand.exists():
            nb_found = cand
            break

    if nb_found:
        p1 = nb_found / "confusion_matrix.png"
        p2 = nb_found / "roc_curves.png"
        p_csv = nb_found / "naive_bayes_predictions.csv"
        p_json = nb_found / "naive_bayes_metrics.json"
        p_report = nb_found / "classification_report.txt"

        if p1.exists():
            ctx["NB_CONFUSION_PNG"] = _file_url_for(p1)
        if p2.exists():
            ctx["NB_ROC_PNG"] = _file_url_for(p2)
        if p_csv.exists():
            ctx["CNN_PRED_CSV"] = ctx.get("CNN_PRED_CSV", "#")  # avoid clobbering CNN keys
            # add a separate key for NB predictions if you want:
            ctx["NB_PRED_CSV"] = _file_url_for(p_csv)
        if p_json.exists():
            try:
                mj = json.loads(p_json.read_text(encoding="utf-8"))
                # convert metrics json to a small HTML table (reuse existing style)
                rows = ["<table>"]
                for k, v in mj.items():
                    rows.append(f"<tr><th style='text-align:left;padding:6px'>{k}</th><td style='padding:6px'>{v}</td></tr>")
                rows.append("</table>")
                ctx["NB_METRICS_HTML"] = "\n".join(rows)
            except Exception:
                ctx["NB_METRICS_HTML"] = "<em>Metrics JSON exists but failed to parse</em>"
        if p_report.exists():
            try:
                ctx["NB_REPORT_STR"] = p_report.read_text(encoding="utf-8")
            except Exception:
                pass


    # If reporter generated textual metrics in a JSON, try reading it (optional)
    metrics_json = output_dir / "report_metrics.json"
    if metrics_json.exists():
        try:
            mj = json.loads(metrics_json.read_text(encoding="utf-8"))
            # simple conversion to small HTML table
            rows = ["<table>"]
            for k, v in mj.items():
                rows.append(f"<tr><th style='text-align:left;padding:6px'>{k}</th><td style='padding:6px'>{v}</td></tr>")
            rows.append("</table>")
            ctx["NB_METRICS_HTML"] = "\n".join(rows)
        except Exception:
            pass

    return ctx

# Flask app and routes
def create_flask_app(output_dir: Path, template_path: Path) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        ctx = build_dashboard_context(output_dir=output_dir)
        # load template file as raw HTML with placeholders
        if template_path.exists():
            tpl_text = template_path.read_text(encoding="utf-8")
            tmpl = Template(tpl_text)
            rendered = tmpl.render(**ctx)
            return Response(rendered, mimetype="text/html")
        else:
            return "<h3>Template not found</h3><p>Check TEMPLATE_PATH setting.</p>", 404

    @app.route("/files/<path:filename>")
    def serve_file(filename):
        # Serve files from project root to allow iframe loading of output HTML/CSV/PNG
        safe_path = Path(filename)
        # disallow path traversal outside project root
        full = (PROJECT_ROOT / safe_path).resolve()
        try:
            full.relative_to(PROJECT_ROOT.resolve())
        except Exception:
            return "Forbidden", 403
        if not full.exists():
            return "Not found", 404
        return send_from_directory(PROJECT_ROOT, safe_path.as_posix(), conditional=True)

    return app

def start_flask_in_thread(app: Flask, host="127.0.0.1", port=5000, open_browser=True):
    def _run():
        # disable debug + reloader for thread-safety
        if open_browser:
            try:
                webbrowser.open_new_tab(f"http://{host}:{port}")
            except Exception:
                pass
        app.run(host=host, port=port, debug=False, use_reloader=False)

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th

# Modify main() behavior: start flask either before pipeline (when enable_monitoring) or after
def main_with_dashboard():
    parser = create_arg_parser()
    args = parser.parse_args()

    if not SYSTEM_READY:
        print("System tidak siap. Ada modul yang hilang.")
        return

    os.makedirs(CONFIG.OUTPUT.directory, exist_ok=True)
    setup_logging(CONFIG.OUTPUT.directory)

    from pathlib import Path
    out = Path(CONFIG.OUTPUT.directory)
    out.mkdir(parents=True, exist_ok=True)  # pastikan folder utama ada


    logging.info("=" * 80)
    logging.info("  VOLCANOAI SYSTEM STARTUP  ".center(80))
    logging.info("=" * 80)

    configure_pipeline_from_args(args, CONFIG)

    # Create Flask app (server will serve files from project root)
    app = create_flask_app(output_dir=Path(CONFIG.OUTPUT.directory), template_path=TEMPLATE_PATH)

    # If monitoring loop will run (blocking), start Flask in background first
    enable_monitoring = getattr(CONFIG.REALTIME, "enable_monitoring", False)
    if enable_monitoring:
        logging.info("[Dashboard] Starting Flask server in background thread (monitoring enabled)...")
        start_flask_in_thread(app, host=FLASK_HOST, port=FLASK_PORT, open_browser=True)

    pipeline = VolcanoAiPipeline(CONFIG)
    pipeline.run()

    # If monitoring not enabled, start Flask AFTER pipeline finishes
    if not enable_monitoring:
        logging.info("[Dashboard] Pipeline finished — launching Flask server (dashboard)...")
        start_flask_in_thread(app, host=FLASK_HOST, port=FLASK_PORT, open_browser=True)
        # Keep main thread alive to keep server running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("[Dashboard] KeyboardInterrupt — exiting.")

# Replace the original main with main_with_dashboard when running as script
# (so existing behavior is preserved but with dashboard)
if __name__ == "__main__":
    main_with_dashboard()
# === END: Flask dashboard integration ===
