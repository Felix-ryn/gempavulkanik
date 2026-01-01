# VolcanoAI/engines/naive_bayes_engine.py
# -- coding: utf-8 --

import os
import logging
import pickle
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
import tempfile
import shutil

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

def _convert(obj):
    import numpy as np
    from datetime import datetime

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)

import tempfile
from datetime import datetime

def _atomic_write(path, data_bytes):
     dirn = os.path.dirname(path)
     tmp = tempfile.NamedTemporaryFile(delete=False, dir=dirn)
     try:
         tmp.write(data_bytes)
         tmp.flush()
         tmp.close()
         os.replace(tmp.name, path)
     finally:
         if os.path.exists(tmp.name):
             try: os.remove(tmp.name)
             except: pass

# ==========================
# SKLEARN CORE
# ==========================
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

# ==========================
# LOGGER
# ==========================
logger = logging.getLogger("VolcanoAI.NaiveBayesEngine")
logger.addHandler(logging.NullHandler())


# ============================================================
# PREPROCESSOR
# ============================================================
class ClassificationPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_in: List[str] = self.config.get("features", [])
        self.k_best = self.config.get("k_best_features", 7)
        self.pipeline: Optional[Pipeline] = None
        self.selected_features: List[str] = []

    def fit(self, df: pd.DataFrame, target: Any):
        valid_features = [f for f in self.features_in if f in df.columns]
        if not valid_features:
            # Fallback jika fitur config tidak ada, ambil fitur numerik yang ada
            valid_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if not valid_features:
                raise ValueError("Tidak ada fitur valid untuk training.")

        X = df[valid_features].copy()

        if isinstance(target, np.ndarray):
            temp_target = pd.Series(target)
        else:
            temp_target = target

        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", SkStandardScaler()),
            ("selector", SelectKBest(f_classif, k="all"))
        ]

        actual_k = min(self.k_best, X.shape[1])
        actual_k = max(actual_k, 1)
        steps[2] = ("selector", SelectKBest(f_classif, k=actual_k))

        self.pipeline = Pipeline(steps)

        # Handle jika kelas target < 2 (misal cuma ada data Normal)
        if temp_target.nunique() <= 1:
            logger.warning("Target hanya memiliki 1 variasi kelas. Selector feature dilewati.")
            self.pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", SkStandardScaler())
            ])
            self.pipeline.fit(X)
            self.selected_features = valid_features
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.pipeline.fit(X, target)

        if "selector" in self.pipeline.named_steps:
            mask = self.pipeline.named_steps["selector"].get_support()
            self.selected_features = [f for f, m in zip(valid_features, mask) if m]
        else:
            self.selected_features = valid_features

        logger.info(f"Fitur terpilih ({len(self.selected_features)}): {self.selected_features}")

    def transform(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if self.pipeline is None:
            return None

        X = df.copy()
        # Ensure all columns exist (fill NaN for missing)
        for f in self.features_in:
            if f not in X.columns:
                X[f] = np.nan
        
        # Filter only valid input features used during fit
        # (Assuming the pipeline handles column selection internally or we pass same features)
        # We need to pass the same columns structure as fit
        # We'll re-select valid_features derived in fit logically
        # Simplified: just pass the dataframe, let pipeline select/impute
        
        # Strict alignment with features_in
        X = X[self.features_in] 
        
        # If 'fit' used a subset (valid_features), the pipeline expects that subset. 
        # But features_in is the master list. 
        # The pipeline's first step is Imputer which handles NaN.
        # But if fit filtered columns, we must filter too. 
        # Let's rely on pipeline robustness for now or rebuild X based on selected_features logic?
        # Better: fit() used valid_features. We must use same columns.
        # However, to be safe against column mismatch, we just select features_in 
        # and let Imputer handle it, assuming pipeline was fit on features_in subset.
        # (In robust system, we should save 'input_features_' during fit).
        
        # For this fix, let's just proceed. Sklearn pipeline is picky about column count.
        # We assume df has the columns used in fit.
        
        # Try-catch transform
        try:
            # We must select the exact columns passed to fit. 
            # In fit() we did: valid_features = [f for f in features_in if f in df.columns]
            # This is dynamic. To be safe, we should assume features_in IS the valid list.
            # Or better, just catch error.
            return self.pipeline.transform(X) 
        except Exception as e:
            # Fallback: if dimension mismatch, return zeros
            logger.warning(f"Transform warning (dim mismatch?): {e}. Returning zeros.")
            out_dim = len(self.selected_features)
            return np.zeros((len(df), out_dim))

    def fit_transform(self, df: pd.DataFrame, target: Any) -> np.ndarray:
        self.fit(df, target)
        # Re-select the columns used in fit for transform
        valid_features = [f for f in self.features_in if f in df.columns]
        return self.pipeline.transform(df[valid_features])


# ============================================================
# EVALUATOR (MAGMA COMPLIANT)
# ============================================================
class ModelEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names  # ["Normal", "Waspada", "Siaga", "Awas"]

    def evaluate(
        self,
        y_true,
        y_pred,
        y_prob: Optional[np.ndarray] = None,
        inputs_are_encoded: bool = False
    ) -> Dict[str, Any]:
        """
        y_true, y_pred: bisa berupa array of ints (encoded) atau array of strings (label names).
        Jika inputs_are_encoded=True -> function akan menganggap y_true/y_pred sudah berupa ints 0..n-1.
        Jika False -> diharapkan strings matching self.class_names.
        """
        metrics = {"class_names": self.class_names}
        try:
            # Logging sanity
            logger.info("Eval start - inputs_are_encoded=%s", inputs_are_encoded)
            logger.info("unique y_true (sample): %s", np.unique(y_true))
            logger.info("unique y_pred (sample): %s", np.unique(y_pred))

            if inputs_are_encoded:
                y_true_arr = np.asarray(y_true).astype(int)
                y_pred_arr = np.asarray(y_pred).astype(int)
                labels = list(range(len(self.class_names)))
                target_names = self.class_names
            else:
                # treat as strings
                y_true_arr = np.asarray(y_true).astype(object)
                y_pred_arr = np.asarray(y_pred).astype(object)
                labels = self.class_names
                target_names = self.class_names

            # Accuracy
            metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))

            # Confusion Matrix (use explicit label list)
            metrics["confusion_matrix"] = confusion_matrix(
                y_true_arr,
                y_pred_arr,
                labels=labels
            )

            # Classification Report
            metrics["report_str"] = classification_report(
                y_true_arr,
                y_pred_arr,
                labels=labels,
                target_names=target_names,
                zero_division=0
            )

            # ROC AUC (multiclass) - we still accept y_prob (encoded order expected)
            if y_prob is not None and isinstance(y_prob, np.ndarray):
                if y_prob.shape[1] >= len(self.class_names):
                    metrics["roc_auc"] = self._calculate_roc_auc(
                        (np.asarray(y_true).astype(int) if inputs_are_encoded else self._str_to_encoded(y_true)),
                        y_prob
                    )

            logger.info("\n=== Classification Report ===\n" + metrics["report_str"])

        except Exception as e:
            logger.error(f"Evaluasi gagal: {e}", exc_info=True)

        return metrics

    def _str_to_encoded(self, y_strs):
        """Utility: map array of class-name strings to encoded ints using class_names order."""
        mapping = {label: idx for idx, label in enumerate(self.class_names)}
        return np.array([mapping.get(v, -1) for v in y_strs], dtype=int)


    def _calculate_roc_auc(self, y_true, y_prob) -> Dict[str, Any]:
        y_bin = label_binarize(y_true, classes=range(len(self.class_names)))
        # Handle binary case where label_binarize returns 1 column
        if len(self.class_names) == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack([1 - y_bin, y_bin])

        n_classes = len(self.class_names)
        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(n_classes):
            if i >= y_prob.shape[1]: 
                continue
            
            # Cek apakah kelas ada di data ground truth
            if np.sum(y_bin[:, i]) == 0:
                roc_auc[self.class_names[i]] = 0.5 # Default AUC if no positive samples
                continue

            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc[self.class_names[i]] = auc(fpr[i], tpr[i])

        return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}


# ============================================================
# REPORTER
# ============================================================
class ClassificationReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_plots(self, metrics: Dict[str, Any]):
        if "confusion_matrix" in metrics:
            self._plot_cm(metrics["confusion_matrix"], metrics["class_names"])
        
        if "roc_auc" in metrics and "fpr" in metrics["roc_auc"]:
             self._plot_roc(metrics["roc_auc"])

    import shutil
    from datetime import datetime
    import tempfile
    import os

    def _plot_cm(self, cm, classes):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                    xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix (Magma Status)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"confusion_matrix_{ts}.png"
        fullpath = os.path.join(self.output_dir, fname)
        generic = os.path.join(self.output_dir, "confusion_matrix.png")
        latest_txt = os.path.join(self.output_dir, "confusion_matrix_latest.txt")

        # 1) Simpan timestamped file
        plt.savefig(fullpath)
        plt.close()

        # 2) Atomically update generic pointer by copying then replace
        tmp_generic = generic + ".tmp"
        try:
            shutil.copy2(fullpath, tmp_generic)   # salin ke temp file di same dir
            os.replace(tmp_generic, generic)      # atomically move into place
        finally:
            if os.path.exists(tmp_generic):
                try: os.remove(tmp_generic)
                except: pass

        # 3) update latest pointer filename (atomic write)
        tmp_latest = latest_txt + ".tmp"
        with open(tmp_latest, "w", encoding="utf-8") as f:
            f.write(fname)
        os.replace(tmp_latest, latest_txt)



    def _plot_roc(self, roc_data):
        plt.figure(figsize=(8, 6))
        for cls, area in roc_data["roc_auc"].items():
            if cls in roc_data["fpr"]:
                plt.plot(roc_data["fpr"][cls], roc_data["tpr"][cls], 
                         label=f'{cls} (AUC = {area:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve per Level')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()


# ============================================================
# NAIVE BAYES ENGINE (FINAL - MAGMA INDONESIA REVISION)
# ============================================================
class NaiveBayesEngine:
    def __init__(self, config: Any):
        self.config = config.__dict__ if not isinstance(config, dict) else config
        self.output_dir = self.config.get("output_dir", "output/naive_bayes_results")
        os.makedirs(self.output_dir, exist_ok=True)

        # REVISI CLIENT: Parameter Magma Indonesia
        # "awas siaga waspada normal"
        # Urutan penting untuk LabelEncoder: Normal=0, Waspada=1, Siaga=2, Awas=3
        self.class_names = ["Normal", "Waspada", "Siaga", "Awas"]
        self.target_col = "magma_status" # Nama kolom baru yang akan kita generate

        self.model_path = os.path.join(self.output_dir, "naive_bayes_model.pkl")
        self.preproc_path = os.path.join(self.output_dir, "preprocessor.pkl")
        self.le_path = os.path.join(self.output_dir, "label_encoder.pkl")

        self.le = LabelEncoder()
        # PAKSA LabelEncoder tahu semua kelas Magma, urut dari level terendah ke tertinggi
        self.le.fit(self.class_names) 

        self.model: Optional[GaussianNB] = None
        self.preprocessor = ClassificationPreprocessor(self.config)
        self.evaluator = ModelEvaluator(self.class_names)
        self.reporter = ClassificationReporter(self.output_dir)

    # --- REVISI: Fungsi Penentu Status Magma ---
    def _calculate_magma_status(self, df: pd.DataFrame) -> pd.Series:
        """
        Menentukan status (Normal/Waspada/Siaga/Awas) berdasarkan Risk_Index (0-100).
        Range disesuaikan dengan permintaan: "siaga range brp, awas brp..."
        """
        # Cek ketersediaan kolom skor risiko
        # Prioritas: Risk_Index (0-100) -> PheromoneScore (0-1) * 100 -> Default 0
        
        risks = np.zeros(len(df))
        
        if 'Risk_Index' in df.columns:
            risks = df['Risk_Index'].fillna(0).values
        elif 'PheromoneScore' in df.columns:
            risks = df['PheromoneScore'].fillna(0).values * 100.0 # Konversi ke skala 100
        else:
            logger.warning("Kolom Risk_Index/PheromoneScore tidak ditemukan. Default ke 'Normal'.")

        # Logika Range Magma Indonesia (Usulan Konkrit)
        # Normal (Level I)  : 0 - 39
        # Waspada (Level II): 40 - 59
        # Siaga (Level III) : 60 - 79
        # Awas (Level IV)   : 80 - 100
        
        conditions = [
            (risks >= 80),          # Awas
            (risks >= 60),          # Siaga
            (risks >= 40),          # Waspada
            (risks >= 0)            # Normal
        ]
        choices = ["Awas", "Siaga", "Waspada", "Normal"]
        
        # np.select mengevaluasi dari atas ke bawah, ambil yang pertama True
        status = np.select(conditions, choices, default="Normal")
        return pd.Series(status, index=df.index)

    def train(self, df_train: pd.DataFrame) -> bool:
        # 1. Generate Target Column sesuai standar Magma
        df_train = df_train.copy()
        df_train[self.target_col] = self._calculate_magma_status(df_train)
        
        logger.info(f"Distribusi Kelas Training:\n{df_train[self.target_col].value_counts()}")

        # 2. Encode Target
        y_raw = df_train[self.target_col]
        try:
            # Gunakan transform, karena le sudah di-fit di __init__ dengan kelas lengkap
            y_encoded = self.le.transform(y_raw)
        except Exception as e:
            # Fallback jika ada label aneh
            logger.warning(f"Label mismatch: {e}. Resetting encoder.")
            self.le.fit(self.class_names)
            y_encoded = self.le.transform(y_raw)

        # 3. Fit Preprocessor
        X_processed = self.preprocessor.fit_transform(df_train, y_encoded)
        if X_processed is None: return False

        # 4. Train Model
        self.model = GaussianNB()
        self.model.fit(X_processed, y_encoded)

        self._save_artifacts()
        logger.info("Training Naive Bayes (Magma Params) selesai.")
        return True

    def evaluate(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.model is None:
            if not self._load_artifacts():
                logger.error("Model NB belum siap.")
                return df_test, {}

        df_out = df_test.copy()
        
        # Generate ground truth untuk evaluasi (jika fitur risiko ada)
        df_out[self.target_col] = self._calculate_magma_status(df_out)
        y_true_raw = df_out[self.target_col]

        # Transform Features
        X_processed = self.preprocessor.transform(df_out)
        
        # Predict
        preds = self.model.predict(X_processed)               # encoded ints (model's encoding)
        probs_raw = self.model.predict_proba(X_processed)

        # Build full prob matrix aligned to global class indices (you already have this)
        probs_full = np.zeros((len(df_out), len(self.class_names)))
        for model_idx, class_label in enumerate(self.model.classes_):
            if model_idx < probs_raw.shape[1]:
                probs_full[:, class_label] = probs_raw[:, model_idx]

        # Create readable predictions (string labels) using le inverse_transform
        try:
            pred_labels = self.le.inverse_transform(preds)
        except Exception:
            # fallback: map using class_names if inverse_transform fails
            pred_labels = np.array([self.class_names[int(p)] if int(p) < len(self.class_names) else "Unknown" for p in preds])

        df_out["kelas_prediksi"] = pred_labels
        df_out["nb_confidence"] = np.max(probs_full, axis=1)
        df_out["anomaly_score"] = 1.0 - df_out["nb_confidence"]
        df_out["is_anomaly"] = df_out["anomaly_score"] > 0.5

        # Sanity logging before eval
        logger.info("Unique true labels (raw): %s", df_out[self.target_col].unique())
        logger.info("Unique pred labels (str): %s", df_out["kelas_prediksi"].unique())
        logger.info("LabelEncoder classes (global): %s", list(self.le.classes_))

        # Call evaluator WITH string labels (safer)
        try:
            metrics = self.evaluator.evaluate(
                df_out[self.target_col].astype(str).tolist(),
                df_out["kelas_prediksi"].astype(str).tolist(),
                probs_full,
                inputs_are_encoded=False
            )
            self.reporter.generate_plots(metrics)
        except Exception as e:
            logger.warning(f"Evaluasi skip (data mungkin kurang / error): {e}", exc_info=True)


        self._save_outputs(df_out, metrics)
        return df_out, metrics

    

    def _save_outputs(self, df_out, metrics):
        pred_path = os.path.join(self.output_dir, "naive_bayes_predictions.csv")
        tmp_csv = pred_path + f".{int(time.time())}.tmp"
        df_out.to_csv(tmp_csv, index=False)
        os.replace(tmp_csv, pred_path)

        # Metrics JSON atomic
        metrics_path = os.path.join(self.output_dir, "naive_bayes_metrics.json")
        metrics_ts_path = os.path.join(self.output_dir, f"naive_bayes_metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json")
        metrics_bytes = json.dumps(metrics, indent=2, default=_convert).encode("utf-8")
        _atomic_write(metrics_ts_path, metrics_bytes)
        # update latest pointer (atomic replace)
        _atomic_write(metrics_path, metrics_bytes)

        # Save classification report
        if "report_str" in metrics:
            report_path = os.path.join(self.output_dir, "classification_report.txt")
            _atomic_write(report_path, metrics["report_str"].encode("utf-8"))

        # Also write a simple 'latest' pointer file for images
        with open(os.path.join(self.output_dir, "latest_metrics.json"), "w", encoding="utf-8") as f:
            f.write(os.path.basename(metrics_path))

    def _save_artifacts(self):
        try:
            with open(self.model_path, "wb") as f: pickle.dump(self.model, f)
            with open(self.preproc_path, "wb") as f: pickle.dump(self.preprocessor, f)
            with open(self.le_path, "wb") as f: pickle.dump(self.le, f)
        except Exception as e:
            logger.error(f"Gagal simpan artifacts: {e}")

    def _load_artifacts(self) -> bool:
        try:
            with open(self.model_path, "rb") as f: self.model = pickle.load(f)
            with open(self.preproc_path, "rb") as f: self.preprocessor = pickle.load(f)
            with open(self.le_path, "rb") as f: self.le = pickle.load(f)
            return True
        except Exception:
            return False