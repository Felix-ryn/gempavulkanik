# VolcanoAI/engines/naive_bayes_engine.py
# -- coding: utf-8 --

import os
import logging
import pickle
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import json 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# ==========================
# SKLEARN CORE
# ==========================
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    LabelEncoder,
    label_binarize
)
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

try:
    from ..config.config import NaiveBayesEngineConfig
except ImportError:
    pass

# ==========================
# LOGGER
# ==========================
logger = logging.getLogger("VolcanoAI.NaiveBayesEngine")
logger.addHandler(logging.NullHandler())


# ============================================================
# PREPROCESSOR
# ============================================================
class ClassificationPreprocessor:
    """
    Bertugas memproses feature hasil pipeline ACO → GA → LSTM → CNN
    sebelum dievaluasi oleh Naive Bayes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_in: List[str] = self.config.get("features", [])
        self.k_best = self.config.get("k_best_features", 7)

        self.pipeline: Optional[Pipeline] = None
        self.selected_features: List[str] = []

    def fit(self, df: pd.DataFrame, target: Any):
        valid_features = [f for f in self.features_in if f in df.columns]
        if not valid_features:
            raise ValueError("Tidak ada fitur valid yang ditemukan.")

        X = df[valid_features].copy()

        if isinstance(target, np.ndarray):
            temp_target = pd.Series(target)
        else:
            temp_target = target

        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", PowerTransformer(method="yeo-johnson")),
            ("selector", SelectKBest(f_classif, k="all"))
        ]

        actual_k = min(self.k_best, X.shape[1])
        actual_k = max(actual_k, 1)
        steps[2] = ("selector", SelectKBest(f_classif, k=actual_k))

        self.pipeline = Pipeline(steps)

        if temp_target.nunique() <= 1:
            logger.warning("Target hanya 1 kelas, selector dilewati.")
            self.pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
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
            logger.warning("Pipeline belum di-fit.")
            return None

        X = df.copy()

        for f in self.features_in:
            if f not in X.columns:
                X[f] = np.nan

        X = X[self.features_in]

        try:
            return self.pipeline.transform(X)
        except Exception as e:
            logger.error(f"Transform gagal: {e}")
            out_dim = len(self.selected_features) or len(self.features_in)
            return np.zeros((len(df), out_dim))

    def fit_transform(self, df: pd.DataFrame, target: Any) -> np.ndarray:
        self.fit(df, target)
        return self.transform(df)


# ============================================================
# EVALUATOR (NUMERIC-SAFE)
# ============================================================
class ModelEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.label_indices = list(range(len(class_names)))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, Any]:

        metrics = {"class_names": self.class_names}

        try:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)

            metrics["confusion_matrix"] = confusion_matrix(
                y_true,
                y_pred,
                labels=self.label_indices
            )

            metrics["report_str"] = classification_report(
                y_true,
                y_pred,
                target_names=self.class_names,
                zero_division=0
            )

            if y_prob is not None:
                metrics["roc_auc"] = self._calculate_roc_auc(y_true, y_prob)

            logger.info("\n=== Classification Report ===\n" + metrics["report_str"])

        except Exception as e:
            logger.error(f"Evaluasi gagal: {e}")

        return metrics

    def _calculate_roc_auc(self, y_true, y_prob) -> Dict[str, Any]:
        y_bin = label_binarize(y_true, classes=self.label_indices)
        n_classes = len(self.class_names)

        fpr, tpr, roc_auc = {}, {}, {}

        for i in range(n_classes):
            if i >= y_prob.shape[1]:
                continue
            if y_bin[:, i].sum() == 0:
                roc_auc[self.class_names[i]] = 0.0
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

    def _plot_cm(self, cm, classes):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()


# ============================================================
# NAIVE BAYES ENGINE (FINAL)
# ============================================================
class NaiveBayesEngine:
    """
    Evaluator akhir pipeline:
    ACO → GA → LSTM → CNN → Naive Bayes
    """

    def __init__(self, config:d Any):
        self.config = config.__dict__ if not isinstance(config, dict) else config

        self.output_dir = self.config.get("output_dir", "output/naive_bayes_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.target_col = self.config.get("target_column", "impact_level")
        self.class_names = self.config.get("class_names", ["Ringan", "Sedang", "Parah"])

        self.model_path = os.path.join(self.output_dir, "naive_bayes_model.pkl")
        self.preproc_path = os.path.join(self.output_dir, "preprocessor.pkl")
        self.le_path = os.path.join(self.output_dir, "label_encoder.pkl")

        self.le = LabelEncoder()
        self.model: Optional[GaussianNB] = None
        self.preprocessor = ClassificationPreprocessor(self.config)

        self.evaluator = ModelEvaluator(self.class_names)
        self.reporter = ClassificationReporter(self.output_dir)

    def train(self, df_train: pd.DataFrame) -> bool:
        if self.target_col not in df_train.columns:
            logger.error("Kolom target tidak ditemukan.")
            return False

        y_raw = df_train[self.target_col]

        try:
            y_encoded = self.le.fit_transform(y_raw)
        except Exception as e:
            logger.error(f"Label encoding gagal: {e}")
            return False

        X_processed = self.preprocessor.fit_transform(df_train, y_encoded)
        if X_processed is None:
            return False

        # ==========================
        # NAIVE BAYES (FINAL MODEL)
        # ==========================
        self.model = GaussianNB()
        self.model.fit(X_processed, y_encoded)

        self._save_artifacts()

        logger.info("Training Naive Bayes selesai.")
        return True

    def evaluate(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.model is None:
            if not self._load_artifacts():
                logger.error("Model tidak ditemukan.")
                return df_test, {}

        df_out = df_test.copy()
        y_true_raw = df_out.get(self.target_col)

        X_processed = self.preprocessor.transform(df_out)
        if X_processed is None:
            X_processed = np.zeros((len(df_out), len(self.preprocessor.selected_features) or 1))

        preds = self.model.predict(X_processed)
        probs = self.model.predict_proba(X_processed)

        df_out["kelas_prediksi"] = self.le.inverse_transform(preds)

        metrics = {}
        # Jika kolom target ada, evaluasi seperti biasa
        if y_true_raw is not None:
            try:
                y_true_encoded = self.le.transform(y_true_raw)
                metrics = self.evaluator.evaluate(y_true_encoded, preds, probs)
                self.reporter.generate_plots(metrics)
            except Exception as e:
                logger.warning(f"Evaluasi gagal: {e}")

        # Simpan output selalu, walau metrics kosong
        self._save_outputs(df_out, metrics)

        return df_out, metrics

    # ============================================================
    # SAVE OUTPUT FILES
    # ============================================================
    def _save_outputs(
        self,
        df_out: pd.DataFrame,
        metrics: Dict[str, Any]
    ):
        """
        Menyimpan hasil prediksi & evaluasi Naive Bayes ke file
        """

        import json  # pastikan import json ada di sini atau di awal file

        # ==========================
        # 1. Simpan hasil prediksi
        # ==========================
        pred_path = os.path.join(self.output_dir, "naive_bayes_predictions.csv")
        df_out.to_csv(pred_path, index=False)

        # ==========================
        # 2. Simpan metrics ke JSON (numpy-safe)
        # ==========================
        def _convert_numpy(obj):
            """Konversi np.ndarray rekursif menjadi list agar JSON-safe"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_convert_numpy(v) for v in obj]
            else:
                return obj

        metrics_safe = _convert_numpy(metrics)
        metrics_path = os.path.join(self.output_dir, "naive_bayes_metrics.json")

        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_safe, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Gagal tulis metrics JSON: {e}")
            # fallback: tulis representasi string supaya file tidak kosong
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write(repr(metrics_safe))
            except Exception as e2:
                logger.error(f"Fallback penulisan metrics juga gagal: {e2}")

        # ==========================
        # 3. Simpan classification report text
        # ==========================
        if "report_str" in metrics:
            report_path = os.path.join(self.output_dir, "classification_report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(metrics["report_str"])

        logger.info("Output Naive Bayes berhasil disimpan.")


    def _save_artifacts(self):
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)
        with open(self.preproc_path, "wb") as f:
            pickle.dump(self.preprocessor, f)
        with open(self.le_path, "wb") as f:
            pickle.dump(self.le, f)

    def _load_artifacts(self) -> bool:
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.preproc_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            with open(self.le_path, "rb") as f:
                self.le = pickle.load(f)
            return True
        except Exception:
            return False
