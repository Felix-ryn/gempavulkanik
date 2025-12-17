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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, label_binarize, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize, PowerTransformer, LabelEncoder # <-- TAMBAHKAN LabelEncoder
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, log_loss
)
from sklearn.pipeline import Pipeline

try:
    from ..config.config import NaiveBayesEngineConfig
except ImportError:
    pass

logger = logging.getLogger("VolcanoAI.NaiveBayesEngine")
logger.addHandler(logging.NullHandler())

class ClassificationPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.features_in: List[str] = self.config.get('features', [])
        self.k_best = self.config.get('k_best_features', 7)
        
        self.pipeline: Optional[Pipeline] = None
        self.selected_features: List[str] = []

    def fit(self, df: pd.DataFrame, target: Any):
        valid_features = [f for f in self.features_in if f in df.columns]
        if not valid_features:
            raise ValueError("Tidak ada fitur valid yang ditemukan di DataFrame.")
            
        X = df[valid_features].copy()
        
        if isinstance(target, np.ndarray):
             temp_target = pd.Series(target)
        else:
             temp_target = target

        # Pipeline: Imputasi -> Transformasi Gaussian -> Seleksi Fitur
        steps = [
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', PowerTransformer(method='yeo-johnson')), 
            ('selector', SelectKBest(f_classif, k='all')) # Placeholder k
        ]
        
        # Adjust K dynamically
        n_feats = X.shape[1]
        actual_k = min(self.k_best, n_feats)
        if actual_k < 1: actual_k = 1
        
        steps[2] = ('selector', SelectKBest(f_classif, k=actual_k))
        
        self.pipeline = Pipeline(steps)
        
        # Handle single class target edge case
        if temp_target.nunique() <= 1:
            logger.warning("Target hanya memiliki 1 kelas. Skip fitting selector.")
            steps = [('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]
            self.pipeline = Pipeline(steps)
            self.pipeline.fit(X)
            self.selected_features = valid_features
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit pipeline dengan target asli (string/array)
            self.pipeline.fit(X, target)
            
        if 'selector' in self.pipeline.named_steps:
            mask = self.pipeline.named_steps['selector'].get_support()
            self.selected_features = [f for f, m in zip(valid_features, mask) if m]
        else:
            self.selected_features = valid_features
            
        logger.info(f"Fitur Terpilih ({len(self.selected_features)}): {self.selected_features}")

    def transform(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Melakukan transformasi data menggunakan Pipeline yang sudah dilatih (Fit).
        Menangani kasus kolom hilang secara otomatis.
        
        Args:
            df: Raw DataFrame input.
            
        Returns:
            np.ndarray: Hasil transformasi, atau array nol jika gagal.
        """
        # 1. Cek Pipeline availability
        if not hasattr(self, 'pipeline') or self.pipeline is None:
            logger.warning("[Transform] Pipeline belum dimuat atau dilatih.")
            return None
            
        # 2. Data Preparation
        X = df.copy()
        
        # Validasi: Pastikan 'features_in' ada. Ini adalah daftar nama kolom input saat training.
        if not hasattr(self, 'features_in'):
             logger.error("[Transform] 'features_in' tidak ditemukan dalam class. Transform dibatalkan.")
             return None

        # [FIX KRITIS]: Penanganan Missing Columns (Robustness)
        # Jika input data inferensi tidak memiliki kolom tertentu yang digunakan saat training,
        # kita buat kolom tersebut dan isikan dengan NaN.
        # Harapannya: Imputer di dalam self.pipeline akan menangani NaN tersebut.
        missing_cols = []
        for f in self.features_in:
            if f not in X.columns:
                X[f] = np.nan
                missing_cols.append(f)
        
        if missing_cols:
            logger.debug(f"[Transform] Kolom hilang (diisi NaN): {len(missing_cols)} kolom")

        # 3. Reordering Columns
        # Wajib mengurutkan kolom agar persis sama dengan urutan saat training
        X = X[self.features_in] 
        
        # 4. Execution & Error Handling
        try:
            # Jalankan pipeline transform
            return self.pipeline.transform(X)
            
        except ValueError as e:
            logger.error(f"[Transform] Value Error saat pipeline.transform: {e}")
            logger.warning("--> Mengembalikan array Zeros sebagai fallback.")
            
            # Tentukan output dimension. 
            # Jika punya atribut 'selected_features', gunakan panjangnya (output size).
            # Jika tidak, gunakan 'features_in' (input size - asusmi scaling saja tanpa seleksi fitur).
            n_feats_out = len(self.selected_features) if hasattr(self, 'selected_features') else len(self.features_in)
            
            return np.zeros((len(df), n_feats_out))
            
        except Exception as e:
            logger.critical(f"[Transform] Unexpected Error: {e}")
            return None

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> np.ndarray:
        self.fit(df, target)
        return self.transform(df)


class ModelEvaluator:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        metrics = {"class_names": self.class_names}
        try:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=self.class_names)
            metrics["report_str"] = classification_report(y_true, y_pred, labels=self.class_names, zero_division=0)
            
            if y_prob is not None:
                metrics["roc_auc"] = self._calculate_roc_auc(y_true, y_prob)
                
            logger.info(f"\n=== Laporan Klasifikasi ===\n{metrics['report_str']}")
        except Exception as e:
            logger.error(f"Evaluasi gagal: {e}")
        return metrics

    def _calculate_roc_auc(self, y_true, y_prob) -> Dict:
        y_bin = label_binarize(y_true, classes=self.class_names)
        n_classes = len(self.class_names)
        
        if n_classes == 2 and y_bin.shape[1] == 1:
            y_bin = np.hstack((1 - y_bin, y_bin))

        fpr, tpr, roc_auc = {}, {}, {}
        
        for i in range(n_classes):
            if i >= y_prob.shape[1]: break
            try:
                if np.sum(y_bin[:, i]) > 0:
                    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    roc_auc[self.class_names[i]] = auc(fpr[i], tpr[i])
                else:
                    roc_auc[self.class_names[i]] = 0.0
            except: pass
            
        try:
            fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        except: pass
        
        return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}


class ClassificationReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_plots(self, metrics: Dict[str, Any], preprocessor: ClassificationPreprocessor):
        if "confusion_matrix" in metrics:
            self._plot_cm(metrics["confusion_matrix"], metrics["class_names"])
        
        if "roc_auc" in metrics:
            self._plot_roc(metrics["roc_auc"], metrics["class_names"])
            
        # Plot feature importance jika selector aktif
        if preprocessor.pipeline and 'selector' in preprocessor.pipeline.named_steps:
            selector = preprocessor.pipeline.named_steps['selector']
            if hasattr(selector, 'scores_'):
                self._plot_feat_imp(selector.scores_, preprocessor.features_in)

    def _plot_cm(self, cm, classes):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

    def _plot_roc(self, roc_data, classes):
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'orange'])
        
        if "micro" in roc_data["fpr"]:
            plt.plot(roc_data["fpr"]["micro"], roc_data["tpr"]["micro"],
                     label=f'Micro-avg (AUC={roc_data["roc_auc"]["micro"]:.2f})',
                     linestyle=':', linewidth=4, color='deeppink')
        
        for i, color in zip(range(len(classes)), colors):
            cls_name = classes[i]
            if cls_name in roc_data["roc_auc"] and i in roc_data["fpr"]:
                auc_score = roc_data["roc_auc"][cls_name]
                plt.plot(roc_data["fpr"][i], roc_data["tpr"][i], color=color, lw=2,
                         label=f'{cls_name} (AUC={auc_score:.2f})')
                         
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_dir, "roc_curves.png"))
        plt.close()

    def _plot_feat_imp(self, scores, all_features):
        scores = np.nan_to_num(scores)
        if len(all_features) != len(scores): return

        df_imp = pd.DataFrame({'Feature': all_features, 'Score': scores})
        df_imp = df_imp.sort_values('Score', ascending=False).head(15)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Score', y='Feature', data=df_imp, palette='viridis')
        plt.title('Feature Importance (F-Score)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"))
        plt.close()


class NaiveBayesEngine:
    def __init__(self, config: Any):
        # [FIX KRITIS 1]: Inisialisasi self.config HARUS di awal
        self.config = config.__dict__ if not isinstance(config, dict) else config
        self.cfg = config # Menyimpan config object asli untuk akses atribut (untuk random_state)
        
        # Inisialisasi lainnya sekarang akan berhasil
        self.output_dir = self.config.get("output_dir", "output/naive_bayes_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.target_col = self.config.get("target_column", "impact_level")
        self.class_names = self.config.get("class_names", ["Ringan", "Sedang", "Parah"])

        # [FIX 2]: Inisialisasi Path Artifacts
        self.model_path = os.path.join(self.output_dir, "naive_bayes_model.pkl") # <-- DITAMBAHKAN
        self.preproc_path = os.path.join(self.output_dir, "preprocessor.pkl")   # <-- DITAMBAHKAN
        
        # Inisialisasi model artifacts dan sub-modules
        self.le = LabelEncoder()
        self.model = None
        self.preprocessor = ClassificationPreprocessor(self.config)
        
        # [FIX 3]: Inisialisasi Evaluator dan Reporter secara eksplisit
        self.evaluator = ModelEvaluator(self.class_names)       # <-- DITAMBAHKAN
        self.reporter = ClassificationReporter(self.output_dir)
        
        
    def train(self, df_train: pd.DataFrame) -> bool:
        if self.target_col not in df_train.columns:
            logger.error(f"Target '{self.target_col}' hilang.")
            return False
        
        y_raw = df_train[self.target_col]
        if len(df_train) < 5: return False

        # [FIX KRITIS LABEL ENCODING]: Transformasi y (string) ke numerik
        try:
            # 1. Lakukan fit_transform pada LabelEncoder
            y_encoded = self.le.fit_transform(y_raw) 
        except Exception as e:
            logger.critical(f"FATAL: Gagal melakukan Label Encoding pada target Y: {e}")
            return False

        # Preprocessing X: Gunakan y_encoded (numerik) untuk f_classif di preprocessor
        X_processed = self.preprocessor.fit_transform(df_train, y_encoded) 
        if X_processed is None: return False
        
        # --- Definisi Model Utama (XGBoost Classifier) ---
        random_state_val = getattr(self.cfg, 'random_state', 42)
        base_model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            use_label_encoder=False, 
            eval_metric='mlogloss', 
            random_state=random_state_val
        )

        # 2. Coba Kalibrasi
        try:
            # Gunakan model XGBoost sebagai estimator untuk Kalibrasi
            self.model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv=3)
            # FIT MENGGUNAKAN y_encoded (numerik)
            self.model.fit(X_processed, y_encoded)
    
        except Exception as e:
            logger.warning(f"Calibrated CV gagal... fallback ke XGBoost biasa. Error: {e}")
            self.model = base_model
            # FIT MENGGUNAKAN y_encoded (numerik)
            self.model.fit(X_processed, y_encoded)
        
        # [FIX TERAKHIR]: Hapus baris fit ganda di sini, karena sudah di-fit di blok try/except
        self._save_artifacts()
        logger.info(f"Training selesai. Model: {type(self.model).__name__} (Classes: {self.le.classes_})")
        return True

    def evaluate(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Diasumsikan df_out = df_test.copy() telah dilakukan di awal evaluate()
        if not self.model:
            if not self._load_artifacts():
                logger.error("Model tidak ditemukan.")
                return df_test, {}
    
        df_out = df_test.copy() # Pastikan df_out diinisialisasi
        y_true_raw = df_out.get(self.target_col)
    
        X_processed = self.preprocessor.transform(df_out)
        if X_processed is None: return df_out, {}
        
        # Prediksi (Output berupa array numerik: 0, 1, 2)
        preds_encoded = self.model.predict(X_processed)
        probs = self.model.predict_proba(X_processed)
    
        # [FIX KRITIS INVERSE TRANSFORM]: Konversi prediksi numerik kembali ke string
        df_out["kelas_prediksi"] = self.le.inverse_transform(preds_encoded)
    
        metrics = {}
        if y_true_raw is not None and y_true_raw.notna().sum() > 0:
        
            # Encode y_true sebelum evaluasi metrik
            try:
                # y_true_encoded adalah array numerik (0, 1, 2)
                y_true_encoded = self.le.transform(y_true_raw.dropna())
            
                # Sesuaikan panjang array jika ada baris NaN yang didrop oleh y_true_raw.dropna()
                preds_encoded_valid = preds_encoded[:len(y_true_encoded)]
                probs_valid = probs[:len(y_true_encoded)]

                # Gunakan y_true_encoded (numerik) dan preds_encoded_valid (numerik) untuk evaluator
                metrics = self.evaluator.evaluate(y_true_encoded, preds_encoded_valid, probs_valid)
                self.reporter.generate_plots(metrics, self.preprocessor)
            except Exception as e:
                logger.error(f"Gagal hitung metrik evaluasi: {e}")
            
        return df_out, metrics

    # CATATAN: LabelEncoder harus disimpan dan dimuat bersama artefak!
    def _save_artifacts(self):
        try:
            with open(self.model_path, "wb") as f: pickle.dump(self.model, f)
            # Simpan juga LabelEncoder
            with open(self.preproc_path.replace(".pkl", "_le.pkl"), "wb") as f: pickle.dump(self.le, f)
            with open(self.preproc_path, "wb") as f: pickle.dump(self.preprocessor, f)
        except Exception as e: logger.error(f"Gagal simpan: {e}")

    def _load_artifacts(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.preproc_path):
            try:
                with open(self.preproc_path.replace(".pkl", "_le.pkl"), "rb") as f: self.le = pickle.load(f)
                with open(self.preproc_path, "rb") as f: self.preprocessor = pickle.load(f)
                return True
            except: pass
        return False