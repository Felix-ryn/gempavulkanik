# VolcanoAI/engines/naive_bayes_engine.py  # file path & name
# -- coding: utf-8 --  # encoding declaration

import os  # operasi filesystem
import logging  # logging runtime
import pickle  # serialisasi objek Python
import time  # utilitas waktu (tidak dipakai banyak di file ini)
import warnings  # kontrol peringatan
from typing import Dict, Any, List, Tuple, Optional  # type hints

import numpy as np  # array numerik
import pandas as pd  # manipulasi DataFrame
import json  # json I/O

import matplotlib  # plotting backend config
matplotlib.use('Agg')  # non-interactive backend untuk server
import matplotlib.pyplot as plt  # plotting
import seaborn as sns  # visualisasi heatmap
from itertools import cycle  # utilitas iterasi (tidak selalu dipakai)

# ==========================
# SKLEARN CORE
# ==========================
from sklearn.naive_bayes import GaussianNB  # model Gaussian Naive Bayes
from sklearn.preprocessing import (
    StandardScaler,  # standard scaling
    PowerTransformer,  # power transform (Yeo-Johnson)
    LabelEncoder,  # encode label string->int
    label_binarize  # untuk ROC multiclass
)
from sklearn.impute import SimpleImputer  # imputasi nilai hilang
from sklearn.feature_selection import SelectKBest, f_classif  # seleksi fitur berdasarkan ANOVA
from sklearn.pipeline import Pipeline  # pipeline preprocessing
from sklearn.metrics import (
    classification_report,  # teks report klasifikasi
    confusion_matrix,  # confusion matrix
    accuracy_score,  # akurasi
    roc_curve,  # ROC curve
    auc  # area under curve
)

try:
    from ..config.config import NaiveBayesEngineConfig  # optional config import relatif
except ImportError:
    pass  # jika tidak ada, tetap jalan

# ==========================
# LOGGER
# ==========================
logger = logging.getLogger("VolcanoAI.NaiveBayesEngine")  # logger khusus modul
logger.addHandler(logging.NullHandler())  # handler default null untuk mencegah double logs


# ============================================================
# PREPROCESSOR
# ============================================================
class ClassificationPreprocessor:  # class untuk preprocessing fitur sebelum NB
    """
    Bertugas memproses feature hasil pipeline ACO → GA → LSTM → CNN
    sebelum dievaluasi oleh Naive Bayes.
    """  # docstring fungsi

    def __init__(self, config: Dict[str, Any]):  # init menerima config dict
        self.config = config  # simpan config
        self.features_in: List[str] = self.config.get("features", [])  # daftar fitur input
        self.k_best = self.config.get("k_best_features", 7)  # jumlah fitur terbaik yang dipilih

        self.pipeline: Optional[Pipeline] = None  # pipeline sklearn nanti
        self.selected_features: List[str] = []  # fitur terpilih setelah fit

    def fit(self, df: pd.DataFrame, target: Any):  # fit pipeline berdasarkan DataFrame & target
        valid_features = [f for f in self.features_in if f in df.columns]  # filter fitur yang ada
        if not valid_features:
            raise ValueError("Tidak ada fitur valid yang ditemukan.")  # error jika tidak ada fitur

        X = df[valid_features].copy()  # salin kolom fitur valid

        if isinstance(target, np.ndarray):
            temp_target = pd.Series(target)  # jika target numpy -> ubah jadi Series
        else:
            temp_target = target  # asumsi target sudah Series/iterable

        steps = [  # default pipeline steps
            ("imputer", SimpleImputer(strategy="median")),  # isi nilai hilang dengan median
            ("scaler", PowerTransformer(method="yeo-johnson")),  # transformasi power
            ("selector", SelectKBest(f_classif, k="all"))  # placeholder selector
        ]

        actual_k = min(self.k_best, X.shape[1])  # pastikan k tidak lebih dari jumlah fitur
        actual_k = max(actual_k, 1)  # minimal 1 fitur
        steps[2] = ("selector", SelectKBest(f_classif, k=actual_k))  # set selector final

        self.pipeline = Pipeline(steps)  # buat pipeline

        if temp_target.nunique() <= 1:  # jika target hanya 1 kelas
            logger.warning("Target hanya 1 kelas, selector dilewati.")  # warning
            self.pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])  # pipeline sederhana tanpa selector
            self.pipeline.fit(X)  # fit pipeline
            self.selected_features = valid_features  # semua fitur dipakai
            return  # keluar

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # sembunyikan peringatan statistik
            self.pipeline.fit(X, target)  # fit pipeline dengan selector

        if "selector" in self.pipeline.named_steps:
            mask = self.pipeline.named_steps["selector"].get_support()  # ambil mask fitur terpilih
            self.selected_features = [f for f, m in zip(valid_features, mask) if m]  # daftar fitur terpilih
        else:
            self.selected_features = valid_features  # fallback semua fitur

        logger.info(f"Fitur terpilih ({len(self.selected_features)}): {self.selected_features}")  # log fitur terpilih

    def transform(self, df: pd.DataFrame) -> Optional[np.ndarray]:  # transform DataFrame -> numpy array siap model
        if self.pipeline is None:
            logger.warning("Pipeline belum di-fit.")  # peringatan jika belum fit
            return None

        X = df.copy()  # salin df

        for f in self.features_in:
            if f not in X.columns:
                X[f] = np.nan  # tambahkan kolom yang tidak ada sebagai NaN

        X = X[self.features_in]  # reorder kolom sesuai features_in

        try:
            return self.pipeline.transform(X)  # transform dan kembalikan array
        except Exception as e:
            logger.error(f"Transform gagal: {e}")  # log error
            out_dim = len(self.selected_features) or len(self.features_in)  # tentukan dimensi output
            return np.zeros((len(df), out_dim))  # fallback zeros agar alur tidak crash

    def fit_transform(self, df: pd.DataFrame, target: Any) -> np.ndarray:  # convenience method
        self.fit(df, target)  # fit
        return self.transform(df)  # transform


# ============================================================
# EVALUATOR (NUMERIC-SAFE)
# ============================================================
class ModelEvaluator:  # class untuk menghitung metrik klasifikasi
    def __init__(self, class_names: List[str]):
        self.class_names = class_names  # simpan nama kelas
        self.label_indices = list(range(len(class_names)))  # indeks label numerik

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray]
    ) -> Dict[str, Any]:  # kembalikan dictionary metrics

        metrics = {"class_names": self.class_names}  # mulai metrics

        try:
            metrics["accuracy"] = accuracy_score(y_true, y_pred)  # hitung akurasi

            metrics["confusion_matrix"] = confusion_matrix(
                y_true,
                y_pred,
                labels=self.label_indices
            )  # confusion matrix sesuai urutan label

            metrics["report_str"] = classification_report(
                y_true,
                y_pred,
                target_names=self.class_names,
                zero_division=0
            )  # teks laporan klasifikasi

            if y_prob is not None:
                metrics["roc_auc"] = self._calculate_roc_auc(y_true, y_prob)  # tambahkan ROC AUC jika ada probabilitas

            logger.info("\n=== Classification Report ===\n" + metrics["report_str"])  # log report

        except Exception as e:
            logger.error(f"Evaluasi gagal: {e}")  # jika error saat evaluasi

        return metrics  # kembalikan metrics

    def _calculate_roc_auc(self, y_true, y_prob) -> Dict[str, Any]:  # hitung ROC AUC multiclass
        y_bin = label_binarize(y_true, classes=self.label_indices)  # binarize labels
        n_classes = len(self.class_names)  # jumlah kelas

        fpr, tpr, roc_auc = {}, {}, {}  # dict penyimpanan

        for i in range(n_classes):  # per kelas
            if i >= y_prob.shape[1]:
                continue  # skip jika probabilitas tidak memiliki kolom ini
            if y_bin[:, i].sum() == 0:
                roc_auc[self.class_names[i]] = 0.0  # jika tidak ada contoh kelas, AUC 0
                continue

            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])  # hitung fpr,tpr
            roc_auc[self.class_names[i]] = auc(fpr[i], tpr[i])  # hitung auc

        return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}  # return struktur ROC


# ============================================================
# REPORTER
# ============================================================
class ClassificationReporter:  # class untuk menyimpan plot hasil evaluasi
    def __init__(self, output_dir: str):
        self.output_dir = output_dir  # direktori output
        os.makedirs(self.output_dir, exist_ok=True)  # buat jika belum ada

    def generate_plots(self, metrics: Dict[str, Any]):
        if "confusion_matrix" in metrics:
            self._plot_cm(metrics["confusion_matrix"], metrics["class_names"])  # generate confusion matrix plot

    def _plot_cm(self, cm, classes):  # plot confusion matrix
        plt.figure(figsize=(8, 6))  # ukuran figure
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)  # heatmap
        plt.title("Confusion Matrix")  # judul
        plt.ylabel("Actual")  # label y
        plt.xlabel("Predicted")  # label x
        plt.tight_layout()  # rapikan layout
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))  # simpan file
        plt.close()  # tutup figure


# ============================================================
# NAIVE BAYES ENGINE (FINAL)
# ============================================================
class NaiveBayesEngine:  # engine akhir yang membungkus preproc, model, evaluator
    """
    Evaluator akhir pipeline:
    ACO → GA → LSTM → CNN → Naive Bayes
    """

    def __init__(self, config:d Any):  # konstruktor menerima config (typo 'config:d' preserved)
        self.config = config.__dict__ if not isinstance(config, dict) else config  # normalisasi config

        self.output_dir = self.config.get("output_dir", "output/naive_bayes_results")  # direktori output default
        os.makedirs(self.output_dir, exist_ok=True)  # buat jika belum ada

        self.target_col = self.config.get("target_column", "impact_level")  # kolom target default
        self.class_names = self.config.get("class_names", ["Ringan", "Sedang", "Parah"])  # nama kelas default

        self.model_path = os.path.join(self.output_dir, "naive_bayes_model.pkl")  # path model
        self.preproc_path = os.path.join(self.output_dir, "preprocessor.pkl")  # path preprocessor
        self.le_path = os.path.join(self.output_dir, "label_encoder.pkl")  # path label encoder

        self.le = LabelEncoder()  # inisialisasi label encoder
        self.model: Optional[GaussianNB] = None  # placeholder model
        self.preprocessor = ClassificationPreprocessor(self.config)  # inisialisasi preprocessor

        self.evaluator = ModelEvaluator(self.class_names)  # evaluator instance
        self.reporter = ClassificationReporter(self.output_dir)  # reporter instance

    def train(self, df_train: pd.DataFrame) -> bool:  # melatih model NB
        if self.target_col not in df_train.columns:  # cek kolom target ada
            logger.error("Kolom target tidak ditemukan.")  # log error
            return False  # gagal

        y_raw = df_train[self.target_col]  # ambil nilai target asli

        try:
            y_encoded = self.le.fit_transform(y_raw)  # encode label ke numerik
        except Exception as e:
            logger.error(f"Label encoding gagal: {e}")  # log error
            return False  # gagal

        X_processed = self.preprocessor.fit_transform(df_train, y_encoded)  # preprocessing fitur
        if X_processed is None:
            return False  # gagal jika preprocessing tidak berhasil

        # ==========================
        # NAIVE BAYES (FINAL MODEL)
        # ==========================
        self.model = GaussianNB()  # instantiate GaussianNB
        self.model.fit(X_processed, y_encoded)  # fit model dengan data

        self._save_artifacts()  # simpan model & preprocessor & encoder

        logger.info("Training Naive Bayes selesai.")  # log selesai
        return True  # sukses

    def evaluate(self, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:  # evaluasi model pada df_test
        if self.model is None:  # jika model belum dimuat
            if not self._load_artifacts():  # coba load artifacts dari disk
                logger.error("Model tidak ditemukan.")  # log error
                return df_test, {}  # return original df dan metrics kosong

        df_out = df_test.copy()  # salin input df
        y_true_raw = df_out.get(self.target_col)  # ambil kolom target jika ada

        X_processed = self.preprocessor.transform(df_out)  # transform fitur
        if X_processed is None:
            X_processed = np.zeros((len(df_out), len(self.preprocessor.selected_features) or 1))  # fallback zeros

        preds = self.model.predict(X_processed)  # prediksi kelas
        probs = self.model.predict_proba(X_processed)  # prediksi probabilitas

        df_out["kelas_prediksi"] = self.le.inverse_transform(preds)  # inverse transform ke label asli

        metrics = {}  # placeholder metrics
        # Jika kolom target ada, evaluasi seperti biasa
        if y_true_raw is not None:
            try:
                y_true_encoded = self.le.transform(y_true_raw)  # encode target ground truth
                metrics = self.evaluator.evaluate(y_true_encoded, preds, probs)  # hitung metrics
                self.reporter.generate_plots(metrics)  # buat plot (confusion matrix)
            except Exception as e:
                logger.warning(f"Evaluasi gagal: {e}")  # warn jika evaluasi gagal

        # Simpan output selalu, walau metrics kosong
        self._save_outputs(df_out, metrics)  # simpan prediksi & metrics

        return df_out, metrics  # kembalikan df hasil & metrics

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
        """  # docstring

        import json  # pastikan import json ada di sini atau di awal file

        # ==========================
        # 1. Simpan hasil prediksi
        # ==========================
        pred_path = os.path.join(self.output_dir, "naive_bayes_predictions.csv")  # path CSV prediksi
        df_out.to_csv(pred_path, index=False)  # simpan CSV tanpa index

        # ==========================
        # 2. Simpan metrics ke JSON (numpy-safe)
        # ==========================
        def _convert_numpy(obj):  # helper rekursif konversi numpy->list
            """Konversi np.ndarray rekursif menjadi list agar JSON-safe"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # array -> list
            elif isinstance(obj, dict):
                return {k: _convert_numpy(v) for k, v in obj.items()}  # rekursif dict
            elif isinstance(obj, list):
                return [_convert_numpy(v) for v in obj]  # rekursif list
            else:
                return obj  # default return

        metrics_safe = _convert_numpy(metrics)  # ubah metrics jadi JSON-safe
        metrics_path = os.path.join(self.output_dir, "naive_bayes_metrics.json")  # path metrics JSON

        try:
            with open(metrics_path, "w", encoding="utf-8") as f:  # tulis metrics
                json.dump(metrics_safe, f, indent=2, ensure_ascii=False)  # dump json
        except Exception as e:
            logger.error(f"Gagal tulis metrics JSON: {e}")  # log error penulisan
            # fallback: tulis representasi string supaya file tidak kosong
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    f.write(repr(metrics_safe))  # tulis string representasi
            except Exception as e2:
                logger.error(f"Fallback penulisan metrics juga gagal: {e2}")  # log error fallback

        # ==========================
        # 3. Simpan classification report text
        # ==========================
        if "report_str" in metrics:
            report_path = os.path.join(self.output_dir, "classification_report.txt")  # path report text
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(metrics["report_str"])  # tulis report

        logger.info("Output Naive Bayes berhasil disimpan.")  # log selesai simpan


    def _save_artifacts(self):  # simpan model & preproc & label encoder ke disk
        with open(self.model_path, "wb") as f:
            pickle.dump(self.model, f)  # simpan model
        with open(self.preproc_path, "wb") as f:
            pickle.dump(self.preprocessor, f)  # simpan preprocessor
        with open(self.le_path, "wb") as f:
            pickle.dump(self.le, f)  # simpan label encoder

    def _load_artifacts(self) -> bool:  # load artifacts dari disk
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)  # load model
            with open(self.preproc_path, "rb") as f:
                self.preprocessor = pickle.load(f)  # load preprocessor
            with open(self.le_path, "rb") as f:
                self.le = pickle.load(f)  # load label encoder
            return True  # sukses
        except Exception:
            return False  # gagal load
