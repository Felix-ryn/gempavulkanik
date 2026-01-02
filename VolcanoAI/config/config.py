# VolcanoAI/config/config.py
# -- coding: utf-8 --

import os
import ast
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field

# =============================================================================
# 1. CONSTANTS & KEYWORDS
# =============================================================================

IMPACT_KEYWORDS: Dict[str, List[str]] = {
    "parah": [
        "letusan aktif", "erupsi aktif", "awan panas", "aliran piroklastik", 
        "guguran awan panas", "status awas", "gempa letusan"
    ],
    "sedang": [
        "vulkanik dangkal", "tremor kuat", "guguran lava", "gempa terasa", 
        "status siaga", "deformasi", "vulkanik dalam"
    ],
    "ringan": [
        "hembusan", "tremor lemah", "tektonik lokal", "asap kawah", 
        "status waspada", "normal", "tektonik jauh"
    ]
}

TYPE_KEYWORDS: Dict[str, List[str]] = {
    "vulkanik": [
        "vulkanik", "letusan", "abu", "hembusan", "lf", "tremor", 
        "erupsi", "guguran", "low frequency", "multiphase", "va", "vb"
    ],
    "tektonik": [
        "tektonik", "jauh", "lokal", "darat", "laut", "subduksi", "tj", "tl"
    ]
}

# =============================================================================
# 2. SUB-CONFIGURATIONS (MODULAR)
# =============================================================================

from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class DataLoaderConfig:
    """Konfigurasi lokasi data dan parameter penggabungan awal."""
    # gunakan path relatif terhadap root project (pakai folder data/)
    volcanic_data_path: str = str(Path("data") / "Data 15 Hari.xlsx")     # file extra (VRP/volcano) atau ganti sesuai
    earthquake_data_path: str = str(Path("data") / "Volcanic_Earthquake_Data.xlsx")  # main earthquake file
    # tambahkan explicit extra path supaya DataLoader tidak bergantung pada fallback string
    earthquake_extra_path: str = str(Path("data") / "Data 15 Hari.xlsx")
    merged_output_path: str = str(Path("output") / "data_merged.xlsx")
    
    date_tolerance_days: int = 30
    distance_tolerance_km: float = 150.0


@dataclass
class DataSplitConfig:
    """Konfigurasi pembagian dataset latih/uji."""
    test_size: float = 0.3
    random_state: int = 42

@dataclass
class OutputConfig:
    """Direktori root output."""
    directory: str = "output"

@dataclass
class PipelineConfig:
    """Saklar utama modul pipeline."""
    run_data_loading: bool = True
    run_feature_engineering: bool = True
    run_aco_engine: bool = True
    run_ga_engine: bool = True
    run_model_training: bool = True
    run_model_evaluation: bool = True
    run_reporting: bool = True

@dataclass
class RealtimeConfig:
    """Konfigurasi 'Sistem Hidup' (Live Monitoring)."""
    enable_monitoring: bool = True
    check_interval_seconds: int = 300  # 5 Menit
    buffer_window_size: int = 90       # Menyimpan 90 data terakhir untuk konteks LSTM

@dataclass
class FeatureEngineeringConfig:
    """Parameter fisika dasar dan aturan thresholding."""
    radius_estimation_defaults: Dict[str, float] = field(
        default_factory=lambda: {"c0": 0.0, "c1": 12.0, "c2": 80.0, "d_min": 1.0}
    )
    ring_multipliers_defaults: Dict[str, float] = field(
        default_factory=lambda: {"m1": 0.25, "m2": 0.55, "m3": 1.00}
    )
    depth_factor_defaults: Dict[str, float] = field(
        default_factory=lambda: {"beta_v": 0.3, "d_ref_v": 30.0, "beta_t": 0.5, "d_ref_t": 40.0}
    )
    impact_thresholds: Dict[str, Any] = field(
        default_factory=lambda: {
            "parah_mag": 6.0,
            "parah_mag_depth_combo": {"mag": 5.0, "depth": 10.0},
            "sedang_mag_min": 4.5,
            "sedang_mag_max": 6.0,
            "sedang_depth_min": 10.0,
            "sedang_depth_max": 30.0,
            "ringan_mag": 4.5,
            "ringan_mag_depth_combo": {"mag": 5.0, "depth": 30.0}
        }
    )
    preprocessor_output_path: str = field(init=False, default="")

    # Kolom yang akan diimputasi jika hilang
    imputation_columns: List[str] = field(
        default_factory=lambda: ['time_since_last_event_days', 'VRP_Max', 'OLI_total (W)', 'MSI_total (W)']
    )
    # Kolom yang akan diskalakan (MinMax/Standard)
    scaling_features: List[str] = field(
        default_factory=lambda: ['Magnitudo', 'Kedalaman (km)', 'seismic_energy_log10', 'time_since_last_event_days']
    )

@dataclass
class AcoEngineConfig:
    """Parameter Ant Colony Optimization (Risk Scoring)."""
    n_ants: int = 100
    n_iterations: int = 200
    evaporation_rate: float = 0.15
    pheromone_influence: float = 1.2
    heuristic_influence: float = 2.8
    
    heuristic_weights: Dict[str, float] = field(
        default_factory=lambda: {"mag": 0.55, "depth": 0.25, "area": 0.2}
    )
    
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 5
    random_seed: Optional[int] = 42
    
    output_dir: str = field(init=False, default="")
    history_output_dir: str = field(init=False, default="")
    visual_output_dir: str = field(init=False, default="")

@dataclass
class GaEngineConfig:
    """
    Parameter Genetic Algorithm (Path Optimization - V6.0 Titan).
    """
    population_size: int = 80
    n_generations: int = 150
    crossover_prob: float = 0.8
    mutation_prob: float = 0.2
    tournament_size: int = 3
    hall_of_fame_size: int = 1
    
    # [UPDATE TITAN] Menambahkan bobot 'mag' dan 'depth' untuk Physics-Aware Fitness
    fitness_weights: Dict[str, float] = field(
        default_factory=lambda: {
            'time': 10000.0, # Penalti waktu (sangat tinggi)
            'space': 1.0,    # Penalti jarak
            'risk': 500.0,   # Prioritas risiko (ACO)
            'mag': 50.0,     # Prioritas magnitudo besar
            'depth': 10.0    # Prioritas gempa dangkal
        }
    )
    
    macro_edge_distance_km: float = 100.0
    output_dir: str = field(init=False, default="")

@dataclass
class LstmEngineConfig:
    """
    Parameter Deep Learning LSTM (Seq2Seq - V5.0 Titan).
    """
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    
    input_seq_len: int = 30
    target_seq_len: int = 1
    
    # Parameter Arsitektur
    latent_dim: int = 64
    use_cnn_encoder: bool = True
    
    # Parameter Data
    target_feature: str = "Magnitudo"
    features: List[str] = field(
        default_factory=lambda: [
            'R3_final', 
            'Magnitudo', 
            'Kedalaman (km)', 
            'seismic_energy_log10', 
            'PheromoneScore', # Hasil ACO
            'Intensity'       # Hasil Feature Eng
        ]
    )
    
    scaler_type: str = "robust" # RobustScaler lebih tahan outlier
    
    # Clustering (DBSCAN Internal)
    clustering_eps: float = 0.05
    clustering_min_samples: int = 3
    clustering_metric: str = "haversine"
    
    # Anomaly Detection
    anomaly_threshold_std: float = 2.5
    
    # Training Control
    validation_split: float = 0.2
    early_stopping_patience: int = 12
    save_history: bool = True
    save_model_checkpoints: bool = True
    
    # Paths
    output_dir: str = field(init=False, default="")
    model_dir: str = field(init=False, default="")
    visuals_dir: str = field(init=False, default="")

@dataclass
class CnnEngineConfig:
    """Parameter Hybrid U-Net CNN."""
    grid_size: int = 64
    domain_km: float = 200.0
    input_channels: int = 3
    epochs: int = 50
    batch_size: int = 8
    
    radius_columns: List[str] = field(
        default_factory=lambda: ["R1_final", "R2_final", "R3_final"]
    )
    
    output_dir: str = field(init=False, default="")
    model_dir: str = field(init=False, default="")

@dataclass
class NaiveBayesEngineConfig:
    """Parameter Klasifikasi Probabilistik Akhir (BINARY)."""
    # target sekarang binary
    target_column: str = "binary_status"
    class_names: List[str] = field(default_factory=lambda: ["Normal", "Tidak Normal"])
    
    features: List[str] = field(
        default_factory=lambda: [
            'Magnitudo', 'Kedalaman (km)', 'isVulkanik', 'isTektonik',
            'R3_final', 'PheromoneScore', 'luas_cnn', 'lstm_prediction'
        ]
    )
    k_best_features: int = 7

    # thresholds untuk heuristik NB (dipakai engine)
    nb_dist_threshold_km: float = 50.0
    nb_mag_threshold: float = 4.5
    nb_pheromone_threshold: float = 0.1
    nb_r3_threshold: float = 0.01

    # output dir default (ProjectConfig akan menimpa output_dir di __post_init__)
    output_dir: str = field(init=False, default="")



@dataclass
class HybridTrainingConfig:
    """
    Konfigurasi Hybrid Training (CNN + LSTM + GA feedback).
    """
    window_days: int = None
    split_ratio: float = 0.7
    min_success_rate: float = 0.8
    max_retries: int = 3

    train_epochs: int = 5
    retrain_epochs: int = 3

    thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            'dist_km': 10.0,
            'angle_deg': 30.0
        }
    )

# =============================================================================
# 3. MAIN PROJECT CONFIGURATION
# =============================================================================

@dataclass
class ProjectConfig:
    """
    Konfigurasi Pusat (Master Config).
    Menggabungkan semua modul konfigurasi di atas.
    """
    PIPELINE: PipelineConfig
    DATA_LOADER: DataLoaderConfig
    DATA_SPLIT: DataSplitConfig
    OUTPUT: OutputConfig
    FEATURE_ENGINEERING: FeatureEngineeringConfig
    ACO_ENGINE: AcoEngineConfig
    GA_ENGINE: GaEngineConfig
    LSTM_ENGINE: LstmEngineConfig
    CNN_ENGINE: CnnEngineConfig
    NAIVE_BAYES_ENGINE: NaiveBayesEngineConfig
    HYBRID: HybridTrainingConfig   
    REALTIME: RealtimeConfig = field(default_factory=RealtimeConfig)

    def __post_init__(self):
        """Membuat direktori output secara otomatis saat inisialisasi."""
        base_output_dir = self.OUTPUT.directory
        try:
            os.makedirs(base_output_dir, exist_ok=True)
        except OSError as e:
            logging.error(f"FATAL: Gagal membuat direktori output utama: {e}")
            raise

        # Set Path Global
        self.DATA_LOADER.merged_output_path = os.path.join(base_output_dir, 'data_merged.xlsx')
        
        # [FIX] Set Path Preprocessor
        self.FEATURE_ENGINEERING.preprocessor_output_path = os.path.join(base_output_dir, "feature_preprocessor.pkl")

        # Set Path ACO
        self.ACO_ENGINE.output_dir = os.path.join(base_output_dir, 'aco_results')
        self.ACO_ENGINE.history_output_dir = os.path.join(self.ACO_ENGINE.output_dir, 'history')
        self.ACO_ENGINE.visual_output_dir = os.path.join(self.ACO_ENGINE.output_dir, 'visuals')

        # Set Path GA
        self.GA_ENGINE.output_dir = os.path.join(base_output_dir, 'ga_results')

        # Set Path LSTM
        self.LSTM_ENGINE.output_dir = os.path.join(base_output_dir, 'lstm_results')
        self.LSTM_ENGINE.model_dir = os.path.join(self.LSTM_ENGINE.output_dir, 'models')
        self.LSTM_ENGINE.visuals_dir = os.path.join(self.LSTM_ENGINE.output_dir, 'visuals')

        # Set Path CNN
        self.CNN_ENGINE.output_dir = os.path.join(base_output_dir, 'cnn_results')
        self.CNN_ENGINE.model_dir = os.path.join(self.CNN_ENGINE.output_dir, 'models')

        # Set Path NB
        self.NAIVE_BAYES_ENGINE.output_dir = os.path.join(base_output_dir, 'naive_bayes_results')

        # Create All Sub-Directories
        all_configs = [
            self.ACO_ENGINE, self.GA_ENGINE, self.LSTM_ENGINE, 
            self.CNN_ENGINE, self.NAIVE_BAYES_ENGINE
        ]
        dir_attrs = ['output_dir', 'model_dir', 'history_output_dir', 'visual_output_dir', 'visuals_dir']
        
        for cfg in all_configs:
            for attr in dir_attrs:
                if hasattr(cfg, attr):
                    path = getattr(cfg, attr)
                    if path: os.makedirs(path, exist_ok=True)

def _get_env(env_name: str, default: Any) -> Any:
    """Membaca environment variable untuk override konfigurasi (DevOps friendly)."""
    value = os.getenv(env_name)
    if value is None: return default
    try: return ast.literal_eval(value)
    except: return value

def load_configuration() -> ProjectConfig:
    """Factory function untuk memuat konfigurasi."""
    logging.info("Memuat konfigurasi proyek VolcanoAI...")
    prefix = "VOLCANOAI"
    
    return ProjectConfig(
        PIPELINE=PipelineConfig(),
        DATA_LOADER=DataLoaderConfig(),
        DATA_SPLIT=DataSplitConfig(test_size=_get_env(f"{prefix}_SPLIT_TEST", 0.3)),
        OUTPUT=OutputConfig(),
        FEATURE_ENGINEERING=FeatureEngineeringConfig(),
        ACO_ENGINE=AcoEngineConfig(
            n_ants=_get_env(f"{prefix}_ACO_ANTS", 100),
            n_iterations=_get_env(f"{prefix}_ACO_ITERS", 200)
        ),
        GA_ENGINE=GaEngineConfig(
            population_size=_get_env(f"{prefix}_GA_POP", 80),
            n_generations=_get_env(f"{prefix}_GA_GEN", 150)
        ),
        LSTM_ENGINE=LstmEngineConfig(
            epochs=_get_env(f"{prefix}_LSTM_EPOCHS", 10)
        ),
        CNN_ENGINE=CnnEngineConfig(
            epochs=_get_env(f"{prefix}_CNN_EPOCHS", 5)
        ),
        NAIVE_BAYES_ENGINE=NaiveBayesEngineConfig(),
        HYBRID=HybridTrainingConfig(),   # 🔥 WAJIB
        REALTIME=RealtimeConfig()
    )

CONFIG = load_configuration()