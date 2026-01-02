# VolcanoAI/config/naivebayes_config.py
# -- coding: utf-8 --

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class NaiveBayesPipelineConfig:
    """Konfigurasi pipeline Naive Bayes (binary: Normal / Tidak Normal)."""
    # Nama kolom target final
    target_column: str = "binary_status"

    # Binary classes: index/order penting untuk LabelEncoder
    class_names: List[str] = field(default_factory=lambda: ["Normal", "Tidak Normal"])

    # Fitur yang dipakai (sesuaikan jika perlu)
    features: List[str] = field(default_factory=lambda: [
        'Kedalaman (km)',
        'isVulkanik',
        'isTektonik',
        'R3_final',
        'PheromoneScore',
        'luas_cnn',
        'lstm_prediction'
    ])

    # Feature selection / preprocessing
    k_best_features: int = 7

    # Train/test split
    test_size: float = 0.3
    random_state: int = 42

    # Output & artifact paths
    output_dir: str = "output/naive_bayes_results"
    model_save_path: str = "output/naive_bayes_results/naive_bayes_model.pkl"
    report_save_path: str = "output/naive_bayes_results/classification_report.json"

    # Thresholds used oleh NaiveBayesEngine heuristics (dapat dioverride lewat env atau ProjectConfig)
    nb_dist_threshold_km: float = 50.0
    nb_mag_threshold: float = 4.5
    nb_pheromone_threshold: float = 0.1
    nb_r3_threshold: float = 0.01

    # Nama kolom jarak alternatif (jika feature eng/ CNN menghasilkan nama berbeda)
    distance_columns: List[str] = field(default_factory=lambda: ["cnn_distance_km", "distance_km"])
