# VolcanoAI/config/lstm_config.py
# -- coding: utf-8 --

from dataclasses import dataclass, field
from typing import List

@dataclass
class LstmPipelineConfig:
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    
    input_seq_len: int = 30
    target_seq_len: int = 1
    latent_dim: int = 64
    
    target_feature: str = "Magnitudo"
    features: List[str] = field(default_factory=lambda: [
        'R3_final', 
        'Magnitudo', 
        'Kedalaman (km)', 
        'seismic_energy_log10', 
        'PheromoneScore'
    ])
    
    scaler_type: str = "minmax"
    use_cnn_encoder: bool = True
    
    clustering_eps: float = 0.05
    clustering_min_samples: int = 3
    clustering_metric: str = "haversine"
    
    anomaly_threshold_std: float = 2.5
    
    validation_split: float = 0.2
    early_stopping_patience: int = 12
    save_history: bool = True
    save_model_checkpoints: bool = True
    
    enable_parallel_training: bool = False
    max_parallel_workers: int = 2
    use_mixed_precision: bool = False
    
    output_dir: str = "output/lstm_results"
    model_dir: str = "output/lstm_results/models"
    visuals_dir: str = "output/lstm_results/visuals"
    cache_dir: str = "output/lstm_results/cache"
    
    volcano_data_path: str = "data/volcano.xlsx"
    earthquake_data_path: str = "data/earthquake.xlsx"
    merged_output_path: str = "output/lstm_pipeline_v5/merged_dataset.xlsx"