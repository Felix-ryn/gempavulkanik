# VolcanoAI/config/cnn_config.py
# -- coding: utf-8 --

from dataclasses import dataclass, field
from typing import List

@dataclass
class CnnPipelineConfig:
    grid_size: int = 64
    domain_km: float = 200.0
    input_channels: int = 3
    
    epochs: int = 50
    batch_size: int = 8
    learning_rate: float = 0.001
    
    radius_columns: List[str] = field(default_factory=lambda: [
        "R1_final", 
        "R2_final", 
        "R3_final"
    ])
    
    min_data_for_training: int = 15
    
    output_dir: str = "output/cnn_results"
    model_dir: str = "output/cnn_results/models"
    visuals_dir: str = "output/cnn_results/visuals"