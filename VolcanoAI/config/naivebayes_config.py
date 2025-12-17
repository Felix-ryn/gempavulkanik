# VolcanoAI/config/naivebayes_config.py
# -- coding: utf-8 --

from dataclasses import dataclass, field
from typing import List

@dataclass
class NaiveBayesPipelineConfig:
    target_column: str = "impact_level"
    class_names: List[str] = field(default_factory=lambda: ["Ringan", "Sedang", "Parah"])
    
    features: List[str] = field(default_factory=lambda: [
        'Magnitudo', 
        'Kedalaman (km)', 
        'isVulkanik', 
        'isTektonik',
        'R3_final', 
        'PheromoneScore', 
        'luas_cnn', 
        'lstm_prediction'
    ])
    
    k_best_features: int = 7
    test_size: float = 0.3
    random_state: int = 42
    
    output_dir: str = "output/naive_bayes_results"
    model_save_path: str = "output/naive_bayes_results/naive_bayes_model.pkl"
    report_save_path: str = "output/naive_bayes_results/classification_report.json"