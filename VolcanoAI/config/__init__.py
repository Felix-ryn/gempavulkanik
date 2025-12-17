# VolcanoAI/config/__init__.py
# -- coding: utf-8 --

# Import konfigurasi spesifik dari file terpisah
# (Pastikan file-file ini ada dan berisi definisi kelas yang benar)
try:
    from .lstm_config import LstmPipelineConfig
except ImportError:
    LstmPipelineConfig = None

try:
    from .cnn_config import CnnPipelineConfig
except ImportError:
    CnnPipelineConfig = None

try:
    from .naivebayes_config import NaiveBayesPipelineConfig
except ImportError:
    NaiveBayesPipelineConfig = None

# Import konfigurasi utama dan sub-konfigurasi dari config.py
from .config import (
    CONFIG, 
    load_configuration, 
    ProjectConfig,
    # Ekspor juga sub-config agar bisa diakses langsung jika perlu
    DataLoaderConfig,
    DataSplitConfig,
    OutputConfig,
    PipelineConfig,
    FeatureEngineeringConfig,
    AcoEngineConfig,
    GaEngineConfig,
    LstmEngineConfig, 
    CnnEngineConfig, 
    NaiveBayesEngineConfig
)

# Ekspor Global agar mudah diakses via 'from VolcanoAI.config import ...'
__all__ = [
    "CONFIG",
    "load_configuration",
    "ProjectConfig",
    
    # Configs dari file terpisah
    "LstmPipelineConfig",
    "CnnPipelineConfig",
    "NaiveBayesPipelineConfig",
    
    # Configs dari config.py (Sub-configs)
    "DataLoaderConfig",
    "DataSplitConfig",
    "OutputConfig",
    "PipelineConfig",
    "FeatureEngineeringConfig",
    "AcoEngineConfig",
    "GaEngineConfig",
    "LstmEngineConfig",
    "CnnEngineConfig",
    "NaiveBayesEngineConfig"
]