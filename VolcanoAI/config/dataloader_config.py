# VolcanoAI/config/dataloader_config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

@dataclass
class DataLoaderConfig:
    """
    Konfigurasi utama untuk proses pemuatan data
    yang digunakan oleh pipeline LSTM, CNN, dan NaiveBayes.
    """

    # Path utama dataset
    volcano_data_path: str = "data/volcano.xlsx"
    earthquake_data_path: str = "data/earthquake.xlsx"
    merged_output_path: str = "output/datasets/merged_dataset.xlsx"

    # Streaming & real-time
    enable_realtime_stream: bool = True
    realtime_buffer_size: int = 256
    stream_update_interval: float = 2.0  # detik

    # Hybrid / Semi-online learning
    hybrid_window_days: int = 15
    hybrid_split_ratio: float = 0.7

    # DataLoader behaviour
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    drop_last: bool = False

    # Data preprocessing
    fill_missing: bool = True
    normalize: bool = True
    scaling_method: str = "minmax"  # atau 'standard'
    feature_columns: List[str] = field(default_factory=lambda: [
        "Magnitudo", "Kedalaman (km)", "num_nearby_eqs",
        "max_nearby_mag", "VRP_Max", "OLI_total (W)", "MSI_total (W)"
    ])
    target_column: str = "Magnitudo"

    # Cache
    use_cache: bool = True
    cache_dir: str = "output/cache"
    overwrite_cache: bool = False

    # Random seed untuk reproducibility
    random_seed: Optional[int] = 42

    def __post_init__(self):
        # Pastikan direktori cache dan dataset ada
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        Path("data").mkdir(parents=True, exist_ok=True)
        Path("output/datasets").mkdir(parents=True, exist_ok=True)
