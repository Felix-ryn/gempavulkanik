# VolcanoAI/utils/setup_logging.py

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(output_dir: str, level: str = "INFO"):
    """
    Konfigurasi logging untuk menyimpan ke file + konsol.
    Menambahkan rotasi otomatis dan dukungan log level environment.
    """
    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_filepath = os.path.join(output_dir, log_filename)
    os.makedirs(output_dir, exist_ok=True)

    # Hapus handler lama
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Rotating handler: max 10 MB per file, simpan 5 backup
    rotating_handler = RotatingFileHandler(
        log_filepath, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    stream_handler = logging.StreamHandler(sys.stdout)

    log_format = (
        "%(asctime)s [%(levelname)-8s] [%(name)-20s] "
        "[%(threadName)s] --- %(message)s"
    )
    formatter = logging.Formatter(log_format)

    rotating_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[rotating_handler, stream_handler],
    )

    logging.info(f"Logging diinisialisasi. File: {log_filepath}")