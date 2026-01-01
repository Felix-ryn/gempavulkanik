# script/clear_cache.py
# ----------------------------------
# Menghapus seluruh cache sistem VolcanoAI
# ----------------------------------

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder cache yang aman dihapus
CACHE_DIRS = [
    PROJECT_ROOT / "output" / "cache",
    PROJECT_ROOT / "output" / "aco_results",
    PROJECT_ROOT / "output" / "ga_results",
    PROJECT_ROOT / "output" / "lstm_results",
    PROJECT_ROOT / "output" / "cnn_results",
    PROJECT_ROOT / "output" / "naive_bayes_results",
]

# File cache individual
CACHE_FILES = [
    PROJECT_ROOT / "output" / "feature_preprocessor.pkl",
    PROJECT_ROOT / "output" / "data_merged.xlsx",
]

def remove_path(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
            print(f"[DELETED DIR] {path}")
        else:
            path.unlink()
            print(f"[DELETED FILE] {path}")
    else:
        print(f"[SKIP] {path} (not found)")

def main():
    print("🔥 Membersihkan cache sistem VolcanoAI...\n")

    for d in CACHE_DIRS:
        remove_path(d)

    for f in CACHE_FILES:
        remove_path(f)

    print("\n✅ Cache berhasil dibersihkan.")

if __name__ == "__main__":
    main()
