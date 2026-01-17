import pandas as pd
from pathlib import Path
from datetime import datetime

def generate_presentation_excel(csv_latest_path: Path, output_dir: Path):
    if not csv_latest_path.exists():
        raise FileNotFoundError("Latest CNN CSV not found")

    df = pd.read_csv(csv_latest_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    excel_path = output_dir / f"cnn_presentation_{ts}.xlsx"
    df.to_excel(excel_path, index=False)

    return excel_path
