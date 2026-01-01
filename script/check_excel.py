# scripts/check_excel.py
from pathlib import Path
import pandas as pd

# ⬇️ ambil PROJECT ROOT (script/ -> project root)
BASE_DIR = Path(__file__).resolve().parents[1]

files = [
    BASE_DIR / "data" / "Volcanic_Earthquake_Data.xlsx",
    BASE_DIR / "data" / "Data 15 Hari.xlsx",
]

for f in files:
    print("== Checking:", f)
    print("exists:", f.exists())
    if f.exists():
        try:
            xls = pd.ExcelFile(f)
            print("sheets:", xls.sheet_names)
            df = pd.read_excel(f, sheet_name=0)
            print("shape:", df.shape)
            print(df.head(3))
        except Exception as e:
            print("read error:", e)
    print()
