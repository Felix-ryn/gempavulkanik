import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_presentation_excel(csv_latest_path: Path, output_dir: Path):
    if not csv_latest_path.exists():
        raise FileNotFoundError("Latest CNN CSV not found")

    # Pastikan CSV dibaca dengan benar oleh Excel user
    df = pd.read_csv(csv_latest_path, encoding="utf-8-sig")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    excel_path = output_dir / f"cnn_presentation_{ts}.xlsx"

    # =========================
    # PRESENTATION COLUMNS (FINAL)
    # =========================
    present_cols = [
        "cluster_id",
        "Acquired_Date",
        "cnn_angle_deg",
        "cnn_distance_km",
        "cnn_cardinal",
        "cnn_dx_km",
        "cnn_dy_km",
        "cnn_direction_text",
    ]
    present_cols = [c for c in present_cols if c in df.columns]

    df_present = df[present_cols].copy()

    if "Acquired_Date" in df_present.columns:
        df_present["Acquired_Date"] = df_present["Acquired_Date"].astype(str)

    # =========================
    # META INFO
    # =========================
    meta_info = {
        "source_file": csv_latest_path.name,
        "rows": len(df_present),
        "generated_at": datetime.now().isoformat(),
        "notes": "Client presentation (CNN prediction summary only)",
    }

    # =========================
    # WRITE MULTI-SHEET EXCEL
    # =========================
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 1️⃣ Sheet utama (INI YANG DILIHAT CLIENT)
        df_present.to_excel(
            writer,
            sheet_name="CNN_Prediction_Summary",
            index=False
        )

        # 2️⃣ Meta
        pd.DataFrame([meta_info]).to_excel(
            writer,
            sheet_name="Meta",
            index=False
        )

        # 3️⃣ Raw CSV (opsional, buat engineer)
        df.to_excel(
            writer,
            sheet_name="Raw_Output",
            index=False
        )

    return excel_path
