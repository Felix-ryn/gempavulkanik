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

    # =========================
    # COLUMN GROUPING (SAFE)
    # =========================
    input_cols = [c for c in df.columns if c.startswith(("aco_", "ga_"))]
    output_cols = [
        c for c in [
            "luas_cnn",
            "cnn_angle_deg",
            "cnn_distance_km",
            "cnn_confidence",
        ] if c in df.columns
    ]

    meta_cols = [
        c for c in [
            "cluster_id",
            "Acquired_Date",
        ] if c in df.columns
    ]

    # =========================
    # PREPARE DATAFRAMES
    # =========================
    df_input = df[input_cols].copy() if input_cols else pd.DataFrame()
    df_output = df[output_cols].copy() if output_cols else pd.DataFrame()
    df_meta = df[meta_cols].copy() if meta_cols else pd.DataFrame()

    meta_info = {
        "source_file": csv_latest_path.name,
        "total_rows": len(df),
        "generated_at": datetime.now().isoformat(),
        "generator": "generate_presentation_excel",
        "notes": "Presentation Excel generated from latest CNN CSV (no retraining)",
    }

    # =========================
    # WRITE MULTI-SHEET EXCEL
    # =========================
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 1️⃣ Meta summary
        pd.DataFrame([meta_info]).to_excel(
            writer, sheet_name="Meta", index=False
        )

        # 2️⃣ CNN Output (utama untuk presentasi)
        if not df_output.empty:
            df_output.to_excel(
                writer, sheet_name="CNN_Output_Pred", index=False
            )

        # 3️⃣ CNN Input (ACO / GA features)
        if not df_input.empty:
            df_input.to_excel(
                writer, sheet_name="CNN_Input_ACO", index=False
            )

        # 4️⃣ Identitas / waktu data
        if not df_meta.empty:
            df_meta.to_excel(
                writer, sheet_name="Data_Info", index=False
            )

        # 5️⃣ Raw data (fallback / audit)
        df.to_excel(
            writer, sheet_name="Raw_Output", index=False
        )

    return excel_path
