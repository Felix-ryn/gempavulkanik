# VolcanoAI/postprocess/cnn_csv_to_json.py
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger("VolcanoAI.postprocess.cnn_csv_to_json")
logger.addHandler(logging.NullHandler())

def deg_to_dir(d):
    dirs = ["N","NE","E","SE","S","SW","W","NW"]
    return dirs[int((d + 22.5) // 45) % 8]

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def run(csv_path: str = None, out_json: str = None, force: bool = False) -> bool:
    """
    Convert cnn_predictions_latest.csv -> cnn_predictions_latest.json
    Returns True if success.
    """
    try:
        # Default paths (if caller tidak memberikan)
        if csv_path is None:
            csv_path = Path("output/cnn_results/results/cnn_predictions_latest.csv")
        else:
            csv_path = Path(csv_path)

        if out_json is None:
            out_json = Path("output/cnn_results/cnn_predictions_latest.json")
        else:
            out_json = Path(out_json)

        # Safety: pastikan folder ada
        out_json.parent.mkdir(parents=True, exist_ok=True)

        if not csv_path.exists():
            logger.warning(f"CSV not found: {csv_path}")
            # tulis status JSON minimal agar UI tidak kosong
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "next_event": None,
                "status": "csv_missing"
            }
            out_json.write_text(json.dumps(payload, indent=2))
            return False

        df = pd.read_csv(csv_path)
        if "Acquired_Date" in df.columns:
            df["Acquired_Date"] = pd.to_datetime(df["Acquired_Date"], errors="coerce")

        df = df.sort_values("Acquired_Date") if "Acquired_Date" in df.columns else df

        if len(df) < 2 and not force:
            logger.warning("Data CNN belum cukup (<2 rows). Writing status JSON instead of next_event.")
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "next_event": None,
                "status": "not_enough_data",
                "rows": len(df)
            }
            out_json.write_text(json.dumps(payload, indent=2))
            return False

        # Ambil dua baris terakhir
        last = df.iloc[-1]
        prev = df.iloc[-2]

        # Jika CSV punya kolom lat/lon gunakan itu; else fallback ke centroid (0)
        if all(c in df.columns for c in ["lat", "lon"]) or all(c in df.columns for c in ["Latitude", "Longitude"]):
            lat_col = "lat" if "lat" in df.columns else ("Latitude" if "Latitude" in df.columns else None)
            lon_col = "lon" if "lon" in df.columns else ("Longitude" if "Longitude" in df.columns else None)
            lat_prev, lon_prev = safe_float(prev.get(lat_col, np.nan)), safe_float(prev.get(lon_col, np.nan))
            lat_last, lon_last = safe_float(last.get(lat_col, np.nan)), safe_float(last.get(lon_col, np.nan))
        else:
            # fallback: gunakan nilai statis (sebaiknya diganti dengan centroid cluster)
            # Jika kamu punya cluster centroid di DB, sebaiknya ambil dari sana
            lat_prev, lon_prev = -7.96, 112.36
            lat_last, lon_last = -7.97, 112.38

        dx = lon_last - lon_prev
        dy = lat_last - lat_prev

        # hitung sudut (catatan: formula sesuai implementasimu)
        angle = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        direction = deg_to_dir(angle)
        distance_km = np.sqrt(dx**2 + dy**2) * 111  # approx degree -> km

        luas = safe_float(last.get("luas_cnn", 0.0))
        # confidence mapping (customize)
        confidence = min(1.0, max(0.0, float(luas))) if luas >= 0 else 0.0
        # clamp to reasonable
        confidence = round(max(0.0, min(1.0, 0.4 + confidence)), 2)

        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "next_event": {
                "lat": round(lat_last, 6),
                "lon": round(lon_last, 6),
                "direction_deg": round(angle, 2),
                "movement": direction,
                "distance_km": round(distance_km, 2),
                "confidence": confidence,
                "seismic": round(luas, 2)
            },
            "status": "ok",
            "rows": len(df)
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        logger.info(f"✅ CNN JSON written: {out_json}")
        return True

        print("🚀 cnn_csv_to_json.py DIJALANKAN")
        print("📄 CSV:", CSV_PATH.resolve())
        print("🧾 JSON:", OUT_JSON.resolve())

    except Exception as e:
        logger.exception(f"Failed to run cnn_csv_to_json: {e}")
        try:
            # tulis status minimal agar UI tetap baca
            fallback = {
                "timestamp": datetime.utcnow().isoformat(),
                "next_event": None,
                "status": "exception",
                "error": str(e)
            }
            out_json.write_text(json.dumps(fallback, indent=2))
        except Exception:
            pass
        return False

if __name__ == "__main__":
    # run with defaults if script dijalankan langsung
    run()
