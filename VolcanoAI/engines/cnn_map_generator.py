# -*- coding: utf-8 -*-
# VolcanoAI/engines/cnn_map_generator.py

import json
import math
from pathlib import Path
import folium
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class CNNMapGenerator:
    """
    Generator peta prediksi CNN (Folium HTML)
    + menulis latest_map.txt agar dashboard tahu peta terbaru
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_json_robust(self, json_path: Path):
        """Read JSON safely with multiple encodings."""
        raw = json_path.read_bytes()
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = raw.decode(enc)
                return json.loads(text), enc
            except Exception:
                continue
        return None, None

    def generate(self, json_path: str):
        json_path = Path(json_path)

        if not json_path.exists():
            logger.warning(f"CNNMapGenerator: JSON not found: {json_path}")
            return None

        data, enc = self._load_json_robust(json_path)
        if data is None:
            logger.error("CNNMapGenerator: Failed to read JSON")
            return None

        nxt = data.get("next_event")
        if not nxt:
            logger.info("CNNMapGenerator: next_event missing")
            return None

        try:
            lat = float(nxt["lat"])
            lon = float(nxt["lon"])
        except Exception:
            logger.error("CNNMapGenerator: invalid lat/lon")
            return None

        bearing = float(nxt.get("direction_deg", 0.0))
        distance = float(nxt.get("distance_km", 0.0))
        confidence = float(nxt.get("confidence", 0.0))

        # =====================
        # MAP CREATION
        # =====================
        m = folium.Map(
            location=[lat, lon],
            zoom_start=8,
            tiles="CartoDB positron"
        )

        popup = (
            "<b>CNN Predicted Event</b><br>"
            f"Lat: {lat}<br>"
            f"Lon: {lon}<br>"
            f"Bearing: {bearing:.1f} deg<br>"
            f"Direction: {nxt.get('movement', '-') }<br>"
            f"Distance: {distance:.2f} km<br>"
            f"Confidence: {confidence:.2f}<br>"
            f"Timestamp: {data.get('timestamp','')}"
        )

        folium.Marker(
            [lat, lon],
            popup=folium.Popup(popup, max_width=300),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

        # Direction arrow
        end_lat = lat + 0.1 * math.cos(math.radians(bearing))
        end_lon = lon + 0.1 * math.sin(math.radians(bearing))

        folium.PolyLine(
            locations=[[lat, lon], [end_lat, end_lon]],
            weight=4,
            opacity=0.7,
        ).add_to(m)

        # =====================
        # SAVE MAP
        # =====================
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_path = self.output_dir / f"cnn_prediction_map_{ts}.html"
        m.save(map_path)

        logger.info(f"CNNMapGenerator: map saved -> {map_path}")

        # 🔥🔥🔥 INI WAJIB UNTUK DASHBOARD
        pointer = self.output_dir / "latest_map.txt"
        pointer.write_text(str(map_path.resolve()), encoding="utf-8")

        logger.info(f"CNNMapGenerator: latest map pointer updated -> {pointer}")

        return str(map_path)
