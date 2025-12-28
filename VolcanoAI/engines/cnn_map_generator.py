# -*- coding: utf-8 -*-  # deklarasi encoding file
# VolcanoAI/engines/cnn_map_generator.py  # path & nama file modul

import json  # modul untuk parsing JSON
import math  # fungsi matematika (cos, sin, radians)
from pathlib import Path  # kelas Path untuk operasi path
import folium  # library peta interaktif Folium
import logging  # logging untuk pesan runtime

logger = logging.getLogger(__name__)  # buat logger modul berdasar nama modul


class CNNMapGenerator:  # kelas pembuat peta prediksi CNN
    def __init__(self, output_dir: str):  # inisialisasi dengan direktori output
        self.output_dir = Path(output_dir)  # simpan Path output
        self.output_dir.mkdir(parents=True, exist_ok=True)  # buat folder jika belum ada
        from datetime import datetime

        self.map_path = self.output_dir / f"cnn_prediction_map_{datetime.now():%Y%m%d_%H%M%S}.html"


    def _load_json_robust(self, json_path: Path):  # baca JSON dengan fallback encoding
        """
        Read JSON safely with multiple encodings.
        """  # docstring fungsi pembaca JSON yang robust
        raw = json_path.read_bytes()  # baca semua byte file JSON
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):  # daftar encoding coba
            try:
                text = raw.decode(enc)  # coba decode dengan encoding saat ini
                return json.loads(text), enc  # kembalikan dict dan encoding yang berhasil
            except Exception:
                continue  # jika gagal, lanjut ke encoding selanjutnya
        return None, None  # jika semua gagal, kembalikan None

    def generate(self, json_path: str):  # fungsi utama generate peta dari file JSON
        json_path = Path(json_path)  # konversi ke Path

        if not json_path.exists():  # cek keberadaan file
            logger.warning(f"CNNMapGenerator: JSON not found: {json_path}")  # log peringatan
            return None  # hentikan fungsi jika file tidak ada

        data, enc = self._load_json_robust(json_path)  # baca JSON secara robust
        if data is None:  # jika pembacaan gagal
            logger.error("CNNMapGenerator: Failed to read JSON")  # log error
            return None  # hentikan fungsi

        nxt = data.get("next_event")  # ambil entry next_event dari JSON
        if not nxt:  # jika tidak ada next_event
            logger.info("CNNMapGenerator: next_event missing")  # log info
            return None  # hentikan fungsi

        try:
            lat = float(nxt["lat"])  # parse latitude sebagai float
            lon = float(nxt["lon"])  # parse longitude sebagai float
        except Exception:
            logger.error("CNNMapGenerator: invalid lat/lon")  # log error parsing koordinat
            return None  # hentikan fungsi jika koordinat invalid

        bearing = float(nxt.get("direction_deg", 0))  # ambil arah/bearing, default 0
        distance = float(nxt.get("distance_km", 0))  # ambil jarak (km), default 0
        confidence = float(nxt.get("confidence", 0))  # ambil confidence, default 0

        # === MAP ===  # komentar pembatas bagian peta
        m = folium.Map(
            location=[lat, lon],  # pusat peta di lat/lon prediksi
            zoom_start=8,  # level zoom awal
            tiles="CartoDB positron"  # style tiles yang dipakai
        )

        popup = (  # HTML popup berisi informasi event
            "<b>CNN Predicted Event</b><br>"  # judul popup
            f"Lat: {lat}<br>"  # tampilkan latitude
            f"Lon: {lon}<br>"  # tampilkan longitude
            f"Bearing: {bearing:.1f} deg<br>"  # tampilkan bearing dengan 1 desimal
            f"Direction: {nxt.get('movement', '-') }<br>"  # movement string jika ada
            f"Distance: {distance:.2f} km<br>"  # tampilkan distance dengan 2 desimal
            f"Confidence: {confidence:.2f}<br>"  # tampilkan confidence 2 desimal
            f"Map generated at (system time): {data.get('timestamp','')}"# timestamp pembuatan jika ada
        )

        folium.Marker(
            [lat, lon],  # koordinat marker
            popup=folium.Popup(popup, max_width=300),  # popup HTML dengan lebar max
            icon=folium.Icon(color="red", icon="info-sign")  # icon marker berwarna merah
        ).add_to(m)  # tambahkan marker ke peta

        # Direction arrow  # komentar menjelaskan garis arah
        end_lat = lat + 0.1 * math.cos(math.radians(bearing))  # hitung lat akhir panah (sederhana)
        end_lon = lon + 0.1 * math.sin(math.radians(bearing))  # hitung lon akhir panah (sederhana)

        folium.PolyLine(
            locations=[[lat, lon], [end_lat, end_lon]],  # garis dari titik ke arah bearing
            weight=4,  # ketebalan garis
            opacity=0.7  # opasitas garis
        ).add_to(m)  # tambahkan polyline ke peta

        m.save(self.map_path)  # simpan peta ke file HTML yang ditentukan
        logger.info(f"CNNMapGenerator: map saved -> {self.map_path}")  # log sukses simpan peta
        return str(self.map_path)  # kembalikan path file sebagai string
