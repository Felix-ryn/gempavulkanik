from VolcanoAI.math.geo_math_core import GeoMathCore
import folium

class MultiLayerVisualizer:

    @staticmethod
    def generate_ga_direction_map(
        center_lat: float,
        center_lon: float,
        bearing_degree: float,
        distance_km: float = 20.0,  # VISUAL SAJA
        output_path: str = "output/ga_results/ga_path_map.html"
    ):
        """
        ⚠️ VISUALISASI ARAH GA
        - BUKAN prediksi lokasi
        - HANYA panah arah dari pusat ACO
        """

        # Titik awal = pusat ACO
        start = (center_lat, center_lon)

        # Titik akhir = proyeksi arah (visual only)
        end = GeoMathCore.destination_point(
            center_lat,
            center_lon,
            bearing_degree,
            distance_km
        )

        # Map
        m = folium.Map(location=start, zoom_start=9)

        # Marker pusat ACO
        folium.Marker(
            start,
            popup="ACO Center",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

        # Panah arah GA
        folium.PolyLine(
            locations=[start, end],
            color="orange",
            weight=4,
            tooltip=f"GA Direction: {bearing_degree:.1f}° (visual)"
        ).add_to(m)

        m.save(output_path)
