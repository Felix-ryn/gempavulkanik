# VolcanoAI/reporting/comprehensive_reporter.py
# -- coding: utf-8 --

import os
import logging
import base64
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import networkx as nx
from folium import plugins
from folium import LayerControl, FeatureGroup, Marker, Icon, Circle
from branca.colormap import LinearColormap
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List, Optional

try:
    from ..config.config import ProjectConfig
except ImportError:
    pass

logger = logging.getLogger("VolcanoAI.Reporter")
logger.addHandler(logging.NullHandler())

# ==============================================================================
# 1. GRAPH VISUALIZER (COMPATIBILITY)
# ==============================================================================
class GraphVisualizer:
    # (Dipertahankan untuk kompatibilitas impor main.py)
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_graphs_to_maps(self, G_macro: nx.Graph, micro_graphs: Dict[str, nx.Graph]) -> Dict[str, str]:
        # Logika export graph GA (jika dipanggil)
        out = {}
        if not G_macro: return out
        try:
            lats = [d.get("latitude") for _, d in G_macro.nodes(data=True)]
            lons = [d.get("longitude") for _, d in G_macro.nodes(data=True)]
            center = [float(np.mean(lats)), float(np.mean(lons))]
            macro_map = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")
            for n, d in G_macro.nodes(data=True):
                folium.CircleMarker(location=[d['latitude'], d['longitude']], radius=6).add_to(macro_map)
            macro_path = os.path.join(self.output_dir, "macro_graph_map.html")
            macro_map.save(macro_path)
            out["macro_map"] = macro_path
        except Exception as e:
            self.logger.warning(f"Gagal export graph map: {e}")
        return out


# ==============================================================================
# 2. ASSET MANAGER
# ==============================================================================
class AssetManager:
    def __init__(self, config: ProjectConfig):
        self.output_dir = config.OUTPUT.directory
        self.engine_dirs = {
            "aco": config.ACO_ENGINE.output_dir,
            "ga": config.GA_ENGINE.output_dir,
            "naive_bayes": config.NAIVE_BAYES_ENGINE.output_dir
        }
        self.paths = {
            "aco_csv": os.path.join(self.engine_dirs["aco"], "aco_dynamic_result.csv"),
            "ga_path_csv": os.path.join(self.engine_dirs["ga"], "ga_optimized_path_for_lstm.csv"),
            "final_xlsx": os.path.join(self.output_dir, "hasil_akhir_analisis.xlsx"), # Mengacu ke XLSX
            "nb_roc": os.path.join(self.engine_dirs["naive_bayes"], "roc_curves.png"),
            "ga_conv": os.path.join(self.engine_dirs["ga"], "ga_convergence_plot.png")
        }

    def get_data(self, key: str) -> pd.DataFrame:
        path = self.paths.get(key)
        if path and os.path.exists(path):
            try:
                # Menggunakan read_excel untuk data XLSX
                if path.endswith('.xlsx'): return pd.read_excel(path)
                return pd.read_csv(path)
            except Exception: return pd.DataFrame()
        return pd.DataFrame()

    def get_image_b64(self, key: str) -> str:
        path = self.paths.get(key)
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception: pass
        return ""


# ==============================================================================
# 3. VISUAL & REPORT GENERATOR
# ==============================================================================
class RealtimePlotter:
    @staticmethod
    def generate_trend_plot(df_buffer: pd.DataFrame, window_size: int = 90) -> str:
        if df_buffer.empty: return ""
        df_view = df_buffer.tail(window_size).copy().reset_index(drop=True)
        plt.figure(figsize=(10, 4))
        
        if 'Magnitudo' in df_view.columns:
            plt.plot(df_view.index, df_view['Magnitudo'], label='Magnitudo Aktual', color='#2c3e50', linewidth=2)
        
        if 'lstm_prediction' in df_view.columns:
            plt.plot(df_view.index, df_view['lstm_prediction'], label='Prediksi LSTM', color='#e74c3c', linestyle='--', linewidth=2)
            
            if 'prediction_sigma' in df_view.columns:
                sigma = df_view['prediction_sigma'].fillna(0)
                upper = df_view['lstm_prediction'] + (1.96 * sigma)
                lower = df_view['lstm_prediction'] - (1.96 * sigma)
                plt.fill_between(df_view.index, lower, upper, color='#e74c3c', alpha=0.1, label='Confidence Interval (95%)')

        if not df_view.empty and 'Magnitudo' in df_view.columns:
            last_idx = df_view.index[-1]
            last_mag = df_view['Magnitudo'].iloc[-1]
            plt.scatter(last_idx, last_mag, s=100, c='red', edgecolors='black', zorder=10, label='EVENT TERKINI')

        plt.title(f"Dinamika Seismik (Jendela {window_size} Data Terakhir)", fontsize=12)
        plt.ylabel("Magnitudo"); plt.xlabel("Urutan Waktu (Buffer)")
        plt.legend(loc='upper left'); plt.grid(True, alpha=0.3, linestyle='--'); plt.tight_layout()
        
        buf = BytesIO(); plt.savefig(buf, format='png'); plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

class MapGenerator:
    """
    Kelas yang bertanggung jawab untuk membuat kedua jenis peta: Statis & Live.
    """
    def __init__(self):
        self.default_center = [-7.8, 112.5]
    
    # ------------------------------------------------------------------
    # HELPER LAYERS (Untuk dipakai di kedua mode)
    # ------------------------------------------------------------------

    def _add_aco_points_layer(self, df_aco: pd.DataFrame, m: folium.Map, name: str = "Area Risiko (ACO)"):
        """Menambahkan titik/lingkaran risiko ACO."""
        fg = FeatureGroup(name=name, show=False)
        if not df_aco.empty and 'PheromoneScore' in df_aco.columns:
            df_sorted = df_aco.sort_values('PheromoneScore')
            for _, row in df_sorted.iterrows():
                score = row.get('PheromoneScore', 0)
                if score < 0.05: continue
                
                color = "#f1c40f" if score > 0.4 else "#2ecc71"
                if score > 0.8: color = "#e74c3c"
                radius = 2 + (score * 12)
                
                popup_txt = f"Risk Score: {score:.4f} | Mag: {row.get('Magnitudo', 0):.1f}"
                folium.CircleMarker([row['EQ_Lintang'], row['EQ_Bujur']], radius=radius, color=color, fill=True, fill_opacity=0.7, weight=1, popup=popup_txt).add_to(fg)
        return fg

    def _add_ga_path_layer(self, df_ga: pd.DataFrame, m: folium.Map, name: str = "Jalur Optimal (GA)"):
        """Menambahkan jalur prediksi GA."""
        fg = FeatureGroup(name=name, show=False)
        if not df_ga.empty:
            try:
                coords = df_ga[['EQ_Lintang', 'EQ_Bujur']].values.tolist()
                if coords:
                    plugins.AntPath(locations=coords, use_polyline=True, dash_array=[10, 20], weight=3, color="blue", pulse_color="cyan", opacity=0.7, name=name).add_to(fg)
            except Exception: pass
        return fg

    # ------------------------------------------------------------------
    # MODE 1: STATIC ANALYSIS (FULL LAYER CONTROL)
    # ------------------------------------------------------------------
    def generate_static_map(self, df_aco_all: pd.DataFrame, df_ga_all: pd.DataFrame, df_historical_points: pd.DataFrame) -> str:
        m = folium.Map(location=self.default_center, zoom_start=8, tiles="CartoDB positron")

        # Layer A: Titik Historis (Titik asli dari data gabungan)
        fg_hist = FeatureGroup(name="A. Titik Historis (Semua Data)", show=True)
        if not df_historical_points.empty:
            for _, row in df_historical_points.head(2000).iterrows(): # Batasi 2000 untuk performa
                 folium.CircleMarker([row['EQ_Lintang'], row['EQ_Bujur']], radius=2, color='#2c3e50', fill=True, popup=f"Mag: {row.get('Magnitudo', 0)}").add_to(fg_hist)
        fg_hist.add_to(m)

        # Layer B: ACO Area (Lingkaran Risiko)
        fg_aco = self._add_aco_points_layer(df_aco_all, m, name="B. Area Risiko ACO (Lingkaran)")
        
        # Layer C: ACO Heatmap (Visualisasi Density)
        fg_heat = FeatureGroup(name="C. Risiko ACO (Heatmap Density)", show=False)
        if not df_aco_all.empty and 'PheromoneScore' in df_aco_all.columns:
            try:
                heat_data = df_aco_all[['EQ_Lintang', 'EQ_Bujur', 'PheromoneScore']].dropna().values.tolist()
                if heat_data:
                    plugins.HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.8: 'orange', 1: 'red'}, name="C. Risiko ACO (Heatmap Density)").add_to(fg_heat)
            except: pass
        fg_heat.add_to(m)

        # Layer D: GA Path
        fg_ga = self._add_ga_path_layer(df_ga_all, m, name="D. Jalur Kausalitas (GA Path)")

        # Layer E: Prediksi Area GA (Contoh: area di sekitar titik GA)
        # Logika ini bisa kompleks, kita gunakan ACO untuk memperkirakan area GA
        
        folium.LayerControl(collapsed=False).add_to(m)
        return m._repr_html_()
    
    # ------------------------------------------------------------------
    # MODE 2: LIVE MONITORING (ALERT FOCUS)
    # ------------------------------------------------------------------
    def generate_live_map(self, df_ga_all: pd.DataFrame, latest_row: pd.Series) -> str:
        """Dashboard Realtime: Fokus pada Alert dan Konteks Jalur."""
        
        # Center di titik baru
        center = [latest_row['EQ_Lintang'], latest_row['EQ_Bujur']] if not latest_row.empty else self.default_center
        m = folium.Map(location=center, zoom_start=9, tiles="CartoDB dark_matter")
        
        # Layer 1: GA Path (Konteks)
        self._add_ga_path_layer(df_ga_all, m, name="Context Path (GA Snake)")

        # Layer 2: LIVE EVENT (Pulsing Marker & CNN Area)
        fg_live = FeatureGroup(name="⚠️ LIVE ALERT", show=True)
        if not latest_row.empty:
            lat, lon = latest_row['EQ_Lintang'], latest_row['EQ_Bujur']
            
            # CNN Area Dampak (Area terprediksi)
            luas_cnn = latest_row.get('luas_cnn', 0)
            if luas_cnn > 0:
                rad_meter = np.sqrt(luas_cnn / np.pi) * 1000
                folium.Circle([lat, lon], radius=rad_meter, color='#8e44ad', fill=True, fill_opacity=0.4).add_to(fg_live)

            # Pulsing Marker di atas semuanya
            folium.CircleMarker(
                location=[lat, lon], radius=12, color='red', fill=True, fill_color='red', fill_opacity=1.0,
                popup=f"<b>LIVE ALERT</b><br>Mag: {latest_row.get('Magnitudo', 0):.1f}"
            ).add_to(fg_live)
                
        fg_live.add_to(m)
        
        # TIDAK ADA LAYER CONTROL di mode live
        return m._repr_html_()

class ComprehensiveReporter(MapGenerator):
    def __init__(self, config: ProjectConfig):
        super().__init__()
        self.cfg = config
        self.output_dir = config.OUTPUT.directory
        self.dashboard_path_live = os.path.join(self.output_dir, "VolcanoAI_Monitor_LIVE.html")
        self.dashboard_path_static = os.path.join(self.output_dir, "VolcanoAI_Analysis_STATIC.html")
        
        self.assets = AssetManager(config)
        self.plotter = RealtimePlotter()
        
        self.GraphVisualizer = GraphVisualizer 

    def run(self, df_final: pd.DataFrame, metrics: dict, anomalies_df: pd.DataFrame):
        logger.info("Generating Dual Dashboard HTML...")
        
        # 1. Load Data Pendukung
        df_aco_all = self.assets.get_data("aco_csv")
        df_ga_all = self.assets.get_data("ga_path_csv")
        df_historical_points = df_final.copy() # Semua data yang pernah diproses
        
        # 2. Ambil Data Terbaru
        if not df_final.empty:
            latest_row = df_final.iloc[-1]
            df_buffer_view = df_final.tail(100)
        else:
            latest_row = pd.Series(); df_buffer_view = pd.DataFrame()

        # 3. Generate Maps
        html_map_static = self.generate_static_map(df_aco_all, df_ga_all, df_historical_points)
        html_map_live = self.generate_live_map(df_ga_all, latest_row)

        # 4. Generate Plot
        img_trend = self.plotter.generate_trend_plot(df_buffer_view)

        # 5. Rakit dan Simpan
        self._build_and_save_dashboard(self.dashboard_path_static, html_map_static, latest_row, img_trend, is_live=False)
        self._build_and_save_dashboard(self.dashboard_path_live, html_map_live, latest_row, img_trend, is_live=True)
        
        logger.info(f"✅ Dual Dashboard Updated. Files: LIVE, STATIC.")

    # ------------------------------------------------------------------
    # HTML WRAPPER (Sama seperti sebelumnya)
    # ------------------------------------------------------------------
    def _build_and_save_dashboard(self, file_path: str, map_html: str, latest_row: pd.Series, img_trend_b64: str, is_live: bool):
        status = latest_row.get('impact_level', 'Unknown') if not latest_row.empty else "No Data"
        status_color = {"Parah": "#e74c3c", "Sedang": "#f39c12", "Ringan": "#27ae60", "Unknown": "#95a5a6"}.get(status, "#95a5a6")
        
        mag_val = f"{latest_row.get('Magnitudo', 0):.1f}" if not latest_row.empty else "-"
        cnn_val = f"{latest_row.get('luas_cnn', 0):.2f} km²" if not latest_row.empty else "-"
        risk_score = f"{latest_row.get('PheromoneScore', 0):.3f}" if not latest_row.empty else "-"
        img_roc = self.assets.get_image_b64("nb_roc")
        
        title = "LIVE MONITOR COMMAND CENTER" if is_live else "HISTORICAL ANALYSIS & VALIDATION"
        
        html_template = f"""
        <!DOCTYPE html><html lang="id"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>VolcanoAI {title}</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background-color: #ecf0f1; }}
            .navbar {{ background-color: #2c3e50; color: white; padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }}
            .status-badge {{ background-color: #e74c3c; padding: 5px 15px; border-radius: 20px; font-weight: bold; animation: pulse 2s infinite; }}
            .grid-container {{ display: grid; grid-template-columns: 300px 1fr; gap: 20px; padding: 20px; height: calc(100vh - 70px); }}
            .card {{ background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); overflow: hidden; margin-bottom: 20px; }}
            .card-header {{ background: #f7f9fc; padding: 15px; border-bottom: 1px solid #eee; font-weight: bold; color: #34495e; }}
            .card-body {{ padding: 20px; }}
            .kpi-box {{ text-align: center; padding: 15px; background: {status_color}; color: white; border-radius: 8px; margin-bottom: 10px; }}
            .kpi-value {{ font-size: 2.5rem; font-weight: bold; }}
            .data-row {{ display: flex; justify-content: space-between; margin-bottom: 10px; border-bottom: 1px dashed #eee; padding-bottom: 5px; }}
            .data-label {{ color: #7f8c8d; }} .data-val {{ font-weight: 600; }}
            iframe {{ width: 100%; height: 100%; border: none; }} img {{ max-width: 100%; height: auto; }}
            @keyframes pulse {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} 100% {{ opacity: 1; }} }}
        </style></head><body>
            <div class="navbar"><h1>🌋 VolcanoAI {title}</h1><div class="status-badge">System Active</div></div>
            <div class="grid-container">
                <div class="sidebar">
                    <div class="card"><div class="card-header">Status Prediksi</div><div class="card-body">
                        <div class="kpi-box"><div class="kpi-value">{status}</div><div>Level Siaga</div></div>
                        <div class="data-row"><span class="data-label">Lokasi</span><span class="data-val">{latest_row.get('Nama', '-')}</span></div>
                        <div class="data-row"><span class="data-label">Waktu</span><span class="data-val">{latest_row.get('Acquired_Date', '-')}</span></div>
                    </div></div>
                    <div class="card"><div class="card-header">Metrik Fisika & AI</div><div class="card-body">
                        <div class="data-row"><span class="data-label">Magnitudo</span><span class="data-val">{mag_val} SR</span></div>
                        <div class="data-row"><span class="data-label">Kedalaman</span><span class="data-val">{latest_row.get('Kedalaman (km)', 0)} km</span></div>
                        <div class="data-row"><span class="data-label">Prediksi Area</span><span class="data-val">{cnn_val}</span></div>
                        <div class="data-row"><span class="data-label">Skor Risiko</span><span class="data-val">{risk_score}</span></div>
                    </div></div>
                    <div class="card" style="flex-grow: 1;"><div class="card-header">Validasi Model</div><div class="card-body" style="text-align: center;"><img src="data:image/png;base64,{img_roc}" alt="ROC Curve"></div></div>
                </div>
                <div class="main-content">
                    <div class="card" style="flex-grow: 2; min-height: 400px;"><div class="card-header">Peta Komando Strategis</div><div style="height: 100%;">{map_html}</div></div>
                    <div class="card" style="flex-grow: 1; min-height: 250px;"><div class="card-header">Dinamika Waktu (LSTM)</div><div class="card-body"><img src="data:image/png;base64,{img_trend_b64}" style="width: 100%; max-height: 250px; object-fit: contain;"></div></div>
                </div>
            </div>
            
            <script>
                // Auto-refresh hanya di mode LIVE (sejalan dengan interval BMKG fetch)
                if ({'true' if is_live else 'false'}) {{
                   setTimeout(function(){{
                       window.location.reload(1);
                   }}, 300000); // 5 menit
                }}
            </script>
        </body></html>
        """

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_template)
            logger.info(f"Dashboard Updated: {file_path}")
        except Exception as e:
            logger.error(f"Dashboard generation failed: {e}")