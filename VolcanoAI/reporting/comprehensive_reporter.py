# VolcanoAI/reporting/comprehensive_reporter.py
# -- coding: utf-8 --

import datetime
import os
import logging
import base64
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import folium
import networkx as nx
from folium import plugins
from folium import LayerControl, FeatureGroup, Marker, Icon, Circle
from branca.colormap import LinearColormap
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
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_graphs_to_maps(self, G_macro: nx.Graph, micro_graphs: Dict[str, nx.Graph]) -> Dict[str, str]:
        out = {}
        if not G_macro: 
            return out
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
            logger.warning(f"Gagal export graph map: {e}")
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
            "final_xlsx": os.path.join(self.output_dir, "hasil_akhir_analisis.xlsx"),
            "nb_roc": os.path.join(self.engine_dirs["naive_bayes"], "roc_curves.png"),
            "ga_conv": os.path.join(self.engine_dirs["ga"], "ga_convergence_plot.png")
        }

    def get_data(self, key: str) -> pd.DataFrame:
        path = self.paths.get(key)
        if path and os.path.exists(path):
            try:
                if path.endswith('.xlsx'):
                    return pd.read_excel(path)
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def get_image_b64(self, key: str) -> str:
        path = self.paths.get(key)
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')
            except Exception:
                pass
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
        plt.ylabel("Magnitudo")
        plt.xlabel("Urutan Waktu (Buffer)")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')


class MapGenerator:
    def __init__(self):
        self.default_center = [-7.8, 112.5]

    def _add_aco_points_layer(self, df_aco: pd.DataFrame, m: folium.Map, name: str = "Area Risiko (ACO)"):
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
        fg = FeatureGroup(name=name, show=False)
        if not df_ga.empty:
            try:
                coords = df_ga[['EQ_Lintang', 'EQ_Bujur']].values.tolist()
                if coords:
                    plugins.AntPath(locations=coords, use_polyline=True, dash_array=[10, 20], weight=3, color="blue", pulse_color="cyan", opacity=0.7, name=name).add_to(fg)
            except Exception:
                pass
        return fg

    def generate_static_map(self, df_aco_all: pd.DataFrame, df_ga_all: pd.DataFrame, df_historical_points: pd.DataFrame) -> str:
        m = folium.Map(location=self.default_center, zoom_start=8, tiles="CartoDB positron")

        fg_hist = FeatureGroup(name="A. Titik Historis (Semua Data)", show=True)
        if not df_historical_points.empty:
            for _, row in df_historical_points.head(2000).iterrows():
                folium.CircleMarker([row['EQ_Lintang'], row['EQ_Bujur']], radius=2, color='#2c3e50', fill=True, popup=f"Mag: {row.get('Magnitudo', 0)}").add_to(fg_hist)
        fg_hist.add_to(m)

        fg_aco = self._add_aco_points_layer(df_aco_all, m, name="B. Area Risiko ACO (Lingkaran)")
        
        fg_heat = FeatureGroup(name="C. Risiko ACO (Heatmap Density)", show=False)
        if not df_aco_all.empty and 'PheromoneScore' in df_aco_all.columns:
            try:
                heat_data = df_aco_all[['EQ_Lintang', 'EQ_Bujur', 'PheromoneScore']].dropna().values.tolist()
                if heat_data:
                    plugins.HeatMap(heat_data, radius=15, blur=10, gradient={0.4:'blue',0.8:'orange',1:'red'}, name="C. Risiko ACO (Heatmap Density)").add_to(fg_heat)
            except:
                pass
        fg_heat.add_to(m)

        fg_ga = self._add_ga_path_layer(df_ga_all, m, name="D. Jalur Kausalitas (GA Path)")

        folium.LayerControl(collapsed=False).add_to(m)
        return m._repr_html_()

    def generate_live_map(self, df_ga_all: pd.DataFrame, latest_row: pd.Series) -> str:
        center = [latest_row['EQ_Lintang'], latest_row['EQ_Bujur']] if not latest_row.empty else self.default_center
        m = folium.Map(location=center, zoom_start=9, tiles="CartoDB dark_matter")
        self._add_ga_path_layer(df_ga_all, m, name="Context Path (GA Snake)")

        fg_live = FeatureGroup(name="⚠️ LIVE ALERT", show=True)
        if not latest_row.empty:
            lat, lon = latest_row['EQ_Lintang'], latest_row['EQ_Bujur']
            luas_cnn = latest_row.get('luas_cnn', 0)
            if luas_cnn > 0:
                rad_meter = np.sqrt(luas_cnn / np.pi) * 1000
                folium.Circle([lat, lon], radius=rad_meter, color='#8e44ad', fill=True, fill_opacity=0.4).add_to(fg_live)
            folium.CircleMarker(location=[lat, lon], radius=12, color='red', fill=True, fill_color='red', fill_opacity=1.0, popup=f"<b>LIVE ALERT</b><br>Mag: {latest_row.get('Magnitudo', 0):.1f}").add_to(fg_live)
        fg_live.add_to(m)
        return m._repr_html_()


# ============================================================================== 
# 4. COMPREHENSIVE REPORTER
# ==============================================================================
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
        self.logger = logger
        self.metrics = {} 

    def run(self, df_final: pd.DataFrame, metrics: dict, anomalies_df: pd.DataFrame):
        self.logger.info("Generating Dual Dashboard HTML...")
        self.metrics = metrics

        df_aco_all = self.assets.get_data("aco_csv")
        df_ga_all = self.assets.get_data("ga_path_csv")
        df_historical_points = df_final.copy()

        if not df_final.empty:
            latest_row = df_final.iloc[-1]
            df_buffer_view = df_final.tail(100)
        else:
            latest_row = pd.Series()
            df_buffer_view = pd.DataFrame()

        html_map_static = self.generate_static_map(df_aco_all, df_ga_all, df_historical_points)
        html_map_live = self.generate_live_map(df_ga_all, latest_row)
        img_trend = self.plotter.generate_trend_plot(df_buffer_view)

        self._build_and_save_dashboard(self.dashboard_path_static, html_map_static, latest_row, img_trend, is_live=False)
        self._build_and_save_dashboard(self.dashboard_path_live, html_map_live, latest_row, img_trend, is_live=True)
        self.logger.info(f"✅ Dual Dashboard Updated. Files: LIVE, STATIC.")

    def _build_and_save_dashboard(self, dashboard_path, html_map, latest_row, img_trend, is_live=False):
        dashboard_path = Path(dashboard_path)

        if is_live:
            template_path = Path(__file__).parent / "templates" / "monitor_live_template.html"
        else:
            template_path = Path(__file__).parent / "static_dashboard_template.html"

        if not template_path.exists():
            self.logger.error(f"Template tidak ditemukan: {template_path}")
            return

        template_html = _safe_read_text(template_path, logger=self.logger)

        def getm(key, default="-"):
            v = self.metrics.get(key)
            if v is None:
                return default
            if isinstance(v, (list, tuple)):
                return ", ".join(map(str, v))
            return str(v)

        data = {
            "TIMESTAMP": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "ACO_IMPACT_CENTER": getm("aco_center"),
            "ACO_IMPACT_AREA": getm("aco_area"),
            "ACO_MAP": html_map or getm("aco_map", ""),
            "GA_MAP": getm("ga_map", ""),
            "GA_PRED_LAT": getm("ga_lat"),
            "GA_PRED_LON": getm("ga_lon"),
            "GA_BEARING": getm("ga_angle"),
            "GA_DISTANCE": getm("ga_distance"),
            "GA_CONFIDENCE": getm("ga_confidence"),
            "LATEST_ROW_HTML": latest_row.to_frame().T.to_html(index=False) if not latest_row.empty else "<i>-</i>",
            "LSTM_MASTER_CSV": getm("lstm_master_csv", ""),
            "LSTM_RECENT_CSV": getm("lstm_recent_csv", ""),
            "LSTM_ANOMALIES_CSV": getm("lstm_anomalies_csv", ""),
            "CNN_PRED_CSV": getm("cnn_pred_csv", ""),
            "CNN_PRED_JSON": getm("cnn_pred_json", ""),
            "NB_REPORT_STR": getm("nb_report", "")
        }

        for k, v in data.items():
            template_html = template_html.replace(f"{{{{{k}}}}}", str(v))

        dashboard_path.parent.mkdir(parents=True, exist_ok=True)
        dashboard_path.write_text(template_html, encoding="utf-8")
        self.logger.info(f"Dashboard berhasil dibuat: {dashboard_path}")


# ============================================================================== 
# 5. HELPER FUNCTION OUTSIDE CLASS
# ==============================================================================
def _safe_read_text(path: Path, logger=None) -> str:
    encodings = ["utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            if logger:
                logger.warning(f"Gagal decode {path} dengan {enc}, mencoba lain...")
        except Exception as e:
            if logger:
                logger.warning(f"Error baca {path} dengan {enc}: {e}")
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace")
