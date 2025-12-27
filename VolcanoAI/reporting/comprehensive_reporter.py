# VolcanoAI/reporting/comprehensive_reporter.py  # File utama untuk pembuatan laporan & dashboard VolcanoAI
# -- coding: utf-8 --  # Deklarasi encoding UTF-8

import datetime  # modul untuk menangani tanggal dan waktu
import os  # modul operasi filesystem dan path
import logging  # modul logging
import base64  # encode/decode base64 (mis. gambar ke HTML)
import glob  # pencarian file dengan pola
import numpy as np  # operasi numerik array
import pandas as pd  # manipulasi DataFrame
import matplotlib  # library plotting
matplotlib.use('Agg')  # set backend non-GUI (aman di server)
import matplotlib.pyplot as plt  # API plotting matplotlib
from pathlib import Path  # kelas Path untuk path modern
import seaborn as sns  # visualisasi statistik (heatmap dll.)
import folium  # library peta interaktif
import networkx as nx  # analisis graf dan jaringan
from folium import plugins  # plugin tambahan folium (HeatMap, AntPath)
from folium import LayerControl, FeatureGroup, Marker, Icon, Circle  # komponen folium yang dipakai
from branca.colormap import LinearColormap  # colormap untuk legenda peta
from io import BytesIO  # buffer in-memory untuk gambar
from typing import Dict, Any, List, Optional  # type hints

try:
    from ..config.config import ProjectConfig  # import konfigurasi proyek jika tersedia
except ImportError:
    pass  # jika tidak ada modul config, lanjut tanpa error

logger = logging.getLogger("VolcanoAI.Reporter")  # buat logger khusus untuk reporter
logger.addHandler(logging.NullHandler())  # tambahkan null handler default untuk menghindari double logging

# ============================================================================== 
# 1. GRAPH VISUALIZER (COMPATIBILITY)
# ==============================================================================
class GraphVisualizer:  # class pembantu untuk mengekspor graph ke peta (compatibility)
    def __init__(self, output_dir: str):
        self.output_dir = output_dir  # simpan direktori output
        os.makedirs(self.output_dir, exist_ok=True)  # pastikan direktori ada
    
    def export_graphs_to_maps(self, G_macro: nx.Graph, micro_graphs: Dict[str, nx.Graph]) -> Dict[str, str]:
        out = {}  # dict untuk menyimpan path hasil
        if not G_macro: 
            return out  # jika graph kosong, langsung return kosong
        try:
            lats = [d.get("latitude") for _, d in G_macro.nodes(data=True)]  # kumpulkan latitude node
            lons = [d.get("longitude") for _, d in G_macro.nodes(data=True)]  # kumpulkan longitude node
            center = [float(np.mean(lats)), float(np.mean(lons))]  # hitung titik pusat peta
            macro_map = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")  # inisialisasi peta
            for n, d in G_macro.nodes(data=True):
                folium.CircleMarker(location=[d['latitude'], d['longitude']], radius=6).add_to(macro_map)  # tambahkan marker
            macro_path = os.path.join(self.output_dir, "macro_graph_map.html")  # tentukan path output HTML
            macro_map.save(macro_path)  # simpan peta sebagai HTML
            out["macro_map"] = macro_path  # masukkan path ke dict hasil
        except Exception as e:
            logger.warning(f"Gagal export graph map: {e}")  # log jika gagal
        return out  # kembalikan dict hasil


# ============================================================================== 
# 2. ASSET MANAGER
# ==============================================================================
class AssetManager:  # class untuk mengelola jalur file aset dari engine lain
    def __init__(self, config: ProjectConfig):
        self.output_dir = config.OUTPUT.directory  # direktori output project
        self.engine_dirs = {
            "aco": config.ACO_ENGINE.output_dir,  # direktori output ACO engine
            "ga": config.GA_ENGINE.output_dir,  # direktori output GA engine
            "naive_bayes": config.NAIVE_BAYES_ENGINE.output_dir  # direktori output Naive Bayes
        }
        self.paths = {
            "aco_csv": os.path.join(self.engine_dirs["aco"], "aco_dynamic_result.csv"),  # path ACO csv
            "ga_path_csv": os.path.join(self.engine_dirs["ga"], "ga_optimized_path_for_lstm.csv"),  # path GA csv
            "final_xlsx": os.path.join(self.output_dir, "hasil_akhir_analisis.xlsx"),  # path final excel
            "nb_roc": os.path.join(self.engine_dirs["naive_bayes"], "roc_curves.png"),  # path ROC image
            "ga_conv": os.path.join(self.engine_dirs["ga"], "ga_convergence_plot.png")  # path GA convergence plot
        }

    def get_data(self, key: str) -> pd.DataFrame:
        path = self.paths.get(key)  # ambil path berdasarkan key
        if path and os.path.exists(path):
            try:
                if path.endswith('.xlsx'):
                    return pd.read_excel(path)  # baca Excel jika berekstensi xlsx
                return pd.read_csv(path)  # baca CSV jika bukan xlsx
            except Exception:
                return pd.DataFrame()  # jika gagal baca, kembalikan DataFrame kosong
        return pd.DataFrame()  # jika path tidak ada, kembalikan kosong

    def get_image_b64(self, key: str) -> str:
        path = self.paths.get(key)  # ambil path image
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return base64.b64encode(f.read()).decode('utf-8')  # kembalikan base64 string
            except Exception:
                pass  # jika baca error, lanjut
        return ""  # fallback string kosong


# ============================================================================== 
# 3. VISUAL & REPORT GENERATOR
# ==============================================================================
class RealtimePlotter:  # class statis untuk plotting tren realtime
    @staticmethod
    def generate_trend_plot(df_buffer: pd.DataFrame, window_size: int = 90) -> str:
        if df_buffer.empty: return ""  # jika buffer kosong, kembalikan string kosong
        df_view = df_buffer.tail(window_size).copy().reset_index(drop=True)  # ambil window terakhir
        plt.figure(figsize=(10, 4))  # buat figure

        if 'Magnitudo' in df_view.columns:
            plt.plot(df_view.index, df_view['Magnitudo'], label='Magnitudo Aktual', color='#2c3e50', linewidth=2)  # plot magnitudo aktual

        if 'lstm_prediction' in df_view.columns:
            plt.plot(df_view.index, df_view['lstm_prediction'], label='Prediksi LSTM', color='#e74c3c', linestyle='--', linewidth=2)  # plot prediksi LSTM

            if 'prediction_sigma' in df_view.columns:
                sigma = df_view['prediction_sigma'].fillna(0)  # isi sigma NaN dengan 0
                upper = df_view['lstm_prediction'] + (1.96 * sigma)  # batas atas CI 95%
                lower = df_view['lstm_prediction'] - (1.96 * sigma)  # batas bawah CI 95%
                plt.fill_between(df_view.index, lower, upper, color='#e74c3c', alpha=0.1, label='Confidence Interval (95%)')  # fill CI

        if not df_view.empty and 'Magnitudo' in df_view.columns:
            last_idx = df_view.index[-1]  # indeks event terakhir dalam view
            last_mag = df_view['Magnitudo'].iloc[-1]  # magnitude terakhir
            plt.scatter(last_idx, last_mag, s=100, c='red', edgecolors='black', zorder=10, label='EVENT TERKINI')  # highlight event terkini

        plt.title(f"Dinamika Seismik (Jendela {window_size} Data Terakhir)", fontsize=12)  # judul
        plt.ylabel("Magnitudo")  # label y
        plt.xlabel("Urutan Waktu (Buffer)")  # label x
        plt.legend(loc='upper left')  # legenda di kiri atas
        plt.grid(True, alpha=0.3, linestyle='--')  # grid ringan
        plt.tight_layout()  # rapikan layout

        buf = BytesIO()  # buffer in-memory
        plt.savefig(buf, format='png')  # simpan figure ke buffer
        plt.close()  # tutup figure
        return base64.b64encode(buf.getvalue()).decode('utf-8')  # kembalikan base64 PNG

class MapGenerator:  # class pembuat peta statis dan live
    def __init__(self):
        self.default_center = [-7.8, 112.5]  # koordinat pusat default (mis. Jawa Timur)

    def _add_aco_points_layer(self, df_aco: pd.DataFrame, m: folium.Map, name: str = "Area Risiko (ACO)"):
        fg = FeatureGroup(name=name, show=False)  # buat feature group untuk layer ACO
        if not df_aco.empty and 'PheromoneScore' in df_aco.columns:
            df_sorted = df_aco.sort_values('PheromoneScore')  # urut berdasarkan score
            for _, row in df_sorted.iterrows():
                score = row.get('PheromoneScore', 0)  # ambil pheromone score
                if score < 0.05: continue  # skip skor sangat kecil
                color = "#f1c40f" if score > 0.4 else "#2ecc71"  # tentukan warna
                if score > 0.8: color = "#e74c3c"  # sangat merah untuk skor tinggi
                radius = 2 + (score * 12)  # radius marker berdasarkan skor
                popup_txt = f"Risk Score: {score:.4f} | Mag: {row.get('Magnitudo', 0):.1f}"  # teks popup
                folium.CircleMarker([row['EQ_Lintang'], row['EQ_Bujur']], radius=radius, color=color, fill=True, fill_opacity=0.7, weight=1, popup=popup_txt).add_to(fg)  # tambahkan marker ke feature group
        return fg  # kembalikan feature group

    def _add_ga_path_layer(self, df_ga: pd.DataFrame, m: folium.Map, name: str = "Jalur Optimal (GA)"):
        fg = FeatureGroup(name=name, show=False)  # buat feature group untuk GA path
        if not df_ga.empty:
            try:
                coords = df_ga[['EQ_Lintang', 'EQ_Bujur']].values.tolist()  # ambil koordinat jalur
                if coords:
                    plugins.AntPath(locations=coords, use_polyline=True, dash_array=[10, 20], weight=3, color="blue", pulse_color="cyan", opacity=0.7, name=name).add_to(fg)  # tambahkan ant path animasi
            except Exception:
                pass  # jika gagal, lewati layer ini
        return fg  # kembalikan feature group

    def generate_static_map(self, df_aco_all: pd.DataFrame, df_ga_all: pd.DataFrame, df_historical_points: pd.DataFrame) -> str:
        m = folium.Map(location=self.default_center, zoom_start=8, tiles="CartoDB positron")  # inisialisasi peta statis

        fg_hist = FeatureGroup(name="A. Titik Historis (Semua Data)", show=True)  # layer titik historis
        if not df_historical_points.empty:
            for _, row in df_historical_points.head(2000).iterrows():
                folium.CircleMarker([row['EQ_Lintang'], row['EQ_Bujur']], radius=2, color='#2c3e50', fill=True, popup=f"Mag: {row.get('Magnitudo', 0)}").add_to(fg_hist)  # tambahkan titik historis (limit 2000)
        fg_hist.add_to(m)  # tambahkan layer historis ke peta

        fg_aco = self._add_aco_points_layer(df_aco_all, m, name="B. Area Risiko ACO (Lingkaran)")  # tambahkan layer ACO
        
        fg_heat = FeatureGroup(name="C. Risiko ACO (Heatmap Density)", show=False)  # layer heatmap
        if not df_aco_all.empty and 'PheromoneScore' in df_aco_all.columns:
            try:
                heat_data = df_aco_all[['EQ_Lintang', 'EQ_Bujur', 'PheromoneScore']].dropna().values.tolist()  # data heatmap
                if heat_data:
                    plugins.HeatMap(heat_data, radius=15, blur=10, gradient={0.4:'blue',0.8:'orange',1:'red'}, name="C. Risiko ACO (Heatmap Density)").add_to(fg_heat)  # tambahkan heatmap plugin
            except:
                pass  # jika gagal, lewati heatmap
        fg_heat.add_to(m)  # tambahkan heatmap ke peta

        fg_ga = self._add_ga_path_layer(df_ga_all, m, name="D. Jalur Kausalitas (GA Path)")  # tambahkan jalur GA

        folium.LayerControl(collapsed=False).add_to(m)  # tambahkan kontrol layer
        return m._repr_html_()  # kembalikan HTML representasi peta

    def generate_live_map(self, df_ga_all: pd.DataFrame, latest_row: pd.Series) -> str:
        print("=== LATEST ROW ===")  # debug print
        print(latest_row)  # debug print latest row
        print("=== COLUMNS LATEST ROW ===")  # debug
        print(latest_row.index)  # debug

        if not latest_row.empty and 'EQ_Lintang' in latest_row and 'EQ_Bujur' in latest_row:
            center = [latest_row['EQ_Lintang'], latest_row['EQ_Bujur']]  # pusat peta pada event terbaru
        else:
            center = self.default_center  # gunakan pusat default jika tidak ada latest
        m = folium.Map(location=center, zoom_start=9, tiles="CartoDB dark_matter")  # peta live dengan style gelap
        self._add_ga_path_layer(df_ga_all, m, name="Context Path (GA Snake)")  # tambahkan jalur konteks

        fg_live = FeatureGroup(name="⚠️ LIVE ALERT", show=True)  # layer live alert
        if not latest_row.empty:
            lat = latest_row.get('EQ_Lintang', 0.0)  # lat event
            lon = latest_row.get('EQ_Bujur', 0.0)  # lon event

            luas_cnn = latest_row.get('luas_cnn', 0)  # luas area prediksi CNN
            if luas_cnn > 0:
                rad_meter = np.sqrt(luas_cnn / np.pi) * 1000  # konversi area km2 ke radius meter
                folium.Circle([lat, lon], radius=rad_meter, color='#8e44ad', fill=True, fill_opacity=0.4).add_to(fg_live)  # gambar zona dampak
            folium.CircleMarker(location=[lat, lon], radius=12, color='red', fill=True, fill_color='red', fill_opacity=1.0, popup=f"<b>LIVE ALERT</b><br>Mag: {latest_row.get('Magnitudo', 0):.1f}").add_to(fg_live)  # marker event
        fg_live.add_to(m)  # tambahkan layer live ke peta
        return m._repr_html_()  # kembalikan HTML peta live

# ============================================================================== 
# 4. COMPREHENSIVE REPORTER
# ============================================================================== 
class ComprehensiveReporter(MapGenerator):  # reporter komprehensif yang mewarisi MapGenerator
    def __init__(self, config: ProjectConfig):
        super().__init__()  # inisialisasi MapGenerator
        self.cfg = config  # simpan konfigurasi project
        self.output_dir = config.OUTPUT.directory  # direktori output
        self.dashboard_path_live = os.path.join(self.output_dir, "VolcanoAI_Monitor_LIVE.html")  # path dashboard live
        self.dashboard_path_static = os.path.join(self.output_dir, "VolcanoAI_Analysis_STATIC.html")  # path dashboard statis
        
        self.assets = AssetManager(config)  # init asset manager
        self.plotter = RealtimePlotter()  # init plotter
        self.GraphVisualizer = GraphVisualizer  # simpan referensi graph visualizer
        self.logger = logger  # gunakan logger modul
        self.metrics = {}  # tempat menyimpan metrics saat run

    def run(self, df_final: pd.DataFrame, metrics: dict, anomalies_df: pd.DataFrame):
        self.logger.info("Generating Dual Dashboard HTML...")  # log start
        self.metrics = metrics  # simpan metrics lokal

        df_aco_all = self.assets.get_data("aco_csv")  # load data ACO
        df_ga_all = self.assets.get_data("ga_path_csv")  # load data GA
        df_historical_points = df_final.copy()  # salin df_final sebagai titik historis

        latest_row = pd.Series()  # placeholder untuk baris terakhir
        df_buffer_view = pd.DataFrame()  # buffer yang akan dipakai untuk plot

        # 1) realtime buffer (paling up-to-date jika monitoring berjalan)
        rt_path = os.path.join(self.output_dir, "realtime", "processed.csv")  # path buffer realtime
        try:
            if os.path.exists(rt_path):
                df_rt = pd.read_csv(rt_path, parse_dates=["Acquired_Date"])  # baca CSV realtime
                if not df_rt.empty:
                    # pastikan datetime dan urut
                    if 'Acquired_Date' in df_rt.columns:
                        df_rt['Acquired_Date'] = pd.to_datetime(df_rt['Acquired_Date'], errors='coerce')  # parse tanggal
                        df_rt.sort_values('Acquired_Date', inplace=True)  # urutkan berdasarkan tanggal
                    latest_row = df_rt.iloc[-1]  # ambil baris terakhir
                    df_buffer_view = df_rt.tail(100)  # ambil 100 baris terakhir sebagai buffer
        except Exception as e:
            self.logger.debug(f"[Reporter] gagal baca realtime buffer: {e}")  # debug log jika gagal

        # 2) jika belum ada, coba file CNN latest (hasil prediksi CNN)
        if latest_row.empty:
            cnn_latest_path = os.path.join(self.output_dir, "cnn_results", "results", "cnn_predictions_latest.csv")  # path prediksi CNN terakhir
            try:
                if os.path.exists(cnn_latest_path):
                    df_cnn = pd.read_csv(cnn_latest_path, parse_dates=["Acquired_Date"])  # baca CSV prediksi CNN
                    if not df_cnn.empty:
                        if 'Acquired_Date' in df_cnn.columns:
                            df_cnn['Acquired_Date'] = pd.to_datetime(df_cnn['Acquired_Date'], errors='coerce')  # parse tanggal
                            df_cnn.sort_values('Acquired_Date', inplace=True)  # urutkan
                        latest_row = df_cnn.iloc[-1]  # ambil baris terakhir dari CNN
                        df_buffer_view = df_cnn.tail(100)  # buffer dari CNN
            except Exception as e:
                self.logger.debug(f"[Reporter] gagal baca cnn latest: {e}")  # debug jika gagal baca file CNN

        # 3) fallback: gunakan df_final yang dipass ke reporter
        if latest_row.empty:
            try:
                if not df_final.empty:
                    df_final_local = df_final.copy()  # salin df_final
                    if 'Acquired_Date' in df_final_local.columns:
                        df_final_local['Acquired_Date'] = pd.to_datetime(df_final_local['Acquired_Date'], errors='coerce')  # parse tanggal
                        df_final_local.sort_values('Acquired_Date', inplace=True)  # urutkan
                    latest_row = df_final_local.iloc[-1] if not df_final_local.empty else pd.Series()  # ambil baris terakhir jika ada
                    df_buffer_view = df_final_local.tail(100) if not df_final_local.empty else pd.DataFrame()  # buffer fallback
            except Exception as e:
                self.logger.debug(f"[Reporter] gagal fallback df_final: {e}")  # debug jika fallback gagal

        html_map_static = self.generate_static_map(df_aco_all, df_ga_all, df_historical_points)  # buat peta statis HTML
        html_map_live = self.generate_live_map(df_ga_all, latest_row)  # buat peta live HTML
        img_trend = self.plotter.generate_trend_plot(df_buffer_view)  # buat plot trend (base64)

        self._build_and_save_dashboard(self.dashboard_path_static, html_map_static, latest_row, img_trend, is_live=False)  # simpan dashboard statis
        self._build_and_save_dashboard(self.dashboard_path_live, html_map_live, latest_row, img_trend, is_live=True)  # simpan dashboard live
        self.logger.info(f"✅ Dual Dashboard Updated. Files: LIVE, STATIC.")  # log selesai

    def _build_and_save_dashboard(self, dashboard_path, html_map, latest_row, img_trend, is_live=False):
        dashboard_path = Path(dashboard_path)  # pastikan path adalah Path object

        if is_live:
            template_path = Path(__file__).parent / "templates" / "monitor_live_template.html"  # template live
        else:
            template_path = Path(__file__).parent / "static_dashboard_template.html"  # template statis

        if not template_path.exists():
            self.logger.error(f"Template tidak ditemukan: {template_path}")  # log error jika template hilang
            return  # hentikan eksekusi

        template_html = _safe_read_text(template_path, logger=self.logger)  # baca template dengan safe reader

        def getm(key, default="-"):
            v = self.metrics.get(key)  # ambil value dari metrics
            if v is None:
                return default  # default jika tidak ada
            if isinstance(v, (list, tuple)):
                return ", ".join(map(str, v))  # gabungkan list/tuple jadi string
            return str(v)  # konversi ke string

        data = {
            "TIMESTAMP": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp sekarang
            "ACO_IMPACT_CENTER": getm("aco_center"),  # center impact ACO
            "ACO_IMPACT_AREA": getm("aco_area"),  # luas impact ACO
            "ACO_MAP": html_map or getm("aco_map", ""),  # embed peta ACO atau fallback
            "GA_MAP": getm("ga_map", ""),  # embed peta GA
            "GA_PRED_LAT": getm("ga_lat"),  # GA predicted lat
            "GA_PRED_LON": getm("ga_lon"),  # GA predicted lon
            "GA_BEARING": getm("ga_angle"),  # GA bearing
            "GA_DISTANCE": getm("ga_distance"),  # GA distance
            "GA_CONFIDENCE": getm("ga_confidence"),  # GA confidence
            "LATEST_ROW_HTML": latest_row.to_frame().T.to_html(index=False) if not latest_row.empty else "<i>-</i>",  # latest row sebagai HTML
            "LSTM_MASTER_CSV": getm("lstm_master_csv", ""),  # link/placeholder LSTM master CSV
            "LSTM_RECENT_CSV": getm("lstm_recent_csv", ""),  # link/placeholder LSTM recent CSV
            "LSTM_ANOMALIES_CSV": getm("lstm_anomalies_csv", ""),  # link/placeholder LSTM anomalies
            "CNN_PRED_CSV": getm("cnn_pred_csv", ""),  # link/placeholder CNN predictions CSV
            "CNN_PRED_JSON": getm("cnn_pred_json", ""),  # link/placeholder CNN predictions JSON
            "NB_REPORT_STR": getm("nb_report", "")  # string laporan Naive Bayes
        }

        for k, v in data.items():
            template_html = template_html.replace(f"{{{{{k}}}}}", str(v))  # replace placeholder dalam template

        dashboard_path.parent.mkdir(parents=True, exist_ok=True)  # pastikan direktori tujuan ada
        dashboard_path.write_text(template_html, encoding="utf-8")  # tulis file dashboard HTML
        self.logger.info(f"Dashboard berhasil dibuat: {dashboard_path}")  # log sukses

# ============================================================================== 
# 5. HELPER FUNCTION OUTSIDE CLASS
# ============================================================================== 
def _safe_read_text(path: Path, logger=None) -> str:
    encodings = ["utf-8", "cp1252", "latin-1"]  # daftar encoding yang dicoba
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)  # coba baca dengan encoding saat ini
        except UnicodeDecodeError:
            if logger:
                logger.warning(f"Gagal decode {path} dengan {enc}, mencoba lain...")  # peringatan decode gagal
        except Exception as e:
            if logger:
                logger.warning(f"Error baca {path} dengan {enc}: {e}")  # log error baca file
    raw = path.read_bytes()  # baca byte mentah jika semua encoding gagal
    return raw.decode("utf-8", errors="replace")  # decode byte dengan replace untuk karakter problematik
