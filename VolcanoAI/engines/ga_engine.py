# ============================================
# VolcanoAI/engines/ga_engine.py
# GA Engine + Vector Prediction + Map Popup
# ============================================

import os
import sys
import json
import time
import math
import random
import shutil
import pickle
import functools
import logging
import warnings
import uuid
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterable

import numpy as np
import pandas as pd
import networkx as nx
import folium
from folium import plugins
from folium.plugins import AntPath, HeatMap, Fullscreen, MiniMap, MeasureControl

from deap import base, creator, tools, algorithms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform


# ===========================
# Logging
# ===========================
logger = logging.getLogger("VolcanoAI.GaEngine")
logger.addHandler(logging.NullHandler())


# ========================================================
# 🔥 FIX: Prevent DEAP Creator Crash (REQUIRED)
# ========================================================
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# ===========================
# Utility Decorators
# ===========================
def execution_monitor(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_ts = time.perf_counter()
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {str(e)}")
            raise e
        finally:
            end_ts = time.perf_counter()
            duration = end_ts - start_ts
            if duration > 1.0:
                logger.debug(f"Long execution detected in {func.__name__}: {duration:.4f}s")
    return wrapper


# ===========================
# GEO MATH CORE
# ===========================
class GeoMathCore:
    R_EARTH_KM = 6371.0088

    @staticmethod
    def to_radians(array_like):
        return np.radians(array_like)

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2):
        """Bearing (sudut) dari titik 1 → titik 2 dalam derajat 0-360."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        diff_lon = math.radians(lon2 - lon1)

        x = math.sin(diff_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(diff_lon))

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    @classmethod
    def haversine(cls, lat1, lon1, lat2, lon2):
        """Jarak permukaan bumi (km) antar dua koordinat."""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return cls.R_EARTH_KM * c


# =====================================
# DATA SANITIZER
# =====================================
class DataSanitizer:
    def __init__(self):
        self.required_columns = [
            'EQ_Lintang', 'EQ_Bujur', 'Acquired_Date',
            'PheromoneScore', 'Magnitudo', 'Kedalaman (km)'
        ]
        self.min_rows = 5

    @execution_monitor
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None:
            raise ValueError("Input DataFrame cannot be None")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df = df.copy()
        df['Acquired_Date'] = pd.to_datetime(df['Acquired_Date'], errors='coerce')
        df = df.dropna(subset=['Acquired_Date', 'EQ_Lintang', 'EQ_Bujur'])

        for c in ['EQ_Lintang', 'EQ_Bujur', 'Magnitudo', 'Kedalaman (km)', 'PheromoneScore']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        df = df[(df['EQ_Lintang'] >= -90) & (df['EQ_Lintang'] <= 90)]
        df = df[(df['EQ_Bujur'] >= -180) & (df['EQ_Bujur'] <= 180)]

        df = df.reset_index(drop=True)

        if len(df) < self.min_rows:
            raise ValueError(f"Min rows not met: {len(df)}")

        return df


# =====================================
# PHYSICS FITNESS ENGINE
# =====================================
class PhysicsFitnessEngine:
    def __init__(self, df: pd.DataFrame, weight_config: Dict[str, float]):
        self.df = df
        self.weights = weight_config

        self.vec_lat = df['EQ_Lintang'].values
        self.vec_lon = df['EQ_Bujur'].values
        self.vec_time = df['Acquired_Date'].values.astype(np.int64)
        self.vec_mag = df['Magnitudo'].values
        self.vec_depth = df['Kedalaman (km)'].values
        self.vec_risk = df['PheromoneScore'].values

        self.num_nodes = len(df)
        self.eps = 1e-6
        self.penalty = 1e15

        # Extract weights
        self.w_time = weight_config.get("time", 10000)
        self.w_space = weight_config.get("space", 1)
        self.w_mag = weight_config.get("mag", 50)
        self.w_depth = weight_config.get("depth", 10)
        self.w_risk = weight_config.get("risk", 500)

    # ---------------------------
    # FITNESS FUNCTION
    # ---------------------------
    def evaluate(self, individual: List[int]) -> Tuple[float]:
        """
        Fungsi Evaluasi (Fitness Function) untuk Algoritma Genetika.
        Tujuan: MEMINIMALKAN Total Cost.
        
        Args:
            individual: List integer yang merepresentasikan urutan kunjungan node.
            
        Returns:
            Tuple (total_cost, ) -> Tuple wajib untuk library DEAP/PyGAD.
        """
        # Konversi ke array index integer
        idx = np.array(individual, dtype=int)

        # 1. Validasi Panjang Kromosom
        if len(idx) != self.num_nodes:
            # Return cost penalty sangat besar jika kromosom cacat
            return (self.penalty, )

        # Ambil data dari vector (lookup table) berdasarkan urutan gen individu
        # Asumsi variable vec_* sudah diinisialisasi di __init__ dan berbentuk numpy array
        curr_mags = self.vec_mag[idx]
        curr_depths = self.vec_depth[idx]
        curr_times = self.vec_time[idx]
        curr_risks = self.vec_risk[idx]

        # -------------------------------------------------------------
        # 1. TEMPORAL VIOLATION COST
        # -------------------------------------------------------------
        # Logika: Sebaiknya node dikunjungi urut waktu.
        # Menghitung berapa kali waktu bergerak 'mundur' (t_next < t_prev)
        # Jika t[i+1] < t[i], itu violation (Boolean True = 1)
        time_violations_count = np.sum(curr_times[1:] < curr_times[:-1])
        
        # [FIX]: Bobot pelanggaran waktu harus sangat besar agar individu valid bertahan
        temporal_cost = self.w_time * float(time_violations_count)

        # -------------------------------------------------------------
        # 2. SPATIAL DISTANCE COST
        # -------------------------------------------------------------
        # Logika: Minimalkan total jarak tempuh (Traveling Salesman Problem standard)
        
        # Ambil koordinat berurut sesuai path
        lat_seq = self.vec_lat[idx]
        lon_seq = self.vec_lon[idx]

        # Hitung jarak antara titik i dan i+1 secara sekuensial
        # Menggunakan loop list comprehension (lebih aman) atau vectorized jika GeoMathCore mendukung
        # Lat1=Steps 0 ke N-1, Lat2=Steps 1 ke N
        distances = []
        for i in range(len(idx) - 1):
            d = GeoMathCore.haversine(
                lat_seq[i], lon_seq[i], 
                lat_seq[i+1], lon_seq[i+1]
            )
            distances.append(d)
        
        total_dist = sum(distances)
        spatial_cost = self.w_space * total_dist

        # -------------------------------------------------------------
        # 3. PHYSICS & PRIORITY COST
        # -------------------------------------------------------------
        
        # [Safety Clip]: Cegah pembagian dengan nol atau nilai tak wajar
        clipped_mag = np.clip(curr_mags, 0.1, None)      # M minimal 0.1
        clipped_depth = np.clip(curr_depths, 1.0, None)  # Depth minimal 1.0 km
        clipped_risk = np.clip(curr_risks, 1e-6, None)   # Risk minimal epsilon

        # A. MAGNITUDE COST
        # Kita ingin PRIORITAS pada Magnitudo BESAR.
        # Cost Function harus MENGECIL saat Magnitudo MEMBESAR.
        # Formula: Cost = Sum(1 / M)
        mag_cost = self.w_mag * np.sum(1.0 / clipped_mag)

        # B. DEPTH COST
        # Kita ingin PRIORITAS pada Kedalaman DANGKAL (Kecil).
        # Cost Function harus MENGECIL saat Kedalaman MENGECIL.
        # Formula: Cost = Sum(Depth)
        depth_cost = self.w_depth * np.sum(clipped_depth)

        # C. RISK SCORE COST (Pheromone/External Risk)
        # Kita ingin PRIORITAS pada Risk TINGGI.
        # Cost Function harus MENGECIL saat Risk MEMBESAR.
        # Formula: Cost = Sum(1 / Risk)
        risk_cost = self.w_risk * np.sum(1.0 / clipped_risk)

        # -------------------------------------------------------------
        # 4. TOTAL COST CALCULATION
        # -------------------------------------------------------------
        total_cost = (
            temporal_cost +
            spatial_cost +
            mag_cost +
            depth_cost +
            risk_cost
        )

        # Pastikan return value aman untuk dikonsumsi algoritma
        if np.isnan(total_cost) or np.isinf(total_cost):
            return (self.penalty, )

        return (total_cost, )

    # -------------------------------------------------------------
    # 🔥 PREDIKSI NEXT EVENT (berdasarkan 3 titik terakhir)
    # -------------------------------------------------------------
    def predict_next_event(self, df_path: pd.DataFrame) -> Dict[str, float]:

        if len(df_path) < 3:
            return {}

        last = df_path.iloc[-1]
        prev = df_path.iloc[-2]
        prev2 = df_path.iloc[-3]

        # Vector perubahan
        dlat1 = prev['EQ_Lintang'] - prev2['EQ_Lintang']
        dlon1 = prev['EQ_Bujur'] - prev2['EQ_Bujur']

        dlat2 = last['EQ_Lintang'] - prev['EQ_Lintang']
        dlon2 = last['EQ_Bujur'] - prev['EQ_Bujur']

        # Rata-rata vektor
        dlat = (dlat1 + dlat2) / 2
        dlon = (dlon1 + dlon2) / 2

        # Skala gerakan berbasis magnitudo & risk
        scale = 0.5 + (last['Magnitudo'] / 10) + (last['PheromoneScore'])
        dlat *= scale
        dlon *= scale

        predicted_lat = last['EQ_Lintang'] + dlat
        predicted_lon = last['EQ_Bujur'] + dlon

        return {
            "pred_lat": float(predicted_lat),
            "pred_lon": float(predicted_lon),
            "base_lat": float(last['EQ_Lintang']),
            "base_lon": float(last['EQ_Bujur']),
            "movement_scale": float(scale)
        }


# ============================================
# BLOCK 2/3 — Evolutionary Controller + Checkpoint
# ============================================

class CheckpointSystem:
    def __init__(self, directory: str, filename: str = "ga_state.pkl"):
        self.directory = os.path.join(directory, "checkpoints")
        self.filename = filename
        self.filepath = os.path.join(self.directory, filename)
        os.makedirs(self.directory, exist_ok=True)

    def save_state(self, population, generation, stats, filename=None):
        fname = filename if filename else self.filename
        fpath = os.path.join(self.directory, fname)

        payload = {
            "version": "6.0",
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "population": population,
            "stats": stats
        }

        try:
            with open(fpath, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_state(self):
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


class StatisticsCollector:
    def __init__(self):
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def get(self):
        return self.stats


class EvolutionaryController:
    def __init__(self, config, fitness_engine, checkpoint_mgr):
        self.cfg = config
        self.fitness_engine = fitness_engine
        self.checkpoint_mgr = checkpoint_mgr

        self.toolbox = base.Toolbox()
        self.stats_col = StatisticsCollector()
        self.stats = self.stats_col.get()

        self._register_operators()

    def _register_operators(self):
        problem_size = self.fitness_engine.num_nodes

        # Gene generator → a permutation
        self.toolbox.register("indices", np.random.permutation, problem_size)

        # Chromosome Initialization
        def init_chromosome(icls, generator):
            return icls(generator().tolist())

        self.toolbox.register("individual", init_chromosome,
                              creator.Individual, self.toolbox.indices)

        self.toolbox.register("population", tools.initRepeat,
                              list, self.toolbox.individual)

        # Fitness
        self.toolbox.register("evaluate", self.fitness_engine.evaluate)

        # Operators
        self.toolbox.register("mate", tools.cxOrdered)
        self.toolbox.register("mutate", tools.mutShuffleIndexes,
                              indpb=self.cfg.mutation_prob)
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.cfg.tournament_size)

    @execution_monitor
    def run(self):
        pop_size = self.cfg.population_size
        n_gen = self.cfg.n_generations
        cx_prob = self.cfg.crossover_prob
        mut_prob = self.cfg.mutation_prob

        logger.info(f"GA Evolution Start → Pop={pop_size}, Gen={n_gen}")

        # Initial population
        pop = self.toolbox.population(n=pop_size)
        hof = tools.HallOfFame(self.cfg.hall_of_fame_size)

        # Load checkpoint if exists
        state = self.checkpoint_mgr.load_state()
        if state:
            logger.info("Resuming GA from checkpoint...")
            pop = state["population"]
            start_gen = state["generation"]
        else:
            start_gen = 0

        # Initial evaluation
        invalid = [ind for ind in pop if not ind.fitness.valid]
        fits = map(self.toolbox.evaluate, invalid)

        for ind, fval in zip(invalid, fits):
            ind.fitness.values = fval

        hof.update(pop)

        # Statistics logbook
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + self.stats.fields

        record = self.stats.compile(pop)
        logbook.record(gen=start_gen, nevals=len(invalid), **record)

        best_so_far = record["min"]
        stagnation = 0

        # GENERATIONAL LOOP
        for gen in range(start_gen + 1, n_gen + 1):

            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            # Adaptive Mutation
            adaptive_mut = mut_prob
            if stagnation > 20:
                adaptive_mut = min(0.9, mut_prob * 2.0)

            for mutant in offspring:
                if random.random() < adaptive_mut:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            newfits = map(self.toolbox.evaluate, invalid)
            for ind, fval in zip(invalid, newfits):
                ind.fitness.values = fval

            # Replace old population
            pop[:] = offspring
            hof.update(pop)

            # Logging statistics
            rec = self.stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid), **rec)

            # Stagnation check
            if rec["min"] < best_so_far:
                best_so_far = rec["min"]
                stagnation = 0
            else:
                stagnation += 1

            # Save every 10 generations
            if gen % 10 == 0:
                self.checkpoint_mgr.save_state(pop, gen, logbook)
                logger.info(f"Gen {gen}: Current Best = {best_so_far:.2f}")

        # Save final state
        self.checkpoint_mgr.save_state(pop, n_gen, logbook, filename="final_ga.pkl")

        best_individual = list(hof[0])
        log_df = pd.DataFrame(logbook)

        return best_individual, log_df, hof


# ============================================
# BLOCK 3/3 — Visualizer + Exporter + GA Engine Wrapper
# ============================================

class MultiLayerVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, "visuals")
        os.makedirs(self.viz_dir, exist_ok=True)

    def clamp_to_east_java(self, lat: float, lon: float) -> Tuple[float, float]:
        LAT_MIN, LAT_MAX = -9.5, -5.5
        LON_MIN, LON_MAX = 110.0, 116.0

        clamped_lat = max(min(lat, LAT_MAX), LAT_MIN)
        clamped_lon = max(min(lon, LON_MAX), LON_MIN)

        return clamped_lat, clamped_lon

    def generate_map(
        self,
        best_path: List[int],
        df: pd.DataFrame,
        pred_info: Dict[str, float],
        out_path: str
    ):
        if not best_path:
            logger.warning("GA Visualizer: best_path kosong, map tidak dibuat.")
            return

        center_lat = df['EQ_Lintang'].mean()
        center_lon = df['EQ_Bujur'].mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None)
        folium.TileLayer('CartoDB positron', name='Light').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)

        # CHAOS LAYER
        chaos = folium.FeatureGroup(name="Chaos Connectivity", show=False)

        coords = df[['EQ_Lintang', 'EQ_Bujur']].values
        if len(coords) > 0:
            sample = coords[: min(400, len(coords))]

            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    dist = GeoMathCore.haversine(
                        sample[i][0], sample[i][1],
                        sample[j][0], sample[j][1]
                    )
                    if dist < 40:
                        folium.PolyLine(
                            [sample[i], sample[j]],
                            color="cyan",
                            weight=0.5,
                            opacity=0.25
                        ).add_to(chaos)

            for r in df.itertuples():
                folium.CircleMarker(
                    [getattr(r, 'EQ_Lintang'), getattr(r, 'EQ_Bujur')],
                    radius=2,
                    color="#ffffff",
                    fill=True,
                    fill_opacity=0.4
                ).add_to(chaos)

        chaos.add_to(m)

        # SNAKE LAYER (Best GA path)
        snake = folium.FeatureGroup(name="Snake Path (Best Sequence)", show=True)

        best_df = df.iloc[best_path].reset_index(drop=True)
        path_coords = best_df[['EQ_Lintang', 'EQ_Bujur']].values.tolist()

        if len(path_coords) >= 2:
            AntPath(
                locations=path_coords,
                color="magenta",
                pulse_color="yellow",
                weight=4,
                delay=600
            ).add_to(snake)

        if len(path_coords) >= 1:
            folium.Marker(
                path_coords[0],
                icon=folium.Icon(color="green", icon="play"),
                popup="START"
            ).add_to(snake)

            folium.Marker(
                path_coords[-1],
                icon=folium.Icon(color="red", icon="stop"),
                popup="END"
            ).add_to(snake)

        # Popup info untuk tiap titik di snake (+ sudut & jarak segmen)
        for i, r in enumerate(best_df.itertuples()):
            row = r._asdict()

            lat = row['EQ_Lintang']
            lon = row['EQ_Bujur']

            # hitung jarak & sudut dari titik sebelumnya (kalau ada)
            if i == 0:
                seg_dist = None
                seg_bearing = None
            else:
                prev = best_df.iloc[i - 1]
                seg_dist = GeoMathCore.haversine(
                    prev['EQ_Lintang'], prev['EQ_Bujur'],
                    lat, lon
                )
                seg_bearing = GeoMathCore.calculate_bearing(
                    prev['EQ_Lintang'], prev['EQ_Bujur'],
                    lat, lon
                )

            dist_str = f"{seg_dist:.2f} km" if seg_dist is not None else "-"
            bearing_str = f"{seg_bearing:.1f}°" if seg_bearing is not None else "-"

            popup = f"""
            <b>Event Detail</b><br>
            Date: {row.get('Acquired_Date', 'N/A')}<br>
            Mag: {row.get('Magnitudo', 'N/A')}<br>
            Depth: {row.get('Kedalaman (km)', 'N/A')} km<br>
            Risk: {row.get('PheromoneScore', 0):.3f}<br>
            Segment Distance: {dist_str}<br>
            Segment Bearing: {bearing_str}<br>
            """

            folium.CircleMarker(
                [lat, lon],
                radius=4,
                color="orange",
                fill=True,
                fill_color="yellow",
                fill_opacity=0.8,
                popup=folium.Popup(popup, max_width=320)
            ).add_to(snake)

        snake.add_to(m)

        # PREDICTION LAYER
        if pred_info and "pred_lat" in pred_info and "pred_lon" in pred_info:
            raw_lat = pred_info["pred_lat"]
            raw_lon = pred_info["pred_lon"]

            pred_lat, pred_lon = self.clamp_to_east_java(raw_lat, raw_lon)

            popup_pred = f"""
            <b>Predicted Next Event</b><br>
            Lat: {pred_lat:.4f}<br>
            Lon: {pred_lon:.4f}<br>
            Scale: {pred_info.get('movement_scale', 0):.3f}
            """

            folium.Marker(
                [pred_lat, pred_lon],
                popup=folium.Popup(popup_pred, max_width=300),
                icon=folium.Icon(color="purple", icon="star")
            ).add_to(m)

            last = best_df.iloc[-1]
            folium.PolyLine(
                [
                    [last['EQ_Lintang'], last['EQ_Bujur']],
                    [pred_lat, pred_lon]
                ],
                color="purple",
                weight=4,
                opacity=0.7
            ).add_to(m)

        # Heatmap risiko (optional)
        try:
            if 'PheromoneScore' in df.columns:
                heat_data = df[['EQ_Lintang', 'EQ_Bujur', 'PheromoneScore']].values.tolist()
                HeatMap(
                    heat_data,
                    name="Risk Heatmap",
                    radius=15,
                    blur=10,
                    show=False
                ).add_to(m)
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MiniMap().add_to(m)

        m.save(out_path)
        logger.info(f"GA Map saved → {out_path}")


class DataExporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.excel_path = os.path.join(output_dir, "ga_report.xlsx")

    def export(self, df_original: pd.DataFrame, df_optimal: pd.DataFrame, meta: Dict[str, Any]):
        try:
            with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                df_original.to_excel(writer, sheet_name="RawData", index=False)
                df_optimal.to_excel(writer, sheet_name="BestPath", index=False)
                pd.DataFrame([meta]).to_excel(writer, sheet_name="Meta", index=False)

            logger.info(f"Excel exported → {self.excel_path}")
        except Exception as e:
            logger.error(f"Excel export failed: {e}")


# ==================================================
# GA ENGINE WRAPPER (Dipanggil Pipeline)
# ==================================================
class GaEngine:
    def __init__(self, config: Any):
        self.cfg = config
        self.output_dir = getattr(config, "output_dir", "output/ga_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.sanitizer = DataSanitizer()
        self.checkpoint_mgr = CheckpointSystem(self.output_dir)
        self.visualizer = MultiLayerVisualizer(self.output_dir)
        self.exporter = DataExporter(self.output_dir)

        self.map_path = os.path.join(self.output_dir, "ga_path_map.html")
        self.log_path = os.path.join(self.output_dir, "ga_log.csv")

    def run(self, df_train: pd.DataFrame) -> Tuple[List[int], Dict[str, Any]]:
        logger.info("\n" + "=" * 80)
        logger.info("=== GA ENGINE START ===".center(80))
        logger.info("=" * 80)

        # 1. Sanitization
        clean_df = self.sanitizer.execute(df_train)

        # 2. Fitness Engine
        fit_engine = PhysicsFitnessEngine(clean_df, self.cfg.fitness_weights)

        # 3. Evolution Process
        evo = EvolutionaryController(self.cfg, fit_engine, self.checkpoint_mgr)
        best_idx, log_df, hof = evo.run()

        if isinstance(best_idx, tuple):
            best_idx = list(best_idx)

        if len(best_idx) == 1 and isinstance(best_idx[0], tuple):
            best_idx = list(best_idx[0])

        best_idx = [int(x) for x in best_idx]

        # 4. Best Path DF
        df_opt = clean_df.iloc[best_idx].reset_index(drop=True)

        # 5. Prediction module
        pred = fit_engine.predict_next_event(df_opt)

        # 6. Visualization
        self.visualizer.generate_map(
            best_idx,
            clean_df,
            pred,
            self.map_path
        )

        # 7. Export Excel + meta
        meta = {
            "Timestamp": datetime.now().isoformat(),
            "Best_Cost": float(log_df["min"].iloc[-1]) if not log_df.empty else None,
            "Node_Count": int(len(clean_df)),
            "PredictedLat": pred.get("pred_lat", None),
            "PredictedLon": pred.get("pred_lon", None),
            "MovementScale": pred.get("movement_scale", None),
        }
        self.exporter.export(clean_df, df_opt, meta)

        # 8. Save GA log
        try:
            log_df.to_csv(self.log_path, index=False)
        except Exception as e:
            logger.error(f"Failed to save GA log CSV: {e}")

        logger.info("=== GA ENGINE COMPLETE ===".center(80))

        return best_idx, {"map": self.map_path, "prediction": pred}
