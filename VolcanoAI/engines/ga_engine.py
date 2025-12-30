# ============================================
# VolcanoAI/engines/ga_engine.py
# GA Engine + Vector Prediction + Map Popup
# ============================================

import os # operating system
import sys # system
import json # JSON handling
import time # time measurement
import math # mathematical functions
import random # random number generation
import shutil # file operations
import pickle # object serialization
import functools # function tools
import logging # logging
import warnings # warnings management
import uuid # unique identifiers
from datetime import datetime # date and time handling
from typing import Dict, Any, List, Tuple, Optional, Union, Callable, Iterable # type hints

import numpy as np # numerical computing
import pandas as pd # data manipulation
import networkx as nx # graph algorithms
import folium # interactive maps
from folium import plugins # folium plugins
from folium.plugins import AntPath, HeatMap, Fullscreen, MiniMap, MeasureControl # folium extras

from deap import base, creator, tools, algorithms # DEAP evolutionary algorithms

import matplotlib # plotting
matplotlib.use("Agg") # non-GUI backend
import matplotlib.pyplot as plt # plotting
import seaborn as sns # statistical data visualization
from scipy.spatial.distance import pdist, squareform # distance computations


# =======
# Logging
# =======
logger = logging.getLogger("VolcanoAI.GaEngine")
logger.addHandler(logging.NullHandler())


# ==========================================
# FIX: Prevent DEAP Creator Crash (REQUIRED)
# ==========================================
if not hasattr(creator, "FitnessMin"): 
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


# ===========================
# Utility Decorators
# ===========================
def execution_monitor(func): # decorator untuk monitor eksekusi fungsi
    @functools.wraps(func)
    def wrapper(*args, **kwargs): # wrapper function untuk mengukur waktu eksekusi
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


# =============
# GEO MATH CORE
# =============
class GeoMathCore: # Kelas utilitas untuk perhitungan geodesik
    R_EARTH_KM = 6371.0088 # radius bumi dalam kilometer

    # ----------------------------
    # ACO → GA: Lightweight Angle Search
    # ----------------------------
    @staticmethod
    def angle_search_from_aco(center_lat: float, center_lon: float, impact_area_km2: float,
                              aco_epicenters_csv: Optional[str] = None,
                              sector_half_width_deg: float = 15.0) -> Dict[str, Any]:
        """
        Simple grid-search over 0..359 degrees:
        - score(angle) = sum(pheromone_score) of ACO epicenters that lie inside
          a sector centered at angle (± sector_half_width_deg) and within radius derived from impact_area.
        Returns pred dict (pred_lat, pred_lon, bearing_degree, distance_km, confidence).
        """
        # compute radius from area
        try:
            radius_km = math.sqrt(max(impact_area_km2, 0.0) / math.pi)
            # if area small, use a small default radius
            radius_km = max(radius_km, 1.0)
        except Exception:
            radius_km = 10.0

        # load ACO epicenters (if available)
        if aco_epicenters_csv and os.path.exists(aco_epicenters_csv):
            try:
                df_aco = pd.read_csv(aco_epicenters_csv)
            except Exception:
                df_aco = pd.DataFrame()
        else:
            df_aco = pd.DataFrame()

        if df_aco.empty or not {'Lintang','Bujur'}.issubset(set(df_aco.columns)):
            pred_distance = radius_km * 0.5
            pred_lat, pred_lon = GeoMathCore.destination_point(center_lat, center_lon, 0.0, pred_distance)
            return {
                "pred_lat": float(pred_lat),
                "pred_lon": float(pred_lon),
                "bearing_degree": 0.0,
                "distance_km": float(pred_distance),
                "confidence": 0.2
            }


        # ensure pheromone column availability
        pher_col = None
        for c in ['PheromoneScore','Pheromone_Score','Pheromone', 'Risk_Index']:
            if c in df_aco.columns:
                pher_col = c
                break
        if pher_col is None:
            df_aco['__pher__'] = 1.0
            pher_col = '__pher__'
        df_aco[pher_col] = pd.to_numeric(df_aco[pher_col].fillna(0.0), errors='coerce')

        lats = df_aco['Lintang'].astype(float).values
        lons = df_aco['Bujur'].astype(float).values
        phers = df_aco[pher_col].astype(float).values

        best_score = -1.0
        best_angle = 0.0

        # precompute bearings & distances from center to all points
        bearings = np.array([GeoMathCore.calculate_bearing(center_lat, center_lon, la, lo) for la, lo in zip(lats, lons)])
        distances = np.array([GeoMathCore.haversine(center_lat, center_lon, la, lo) for la, lo in zip(lats, lons)])

        for ang in range(0, 360):
            # compute angular diff (0..180)
            dif = np.abs(bearings - ang)
            dif = np.minimum(dif, 360.0 - dif)
            within_sector = dif <= sector_half_width_deg
            within_dist = distances <= radius_km
            mask = within_sector & within_dist
            score = float(np.nansum( phers[mask] ))
            if score > best_score:
                best_score = score
                best_angle = float(ang)

        # project predicted point a fraction of radius_km along best_angle
        pred_distance = radius_km * 0.6
        pred_lat, pred_lon = GeoMathCore.destination_point(center_lat, center_lon, best_angle, pred_distance)

        # confidence: normalized best_score vs total pheromone mass
        total_pher = float(np.nansum(phers)) + 1e-9
        conf = float(min(1.0, best_score / total_pher)) if total_pher > 0 else 0.2

        return {
            "pred_lat": float(pred_lat),
            "pred_lon": float(pred_lon),
            "bearing_degree": float(best_angle),
            "distance_km": float(pred_distance),
            "confidence": float(conf)
        }

    @staticmethod # konversi derajat ke radian
    def to_radians(array_like): # mengonversi array derajat ke radian
        return np.radians(array_like) # menggunakan numpy untuk konversi

    @staticmethod
    def calculate_bearing(lat1, lon1, lat2, lon2): # menghitung bearing antara dua titik
        """Bearing (sudut) dari titik 1 → titik 2 dalam derajat 0-360 (geodesic)."""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        diff_lon = math.radians(lon2 - lon1)

        x = math.sin(diff_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(diff_lon))

        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360.0) % 360.0

    @classmethod
    def haversine(cls, lat1, lon1, lat2, lon2): # menghitung jarak haversine antara dua titik
        """Jarak permukaan bumi (km) antar dua koordinat (great-circle)."""
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

    @classmethod
    def destination_point(cls, lat1, lon1, bearing_deg, distance_km): # menghitung titik tujuan berdasarkan bearing dan jarak
        """
        Hitung titik tujuan (lat2, lon2) dari lat1,lon1, bearing (deg) dan distance (km).
        Rumus spherial: lat2 = asin(sin(lat1)*cos(d/R) + cos(lat1)*sin(d/R)*cos(brng))
        """
        if distance_km == 0:
            return float(lat1), float(lon1)

        brng = math.radians(bearing_deg)
        d_div_r = float(distance_km) / cls.R_EARTH_KM

        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)

        lat2_r = math.asin(math.sin(lat1_r) * math.cos(d_div_r) +
                           math.cos(lat1_r) * math.sin(d_div_r) * math.cos(brng))

        lon2_r = lon1_r + math.atan2(math.sin(brng) * math.sin(d_div_r) * math.cos(lat1_r),
                                     math.cos(d_div_r) - math.sin(lat1_r) * math.sin(lat2_r))

        lat2 = math.degrees(lat2_r)
        lon2 = math.degrees(lon2_r)
        # Normalize lon to -180..180
        lon2 = (lon2 + 180) % 360 - 180
        return float(lat2), float(lon2)


# ==============
# DATA SANITIZER
# ==============
class DataSanitizer: # Kelas untuk membersihkan dan memvalidasi data input
    def __init__(self):
        self.required_columns = [
            'EQ_Lintang', 'EQ_Bujur', 'Acquired_Date',
            'PheromoneScore', 'Magnitudo', 'Kedalaman (km)'
        ] # kolom yang diperlukan dalam DataFrame
        self.min_rows = 5

    @execution_monitor # dekorator untuk memonitor eksekusi
    def execute(self, df: pd.DataFrame) -> pd.DataFrame: # metode utama untuk membersihkan data
        if df is None:
            raise ValueError("Input DataFrame cannot be None")
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df = df.copy()
        df['Acquired_Date'] = pd.to_datetime(df['Acquired_Date'], errors='coerce')
        df = df.dropna(subset=['EQ_Lintang', 'EQ_Bujur']).reset_index(drop=True)


        for c in ['EQ_Lintang', 'EQ_Bujur', 'Magnitudo', 'Kedalaman (km)', 'PheromoneScore']:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

        df = df[(df['EQ_Lintang'] >= -90) & (df['EQ_Lintang'] <= 90)]
        df = df[(df['EQ_Bujur'] >= -180) & (df['EQ_Bujur'] <= 180)]

        df = df.reset_index(drop=True)

        if len(df) < self.min_rows:
            raise ValueError(f"Min rows not met: {len(df)}")

        return df


# ======================
# PHYSICS FITNESS ENGINE
# ======================
class PhysicsFitnessEngine: 
    def __init__(self, df: pd.DataFrame, weight_config: Dict[str, float]):
        self.df = df
        self.weights = weight_config

        self.vec_lat = df['EQ_Lintang'].values
        self.vec_lon = df['EQ_Bujur'].values
        self.vec_time = df['Acquired_Date'].values.astype(np.int64)
        # REVISI: Hapus load vector Magnitudo dan Depth karena tidak dipakai di fitness
        self.vec_risk = df['PheromoneScore'].values

        self.num_nodes = len(df)
        self.eps = 1e-6
        self.penalty = 1e15

        # REVISI: Hapus w_mag dan w_depth dari config
        self.w_time = weight_config.get("time", 10000)
        self.w_space = weight_config.get("space", 1)
        self.w_risk = weight_config.get("risk", 500)

    def evaluate(self, individual: List[int]) -> Tuple[float]:
        """
        Fungsi Evaluasi Revisi:
        Hanya memperhitungkan Waktu, Jarak Spasial, dan Risiko (Pheromone).
        Tidak menggunakan Magnitudo atau Kedalaman.
        """
        idx = np.array(individual, dtype=int)

        if len(idx) != self.num_nodes:
            return (self.penalty, )

        # Ambil data lookup
        curr_times = self.vec_time[idx]
        curr_risks = self.vec_risk[idx]

        # 1. TEMPORAL COST (Tetap)
        time_violations_count = np.sum(curr_times[1:] < curr_times[:-1])
        temporal_cost = self.w_time * float(time_violations_count)

        # 2. SPATIAL DISTANCE COST (Tetap)
        lat_seq = self.vec_lat[idx]
        lon_seq = self.vec_lon[idx]
        
        # Hitung total jarak path
        total_dist = 0.0
        for i in range(len(idx) - 1):
            total_dist += GeoMathCore.haversine(
                lat_seq[i], lon_seq[i], 
                lat_seq[i+1], lon_seq[i+1]
            )
        spatial_cost = self.w_space * total_dist

        # 3. RISK SCORE COST (Pheromone Only)
        # REVISI: Input GA hanya dari output ACO (Pheromone/Area).
        # Cost mengecil jika Risiko TINGGI (Prioritas area terdampak).
        clipped_risk = np.clip(curr_risks, 1e-6, None)
        risk_cost = self.w_risk * np.sum(1.0 / clipped_risk)

        # 4. TOTAL COST (Tanpa Mag & Depth)
        total_cost = (
            temporal_cost +
            spatial_cost +
            risk_cost
        )
        
        if np.isnan(total_cost) or np.isinf(total_cost):
            return (self.penalty, )

        return (total_cost, )

    # -------------------------------------------------------------
    # PREDIKSI NEXT EVENT (berdasarkan 3 titik terakhir)
    # -------------------------------------------------------------
    def predict_next_event(self, df_path: pd.DataFrame, n_seg: Optional[int] = 5) -> Dict[str, float]:
            if df_path is None or len(df_path) < 2:
                # Return default zero prediction
                return {
                    "pred_lat": 0.0, "pred_lon": 0.0,
                    "bearing_degree": 0.0, "distance_km": 0.0, "confidence": 0.0
                }

            if n_seg is None:
                n_seg = min(5, len(df_path))
            else:
                n_seg = min(max(2, int(n_seg)), len(df_path))

            df_seg = df_path.iloc[-n_seg:].reset_index(drop=True)
            return self.compute_vector_from_segment(df_seg)    # komputasi vektor dari segmen data

    def compute_vector_from_segment(self, df_seg: pd.DataFrame) -> Dict[str, float]:
            """
            REVISI: Perhitungan vektor arah HANYA berbasis Spasial dan Pheromone (Output ACO).
            Magnitudo dihapus dari pembobotan (weights) dan scaling.
            """
            if df_seg is None or len(df_seg) < 2:
                return {}

            lats = df_seg['EQ_Lintang'].astype(float).values
            lons = df_seg['EQ_Bujur'].astype(float).values
            # REVISI: Hapus pengambilan kolom Magnitudo
            risks = df_seg['PheromoneScore'].astype(float).values if 'PheromoneScore' in df_seg.columns else np.ones(len(df_seg)) * 0.1

            bearings = []
            distances = []
            weights = []

            for i in range(len(lats) - 1):
                lat_a, lon_a = lats[i], lons[i] 
                lat_b, lon_b = lats[i + 1], lons[i + 1]
                dkm = GeoMathCore.haversine(lat_a, lon_a, lat_b, lon_b)
                bdeg = GeoMathCore.calculate_bearing(lat_a, lon_a, lat_b, lon_b)
            
                # REVISI: Bobot (Weight) hanya berdasarkan Pheromone Score (ACO Output)
                # Semakin tinggi pheromone, semakin kuat arah tersebut mempengaruhi prediksi.
                w = ((risks[i] + risks[i + 1]) / 2.0) + 1e-9
            
                distances.append(float(dkm))
                bearings.append(float(bdeg))
                weights.append(float(w))

            distances = np.array(distances, dtype=float)
            weights = np.array(weights, dtype=float)
            # Normalize weights
            weights = weights / (np.sum(weights) + 1e-12)

            # --- Circular Mean (Arah Rata-rata) ---
            thetas = np.radians(np.array(bearings))
            x = np.sum(weights * np.cos(thetas))
            y = np.sum(weights * np.sin(thetas))
            mean_theta = math.atan2(y, x) if not (x == 0 and y == 0) else 0.0
            mean_bearing_deg = (math.degrees(mean_theta) + 360.0) % 360.0

            # Angular concentration R
            R = math.sqrt(x * x + y * y)

            # --- Weighted Average Distance ---
            mean_distance_km = float(np.sum(distances * weights)) if len(distances) > 0 else 0.0

            # REVISI: Scaling Factor (Seberapa jauh prediksi ke depan)
            # HAPUS pengaruh Magnitudo. Gunakan hanya Pheromone (Risk).
            last_risk = float(risks[-1]) if len(risks) > 0 else 0.1
            # Scale: Base 0.5 + Risk factor. Jika area High Risk (Output ACO), prediksi lebih jauh.
            scale = 0.5 + float(last_risk) 

            pred_distance_km = float(mean_distance_km * scale)

            # Hitung titik prediksi (Lat/Lon)
            # Perhatikan: Client minta Output GA cuma arah & sudut. 
            # Lat/Lon ini hanya result internal untuk plotting vektor, 
            # output JSON nanti difilter di method `run`.
            pred_lat, pred_lon = GeoMathCore.destination_point(float(lats[-1]), float(lons[-1]), mean_bearing_deg, pred_distance_km)

            # Confidence Calculation
            conf_risk = self.compute_confidence(df_seg)
            combined_conf = conf_risk * (0.5 + 0.5 * R)
            combined_conf = min(max(0.0, combined_conf), 0.85)

            return {
                "pred_lat": float(pred_lat),
                "pred_lon": float(pred_lon),
                "base_lat": float(lats[-1]),
                "base_lon": float(lons[-1]),
                "movement_scale": float(scale),
                "bearing_degree": float(mean_bearing_deg),
                "distance_km": float(pred_distance_km),
                "movement_direction": self.bearing_to_compass(mean_bearing_deg),
                "confidence": float(combined_conf)
            }

    @staticmethod
    def bearing_to_compass(bearing: float) -> str:
        dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        ix = int((bearing + 22.5) // 45) % 8
        return dirs[ix]

    def compute_confidence(self, df_seg: pd.DataFrame) -> float:
        try:
            risks = df_seg['PheromoneScore'].astype(float).values
            if np.all(risks == 0): return 0.4
            mean_r = float(np.mean(risks))
            std_r = float(np.std(risks)) + 1e-6
            conf = mean_r / (mean_r + std_r)
            return max(0.3, min(conf, 1.0))
        except Exception: 
            return 0.4

# ============================================
# BLOCK 2/3 — Evolutionary Controller + Checkpoint
# ============================================
# Checkpoint System
class CheckpointSystem:
    def __init__(self, directory: str, filename: str = "ga_state.pkl"): # inisialisasi sistem checkpoint
        self.directory = os.path.join(directory, "checkpoints") # direktori penyimpanan checkpoint
        self.filename = filename # nama file checkpoint default
        self.filepath = os.path.join(self.directory, filename) # path lengkap file checkpoint 
        os.makedirs(self.directory, exist_ok=True) # buat direktori jika belum ada
    # simpan state ke file checkpoint
    def save_state(self, population, generation, stats, filename=None): # simpan state ke file checkpoint
        fname = filename if filename else self.filename # gunakan nama file khusus jika diberikan
        fpath = os.path.join(self.directory, fname) # path lengkap file checkpoint

        payload = {
            "version": "6.0",
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "population": population,
            "stats": stats
        } # data yang akan disimpan
        # simpan ke file menggunakan pickle
        try:
            with open(fpath, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    # muat state dari file checkpoint
    def load_state(self):
        if not os.path.exists(self.filepath):
            return None
        try:
            with open(self.filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

# Statistics Collector untuk GA Engine 
class StatisticsCollector:
    def __init__(self):
        self.stats = tools.Statistics(lambda ind: ind.fitness.values) # mengumpulkan nilai fitness individu
        self.stats.register("avg", np.mean) # rata-rata fitness
        self.stats.register("std", np.std)# standar deviasi fitness
        self.stats.register("min", np.min) # nilai fitness minimum
        self.stats.register("max", np.max)  # nilai fitness maksimum
    # dapatkan objek statistik
    def get(self):
        return self.stats

# Evolutionary Controller untuk menjalankan GA
class EvolutionaryController:
    def __init__(self, config, fitness_engine, checkpoint_mgr): # inisialisasi controller evolusi
        self.cfg = config # konfigurasi GA
        self.fitness_engine = fitness_engine # engine fitness
        self.checkpoint_mgr = checkpoint_mgr # manajer checkpoint

        self.toolbox = base.Toolbox() # toolbox DEAP untuk operator GA
        self.stats_col = StatisticsCollector() # kolektor statistik
        self.stats = self.stats_col.get() # objek statistik
        # daftarkan operator GA
        self._register_operators()
    # daftarkan operator GA
    def _register_operators(self):
        problem_size = self.fitness_engine.num_nodes # ukuran masalah (jumlah node)

        # Index Generator (Perumutan) untuk kromosom permutasi 
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

    @execution_monitor # dekorator untuk memonitor eksekusi
    def run(self): # metode utama untuk menjalankan evolusi GA
        pop_size = self.cfg.population_size # ukuran populasi
        n_gen = self.cfg.n_generations # jumlah generasi
        cx_prob = self.cfg.crossover_prob # probabilitas crossover
        mut_prob = self.cfg.mutation_prob # probabilitas mutasi
        # Logging awal
        logger.info(f"GA Evolution Start → Pop={pop_size}, Gen={n_gen}")

        
        pop = self.toolbox.population(n=pop_size) # inisialisasi populasi
        hof = tools.HallOfFame(self.cfg.hall_of_fame_size) # hall of fame untuk individu terbaik

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
                fname = f"ga_state_gen_{gen}.pkl"
                self.checkpoint_mgr.save_state(
                    pop,
                    gen,
                    logbook,
                    filename=fname
                )

        # Save final state
        self.checkpoint_mgr.save_state(pop, n_gen, logbook, filename="final_ga.pkl")

        best_individual = list(hof[0])
        log_df = pd.DataFrame(logbook)

        return best_individual, log_df, hof


# ============================================
# BLOCK 3/3 — Visualizer + Exporter + GA Engine Wrapper
# ============================================
# Multi-Layer Visualizer untuk peta interaktif
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
        # --- LOGIC FIX: Penentuan Titik Pusat Peta (Center) ---
        # Prioritas 1: Gunakan ACO Center (jika Mode ACO)
        center_lat = pred_info.get("aco_center_lat")
        center_lon = pred_info.get("aco_center_lon")

        # Prioritas 2: Gunakan Base Location (Titik terakhir data, jika Mode GA Standard)
        if center_lat is None or center_lon is None:
            center_lat = pred_info.get("base_lat")
            center_lon = pred_info.get("base_lon")

        # Prioritas 3: Gunakan Prediksi Lokasi
        if center_lat is None or center_lon is None:
            center_lat = pred_info.get("pred_lat")
            center_lon = pred_info.get("pred_lon")

        # Prioritas 4: Gunakan Rata-rata Data (Fallback terakhir)
        if (center_lat is None or center_lon is None) and not df.empty:
            center_lat = df['EQ_Lintang'].mean()
            center_lon = df['EQ_Bujur'].mean()

        # Jika masih gagal (Data kosong dan tidak ada prediksi), hentikan pembuatan peta tanpa error fatal
        if center_lat is None or center_lon is None:
            logger.warning("Map center could not be determined. Skipping map generation.")
            return

        if best_path is None:
            best_path = []

        # Buat objek peta folium
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
                            color="cyan", weight=0.5, opacity=0.25
                        ).add_to(chaos)

            for r in df.itertuples():
                folium.CircleMarker(
                    [getattr(r, 'EQ_Lintang'), getattr(r, 'EQ_Bujur')],
                    radius=2, color="#ffffff", fill=True, fill_opacity=0.4
                ).add_to(chaos)

        chaos.add_to(m)

        # SNAKE LAYER (Best GA path)
        snake = folium.FeatureGroup(name="Snake Path (Best Sequence)", show=True)

        max_idx = len(df) - 1
        safe_path = [i for i in best_path if 0 <= i <= max_idx]

        # Jika tidak ada path (misal mode ACO only), kita tetap bisa plot titik-titik
        if not safe_path and not df.empty:
             # Fallback: Plot 5 titik terakhir sebagai jejak jika path kosong
             safe_path = list(range(max(0, len(df)-5), len(df)))

        if safe_path:
            best_df = df.iloc[safe_path].reset_index(drop=True)
            path_coords = best_df[['EQ_Lintang', 'EQ_Bujur']].values.tolist()

            if len(path_coords) >= 2:
                AntPath(
                    locations=path_coords,
                    color="magenta", pulse_color="yellow", weight=4, delay=600
                ).add_to(snake)

            if len(path_coords) >= 1:
                folium.Marker(path_coords[0], icon=folium.Icon(color="green", icon="play"), popup="START").add_to(snake)
                folium.Marker(path_coords[-1], icon=folium.Icon(color="red", icon="stop"), popup="END").add_to(snake)

            # Popup info
            for i in range(len(best_df)):
                row = best_df.iloc[i]
                lat = float(row.get('EQ_Lintang', float('nan')))
                lon = float(row.get('EQ_Bujur', float('nan')))
                
                # ... (Perhitungan popup tetap sama) ...
                depth_val = row.get('Kedalaman (km)', row.get('depth', None))
                depth_str = f"{depth_val}" if depth_val is not None else "N/A"
                
                popup = f"""
                <b>Event Detail</b><br>
                Date: {row.get('Acquired_Date', 'N/A')}<br>
                Mag: {row.get('Magnitudo', 'N/A')}<br>
                Depth: {depth_str} km<br>
                Risk: {row.get('PheromoneScore', 0):.3f}<br>
                """
                folium.CircleMarker(
                    [lat, lon], radius=4, color="orange", fill=True, fill_color="yellow", fill_opacity=0.8,
                    popup=folium.Popup(popup, max_width=320)
                ).add_to(snake)

        snake.add_to(m)

        # DIRECTION VISUALIZATION
        if pred_info and "bearing_degree" in pred_info:
            # Gunakan center yang sudah ditentukan di awal fungsi
            start_lat = center_lat
            start_lon = center_lon

            bearing = pred_info.get("bearing_degree")
            distance_km = pred_info.get("distance_km", 3.0)

            end_lat, end_lon = GeoMathCore.destination_point(
                start_lat, start_lon, bearing, distance_km
            )

            popup_dir = f"<b>Predicted Direction</b><br>Bearing: {bearing:.1f}°"

            folium.PolyLine(
                [[start_lat, start_lon], [end_lat, end_lon]],
                color="orange", weight=4, opacity=0.8, tooltip="Prediction Vector"
            ).add_to(m)

            folium.Marker(
                [end_lat, end_lon],
                popup=folium.Popup(popup_dir, max_width=280),
                icon=folium.DivIcon(html="<div style='font-size:20px'>➤</div>")
            ).add_to(m)

        # Heatmap
        try:
            if 'PheromoneScore' in df.columns:
                heat_data = df[['EQ_Lintang', 'EQ_Bujur', 'PheromoneScore']].values.tolist()
                HeatMap(heat_data, name="Risk Heatmap", radius=15, blur=10, show=False).add_to(m)
        except Exception: pass

        folium.LayerControl().add_to(m)
        plugins.Fullscreen().add_to(m)
        plugins.MiniMap().add_to(m)

        m.save(out_path)
        logger.info(f"GA Map saved → {out_path}")
# Excel Data Exporter
class DataExporter:
    def __init__(self, output_dir: str): # inisialisasi exporter
        self.output_dir = output_dir # direktori output
        self.excel_path = os.path.join(output_dir, "ga_report.xlsx") # path file excel output
    # metode utama untuk mengekspor data ke file excel
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
class GaEngine: # GA Engine utama
    def __init__(self, config: Any): # inisialisasi GA Engine
        self.cfg = config # konfigurasi GA
        self.output_dir = getattr(config, "output_dir", "output/ga_results") # direktori output
        os.makedirs(self.output_dir, exist_ok=True) # buat direktori output jika belum ada

        # MAPS directory (UNTUK FOLIUM MAP)
        self.maps_dir = os.path.join(self.output_dir, "maps")
        os.makedirs(self.maps_dir, exist_ok=True)

        # Path FINAL map
        self.map_path = os.path.join(self.maps_dir, "ga_path_map.html")

        self.sanitizer = DataSanitizer() # inisialisasi data sanitizer
        self.checkpoint_mgr = CheckpointSystem(self.output_dir) # inisialisasi manajer checkpoint
        self.visualizer = MultiLayerVisualizer(self.output_dir) # inisialisasi visualizer peta
        self.exporter = DataExporter(self.output_dir)

        self.map_path = os.path.join(self.output_dir, "ga_path_map.html")
        self.log_path = os.path.join(self.output_dir, "ga_log.csv")
    # Tulis hasil prediksi ke aco_to_ga.json
    def _write_back_to_aco_json(self, pred: Dict[str, Any]):
        """
        GA OUTPUT (client-compliant):
        - HANYA arah dan sudut pergerakan
        - Tidak ada lokasi, magnitudo, confidence
        """

        import os
        import json
        import numpy as np
        from datetime import datetime

        aco_json_path = os.path.join(
            os.path.dirname(self.output_dir),
            "aco_results",
            "aco_to_ga.json"
        )

        os.makedirs(os.path.dirname(aco_json_path), exist_ok=True)

        def safe_float(x, default=0.0):
            try:
                fx = float(x)
                return fx if np.isfinite(fx) else default
            except Exception:
                return default

        # === GA OUTPUT SESUAI CLIENT ===
        next_event = {
            "direction_deg": safe_float(pred.get("bearing_degree")),
            "distance_km": safe_float(pred.get("distance_km"))  # hanya visual panah
        }

        try:
            # load ACO json
            if os.path.exists(aco_json_path):
                with open(aco_json_path, "r") as f:
                    data = json.load(f)
            else:
                data = {}

            # PERTAHANKAN ACO OUTPUT
            data["next_event"] = next_event
            data["_ga_generated_at"] = datetime.now().isoformat()

            with open(aco_json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)

            logger.info("[GA] next_event (direction-only) ditulis ke aco_to_ga.json")

        except Exception:
            logger.error("[GA] Gagal menulis next_event GA", exc_info=True)


    # Metode utama untuk menjalankan GA Engine
    def run(self, df_train: pd.DataFrame) -> Tuple[List[int], Dict[str, Any]]:
        logger.info("\n" + "=" * 80)
        logger.info("=== GA ENGINE START (Input from ACO Only) ===".center(80))
        logger.info("=" * 80)

        # --- LOGIKA REVISI: INPUT DARI OUTPUT ACO AJA ---
        aco_results_dir = os.path.join(os.path.dirname(self.output_dir), "aco_results")
        aco_json_path = os.path.join(aco_results_dir, "aco_to_ga.json")
        aco_epicenters_csv = os.path.join(aco_results_dir, "aco_epicenters.csv")

        # Cek apakah ACO output tersedia
        if os.path.exists(aco_json_path):
            try:
                with open(aco_json_path, "r", encoding="utf-8") as f:
                    aco_payload = json.load(f)
                
                # Input GA: Pusat ACO dan Area ACO
                center_lat = float(aco_payload.get("center_lat", 0.0))
                center_lon = float(aco_payload.get("center_lon", 0.0))
                impact_area = float(aco_payload.get("impact_area_km2", aco_payload.get("impact_area", 0.0)))

                logger.info(f"[GA] Using ACO Input: Center=({center_lat}, {center_lon}), Area={impact_area}")

                # JALANKAN PROSES GA (Angle Search) MENGGUNAKAN INPUT ACO
                pred = GeoMathCore.angle_search_from_aco(
                    center_lat, center_lon, impact_area, aco_epicenters_csv
                )
                
                # Tambahkan info center ACO untuk map
                pred["aco_center_lat"] = float(center_lat)
                pred["aco_center_lon"] = float(center_lon)

                # Simpan output ke JSON (hanya arah & sudut sesuai request)
                self._write_back_to_aco_json(pred)

                # Simpan vektor untuk LSTM
                try:
                    vector_out = {
                        "_generated_at": datetime.now().isoformat(),
                        "bearing_degree": float(pred.get("bearing_degree", 0.0)),
                        "distance_km": float(pred.get("distance_km", 0.0)),
                        "note": "Output GA based on ACO Input only"
                    }
                    with open(os.path.join(self.output_dir, "ga_vector.json"), "w", encoding="utf-8") as vf:
                        json.dump(vector_out, vf, indent=2)
                except Exception as e:
                    logger.warning(f"[GA] Failed to save ga_vector.json: {e}")

                # --- FIX 1: Generate Map dengan Handling Nama Kolom ---
                try:
                    if os.path.exists(aco_epicenters_csv):
                        df_aco = pd.read_csv(aco_epicenters_csv)
                        
                        # PERBAIKAN: Rename kolom Lintang/Bujur ke EQ_Lintang/EQ_Bujur
                        # Ini mengatasi error KeyError saat Visualizer membaca file CSV ACO
                        rename_map = {
                            'Lintang': 'EQ_Lintang', 'Bujur': 'EQ_Bujur',
                            'Latitude': 'EQ_Lintang', 'Longitude': 'EQ_Bujur'
                        }
                        df_aco.rename(columns=rename_map, inplace=True)
                        
                        # Fallback ekstra jika rename tidak lengkap
                        if 'EQ_Lintang' not in df_aco.columns and 'Lintang' in df_aco.columns:
                            df_aco['EQ_Lintang'] = df_aco['Lintang']
                        if 'EQ_Bujur' not in df_aco.columns and 'Bujur' in df_aco.columns:
                            df_aco['EQ_Bujur'] = df_aco['Bujur']
                    else:
                        df_aco = df_train
                except Exception as e:
                    logger.warning(f"[GA] Failed to load ACO CSV for map: {e}")
                    df_aco = df_train

                out_map_path = os.path.join(self.output_dir, "ga_from_aco_map.html")
                self.visualizer.generate_map([], df_aco, pred, out_map_path)

                logger.info("=== GA ENGINE COMPLETE (ACO Mode) ===".center(80))

                return [], {
                    "map": out_map_path,
                    "prediction": {
                        "bearing_degree": pred["bearing_degree"],
                        "distance_km": pred["distance_km"]
                    }
                }

            except Exception as e:
                logger.error(f"[GA] Error processing ACO input: {e}", exc_info=True)
                # Fallback ke standar jika error
        
        # --- FALLBACK LOGIC (Standard GA) ---
        logger.warning("[GA] ACO Input not found or Error. Running Standard GA.")

        # 1. Sanitization
        clean_df = self.sanitizer.execute(df_train)

        # 2. Fitness Engine
        fit_engine = PhysicsFitnessEngine(clean_df, self.cfg.fitness_weights)

        # 3. Evolution
        evo = EvolutionaryController(self.cfg, fit_engine, self.checkpoint_mgr)
        
        # --- FIX 2: Pass 'clean_df' sebagai argumen ---
        # Ini mengatasi error TypeError: missing 1 required positional argument: 'df_train'
        best_idx, log_df, hof = evo.run(clean_df)
        
        # Clean indices
        if isinstance(best_idx, tuple): best_idx = list(best_idx)
        if len(best_idx) == 1 and isinstance(best_idx[0], tuple): best_idx = list(best_idx[0])
        best_idx = [int(x) for x in best_idx]

        max_idx = len(clean_df) - 1
        safe_idx = [i for i in best_idx if isinstance(i, int) and 0 <= i <= max_idx] 

        if not safe_idx:
            df_opt = clean_df.copy().reset_index(drop=True)
        else:
            df_opt = clean_df.iloc[safe_idx].reset_index(drop=True)

        # 5. Prediction
        pred = fit_engine.predict_next_event(df_opt, n_seg=getattr(self.cfg, "ga_segment_window", 5))
        
        if isinstance(pred, dict) and pred:
            self._write_back_to_aco_json(pred)

        # 6. Visualization
        self.visualizer.generate_map(safe_idx, clean_df, pred, self.map_path)

        # 7. Export
        meta = {
            "Timestamp": datetime.now().isoformat(),
            "PredictedBearing": pred.get("bearing_degree", None),
            "PredictedDistanceKM": pred.get("distance_km", None),
        }
        self.exporter.export(clean_df, df_opt, meta)

        try:
            log_df.to_csv(self.log_path, index=False)
        except Exception: pass

        logger.info("=== GA ENGINE COMPLETE (Fallback Mode) ===".center(80))

        return best_idx, {"map": self.map_path, "prediction": pred}