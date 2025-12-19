"""
TectonicAI - ACO Engine (Titanium Edition).
Module: Advanced Ant Colony Optimization for Seismic Risk Zoning.
"""
import json
import numpy as np
import pandas as pd
import logging
import os
import time
import math
import pickle
import folium
from folium.plugins import HeatMap
from typing import List, Dict, Any, Tuple, Optional, Union

# --- KONSTANTA ---
R_EARTH_KM = 6371.0
EPSILON = 1e-12
DEFAULT_PHEROMONE = 0.1
MAX_PHEROMONE = 10.0
MIN_PHEROMONE = 0.01

logger = logging.getLogger("ACO_Engine_Master")
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)

# ==========================================
# 1. GEO UTILITIES
# ==========================================

class GeoMath:
    @staticmethod
    def haversine_vectorized(lat_array, lon_array):
        lat_rad = np.radians(lat_array)
        lon_rad = np.radians(lon_array)

        dlat = lat_rad[:, np.newaxis] - lat_rad
        dlon = lon_rad[:, np.newaxis] - lon_rad

        a = (
            np.sin(dlat / 2.0) ** 2 +
            np.cos(lat_rad[:, np.newaxis]) * np.cos(lat_rad) *
            np.sin(dlon / 2.0) ** 2
        )
        a = np.clip(a, 0.0, 1.0)

        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        dist_matrix = R_EARTH_KM * c

        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix + EPSILON  # hindari 0 murni


# ==========================================
# 2. ANT AGENT
# ==========================================

class AntAgent:
    def __init__(self, ant_id: int, start_node: int, alpha: float, beta: float, role: str = "Worker"):
        self.id = ant_id
        self.start_node = start_node
        self.current_node = start_node
        self.role = role
        self.alpha = alpha
        self.beta = beta
        self.path = [start_node]
        self.visited_mask = None
        self.accumulated_risk = 0.0

    def reset(self, new_start_node: int, n_nodes: int):
        self.start_node = new_start_node
        self.current_node = new_start_node
        self.path = [new_start_node]
        self.visited_mask = np.zeros(n_nodes, dtype=bool)
        self.visited_mask[new_start_node] = True
        self.accumulated_risk = 0.0

    def move_to(self, next_node: int, risk_val: float):
        self.path.append(next_node)
        self.visited_mask[next_node] = True
        self.current_node = next_node
        self.accumulated_risk += float(max(risk_val, 0.0))


# ==========================================
# 3. ENVIRONMENT MANAGER
# ==========================================

class EnvironmentManager:
    def __init__(self, df: pd.DataFrame, logger_obj):
        self.df = df.copy()
        self.logger = logger_obj
        self.n_nodes = len(df)

        self.dist_matrix = None
        self.heuristic_matrix = None
        self.pheromone_matrix = None

        if self.n_nodes > 0:
            self._normalize_geo_columns()
            self._build_distance_matrix()
            self._build_heuristic_matrix()
            self._init_pheromone_matrix()

    # ---------------------- NEW ---------------------- #
    def _normalize_geo_columns(self):
        """
        Samakan nama kolom geospasial untuk ACO:
        - Lintang ← EQ_Lintang / lat / Latitude
        - Bujur   ← EQ_Bujur / lon / Longitude
        """
        col_map_lat = ['Lintang', 'EQ_Lintang', 'lat', 'Latitude']
        col_map_lon = ['Bujur', 'EQ_Bujur', 'lon', 'Longitude']

        lat_col = next((c for c in col_map_lat if c in self.df.columns), None)
        lon_col = next((c for c in col_map_lon if c in self.df.columns), None)

        if lat_col is None or lon_col is None:
            raise KeyError(
                f"[ACO] Tidak menemukan kolom koordinat. "
                f"Cari salah satu dari {col_map_lat} dan {col_map_lon}"
            )

        # Buat alias standar
        self.df['Lintang'] = self.df[lat_col].astype(float)
        self.df['Bujur'] = self.df[lon_col].astype(float)

    # ---------------------- FIXED ---------------------- #
    def _build_distance_matrix(self):
        lats = self.df['Lintang'].values
        lons = self.df['Bujur'].values
        self.dist_matrix = GeoMath.haversine_vectorized(lats, lons)


    # ----------------------------------
    # [FIX] Pastikan indentasi metode ini sudah benar (Level 1 di dalam class)
    def _build_heuristic_matrix(self):
        """
        Membangun Matrix Heuristic (Eta) untuk ACO berbasis Fisika Gempa.
        
        Rumus Heuristik:
        Eta_ij = (Energi_j * Faktor_Kedalaman_j) / Jarak_ij
        
        Dimana:
        - Energi ~ Magnitudo^2.5
        - Faktor Kedalaman ~ 1 / sqrt(Kedalaman). (Semakin dangkal -> semakin bahaya/prioritas).
        """
        import numpy as np # Pastikan numpy tersedia

        # -----------------------------------------------------------
        # 1. COLUMN IDENTIFICATION (Dengan Fallback yang Robust)
        # -----------------------------------------------------------
        # Cari nama kolom Magnitudo yang tersedia
        mag_col = next((c for c in ['Magnitudo_Original', 'Magnitudo', 'magnitude', 'mag'] if c in self.df.columns), None)
        
        # Cari nama kolom Kedalaman yang tersedia
        depth_cols_candidates = ['Kedalaman_Original', 'Kedalaman_km', 'Kedalaman (km)', 'depth', 'depth_km']
        depth_col = next((c for c in depth_cols_candidates if c in self.df.columns), None)

        # Jika kolom vital tidak ditemukan, raise Error agar pipeline berhenti
        if mag_col is None or depth_col is None:
            raise KeyError(
                f"[ACO Error] Kolom Magnitudo ('{mag_col}') atau Kedalaman ('{depth_col}') tidak ditemukan di DataFrame."
            )

        # -----------------------------------------------------------
        # 2. PHYSICS CALCULATION
        # -----------------------------------------------------------
        # [FIX KRITIS]: Clip nilai untuk safety matematika.
        # Magnitudo minimal 0.1 (agar tidak 0 ^ n), Kedalaman minimal 1.0 km (hindari pembagian 0)
        mags = np.clip(self.df[mag_col].values.astype(float), 0.1, None)
        depths = np.clip(self.df[depth_col].values.astype(float), 1.0, None)

        # Hitung Komponen Attractiveness (Daya Tarik Node Tujuan)
        # a. Energi relatif (Heuristik: Magnitudo ^ 2.5)
        energy_score = np.power(mags, 2.5)

        # b. Faktor Kedalaman 
        # Logika: Gempa dangkal lebih berdampak -> Score lebih tinggi.
        # depth_factor = 1 / sqrt(depth)
        depth_factor = 1.0 / np.power(depths, 0.5)
        
        # Normalisasi depth_factor agar skalanya 0-1
        if np.max(depth_factor) > 0:
            depth_factor /= np.max(depth_factor)

        # Total 'Attractiveness' Node j (tanpa memperhitungkan jarak)
        attractiveness = energy_score * depth_factor

        # -----------------------------------------------------------
        # 3. BUILD MATRIX (BROADCASTING)
        # -----------------------------------------------------------
        # Kita butuh matrix (N, N) dimana element [i, j] berisi nilai attractiveness dari node j.
        # Tile vector (N,) menjadi (N, N) baris-per-baris
        attr_matrix = np.tile(attractiveness, (self.n_nodes, 1))

        # [FIX] Division by Distance
        # Eta_ij = Attractiveness_j / Distance_ij
        # Tambahkan epsilon kecil pada dist_matrix agar tidak error saat distance=0
        dist_safe = self.dist_matrix.copy()
        # Hindari pembagian dengan nol dengan mengganti 0 dengan nilai sangat kecil (atau tangani infinity nanti)
        dist_safe[dist_safe < 1e-6] = 1e-6 
        
        self.heuristic_matrix = attr_matrix / dist_safe

        # -----------------------------------------------------------
        # 4. CLEANUP & NORMALIZATION
        # -----------------------------------------------------------
        # Diagonal utama (jarak ke diri sendiri) tidak relevan dalam ACO untuk TSP/Pathfinding
        np.fill_diagonal(self.heuristic_matrix, 0.0)

        # Validasi Nilai Ekstrim
        # Mengganti Infinity (akibat jarak ~0 selain diagonal) dengan nilai maksimum yang valid
        finite_vals = self.heuristic_matrix[np.isfinite(self.heuristic_matrix)]
        if len(finite_vals) > 0:
            max_finite = np.max(finite_vals)
            self.heuristic_matrix[~np.isfinite(self.heuristic_matrix)] = max_finite
        else:
            max_finite = 1.0 # Fallback jika matriks rusak

        # Normalisasi Min-Max Skalar ke rentang [0, 1] agar probabilitas stabil
        max_val = np.max(self.heuristic_matrix)
        
        if max_val <= 1e-9 or np.isnan(max_val):
            self.logger.warning("[ACO] Heuristic Matrix kosong/flat/NaN. Menggunakan Fallback Uniform distribution.")
            self.heuristic_matrix.fill(1.0) # Uniform heuristic
            np.fill_diagonal(self.heuristic_matrix, 0.0)
        else:
            self.heuristic_matrix /= max_val

        # [Safety Clip Final]: Jangan biarkan ada nilai murni 0 (kecuali diagonal)
        # agar log probability tidak error nanti
        # Diagonal tetap dibiarkan 0 agar semut tidak diam di tempat
        mask_diag = np.eye(self.n_nodes, dtype=bool)
        self.heuristic_matrix[~mask_diag] = np.clip(self.heuristic_matrix[~mask_diag], 1e-4, 1.0)


    # ----------------------------------
    def _init_pheromone_matrix(self):
        # mulai dari nilai konstan
        self.pheromone_matrix = np.full((self.n_nodes, self.n_nodes), DEFAULT_PHEROMONE, dtype=float)
        np.fill_diagonal(self.pheromone_matrix, 0.0)

    # ----------------------------------
    def get_transition_probabilities(self, current_node: int, ant: AntAgent):
        tau = self.pheromone_matrix[current_node]
        eta = self.heuristic_matrix[current_node]

        # kombinasi pheromone & heuristic
        prob = np.power(tau, ant.alpha) * np.power(eta, ant.beta)

        if ant.visited_mask is not None and len(ant.visited_mask) == len(prob):
            prob[ant.visited_mask] = 0.0

        # jika semua nol → fallback uniform
        if not np.isfinite(prob).any() or np.sum(prob) <= 0:
            prob = np.ones_like(prob, dtype=float)

        return prob

    # ----------------------------------
    def apply_global_update(self, evaporation_rate: float, deposit_matrix: np.ndarray):
        # evaporasi
        self.pheromone_matrix *= (1.0 - evaporation_rate)
        # deposit
        self.pheromone_matrix += deposit_matrix

        # jaga range
        self.pheromone_matrix = np.clip(self.pheromone_matrix, MIN_PHEROMONE, MAX_PHEROMONE)
        np.fill_diagonal(self.pheromone_matrix, 0.0)

    # ----------------------------------
    def reset_pheromone_smooth(self):
        avg = float(np.mean(self.pheromone_matrix))
        self.pheromone_matrix = 0.5 * self.pheromone_matrix + 0.5 * avg
        np.fill_diagonal(self.pheromone_matrix, 0.0)


# ==========================================
# 4. MAIN ENGINE
# ==========================================

class DynamicAcoEngine:

    def __init__(self, config):
        self.logger = logging.getLogger("ACO_Engine_Master")

        # =====================================
        # FIX – ubah config object → dictionary
        # =====================================
        if isinstance(config, dict):
            self.aco_cfg = config
        elif hasattr(config, "__dict__"):
            # menggunakan dict(vars(config)) lebih aman untuk dataclass
            self.aco_cfg = {k: v for k, v in vars(config).items() if not k.startswith('__')}
        else:
            self.logger.warning("[ACO] Config tidak dikenali, memakai config kosong.")
            self.aco_cfg = {}

        # Lanjutkan proses internal
        self._load_parameters()

        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output'))
        base_output_dir = self.aco_cfg.get('output_dir', os.path.abspath(os.path.join(os.path.dirname(__file__), '../../output/aco_results')))
        
        # Penamaan file output agar jelas (Training vs Evaluation)
        tag = self.aco_cfg.get('run_tag', 'training')
        self.output_paths = {
            'aco_zoning_excel': os.path.join(base_path, 'aco_results/aco_zoning_data_for_lstm.xlsx'),
            'aco_epicenters_csv': os.path.join(base_path, 'aco_results/aco_epicenters.csv'),
            'aco_state_file': os.path.join(base_path, 'aco_results/aco_brain_state.pkl'),
            'aco_impact_html': os.path.join(base_path, 'aco_results/aco_impact_zones.html')
        }
        os.makedirs(os.path.dirname(self.output_paths['aco_zoning_excel']), exist_ok=True)

        self.env_manager: Optional[EnvironmentManager] = None
        self.colony: List[AntAgent] = []
        self.best_global_score = -np.inf
        self.stagnation_counter = 0

    # ----------------------------------
    def _load_parameters(self):
        self.n_ants = int(self.aco_cfg.get('n_ants', 50))
        self.n_iterations = int(self.aco_cfg.get('n_iterations', 100))
        self.n_steps = int(self.aco_cfg.get('n_epicenters', 20))

        self.alpha_base = float(self.aco_cfg.get('alpha', 1.0))
        self.beta_base = float(self.aco_cfg.get('beta', 2.0))
        self.rho_base = float(self.aco_cfg.get('evaporation_rate', 0.1))
        self.Q = float(self.aco_cfg.get('pheromone_deposit', 100.0))
        self.risk_threshold = float(self.aco_cfg.get('risk_threshold', 0.7))

    # ----------------------------------
    def _initialize_colony(self, n_nodes: int):
        self.colony = []

        if n_nodes <= 0:
            return

        start_scores = self.env_manager.heuristic_matrix.sum(axis=1)
        # normalisasi aman
        start_scores = np.clip(start_scores, 0.0, None)
        total = float(start_scores.sum())
        if total <= 0 or not np.isfinite(total):
            start_probs = np.ones(n_nodes, dtype=float) / max(n_nodes, 1)
        else:
            start_probs = start_scores / total

        for i in range(self.n_ants):
            if i < int(self.n_ants * 0.2):
                role = "Explorer"
                alpha = self.alpha_base * 0.5
                beta = self.beta_base * 1.5
            else:
                role = "Exploiter"
                alpha = self.alpha_base * 1.2
                beta = self.beta_base * 0.9

            start_node = int(np.random.choice(np.arange(n_nodes), p=start_probs))
            ant = AntAgent(i, start_node, alpha, beta, role)
            ant.reset(start_node, n_nodes)
            self.colony.append(ant)

    # ----------------------------------
    def _step_ants(self):
        active_ants = 0
        for ant in self.colony:
            if ant.current_node == -1:
                continue

            probs = self.env_manager.get_transition_probabilities(ant.current_node, ant)
            total = float(np.sum(probs))

            if total <= 0 or not np.isfinite(total):
                ant.current_node = -1
                continue

            norm = probs / total
            next_node = int(np.random.choice(np.arange(self.env_manager.n_nodes), p=norm))

            risk_val = self.env_manager.heuristic_matrix[ant.current_node, next_node]
            ant.move_to(next_node, risk_val)
            active_ants += 1

        return active_ants

    # ----------------------------------
    def _manage_lifecycle(self, iteration: int):
        """
        Mengelola siklus hidup koloni per iterasi:
        1. Menghitung deposit pheromone berdasarkan performa semut.
        2. Evaporasi & Update Global Pheromone.
        3. Mendeteksi Stagnasi (Convergence trap).
        4. Re-spawn (Reset posisi) semut untuk iterasi berikutnya.
        """
        # Matriks Delta Pheromone (Akan dijumlahkan ke pheromone utama)
        deposit_matrix = np.zeros_like(self.env_manager.pheromone_matrix, dtype=float)
        iter_best = -np.inf
        
        # 1. LOOP ANALISIS PERFORMANCE SEMUT
        for ant in self.colony:
            score = float(ant.accumulated_risk)
            
            # Abaikan solusi tidak valid (NaN atau <= 0)
            if not np.isfinite(score) or score <= 0:
                continue
            
            # Track best score local di iterasi ini
            if score > iter_best:
                iter_best = score
                
            # Hitung Kualitas Solusi untuk deposit
            # Quality = Total Risk / (Panjang Path).
            # Logika: Risiko tinggi yang dicapai dalam langkah sedikit = Lebih Efisien.
            path_len = len(ant.path)
            ant_quality = score / (path_len + 1e-6)
            
            if ant_quality > 0:
                # Faktor Q mempengaruhi seberapa kuat jejak ditinggalkan
                dep = self.Q * ant_quality 
                
                # Deposit pheromone di sepanjang edge yang dilalui
                path = ant.path
                for k in range(len(path) - 1):
                    u, v = path[k], path[k + 1]
                    # Update undirected (dua arah)
                    deposit_matrix[u, v] += dep
                    deposit_matrix[v, u] += dep

        # 2. ADAPTIVE EVAPORATION (DYNAMIC RHO)
        # Jika stagnasi tinggi, tingkatkan evaporasi agar jejak lama cepat hilang (exploration mode)
        current_rho = self.rho_base
        if self.stagnation_counter > 5:
            # Batas rho max 0.9 agar tidak menghapus total
            current_rho = min(0.9, self.rho_base * 1.5)
            
        # Terapkan update global ke Environment (Evaporasi + Deposit baru)
        self.env_manager.apply_global_update(current_rho, deposit_matrix)

        # 3. STAGNATION CHECK
        if iter_best > self.best_global_score:
            self.best_global_score = iter_best
            self.stagnation_counter = 0 # Reset counter jika ada kemajuan
        else:
            self.stagnation_counter += 1
            
        # Mekanisme Anti-Stagnasi: Soft Reset Pheromone
        if self.stagnation_counter > (self.n_iterations * 0.2):
            self.logger.info(f"[ACO Iter-{iteration}] Stagnasi ({self.stagnation_counter}) → Smooth Reset Pheromone.")
            self.env_manager.reset_pheromone_smooth()
            self.stagnation_counter = 0

        # 4. RE-SPAWN ANTS (POSITION RESET)
        n_nodes = self.env_manager.n_nodes
        if n_nodes <= 0:
            return

        # Tentukan start node berikutnya berdasarkan heuristic (Semakin bahaya node, semakin mungkin jadi start)
        # Sum axis=1: Total attractiveness node tersebut
        if hasattr(self.env_manager, 'heuristic_matrix'):
            start_scores = np.sum(self.env_manager.heuristic_matrix, axis=1)
        else:
            start_scores = np.ones(n_nodes)
            
        # Sanitasi Score (hapus NaN/Negatif)
        start_scores = np.nan_to_num(start_scores, nan=0.0)
        start_scores = np.clip(start_scores, 0.0, None)
        
        total_score = float(start_scores.sum())
        
        # Buat Probabilitas Start
        if total_score <= 1e-9:
            # Fallback: Uniform Probability jika semua 0
            start_probs = np.ones(n_nodes, dtype=float) / n_nodes
        else:
            start_probs = start_scores / total_score
            # [FIX CRITICAL]: Re-normalize sum agar PERSIS 1.0 (numpy choice rewel soal ini)
            start_probs /= start_probs.sum()

        # Reset posisi setiap semut untuk iterasi depan
        for ant in self.colony:
            try:
                new_start = np.random.choice(np.arange(n_nodes), p=start_probs)
            except ValueError:
                # Jika masih error floating point sum != 1, fallback random uniform
                new_start = np.random.randint(0, n_nodes)
                
            ant.reset(int(new_start), n_nodes)
    def _compute_impact_center(self, df):
        """
        Hitung pusat area terdampak berbobot risiko
        (robust terhadap variasi nama kolom koordinat)
        """

        # -------------------------------
        # Deteksi nama kolom koordinat
        # -------------------------------
        lat_col = None
        lon_col = None

        for c in df.columns:
            c_low = c.lower()
            if c_low in ['lintang', 'latitude', 'lat']:
                lat_col = c
            if c_low in ['bujur', 'longitude', 'lon']:
                lon_col = c

        if lat_col is None or lon_col is None:
            raise KeyError(
                f"[ACO] Kolom koordinat tidak ditemukan. "
                f"Kolom tersedia: {list(df.columns)}"
            )

        weights = df['PheromoneScore'].values

        lat_center = np.average(df[lat_col].values, weights=weights)
        lon_center = np.average(df[lon_col].values, weights=weights)

        return {
            "center_lat": float(lat_center),
            "center_lon": float(lon_center),
            "lat_column_used": lat_col,
            "lon_column_used": lon_col
        }


    def _compute_impact_area(self, df):
        """
        Estimasi luas area terdampak (km²)
        menggunakan radius efektif berbobot risiko
        """
        if df.empty:
            return {
                "effective_radius_km": None,
                "impact_area_km2": None
            }

        r_eff = np.average(
            df['Radius_Visual_KM'].values,
            weights=df['PheromoneScore'].values
        )

        area_km2 = math.pi * (r_eff ** 2)

        return {
            "effective_radius_km": float(r_eff),
            "impact_area_km2": float(area_km2)
        }
    def _export_for_ga(self, df, center_info):
        """
        Export hasil ACO sebagai input GA
        """
        # Hitung area terdampak
        area_info = self._compute_impact_area(df)

        ga_input = {
            "center_lat": center_info["center_lat"],
            "center_lon": center_info["center_lon"],
            "risk_mean": float(df['Risk_Index'].mean()),
            "risk_max": float(df['Risk_Index'].max()),
            "n_events": int(len(df)),
            "impact_area_km2": area_info["impact_area_km2"]  # <<< PENAMBAHAN
        }

        output_dir = os.path.dirname(self.output_paths['aco_state_file'])
        ga_path = os.path.join(output_dir, "aco_to_ga.json")

        with open(ga_path, "w") as f:
            json.dump(ga_input, f, indent=2)

        self.logger.info(f"[ACO] Output GA tersimpan → {ga_path}")

        return ga_input

    # ----------------------------------
    def run(self, df: pd.DataFrame):
        if df is None or df.empty:
            self.logger.warning("[ACO] DataFrame kosong.")
            return df, {}

        self.env_manager = EnvironmentManager(df, self.logger)

        if self.env_manager.n_nodes <= 1:
            if os.path.exists(self.output_paths['aco_state_file']):
                try:
                    with open(self.output_paths['aco_state_file'], 'rb') as f:
                        state = pickle.load(f)
                    
                    # Logika ini HANYA JALAN PADA SINGLE EVENT
                    if 'PheromoneScore' in df.columns:
                         # Ambil Skor PheromoneScore yang sudah dihitung di FE sebelumnya
                         # Jika ACO dijalankan, skornya pasti sudah ada
                         final_score = df['PheromoneScore'].values 
                         return df, {"pheromone_matrix": state.get('pheromone_matrix')}
                         
                except Exception as e:
                    self.logger.warning(f"[ACO] Gagal memproses live event dengan state: {e}. Mengembalikan skor 0.")
                    df['PheromoneScore'] = 1e-4 # Fallback minimal
                    df['Pheromone_Score'] = 1e-4
                    df['Risk_Index'] = 0.01
                    return df, {}
            else:
                 # Jika tidak ada state, tidak ada ACO.
                 df['PheromoneScore'] = 1e-4
                 df['Pheromone_Score'] = 1e-4
                 df['Risk_Index'] = 0.01
                 self.logger.warning("[ACO] Live Mode: State tidak ditemukan. Menggunakan skor risiko minimum.")
                 return df, {}

        # coba load state lama
        if os.path.exists(self.output_paths['aco_state_file']):
            try:
                with open(self.output_paths['aco_state_file'], 'rb') as f:
                    state = pickle.load(f)
                old_matrix = state.get('pheromone_matrix')
                if isinstance(old_matrix, np.ndarray) and old_matrix.shape == self.env_manager.pheromone_matrix.shape:
                    self.logger.info("[ACO] Melanjutkan dari brain state sebelumnya.")
                    self.env_manager.pheromone_matrix = old_matrix
            except Exception as e:
                self.logger.warning(f"[ACO] Gagal load state lama: {e}")

        self._initialize_colony(self.env_manager.n_nodes)

        for it in range(self.n_iterations):
            for _ in range(max(self.n_steps - 1, 1)):
                if self._step_ants() == 0:
                    break
            self._manage_lifecycle(it)

        # simpan brain state
        try:
            with open(self.output_paths['aco_state_file'], 'wb') as f:
                pickle.dump({'pheromone_matrix': self.env_manager.pheromone_matrix}, f)
        except Exception as e:
            self.logger.warning(f"[ACO] Gagal simpan brain state: {e}")

        return self._finalize_results(df)

    # ==========================================
    # FINAL OUTPUT + RADIUS FIXED
    # ==========================================

    def _finalize_results(self, df: pd.DataFrame):
        """
        Hitung skor risiko node dari matriks pheromone,
        skala ke [1e-4, 1] dan juga ke indeks 0–100.
        """
        node_importance = np.sum(self.env_manager.pheromone_matrix, axis=0)

        # quantile untuk buang outlier ekstrem
        q01, q99 = np.quantile(node_importance, [0.01, 0.99])
        if not np.isfinite(q01):
            q01 = float(node_importance.min())
        if not np.isfinite(q99):
            q99 = float(node_importance.max())

        if q99 <= q01 + 1e-9:
            norm_scores = np.ones_like(node_importance, dtype=float) * 0.5
        else:
            clipped = np.clip(node_importance, q01, q99)
            norm_scores = (clipped - q01) / (q99 - q01)

        # jaga supaya tidak nol total
        norm_scores = np.clip(norm_scores, 1e-4, 1.0)

        df_out = self.env_manager.df.copy()

        # 🔑 KOMPATIBILITAS NAMA KOLUMN
        df_out['PheromoneScore'] = norm_scores  # dipakai FE + NB + cache
        df_out['Pheromone_Score'] = norm_scores  # kalau mau style snake_case

        # indeks 0–100 biar lebih intuitif
        df_out['Risk_Index'] = (norm_scores * 100.0).round(2)

        df_out['Status_Zona'] = df_out['PheromoneScore'].apply(
            lambda x: 'Terdampak' if x >= self.risk_threshold else 'Aman'
        )
        

        # ------------------ RADIUS VISUAL ------------------
        mags = (
            df_out['Magnitudo_Original']
            if 'Magnitudo_Original' in df_out.columns
            else df_out['Magnitudo']
        )
        pher = df_out['Pheromone_Score']

        base_r = np.power(np.clip(mags, 0.0, None), 1.3)  # sedikit lebih kecil
        radius_km = base_r * (1.0 + 0.5 * pher)  # dipengaruhi risiko
        radius_km = np.clip(radius_km, 3.0, 80.0)  # min 3km, max 80km

        df_out['Radius_Visual_KM'] = radius_km
        # --------------------------------------------------

        self._save_to_disk(df_out)
        self._generate_visuals(df_out)

        center_info = self._compute_impact_center(df_out)
        area_info = self._compute_impact_area(df_out)
        ga_payload = self._export_for_ga(df_out, center_info)

        self.logger.info(
            f"[ACO] Impact Center: {center_info} | "
            f"Impact Area (km²): {area_info['impact_area_km2']}"
        )

        return df_out, {
            "pheromone_matrix": self.env_manager.pheromone_matrix,
            "impact_center": center_info,
            "impact_area": area_info,
            "ga_input": ga_payload
        }

    # ==========================================
    # SAVE FILES
    # ==========================================

    def _save_to_disk(self, df):
        mag_col = 'Magnitudo_Original' if 'Magnitudo_Original' in df.columns else 'Magnitudo'
        depth_col = 'Kedalaman_Original' if 'Kedalaman_Original' in df.columns else 'Kedalaman_km'

        cols = [
            'Tanggal', 'Lintang', 'Bujur', mag_col, depth_col, 'Lokasi',
            'PheromoneScore', 'Risk_Index', 'Status_Zona', 'Radius_Visual_KM'
        ]
        
        # Tambahkan kolom yang mungkin sudah di-rename oleh FeatureEngineer
        final_df = df.copy()
        if 'EQ_Lintang' in df.columns and 'Lintang' not in df.columns:
            final_df['Lintang'] = df['EQ_Lintang']
        if 'EQ_Bujur' in df.columns and 'Bujur' not in df.columns:
            final_df['Bujur'] = df['EQ_Bujur']
        if 'Nama' in df.columns and 'Lokasi' not in df.columns:
            final_df['Lokasi'] = df['Nama']
        if 'Acquired_Date' in df.columns and 'Tanggal' not in df.columns:
            final_df['Tanggal'] = df['Acquired_Date']

        final_df = final_df[[c for c in cols if c in final_df.columns]].rename(columns={mag_col: 'Magnitudo', depth_col: 'Kedalaman'})

        try:
            final_df.to_excel(self.output_paths['aco_zoning_excel'], index=False)
            final_df.to_csv(self.output_paths['aco_epicenters_csv'], index=False)
        except Exception as e:
            self.logger.error(f"Gagal menyimpan output ACO: {e}")

    # ==========================================
    # VISUAL: CIRCLE KUNING + POPUP LENGKAP
    # ==========================================

    def _generate_visuals(self, df):
        """
        Visual:
        - Circle orange (zona) + titik pusat merah
        - Popup: lokasi, tanggal, M, kedalaman, radius, Risk Score, Risk Index
        """
        if df.empty or 'Lintang' not in df.columns or 'Bujur' not in df.columns:
            self.logger.warning("[ACO] Dataframe kosong atau kurang kolom untuk visualisasi.")
            return

        try:
            center = [float(df['Lintang'].mean()), float(df['Bujur'].mean())]
            m = folium.Map(location=center, zoom_start=7, tiles='CartoDB positron')

            for _, r in df.iterrows():
                radius_m = float(r['Radius_Visual_KM']) * 1000.0

                # Fallbacks untuk kolom yang mungkin tidak ada di df input
                lokasi = r.get('Lokasi', r.get('Nama', '-'))
                tanggal = r.get('Tanggal', r.get('Acquired_Date', '-'))
                mag = r.get('Magnitudo', r.get('Magnitudo_Original', '-'))
                depth = r.get('Kedalaman', r.get('Kedalaman (km)', r.get('Kedalaman_km', '-')))
                
                # Pastikan numerik
                try: mag = f"{float(mag):.1f}" if mag != '-' else mag
                except: mag = str(mag)
                try: depth = f"{float(depth):.0f}" if depth != '-' else depth
                except: depth = str(depth)
                
                rad = float(r['Radius_Visual_KM'])
                risk = float(r['Pheromone_Score'])
                risk_idx = float(r.get('Risk_Index', risk * 100.0))

                popup_html = f"""
                <b>Lokasi:</b> {lokasi}<br>
                <b>Tanggal:</b> {tanggal}<br>
                <b>Magnitudo:</b> {mag}<br>
                <b>Kedalaman:</b> {depth} km<br>
                <b>Radius Prediksi:</b> {rad:.1f} km<br>
                <b>Risk Score (0–1):</b> {risk:.4f}<br>
                <b>Risk Index (0–100):</b> {risk_idx:.2f}
                """

                popup = folium.Popup(popup_html, max_width=320)

                # CIRCLE ZONA
                folium.Circle(
                    location=[r['Lintang'], r['Bujur']],
                    radius=radius_m,
                    color='orange',
                    fill=True,
                    fill_opacity=0.25,
                    weight=1.5,
                    popup=popup
                ).add_to(m)

                # TITIK PUSAT
                folium.CircleMarker(
                    location=[r['Lintang'], r['Bujur']],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_opacity=1.0
                ).add_to(m)

            m.save(self.output_paths['aco_impact_html'])
            self.logger.info(f"Visual ACO tersimpan: {self.output_paths['aco_impact_html']}")

        except Exception as e:
            self.logger.error(f"Gagal membuat visual ACO: {e}")