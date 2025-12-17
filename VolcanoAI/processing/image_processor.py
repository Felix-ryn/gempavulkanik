# VolcanoAI/processing/image_processor.py
# -- coding: utf-8 --

# ==============================================================================
# IMPOR PUSTAKA (LIBRARY) YANG DIBUTUHKAN
# ==============================================================================
import os
import re
import logging
import yaml
import cv2
import pytesseract
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from tqdm import tqdm
from scipy.spatial import distance

# ==============================================================================
# PEMUATAN DAN VALIDASI KONFIGURASI
# ==============================================================================
@dataclass
class ImageProcessorConfig:
    """
    Dataclass untuk menyimpan dan memvalidasi konfigurasi dari file YAML.
    Ini memastikan semua parameter yang dibutuhkan ada dan dalam format yang benar.
    """
    settings: Dict[str, Any]
    roi: Dict[str, List[int]]
    graph_axis_calibration: Dict[str, Any]
    detection_targets: List[Dict[str, Any]]
    final_columns_order: List[str]

    def __post_init__(self):
        """Validasi dan konversi tipe data otomatis setelah inisialisasi."""
        calib = self.graph_axis_calibration
        calib['x_date_start'] = datetime.strptime(calib['x_date_start'], '%Y-%m-%d')
        calib['x_date_end'] = datetime.strptime(calib['x_date_end'], '%Y-%m-%d')
        calib['y_vrp_min'] = 10**calib['y_vrp_log_min']
        calib['y_vrp_max'] = 10**calib['y_vrp_log_max']
        for target in self.detection_targets:
            target['hsv_lower'] = np.array(target['hsv_lower'], dtype=np.uint8)
            target['hsv_upper'] = np.array(target['hsv_upper'], dtype=np.uint8)

def load_config_from_yaml(config_path: str) -> ImageProcessorConfig:
    """Memuat konfigurasi dari file YAML ke dalam objek ImageProcessorConfig."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"File konfigurasi '{config_path}' tidak ditemukan.")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        return ImageProcessorConfig(**raw_config)
    except Exception as e:
        raise ValueError(f"Gagal memuat atau memvalidasi konfigurasi YAML: {e}") from e

# ==============================================================================
# KELAS UTAMA UNTUK EKSTRAKSI DATA GAMBAR
# ==============================================================================
class AdvancedVolcanoExtractor:
    """
    Kelas ini mengorkestrasi seluruh proses ekstraksi data dari gambar-gambar vulkanik,
    mulai dari pembacaan gambar, ekstraksi teks (OCR), deteksi titik data pada grafik,
    hingga agregasi hasil ke dalam format tabel.
    """
    def __init__(self, config: ImageProcessorConfig, base_folder: str, sub_folders: List[str], output_filename: str):
        self.config = config
        self.base_folder = base_folder
        self.sub_folders = sub_folders
        self.output_path = os.path.join(self.base_folder, output_filename)
        self.debug_dir = os.path.join("output", "debug_images_ocr")
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.config.settings.get('generate_debug_images', False):
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # Pendekatan efisien: kumpulkan baris data yang sudah jadi dalam satu list.
        self.final_data_rows: List[Dict[str, Any]] = []

    # --- Bagian Strategi Preprocessing OCR ---
    
    def _preprocess_strategy_simple_gray(self, image: np.ndarray) -> np.ndarray:
        """Strategi 1: Paling lembut, hanya grayscale dan perbesar resolusi."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    def _preprocess_strategy_simple_invert(self, image: np.ndarray) -> np.ndarray:
        """Strategi 2: Inversi warna sederhana. Efektif jika teks berwarna terang di atas background gelap."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        return cv2.bitwise_not(resized)

    def _preprocess_strategy_adaptive_thresh(self, image: np.ndarray) -> np.ndarray:
        """Strategi 3: Menggunakan adaptive thresholding, sangat baik untuk pencahayaan yang tidak merata."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.medianBlur(resized, 3) # Kurangi noise sebelum thresholding
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    def _preprocess_strategy_otsu_inverted(self, image: np.ndarray) -> np.ndarray:
        """Strategi 4: Pendekatan agresif dengan sharpening dan Otsu's binarization."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        resized = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_LANCZOS4)
        gaussian_blur = cv2.GaussianBlur(resized, (0,0), 3.0)
        sharpened = cv2.addWeighted(resized, 1.5, gaussian_blur, -0.5, 0)
        _, thresholded = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresholded
        
    def _extract_text_from_roi(self, full_image: np.ndarray, roi_name: str, pattern: str, filename_for_debug: str, ocr_config: str = "--oem 3 --psm 6") -> Optional[str]:
        """
        Fungsi OCR utama yang secara cerdas mencoba beberapa strategi preprocessing secara berurutan
        hingga berhasil mengekstrak teks yang cocok dengan pola yang diberikan.
        """
        try:
            y1, y2, x1, x2 = self.config.roi[roi_name]
            roi_image = full_image[y1:y2, x1:x2]
            
            strategies: List[Tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
                ("adaptive_thresh", self._preprocess_strategy_adaptive_thresh),
                ("simple_invert", self._preprocess_strategy_simple_invert),
                ("simple_gray", self._preprocess_strategy_simple_gray),
                ("otsu_inverted", self._preprocess_strategy_otsu_inverted),
            ]
            last_detected_text = ""
            for strategy_name, preprocess_func in strategies:
                processed_roi = preprocess_func(roi_image)
                detected_text = pytesseract.image_to_string(processed_roi, config=ocr_config, lang='eng').strip().replace("\n", " ")
                last_detected_text = detected_text

                if detected_text:
                    match = re.search(pattern, detected_text, re.IGNORECASE)
                    if match:
                        # Mengambil hasil dari group pertama jika ada, jika tidak, ambil seluruh kecocokan.
                        extracted_value = match.group(1).strip() if match.groups() else match.group(0).strip()
                        self.logger.info(f"  -> ROI '{roi_name}' [Strategi: {strategy_name}]: SUKSES! Teks: '{detected_text}' -> Ditemukan: '{extracted_value}'")
                        return extracted_value
            
            # Hanya tampilkan log peringatan jika SEMUA strategi gagal.
            self.logger.warning(f"  -> ROI '{roi_name}': GAGAL setelah mencoba {len(strategies)} strategi. Teks terakhir ({strategies[-1][0]}): '{last_detected_text}'")
            return None
        except Exception as e:
            self.logger.error(f"  -> Error fatal saat OCR di ROI '{roi_name}' untuk file '{filename_for_debug}': {e}", exc_info=True)
            return None

    # --- Bagian Deteksi Titik Grafik ---
    @staticmethod
    def _is_shape_valid(contour: np.ndarray, expected_shape: str, area: float) -> bool:
        """Memvalidasi kontur berdasarkan bentuk geometris yang diharapkan (segitiga, kotak, lingkaran)."""
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: return False
        approx_poly = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

        if expected_shape == 'triangle': return len(approx_poly) == 3
        elif expected_shape == 'square':
            if len(approx_poly) == 4:
                _, _, w, h = cv2.boundingRect(approx_poly)
                aspect_ratio = w / float(h) if h > 0 else 0
                return 0.8 <= aspect_ratio <= 1.2
            return False
        elif expected_shape == 'circle':
            if len(approx_poly) > 5:
                circularity = 4 * np.pi * area / (perimeter**2)
                return 0.75 <= circularity <= 1.25
            return False
        return False

    def _filter_close_points(self, centroids: List[Tuple[int, int]], min_dist: int = 10) -> List[Tuple[int, int]]:
        """Menghapus titik duplikat yang terdeteksi terlalu berdekatan."""
        if not centroids: return []
        points = sorted(centroids, key=lambda p: p[0])
        filtered_points = []
        while points:
            ref_point = points.pop(0)
            filtered_points.append(ref_point)
            points = [p for p in points if distance.euclidean(ref_point, p) >= min_dist]
        return filtered_points

    def _detect_graph_points(self, hsv_graph_crop: np.ndarray, target: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Mendeteksi titik data pada crop-an grafik dengan filter berlapis untuk akurasi tinggi."""
        mask = cv2.inRange(hsv_graph_crop, target['hsv_lower'], target['hsv_upper'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_cleaned = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_centroids = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < target['min_area']: continue
            
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = area / float(hull_area) if hull_area > 0 else 0
            if solidity < 0.85: continue # Filter bentuk yang tidak solid/kompak
            
            if self._is_shape_valid(c, target['shape'], area):
                m = cv2.moments(c)
                if m["m00"] != 0: 
                    cx = int(m["m10"] / m["m00"])
                    cy = int(m["m01"] / m["m00"])
                    valid_centroids.append((cx, cy))

        return self._filter_close_points(valid_centroids)
        
    def _convert_pixel_to_value(self, px: int, py: int) -> Tuple[datetime.date, float]:
        """Mengonversi koordinat piksel (x, y) menjadi nilai nyata (tanggal dan VRP)."""
        p = self.config.graph_axis_calibration
        px_c, py_c = np.clip(px, p['x_pixel_min'], p['x_pixel_max']), np.clip(py, p['y_pixel_max'], p['y_pixel_min'])
        
        pixel_range_x = float(p['x_pixel_max'] - p['x_pixel_min'])
        x_fraction = (px_c - p['x_pixel_min']) / pixel_range_x
        date_delta_days = (p['x_date_end'] - p['x_date_start']).days
        calculated_date = p['x_date_start'] + pd.to_timedelta(x_fraction * date_delta_days, unit='d')
        
        pixel_range_y = float(p['y_pixel_min'] - p['y_pixel_max'])
        y_fraction = (p['y_pixel_min'] - py_c) / pixel_range_y
        log_range = p['y_vrp_log_max'] - p['y_vrp_log_min']
        log_vrp_value = p['y_vrp_log_min'] + (y_fraction * log_range)
        
        return calculated_date.date(), 10**log_vrp_value

    def _process_single_image(self, img_path: str):
        """Alur kerja utama untuk memproses satu file gambar."""
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            self.logger.error(f"Gagal membaca file gambar: {filename}")
            return

        ocr_config_numeric = "--oem 3 --psm 7 -c tessedit_char_whitelist==0123456789"
        ocr_config_general = "--oem 3 --psm 6"

        s2pix_pattern = r'=?\s*(\d+)'
        s2pix_summit = self._extract_text_from_roi(img, "s2pix_summit", s2pix_pattern, filename, ocr_config_numeric)
        s2pix_total = self._extract_text_from_roi(img, "s2pix_total", s2pix_pattern, filename, ocr_config_numeric)

        date_pattern = r'(\d{1,2}\s*-\s*(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*-\s*\d{4}\s+\d{2}:\d{2}:\d{2})'
        acq_date_str = self._extract_text_from_roi(img, "acquisition_info", date_pattern, filename, ocr_config_general)
        
        sensor_pattern = r'\b(OLI|MSI|LANDSAT|MODIS|VIIRS|TIRS[A-Z\s/0-9\-]*)\b'
        sensor_type = self._extract_text_from_roi(img, "sensor_type_info", sensor_pattern, filename, ocr_config_general)
        
        volcano_name = os.path.basename(os.path.dirname(img_path)).capitalize()
        acquired_date = None
        if acq_date_str:
            try:
                normalized_date_str = re.sub(r'\s+', ' ', acq_date_str.replace('-', ' ')).strip()
                acquired_date = datetime.strptime(normalized_date_str, '%d %b %Y %H:%M:%S')
            except ValueError:
                self.logger.warning(f"Gagal konversi format tanggal tidak valid: '{acq_date_str}' (dari file '{filename}')")

        base_metadata = {
            "Nama": volcano_name, "Acquired_Date": acquired_date, "Source_Sensor_Type": sensor_type,
            "S2Pix_summit": int(s2pix_summit) if s2pix_summit and s2pix_summit.isdigit() else 0,
            "S2Pix_total": int(s2pix_total) if s2pix_total and s2pix_total.isdigit() else 0, 
            "Source_Filename": filename
        }

        y1, y2, x1, x2 = self.config.roi["graph_last_2_years"]
        graph_crop = img[y1:y2, x1:x2]
        hsv_graph = cv2.cvtColor(graph_crop, cv2.COLOR_BGR2HSV)
        graph_points = []
        for target in self.config.detection_targets:
            points = self._detect_graph_points(hsv_graph, target)
            for px, py in points:
                try:
                    date_val, vrp_val = self._convert_pixel_to_value(px, py)
                    graph_points.append({"Data_Point_Date_on_Graph": date_val, "Tipe": target['name'], "Nilai": vrp_val})
                except Exception as e:
                    self.logger.warning(f"Gagal konversi piksel ke nilai di '{filename}': {e}")

        if not graph_points:
            self.final_data_rows.append(base_metadata) 
        else:
            df_graph = pd.DataFrame(graph_points).pivot_table(
                index="Data_Point_Date_on_Graph", 
                columns='Tipe', 
                values='Nilai',
                aggfunc='sum').reset_index()

            for _, row in df_graph.iterrows():
                self.final_data_rows.append({**base_metadata, **row.to_dict()})

    def _aggregate_and_format_results(self) -> pd.DataFrame:
        """Mengubah list hasil menjadi DataFrame final dan melakukan pembersihan akhir."""
        if not self.final_data_rows:
            self.logger.warning("Ekstraksi tidak menghasilkan data. DataFrame akan kosong.")
            return pd.DataFrame()
        
        self.logger.info(f"Membuat DataFrame final dari {len(self.final_data_rows)} baris data yang terkumpul...")
        final_df = pd.DataFrame(self.final_data_rows)
        

        df_cols = {}
        for col in final_df.columns:
            base_col = col.split('_alt')[0]
            if base_col not in df_cols:
                df_cols[base_col] = final_df[col].copy()
            else:
                df_cols[base_col].fillna(final_df[col], inplace=True)
        
        final_df = pd.DataFrame(df_cols)

        for col in self.config.final_columns_order:
            if col not in final_df.columns:
                final_df[col] = np.nan
        final_df = final_df[self.config.final_columns_order]

        for col in ['S2Pix_summit', 'S2Pix_total']:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0).astype(int)
        for col in ['Acquired_Date', 'Data_Point_Date_on_Graph']:
            final_df[col] = pd.to_datetime(final_df[col], errors='coerce').dt.strftime('%Y-%m-%d').replace('NaT', None)
            
        return final_df.sort_values(by=['Nama', 'Acquired_Date', 'Data_Point_Date_on_Graph'], na_position='first').reset_index(drop=True)

    def run_extraction_and_export(self):
        """Menjalankan seluruh pipeline ekstraksi: mencari file, memproses, dan menyimpan ke Excel."""
        self.logger.info("="*70 + "\n" + " MEMULAI PIPELINE EKSTRAKSI GAMBAR (VERSI FINAL TERKALIBRASI) ".center(70) + "\n" + "="*70)
        
        image_paths = []
        for folder in self.sub_folders:
            full_path = os.path.join(self.base_folder, folder)
            if os.path.isdir(full_path):
                for f in sorted(os.listdir(full_path)):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        image_paths.append(os.path.join(full_path, f))

        if not image_paths:
            self.logger.critical("Kritis: Tidak ada file gambar yang ditemukan di subfolder yang ditentukan. Proses dihentikan.")
            return

        for img_path in tqdm(image_paths, desc="Mengekstrak Data dari Gambar"):
            try:
                self._process_single_image(img_path)
            except Exception as e:
                self.logger.error(f"Terjadi error tak terduga saat memproses {os.path.basename(img_path)}: {e}", exc_info=True)

        final_df = self._aggregate_and_format_results()
        
        if final_df.empty:
            self.logger.error("Ekstraksi tidak menghasilkan data apa pun. File Excel tidak akan dibuat.")
        else:
            try:
                final_df.to_excel(self.output_path, index=False, engine='openpyxl')
                self.logger.info(f"✅ SUKSES! {len(final_df)} baris data telah disimpan di: '{self.output_path}'")
            except Exception as e:
                self.logger.error(f"❌ GAGAL menyimpan file Excel. Error: {e}", exc_info=True)
        
        self.logger.info("\n" + "="*70 + "\n" + " LAPORAN AKHIR PROSES EKSTRAKSI ".center(70) + "\n" + "-"*70 +
                        f"\n  Total gambar diproses    : {len(image_paths)}" +
                        f"\n  Total baris data dihasilkan: {len(final_df) if not final_df.empty else 0}\n" + "="*70)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s - [%(name)-25s] - %(message)s')
    
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR)) 
        CONFIG_YAML_PATH = os.path.join(PROJECT_ROOT, "VolcanoAI", "mirova_config.yaml")
        BASE_FOLDER_PATH = PROJECT_ROOT
        VOLCANO_SUBFOLDERS = ["data_volcano/ijen", "data_volcano/kelut", "data_volcano/raung", "data_volcano/semeru"]
        OUTPUT_EXCEL_FILENAME = "data_volcano/Hasil_Ekstraksi_Data_Vulkanik_Final.xlsx"
        
        config_obj = load_config_from_yaml(CONFIG_YAML_PATH)

        extractor = AdvancedVolcanoExtractor(
            config=config_obj,
            base_folder=BASE_FOLDER_PATH,
            sub_folders=VOLCANO_SUBFOLDERS,
            output_filename=OUTPUT_EXCEL_FILENAME
        )
        extractor.run_extraction_and_export()

    except FileNotFoundError as e:
        logging.critical(f"Proses utama tidak dapat dimulai karena file penting tidak ditemukan: {e}")
    except Exception as e:
        logging.critical(f"Terjadi error fatal yang tidak terduga pada level eksekusi utama: {e}", exc_info=True)

logging.info("[image_processor.py] Eksekusi skrip selesai.")