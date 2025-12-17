# VolcanoAI/processing/fetch_data.py
# -- coding: utf-8 --

import os
import logging
import json
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

API_ENDPOINT = "https://magma.esdm.go.id/api/v1/gempa-bumi"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'Volcanic.xlsx')
REQUEST_TIMEOUT_SECONDS = 30
HTTP_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class DataFetcher:
    """
    Kelas yang bertanggung jawab untuk mengambil, mem-parsing, dan menyimpan
    data gempa bumi dari API eksternal.
    """

    def __init__(self, start_date: datetime, end_date: datetime, output_path: str):
        """
        Inisialisasi DataFetcher.

        Args:
            start_date (datetime): Tanggal mulai untuk pengambilan data.
            end_date (datetime): Tanggal akhir untuk pengambilan data.
            output_path (str): Path file untuk menyimpan hasil Excel.
        """
        if start_date >= end_date:
            raise ValueError("Tanggal mulai harus sebelum tanggal akhir.")
            
        self.start_date_str = start_date.strftime('%Y-%m-%d')
        self.end_date_str = end_date.strftime('%Y-%m-%d')
        self.output_path = output_path
        self.api_params = {
            "startdate": self.start_date_str,
            "enddate": self.end_date_str
        }
        logging.info(f"DataFetcher diinisialisasi untuk mengambil data dari {self.start_date_str} hingga {self.end_date_str}")

    def _make_api_request(self) -> Optional[Dict[str, Any]]:
        """
        Melakukan panggilan HTTP GET ke API (disimulasikan) dan menangani respons.
        """
        logging.info(f"Mengirim permintaan ke {API_ENDPOINT} dengan parameter: {self.api_params}")

        logging.warning("MODE SIMULASI AKTIF: Menggunakan data respons API palsu.")
        fake_response_data = {
            "status": 200, "message": "success",
            "data": [
                {"id": 1, "date": "2023-10-26 10:00:00", "ot": "1698285600", "lat": "-8.2", "lon": "112.9", "depth": "10", "mag": "3.5", "type": "Vulkanik Dalam", "remark": "Gunung Semeru"},
                {"id": 2, "date": "2023-10-26 11:30:00", "ot": "1698291000", "lat": "-8.3", "lon": "115.5", "depth": "80", "mag": "4.1", "type": "Tektonik Jauh", "remark": "Wilayah Bali"},
                {"id": 3, "date": "2023-10-27 08:00:00", "ot": "1698364800", "lat": "-7.9", "lon": "112.4", "depth": "5", "mag": "2.8", "type": "Hembusan", "remark": "Gunung Kelut"},
            ]
        }
        return fake_response_data

    def _parse_response_to_dataframe(self, response_json: Dict[str, Any]) -> pd.DataFrame:
        """
        Mengubah respons JSON mentah yang kompleks menjadi DataFrame Pandas yang bersih.
        Fungsi ini secara spesifik menangani struktur data dari sumber.
        """
        if not response_json or "data" not in response_json or not isinstance(response_json["data"], list):
            logging.warning("Respons JSON tidak valid atau tidak berisi data.")
            return pd.DataFrame()

        logging.info(f"Mem-parsing {len(response_json['data'])} record dari respons API.")
        
        parsed_records: List[Dict[str, Any]] = []
        for item in response_json["data"]:
            try:
                record = {
                    'Tanggal': pd.to_datetime(item.get('date')),
                    'Lintang': float(item.get('lat', 0)),
                    'Bujur': float(item.get('lon', 0)),
                    'Magnitudo': float(item.get('mag', 0)),
                    'Kedalaman (km)': int(item.get('depth', 999)),
                    'Keterangan': f"Jenis: {item.get('type', 'N/A')}. Lokasi: {item.get('remark', 'N/A')}",
                    'Source_API_ID': item.get('id')
                }
                parsed_records.append(record)
            except (ValueError, TypeError) as e:
                logging.warning(f"Melewatkan record karena data tidak valid: {item}. Error: {e}")
                continue
        
        if not parsed_records:
            return pd.DataFrame()
            
        df = pd.DataFrame(parsed_records)
        standard_columns = ['Tanggal', 'Lintang', 'Bujur', 'Magnitudo', 'Kedalaman (km)', 'Keterangan', 'Source_API_ID']
        df = df[standard_columns]
        
        return df

    def run(self) -> bool:
        """
        Orkestrator utama untuk menjalankan seluruh alur kerja:
        1. Membuat permintaan API.
        2. Mem-parsing respons.
        3. Menyimpan ke file Excel.

        Returns:
            bool: True jika berhasil, False jika gagal.
        """
        logging.info("="*60)
        logging.info("Memulai Proses Akuisisi Data Gempa Terbaru")
        logging.info("="*60)

        api_response = self._make_api_request()
        if not api_response:
            logging.error("Akuisisi data gagal: tidak ada respons dari API.")
            return False

        df = self._parse_response_to_dataframe(api_response)
        if df.empty:
            logging.warning("Tidak ada data valid yang dapat diparsing. Proses dihentikan.")
            return True 

        logging.info(f"Menyimpan {len(df)} record baru ke file: {self.output_path}")
        try:
            output_dir = os.path.dirname(self.output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            df.to_excel(self.output_path, index=False, engine='openpyxl')
            logging.info(f"✅ SUKSES: Data terbaru berhasil disimpan.")
            return True
        except Exception as e:
            logging.error(f"Gagal menyimpan file Excel di '{self.output_path}': {e}", exc_info=True)
            return False

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] - %(message)s')
    
    parser = argparse.ArgumentParser(
        description="Mengambil data gempa bumi vulkanik dari API dan menyimpannya secara lokal.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-d', '--days',
        type=int,
        default=7,
        help="Jumlah hari ke belakang dari sekarang untuk mengambil data.\nContoh: --days 30 (untuk mengambil data 30 hari terakhir)."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=OUTPUT_PATH,
        help=f"Path file output Excel.\nDefault: {OUTPUT_PATH}"
    )
    args = parser.parse_args()

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        fetcher = DataFetcher(start_date, end_date, args.output)
        success = fetcher.run()
        
        if success:
            print("\nProses akuisisi data selesai.")
        else:
            print("\nProses akuisisi data gagal. Periksa log di atas untuk detail.")

    except ValueError as e:
        logging.error(f"Error pada parameter input: {e}")
    except Exception as e:
        logging.critical(f"Terjadi error fatal yang tidak terduga: {e}", exc_info=True)