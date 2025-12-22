import pandas as pd
import re

def preprocess_earthquake_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse kolom 'Keterangan' menjadi kolom numerik per jenis gempa.
    """
    # Pastikan kolom Keterangan ada
    if "Keterangan" not in df.columns:
        df["Keterangan"] = ""

    # Fungsi parse per baris
    def parse_keterangan(ket: str) -> dict:
        result = {
            "Hembusan": 0,
            "Tektonik_Lokal": 0,
            "Tektonik_Jauh": 0,
            "Tremor_Menerus": 0,
            "Vulkanik_Dangkal": 0,
            "Vulkanik_Dalam": 0,
            "Letusan_Erupsi": 0,
            "Guguran": 0,
            "Harmonik": 0,
            "Tremor_Non_Harmonik": 0
        }
        for line in str(ket).split("\n"):
            match = re.match(
                r"(\d+)\s+.*(Hembusan|Tektonik Lokal|Tektonik Jauh|Tremor Menerus|Vulkanik Dangkal|Vulkanik Dalam|Letusan/Erupsi|Guguran|Harmonik|Tremor Non-Harmonik)",
                line, re.IGNORECASE)
            if match:
                jumlah, jenis = match.groups()
                jenis_key = jenis.replace("/", "_").replace(" ", "_").title()
                if jenis_key in result:
                    result[jenis_key] = int(jumlah)
        return result

    # Terapkan parsing ke DataFrame
    parsed = df["Keterangan"].apply(parse_keterangan).apply(pd.Series)

    # Gabungkan dengan df asli
    df = pd.concat([df, parsed], axis=1)

    # Drop kolom Keterangan lama
    df.drop(columns=["Keterangan"], inplace=True)

    # Pastikan numeric semua kolom penting
    num_cols = ["EQ_Lintang", "EQ_Bujur", "Magnitudo", "Kedalaman (km)"] + list(parsed.columns)
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Format tanggal
    if "Acquired_Date" in df.columns:
        df["Acquired_Date"] = pd.to_datetime(df["Acquired_Date"], errors="coerce")
        df = df.dropna(subset=["Acquired_Date"])
        df["Acquired_Date"] = df["Acquired_Date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df