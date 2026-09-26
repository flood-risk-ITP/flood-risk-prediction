"""
Konfigurasi statis inference pipeline.

Nilai-nilai di sini bersumber dari INFERENCE_CONTRACT.md dan
STAGE_DEPLOYMENT_MAP.md (CHECKPOINT 01). Dilarang mengubah FEATURE_COLUMNS,
LOOKBACK, atau CLASS_NAMES tanpa audit ulang terhadap artefak model/scaler.
"""

from __future__ import annotations

import os

# -----------------------------------------------------------------------
# PATH
# -----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(MODELS_DIR, "model_final_4_class.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
STAGE7_MEDIANS_PATH = os.path.join(CONFIG_DIR, "stage7_monthly_medians.json")

# Direktori cache akuisisi data (Ogimet/Wyoming), dibuat saat runtime jika
# belum ada. Bukan bagian dari kontrak model, hanya kemudahan operasional.
CACHE_DIR = os.path.join(BASE_DIR, ".cache")

# -----------------------------------------------------------------------
# KONTRAK MODEL (INFERENCE_CONTRACT.md Bagian 1-2-3-5-6) — JANGAN UBAH
# -----------------------------------------------------------------------
# Identitas stasiun pengamatan BMKG (Stasiun Meteorologi Minangkabau / Padang)
STATION_ID = "96163"

# 8 fitur input yang digunakan oleh model LSTM untuk prediksi risiko banjir.
# Urutan ini wajib identik dengan scaler dan model agar input data konsisten.
FEATURE_COLUMNS = [
    "rr",       # curah hujan harian (Ogimet)
    "tavg",     # suhu udara rata-rata harian (Ogimet)
    "rh",       # kelembapan relatif (Ogimet)
    "cin",      # SBCIN (SounderPy)
    "kindex",   # K-Index (SHARPpy direct)
    "li",       # LI_SB_500 (SHARPpy direct)
    "tt",       # Total Totals (SHARPpy direct)
    "sweat",    # SWEAT (SHARPpy direct)
]

# Pemisahan variabel permukaan (Ogimet) dan variabel atmosfer atas (Sounding) untuk strategi penanganan data hilang.
OGIMET_FEATURE_COLUMNS = ["rr", "tavg", "rh"]
SOUNDING_FEATURE_COLUMNS = ["cin", "kindex", "li", "tt", "sweat"]

# Jumlah hari historis (D-7 hingga D-1) yang digunakan sebagai rentang waktu (look-back) input model LSTM.
LOOKBACK = 7  # D-7 ... D-1, target D tidak termasuk window

# 4 kategori tingkat risiko banjir yang menjadi hasil keluaran klasifikasi model.
CLASS_NAMES = ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"]

# -----------------------------------------------------------------------
# STAGE 2 — placeholder standardisasi Ogimet (WAJIB direplikasi persis)
# -----------------------------------------------------------------------
OGIMET_PLACEHOLDER_MAP = {
    "Tr": 0.0,
    "----": None,
    "-----": None,
}

# -----------------------------------------------------------------------
# STAGE 4 / wyouming_downloader.py — prioritas sumber & jam observasi
# -----------------------------------------------------------------------
SRC_ORDER = ["FM35", None]
HOUR_PRIORITY = [12, 0]  # coba 12Z dulu, fallback 00Z
# -----------------------------------------------------------------------
# STAGE 7 — missing-value handling (WAJIB direplikasi persis)
# -----------------------------------------------------------------------
NO_SOUNDING_STATUS = "NO_SOUNDING"
SELECTED_STATUS = "SELECTED"

# OPEN DECISION 1 (INFERENCE_CONTRACT.md #10): ukuran buffer sebelum D-7
# untuk interpolate(method="time", limit_direction="both") BELUM diputuskan
# pada CHECKPOINT 01. Implementasi ini TIDAK mengarang nilai baru — buffer
# harus disuplai eksplisit oleh caller (lihat src/preprocessing.py) dan
# kegagalan menyuplai cukup data historis akan membuat validation gate
# menolak prediksi, bukan diam-diam memakai default yang tidak diverifikasi.
INTERPOLATION_BUFFER_DAYS = None  # sengaja None -- lihat OPEN DECISION 1

# Path ke integrated dataset historis
HISTORICAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "integrated", "integrated_dataset.csv")


