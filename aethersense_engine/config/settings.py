"""
Konfigurasi terpusat untuk inference pipeline Sistem Penilaian Risiko Banjir
Kota Padang (Random Forest + XGBoost Voting Ensemble).
Subpackage: aethersense_engine
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from aethersense_engine.canvas.step8_final import FEATURE_COLUMNS  # noqa: E402

MODELS_DIR = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.joblib")

# WMO ID stasiun Padang (Minangkabau) - sama untuk Ogimet maupun Wyoming.
STATION_ID = "96163"

# Jam sounding yang dipakai untuk indeks atmosfer resmi (Subbab 3.4.2 proposal).
SOUNDING_HOUR = 12  # UTC

# Label 1-4 -> nama kelas, urutan sesuai target_names di step9_model_training.py.
CLASS_NAMES = {
    1: "Low",
    2: "Medium",
    3: "High",
    4: "Very High",
}
CLASS_ORDER = ["Low", "Medium", "High", "Very High"]

# Sistem retrospektif (Bagian 6 spesifikasi): tanggal minimal = awal cakupan
# data historis yang dipakai untuk melatih model.
MIN_SUPPORTED_DATE = "2017-01-01"

# Sounding 12Z = 19:00 WIB -> data hari ini baru "final" setelah jam ini.
WIB_UTC_OFFSET_HOURS = 7
SOUNDING_AVAILABLE_HOUR_WIB = 19
