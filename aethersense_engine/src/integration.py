"""
Orkestrasi Bagian 3 spesifikasi: gabungkan Ogimet + Wyoming + indeks atmosfer
+ skor ordinal + rolling window jadi satu vektor 16 fitur.
Subpackage: aethersense_engine
"""

import pandas as pd

from aethersense_engine.config.settings import FEATURE_COLUMNS
from aethersense_engine.src.atmospheric_indices import calculate_daily_indices
from aethersense_engine.src.ogimet import fetch_ogimet_daily
from aethersense_engine.src.ordinal_scoring import calculate_ordinal_scores
from aethersense_engine.src.rolling_features import compute_rolling_features
from aethersense_engine.src.wyoming import WyomingFetchError, fetch_wyoming_sounding


def build_feature_vector(target_date):
    """Ambil semua data mentah & hitung 16 fitur final untuk satu tanggal.

    Returns:
        X_row: pd.DataFrame, shape (1, 16), kolom = FEATURE_COLUMNS
        raw_values: dict berisi 10 nilai mentah (RR, T_avg, RH, CAPE, CIN,
                    K_Index, LI, TT, SWEAT, PW) untuk direview pengguna.
    """
    ogimet_data = fetch_ogimet_daily(target_date)  # {"RR", "T_avg", "RH"}

    profile = fetch_wyoming_sounding(target_date)
    indices = calculate_daily_indices(profile)
    if indices["status"] not in ("OK", "DICURIGAI"):
        raise WyomingFetchError(
            f"Indeks atmosfer gagal dihitung untuk {target_date.isoformat()} "
            f"(status: {indices['status']})."
        )

    df_one = pd.DataFrame([{
        "date": target_date,
        "CAPE": indices["CAPE"],
        "K_Index": indices["K_Index"],
        "LI": indices["LI"],
        "TT": indices["TT"],
        "SWEAT": indices["SWEAT"],
    }])
    ordinal_row = calculate_ordinal_scores(df_one).iloc[0]

    rolling = compute_rolling_features(
        target_date,
        cape_today=indices["CAPE"],
        pw_today=indices["PW"],
        li_today=indices["LI"],
    )

    feature_values = {
        "CAPE": indices["CAPE"],
        "CIN": indices["CIN"],
        "LI": indices["LI"],
        "K_Index": indices["K_Index"],
        "TT": indices["TT"],
        "SWEAT": indices["SWEAT"],
        "PW": indices["PW"],
        "T_avg": ogimet_data["T_avg"],
        "RH": ogimet_data["RH"],
        "score_CAPE": ordinal_row["score_CAPE"],
        "score_K_Index": ordinal_row["score_K_Index"],
        "score_LI": ordinal_row["score_LI"],
        "score_TT": ordinal_row["score_TT"],
        "score_SWEAT": ordinal_row["score_SWEAT"],
        "PW_mean_3d": rolling["PW_mean_3d"],
        "LI_min_3d": rolling["LI_min_3d"],
    }
    X_row = pd.DataFrame([feature_values])[FEATURE_COLUMNS]

    raw_values = {
        "RR": ogimet_data["RR"],
        "T_avg": ogimet_data["T_avg"],
        "RH": ogimet_data["RH"],
        "CAPE": indices["CAPE"],
        "CIN": indices["CIN"],
        "K_Index": indices["K_Index"],
        "LI": indices["LI"],
        "TT": indices["TT"],
        "SWEAT": indices["SWEAT"],
        "PW": indices["PW"],
    }
    return X_row, raw_values
