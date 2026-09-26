# ============================================================
# STEP 5 - FEATURE ENGINEERING
# Skor ordinal bulanan (Nuryadi et al., 2025) + Rolling Window Statistics
# (CAPE_mean_3d, CAPE_max_3d, PW_mean_3d, LI_min_3d)
# ============================================================
#
# INPUT (hasil Tahap 3):
#   /content/output/indices_clean.csv
#
# OUTPUT:
#   /content/output/ordinal_features.csv       (tanggal + score_CAPE..score_SWEAT)
#   /content/output/rolling_features.csv       (tanggal + CAPE_mean_3d, dst)

import numpy as np
import pandas as pd

OUTPUT_DIR = "../output"

# ============================================================
# 5a. TABEL AMBANG BATAS BULANAN (Nuryadi, Sudiar, Hamdi, & Amir, 2025)
# Sumber: Tabel 3.4 (Januari) + Lampiran A (Februari-Desember)
# Struktur: {bulan: {indeks: (batas_stabil, batas_tidak_stabil)}}
# Untuk all_items indeks KECUALI LI: skor 0 jika nilai < batas_stabil,
#                                 skor 2 jika nilai > batas_tidak_stabil,
#                                 skor 1 di antaranya.
# Untuk LI, arahnya terbalik: skor 0 jika nilai > batas_stabil (kurang negatif),
#                              skor 2 jika nilai < batas_tidak_stabil (lebih negatif).
# ============================================================

MONTHLY_THRESHOLDS = {
    1:  {"CAPE": (1141, 1306), "K_Index": (32.8, 34.8), "LI": (-2.56, -3.21), "TT": (43.1, 44.5), "SWEAT": (204.9, 212.3)},
    2:  {"CAPE": (975, 1198),  "K_Index": (33.3, 35.6), "LI": (-2.10, -3.07), "TT": (43.0, 44.6), "SWEAT": (198.2, 211.8)},
    3:  {"CAPE": (607, 1255),  "K_Index": (33.0, 34.3), "LI": (-1.87, -3.16), "TT": (43.5, 44.4), "SWEAT": (204.5, 212.7)},
    4:  {"CAPE": (890, 1188),  "K_Index": (32.6, 36.2), "LI": (-1.95, -3.14), "TT": (42.6, 44.5), "SWEAT": (202.2, 220.8)},
    5:  {"CAPE": (889, 1174),  "K_Index": (30.9, 35.6), "LI": (-1.52, -3.04), "TT": (42.4, 44.6), "SWEAT": (202.3, 217.5)},
    6:  {"CAPE": (526, 938),   "K_Index": (32.3, 35.3), "LI": (-1.22, -3.04), "TT": (42.3, 44.0), "SWEAT": (187.0, 266.6)},
    7:  {"CAPE": (665, 1167),  "K_Index": (29.4, 34.7), "LI": (-1.67, -3.38), "TT": (42.8, 45.0), "SWEAT": (183.0, 208.2)},
    8:  {"CAPE": (635, 1183),  "K_Index": (29.9, 34.9), "LI": (-2.36, -3.36), "TT": (42.8, 45.0), "SWEAT": (184.5, 213.7)},
    9:  {"CAPE": (547, 862),   "K_Index": (30.0, 33.9), "LI": (-1.19, -2.51), "TT": (42.3, 44.5), "SWEAT": (190.9, 205.4)},
    10: {"CAPE": (418, 629),   "K_Index": (32.6, 34.6), "LI": (-1.27, -1.95), "TT": (42.9, 43.6), "SWEAT": (197.4, 208.9)},
    11: {"CAPE": (438, 524),   "K_Index": (32.7, 34.4), "LI": (-1.22, -1.82), "TT": (41.7, 43.3), "SWEAT": (196.5, 213.4)},
    12: {"CAPE": (943, 1486),  "K_Index": (34.7, 36.1), "LI": (-2.44, -3.48), "TT": (42.8, 44.3), "SWEAT": (210.4, 215.8)},
}


def ordinal_score_single_value(index_name, value, month):
    """Calculate 0/1/2 score for one index in one month."""
    if pd.isna(value) or month not in MONTHLY_THRESHOLDS:
        return np.nan
    stable_threshold, unstable_threshold = MONTHLY_THRESHOLDS[month][index_name]

    if index_name == "LI":
        # reversed direction: more negative = more unstable
        if value > stable_threshold:
            return 0
        elif value < unstable_threshold:
            return 2
        else:
            return 1
    else:
        if value < stable_threshold:
            return 0
        elif value > unstable_threshold:
            return 2
        else:
            return 1


def calculate_ordinal_scores(df_indices_clean):
    """Add 5 columns: score_CAPE, score_K_Index, score_LI, score_TT, score_SWEAT."""
    df = df_indices_clean.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month

    for index_name in ["CAPE", "K_Index", "LI", "TT", "SWEAT"]:
        df[f"score_{index_name}"] = df.apply(
            lambda row: ordinal_score_single_value(index_name, row[index_name], row["month"]), axis=1
        )

    df["total_score"] = df[["score_CAPE", "score_K_Index", "score_LI", "score_TT", "score_SWEAT"]].sum(
        axis=1, skipna=False
    )
    columns = ["date", "month", "score_CAPE", "score_K_Index", "score_LI", "score_TT",
             "score_SWEAT", "total_score"]
    return df[columns]


# ============================================================
# 5b. ROLLING WINDOW STATISTICS (window 3 hari kalender)
# Dihitung pada deret HARIAN PENUH (reindex tiap tanggal kalender) agar window
# 3 hari benar-benar 3 hari BERURUTAN, bukan 3 rows yang kebetulan berdekatan
# setelah ada tanggal yang bolong (hari tanpa sounding).
# ============================================================

def calculate_rolling_window(df_indices_clean, window=3):
    df = df_indices_clean.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # reindex ke kalender harian penuh -> tanggal yang bolong otomatis jadi NaN
    full_calendar = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df_full = df.reindex(full_calendar)

    hasil = pd.DataFrame(index=df_full.index)
    hasil["CAPE_mean_3d"] = df_full["CAPE"].rolling(window, min_periods=window).mean()
    hasil["CAPE_max_3d"] = df_full["CAPE"].rolling(window, min_periods=window).max()
    hasil["PW_mean_3d"] = df_full["PW"].rolling(window, min_periods=window).mean()
    hasil["LI_min_3d"] = df_full["LI"].rolling(window, min_periods=window).min()

    hasil = hasil.reset_index().rename(columns={"index": "date"})
    # hanya kembalikan rows yang memang ada di indeks_bersih (bukan tanggal isian kalender)
    hasil = hasil[hasil["date"].isin(df.index)].reset_index(drop=True)
    return hasil


if __name__ == "__main__":
    df_indices_clean = pd.read_csv(f"{OUTPUT_DIR}/indices_clean.csv", parse_dates=["date"])

    df_ordinal = calculate_ordinal_scores(df_indices_clean)
    df_rolling = calculate_rolling_window(df_indices_clean)

    df_ordinal.to_csv(f"{OUTPUT_DIR}/ordinal_features.csv", index=False)
    df_rolling.to_csv(f"{OUTPUT_DIR}/rolling_features.csv", index=False)

    print(f"ordinal_features.csv    : {len(df_ordinal)} rows")
    print(f"rolling_features.csv    : {len(df_rolling)} rows "
          f"({df_rolling['CAPE_mean_3d'].isna().sum()} NaN karena window belum penuh 3 hari)")

    print("\nContoh fitur_ordinal:")
    print(df_ordinal.head())
    print("\nContoh fitur_rolling:")
    print(df_rolling.head())
