# ============================================================
# STEP 8 - FEATURE FILTERING & PENANGANAN AKHIR
# Menghasilkan X, y, sample_weight per subset (Latih/Validasi/Uji)
# Masih tahap treatment data - BELUM pelatihan model.
# ============================================================
#
# INPUT (hasil Tahap 7):
#   /content/output/train_dataset.csv
#   /content/output/val_dataset.csv
#   /content/output/test_dataset.csv
#
# OUTPUT (per subset):
#   /content/output/X_<subset>.csv
#   /content/output/y_<subset>.csv
#   /content/output/sample_weight_<subset>.csv   (khusus subset Latih)

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

OUTPUT_DIR = "../output"

# 18 columns fitur final (lihat Tahap 5 markdown)
# 16 columns fitur final (CAPE_mean_3d dan CAPE_max_3d DIKELUARKAN - bukti
# permutation importance menunjukkan keduanya negatif di RF & XGBoost,
# artinya menambah noise, bukan sinyal, untuk model ini).
FEATURE_COLUMNS_CORE = [
    "CAPE", "CIN", "LI", "K_Index", "TT", "SWEAT", "PW",       # indeks mentah (7)
    "T_avg", "RH",                                          # permukaan (2)
    "score_CAPE", "score_K_Index", "score_LI", "score_TT", "score_SWEAT",  # ordinal (5)
    "PW_mean_3d", "LI_min_3d",   # rolling window (2)
]

# EXPERIMENTAL: fitur harian dari data 3-jaman (step1b_hourly_features.py).
# Belum diuji permutation importance - jangan diaktifkan begitu saja tanpa
# nanti mengecek apakah benar-benar membantu (sama seperti CAPE_mean_3d/
# CAPE_max_3d yang dulu dibuang setelah terbukti negatif).
FEATURE_COLUMNS_HOURLY_EXPERIMENTAL = [
    "temp_range", "peak_temp_hour", "wind_max", "pressure_tendency_max",
]

# Ganti ke True SETELAH menjalankan step1b_hourly_features.py dan step6
# (dengan df_hourly terisi) - kalau kolomnya belum ada, split_X_y akan
# error dengan pesan jelas, bukan diam-diam salah.
USE_HOURLY_FEATURES = False

FEATURE_COLUMNS = FEATURE_COLUMNS_CORE + (
    FEATURE_COLUMNS_HOURLY_EXPERIMENTAL if USE_HOURLY_FEATURES else []
)

NON_FEATURE_COLUMNS = ["RR", "label", "label_name", "date", "month", "total_score"]


def split_X_y(df):
    """8a: penyaringan columns -> X (fitur) dan y (label)."""
    missing = [k for k in FEATURE_COLUMNS if k not in df.columns]
    if missing:
        raise ValueError(
            f"Kolom fitur berikut tidak ditemukan di dataset: {missing}\n"
            f"(Kalau ini kolom fitur harian eksperimental, pastikan sudah "
            f"jalankan step1b_hourly_features.py dan step6_integration.py "
            f"ulang sebelum mengaktifkan USE_HOURLY_FEATURES=True)"
        )

    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].astype(int).copy()
    return X, y


def calculate_sample_weight(y_train):
    """8e: bobot per-sampel untuk XGBoost (compute_sample_weight)."""
    return compute_sample_weight(class_weight="balanced", y=y_train)


def summarize_class_weight(y_train):
    """Tampilkan bobot per kelas (dipakai langsung oleh class_weight='balanced' di RF)."""
    n_total = len(y_train)
    ringkasan = []
    for kelas in sorted(y_train.unique()):
        n_kelas = (y_train == kelas).sum()
        bobot = n_total / (len(y_train.unique()) * n_kelas)
        ringkasan.append({"kelas": kelas, "n_sampel": n_kelas, "bobot_class_weight": round(bobot, 4)})
    return pd.DataFrame(ringkasan)


if __name__ == "__main__":
    # maps: internal subset column value (from step7) -> (dataset file prefix, output suffix)
    subset_mapping = [
        ("Latih", "train", "train"),
        ("Validasi", "val", "val"),
        ("Uji", "test", "test"),
    ]

    for internal_name, file_prefix, out_suffix in subset_mapping:
        path = f"{OUTPUT_DIR}/{file_prefix}_dataset.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Skipping {out_suffix}: file {path} not found.")
            continue

        if len(df) == 0:
            print(f"Skipping {out_suffix}: dataset is empty.")
            continue

        X, y = split_X_y(df)
        print(f"\n=== Subset: {out_suffix.upper()} ===")
        print(f"X shape: {X.shape}  |  y shape: {y.shape}")
        print(f"Class distribution:\n{y.value_counts().sort_index()}")

        X.to_csv(f"{OUTPUT_DIR}/X_{out_suffix}.csv", index=False)
        y.to_csv(f"{OUTPUT_DIR}/y_{out_suffix}.csv", index=False)
        print(f"Saved: X_{out_suffix}.csv, y_{out_suffix}.csv")

        if out_suffix == "train":
            sample_weight = calculate_sample_weight(y)
            pd.Series(sample_weight, name="sample_weight").to_csv(
                f"{OUTPUT_DIR}/sample_weight_{out_suffix}.csv", index=False
            )
            print(f"Saved: sample_weight_{out_suffix}.csv")

            print("\nclass_weight='balanced' weight summary (for Random Forest):")
            print(summarize_class_weight(y).to_string(index=False))

    print("\n=== STEP 8 COMPLETE ===")
    print("X_train.csv / y_train.csv / sample_weight_train.csv are ready for the model training step")
    print("(Random Forest + XGBoost, voting ensemble - outside the scope of this script).")
