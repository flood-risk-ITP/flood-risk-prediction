# ============================================================
# STEP 9 - MODEL TRAINING
# Random Forest + XGBoost, GridSearchCV dengan k-fold berbasis blok tahun,
# digabung melalui Voting Ensemble (soft voting)
# ============================================================
#
# INPUT (hasil Tahap 8, dan train_dataset.csv dari Tahap 7 untuk info tanggal):
#   ../output/X_train.csv, y_train.csv, sample_weight_train.csv
#   ../output/X_val.csv, y_val.csv
#   ../output/X_test.csv, y_test.csv
#   ../output/train_dataset.csv   (untuk columns tanggal -> dibuat blok CV per tahun)
#
# OUTPUT:
#   ../output/rf_model.joblib
#   ../output/xgb_model.joblib
#   ../output/evaluation_results.csv

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib

OUTPUT_DIR = "../output"

# ============================================================
# 1. MUAT DATA
# ============================================================

def load_data():
    X_train = pd.read_csv(f"{OUTPUT_DIR}/X_train.csv")
    y_train = pd.read_csv(f"{OUTPUT_DIR}/y_train.csv").iloc[:, 0]
    sample_weight_train = pd.read_csv(f"{OUTPUT_DIR}/sample_weight_train.csv").iloc[:, 0]

    X_val = pd.read_csv(f"{OUTPUT_DIR}/X_val.csv")
    y_val = pd.read_csv(f"{OUTPUT_DIR}/y_val.csv").iloc[:, 0]

    X_test = pd.read_csv(f"{OUTPUT_DIR}/X_test.csv")
    y_test = pd.read_csv(f"{OUTPUT_DIR}/y_test.csv").iloc[:, 0]

    # tanggal dipakai HANYA untuk menyusun blok CV per tahun, tidak ikut jadi fitur
    df_train_tanggal = pd.read_csv(f"{OUTPUT_DIR}/train_dataset.csv", parse_dates=["date"])
    train_years = df_train_tanggal["date"].dt.year.values

    assert len(train_years) == len(X_train), (
        "Jumlah rows train_dataset.csv dan X_train.csv tidak sama - "
        "urutan rows harus identik untuk penyusunan CV per tahun."
    )

    return (X_train, y_train, sample_weight_train, train_years,
            X_val, y_val, X_test, y_test)


# ============================================================
# 2. SUSUN CV BERBASIS BLOK TAHUN (leave-one-year-out di dalam Latih)
# ============================================================

def build_yearly_cv(year_array):
    """Kembalikan list (idx_train, idx_val) - satu tahun jadi validasi,
    sisanya jadi latih, bergantian untuk tiap tahun di periode Latih.
    Sesuai Subbab 3.7.3 proposal: k-fold berbasis blok waktu, satu per tahun."""
    unique_years = sorted(set(year_array))
    folds = []
    for val_year in unique_years:
        idx_val = np.where(year_array == val_year)[0]
        idx_train = np.where(year_array != val_year)[0]
        folds.append((idx_train, idx_val))
    print(f"CV per tahun disusun: {len(folds)} fold ({unique_years})")
    return folds, unique_years


def show_yearly_scores(grid, year_order):
    """Break down F1-macro CV score PER YEAR (not just the average) for the
    BEST hyperparameter configuration - to see whether any particular year
    is much worse (indication of temporal instability, not model
    complexity overfitting)."""
    best_idx = grid.best_index_
    results = grid.cv_results_
    print("\n  F1-macro CV score per year (best configuration):")
    for i, year in enumerate(year_order):
        score = results[f"split{i}_test_score"][best_idx]
        print(f"    Year {year} as validation: F1-macro = {score:.4f}")


# ============================================================
# 3. GRID HIPERPARAMETER
# Grid awal (n_estimators, max_depth, min_samples_split / learning_rate,
# subsample, colsample_bytree) mengikuti Tabel 3.6 proposal.
#
# PERLUASAN (setelah ditemukan overfitting - recall sempurna di data Latih
# tapi F1-macro jatuh di Validasi/Uji): ditambahkan min_samples_leaf (RF)
# dan min_child_weight (XGBoost) - keduanya parameter regularisasi yang
# secara langsung mengontrol seberapa "spesifik" sebuah leaf/daun boleh
# terbentuk, sehingga model tidak menghafal rows individual dari kelas
# minoritas kecil (Sangat Lebat, dll).
#
# Dasar penambahan:
# - min_samples_leaf: Probst, P., Boulesteix, A.-L., & Bischl, B. (2019).
#   Hyperparameters and Tuning Strategies for Random Forest. WIREs Data
#   Mining and Knowledge Discovery - mengulas "minimum number of
#   observations in a node" sebagai hiperparameter kunci RF.
# - min_child_weight: Dokumentasi resmi XGBoost - "the larger, the more
#   conservative the algorithm will be" (analog min_samples_leaf untuk
#   gradient boosting).
# - Keterkaitan learning_rate <-> n_estimators: Probst, P., Boulesteix,
#   A.-L., & Bischl, B. (2019). Tunability: Importance of Hyperparameters
#   of Machine Learning Algorithms. JMLR, 20(53), 1-32.
#
# max_depth=None (RF) dropped dari grid - itu penyebab langsung overfitting
# pada percobaan before_countnya (pohon tumbuh tanpa batas kedalaman).
# subsample & colsample_bytree (XGBoost) dikecilkan jadi 2 nilai (perannya
# tumpang tindih - keduanya soal subsampling acak) supaya total kombinasi
# tetap terkendali setelah min_child_weight ditambahkan.
# ============================================================

GRID_RF = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10, 20],
}

GRID_XGB = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 5, 10],
}


# ============================================================
# 4. LATIH RANDOM FOREST
# ============================================================

def train_random_forest(X_train, y_train, cv_folds, year_order):
    rf_dasar = RandomForestClassifier(class_weight="balanced", random_state=42)
    grid = GridSearchCV(
        rf_dasar, GRID_RF, cv=cv_folds, scoring="f1_macro",
        n_jobs=-1, refit=True, verbose=1,
    )
    grid.fit(X_train, y_train)
    print(f"\n[Random Forest] Best hyperparameters: {grid.best_params_}")
    print(f"[Random Forest] Best average F1-macro CV: {grid.best_score_:.4f}")
    show_yearly_scores(grid, year_order)
    return grid.best_estimator_, grid.best_params_


# ============================================================
# 5. LATIH XGBOOST
# ============================================================

def train_xgboost(X_train, y_train, sample_weight_train, cv_folds, year_order):
    # XGBoost butuh label mulai dari 0, bukan 1 - konversi sementara
    y_train_xgb = y_train - 1

    # n_jobs=1 (bukan XGBClassifier default 0/all-core) - mencegah XGBoost
    # memakai banyak thread internal DI DALAM tiap proses paralel GridSearchCV,
    # yang di Windows sering menyebabkan crash ("access violation"/MemoryError)
    # karena terlalu banyak proses berebut memori/CPU sekaligus.
    xgb_dasar = xgb.XGBClassifier(
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", random_state=42, n_jobs=1,
    )
    # GridSearchCV juga TIDAK dijalankan paralel (n_jobs=1) untuk XGBoost -
    # lebih lambat dari Random Forest, tapi jauh lebih stabil di Windows.
    # (Random Forest tetap n_jobs=-1 di fungsi train_random_forest karena
    # terbukti tidak bermasalah.)
    grid = GridSearchCV(
        xgb_dasar, GRID_XGB, cv=cv_folds, scoring="f1_macro",
        n_jobs=1, refit=True, verbose=1,
    )
    # sample_weight diteruskan lewat fit_params supaya dipakai tiap fold
    grid.fit(X_train, y_train_xgb, sample_weight=sample_weight_train)
    print(f"\n[XGBoost] Best hyperparameters: {grid.best_params_}")
    print(f"[XGBoost] Best average F1-macro CV: {grid.best_score_:.4f}")
    show_yearly_scores(grid, year_order)
    return grid.best_estimator_, grid.best_params_


# ============================================================
# 6. VOTING ENSEMBLE (soft voting manual - P_RF + P_XGB) / 2
# ============================================================

def voting_predict(rf_model, xgb_model, X):
    """Kembalikan (label_prediksi, probabilitas gabungan) - generik untuk
    berapapun jumlah kelas (mengikuti label asli, mis. 1-3 setelah kelas
    Lebat+Sangat Lebat digabung)."""
    proba_rf = rf_model.predict_proba(X)          # urutan sesuai rf_model.classes_
    proba_xgb = xgb_model.predict_proba(X)          # urutan 0..n-1 mewakili label asli 1..n

    # samakan urutan columns kelas RF dengan urutan XGB
    urutan_rf = rf_model.classes_
    proba_rf_terurut = np.zeros_like(proba_xgb)
    for i, kelas in enumerate(urutan_rf):
        proba_rf_terurut[:, int(kelas) - 1] = proba_rf[:, i]

    proba_voting = (proba_rf_terurut + proba_xgb) / 2
    label_prediksi = np.argmax(proba_voting, axis=1) + 1   # balik ke label asli (mulai dari 1)
    return label_prediksi, proba_voting


# ============================================================
# 7. EVALUASI
# ============================================================

def evaluate(subset_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n=== Evaluation on subset {subset_name} ===")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1-macro  : {f1_macro:.4f}")
    print("\nConfusion Matrix (rows=actual, columns=predicted, class order 1-4):")
    print(confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4]))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Ringan", "Sedang", "Lebat", "Sangat Lebat"]))
    return {"subset": subset_name, "accuracy": accuracy, "f1_macro": f1_macro}


# ============================================================
# EKSEKUSI
# ============================================================

if __name__ == "__main__":
    (X_train, y_train, sample_weight_train, train_years,
     X_val, y_val, X_test, y_test) = load_data()

    cv_folds, year_order = build_yearly_cv(train_years)

    print("\n########## TRAINING RANDOM FOREST ##########")
    rf_model, rf_params = train_random_forest(X_train, y_train, cv_folds, year_order)

    print("\n########## TRAINING XGBOOST ##########")
    xgb_model, xgb_params = train_xgboost(X_train, y_train, sample_weight_train, cv_folds, year_order)

    print("\n########## VOTING ENSEMBLE EVALUATION ##########")
    all_results = []

    pred_train, _ = voting_predict(rf_model, xgb_model, X_train)
    all_results.append(evaluate("Train", y_train, pred_train))

    pred_val, _ = voting_predict(rf_model, xgb_model, X_val)
    all_results.append(evaluate("Validation", y_val, pred_val))

    pred_test, _ = voting_predict(rf_model, xgb_model, X_test)
    all_results.append(evaluate("Test", y_test, pred_test))

    # ---- simpan model & ringkasan ----
    joblib.dump(rf_model, f"{OUTPUT_DIR}/rf_model.joblib")
    joblib.dump(xgb_model, f"{OUTPUT_DIR}/xgb_model.joblib")
    pd.DataFrame(all_results).to_csv(f"{OUTPUT_DIR}/evaluation_results.csv", index=False)

    print(f"\nSaved: {OUTPUT_DIR}/rf_model.joblib")
    print(f"Saved: {OUTPUT_DIR}/xgb_model.joblib")
    print(f"Saved: {OUTPUT_DIR}/evaluation_results.csv")

    print(f"\nHiperparameter final RF : {rf_params}")
    print(f"Hiperparameter final XGB: {xgb_params}")
