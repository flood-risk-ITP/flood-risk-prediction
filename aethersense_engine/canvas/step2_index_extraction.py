# ============================================================
# STEP 2 - ATMOSPHERIC INDEX EXTRACTION
# CAPE, CIN, K-Index, LI, TT, SWEAT, PW dari profil sounding (MetPy)
# Jam sounding mengikuti SOUNDING_HOUR yang diset di Tahap 1
# ============================================================
#
# INPUT (hasil Tahap 1):
#   /content/output/wyoming_sounding_raw.pkl
#
# OUTPUT:
#   /content/output/atmospheric_indices.csv
#   (tanggal, CAPE, CIN, K_Index, LI, TT, SWEAT, PW, level_count, status)

import pickle
import warnings
import numpy as np
import pandas as pd
import metpy.calc as mpcalc
from metpy.units import units

OUTPUT_DIR = "../output"

MIN_LEVEL = 10  # sesuai Tahap 3 markdown: level sounding < 10 -> status GAGAL


def calculate_sweat(td850, tt, dd850, ff850, dd500, ff500):
    """SWEAT manual sesuai Formula 2.7 proposal (Miller, 1972)."""
    nilai = [td850, tt, dd850, ff850, dd500, ff500]
    if any(pd.isna(v) for v in nilai):
        return np.nan
    term1 = 12 * td850 if td850 > 0 else 0
    term2 = 20 * (tt - 49) if tt > 49 else 0
    term3 = 2 * ff850
    term4 = ff500
    shear_ok = (
        130 <= dd850 <= 250 and 210 <= dd500 <= 310
        and (dd500 - dd850) > 0 and ff850 >= 15 and ff500 >= 15
    )
    term5 = 125 * (np.sin(np.radians(dd500 - dd850)) + 0.2) if shear_ok else 0
    return term1 + term2 + term3 + term4 + term5


def get_level(prof, target_hpa, toleransi=2.0):
    """Ambil rows level tekanan tertentu (850/500 mb); interpolasi bila level
    persis tidak tersedia."""
    cocok = prof[np.isclose(prof["pres_hpa"], target_hpa, atol=toleransi)]
    if len(cocok) > 0:
        return cocok.iloc[0]
    # interpolasi sederhana bila level mandatory tidak eksis persis
    if prof["pres_hpa"].min() <= target_hpa <= prof["pres_hpa"].max():
        interp = {}
        for col in ["temp_c", "dwpt_c", "drct_deg", "sknt_knot"]:
            interp[col] = np.interp(
                target_hpa, prof["pres_hpa"][::-1], prof[col][::-1]
            )
        return pd.Series(interp)
    return None


def calculate_daily_indices(prof):
    """Hitung 7 indeks untuk satu profil (DataFrame level tekanan).
    Warning MetPy ('Interpolation point out of data bounds') DITANGKAP dan
    dihitung, bukan dibiarkan mencetak berantakan - lihat columns
    'warning_count' dan 'extrapolation' pada hasil untuk QC."""
    prof = prof.dropna(subset=["pres_hpa", "temp_c", "dwpt_c"]).drop_duplicates("pres_hpa")
    prof = prof.sort_values("pres_hpa", ascending=False).reset_index(drop=True)

    level_count = len(prof)
    if level_count < MIN_LEVEL:
        return {
            "CAPE": np.nan, "CIN": np.nan, "K_Index": np.nan, "LI": np.nan,
            "TT": np.nan, "SWEAT": np.nan, "PW": np.nan,
            "level_count": level_count, "status": "GAGAL",
            "warning_count": 0, "extrapolation": False, "cape_corrected": False,
        }

    min_pressure = prof["pres_hpa"].min()
    max_pressure = prof["pres_hpa"].max()

    try:
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.simplefilter("always")

            p = prof["pres_hpa"].values * units.hPa
            T = (prof["temp_c"].values * units.degC).to("kelvin")
            Td = (prof["dwpt_c"].values * units.degC).to("kelvin")

            parcel = mpcalc.parcel_profile(p, T[0], Td[0]).to("kelvin")
            cape, cin = mpcalc.surface_based_cape_cin(p, T, Td)
            k_idx = mpcalc.k_index(p, T, Td)
            li = mpcalc.lifted_index(p, T, parcel)
            tt = mpcalc.total_totals_index(p, T, Td)
            pw = mpcalc.precipitable_water(p, Td).to("mm")

        warning_count = len(warning_list)
        ekstrapolasi = warning_count > 0 or min_pressure > 850 or max_pressure < 500

        # CAPE secara definisi (Formula 2.2) adalah integral pada rentang LFC-EL
        # di mana parsel > lingkungan, sehingga TIDAK PERNAH negatif secara fisis.
        # Nilai negatif yang muncul adalah artefak numerik integrasi trapesium
        # (bukan sinyal meteorologis, beda dari CIN/LI yang memang didesain negatif)
        # -> dikoreksi ke 0, bukan dropped, karena sisa profil tetap valid.
        cape_val = float(cape.magnitude)
        cape_dikoreksi = cape_val < 0
        if cape_dikoreksi:
            cape_val = 0.0

        li_val = float(np.atleast_1d(li.magnitude)[0])
        tt_val = float(tt.magnitude)

        lvl850 = get_level(prof, 850.0)
        lvl500 = get_level(prof, 500.0)
        if lvl850 is not None and lvl500 is not None:
            sweat_val = calculate_sweat(
                td850=lvl850["dwpt_c"], tt=tt_val,
                dd850=lvl850["drct_deg"], ff850=lvl850["sknt_knot"],
                dd500=lvl500["drct_deg"], ff500=lvl500["sknt_knot"],
            )
        else:
            sweat_val = np.nan

        k_val = float(k_idx.magnitude)
        status = "OK"
        if k_val == 0 and tt_val == 0:
            status = "DICURIGAI"

        return {
            "CAPE": cape_val, "CIN": float(cin.magnitude),
            "K_Index": k_val, "LI": li_val, "TT": tt_val,
            "SWEAT": sweat_val, "PW": float(pw.magnitude),
            "level_count": level_count, "status": status,
            "warning_count": warning_count, "extrapolation": ekstrapolasi,
            "cape_corrected": cape_dikoreksi,
        }
    except Exception as e:
        return {
            "CAPE": np.nan, "CIN": np.nan, "K_Index": np.nan, "LI": np.nan,
            "TT": np.nan, "SWEAT": np.nan, "PW": np.nan,
            "level_count": level_count, "status": f"GAGAL_HITUNG: {e}",
            "warning_count": 0, "extrapolation": False, "cape_corrected": False,
        }


def process_all_profiles(profile_dict):
    hasil = []
    for tanggal, prof in sorted(profile_dict.items()):
        indeks = calculate_daily_indices(prof)
        indeks["date"] = tanggal
        hasil.append(indeks)

    columns = ["date", "CAPE", "CIN", "K_Index", "LI", "TT", "SWEAT", "PW",
             "level_count", "status", "warning_count", "extrapolation", "cape_corrected"]
    return pd.DataFrame(hasil)[columns]


if __name__ == "__main__":
    with open(f"{OUTPUT_DIR}/wyoming_sounding_raw.pkl", "rb") as f:
        profile_dict = pickle.load(f)

    df_indices = process_all_profiles(profile_dict)

    print("=== STATUS SUMMARY ===")
    print(df_indices["status"].value_counts())

    n_extrapolation = df_indices["extrapolation"].sum()
    print(f"\n=== EXTRAPOLATION SUMMARY ===")
    print(f"Baris dengan ekstrapolasi (level 850/500 mb di luar jangkauan profil): "
          f"{n_extrapolation} dari {len(df_indices)} "
          f"({100*n_extrapolation/len(df_indices):.1f}%)")
    print(f"Total warning MetPy tertangkap: {df_indices['warning_count'].sum()}")

    n_cape_corrected = df_indices["cape_corrected"].sum()
    print(f"\n=== NEGATIVE CAPE CORRECTION SUMMARY ===")
    print(f"Baris dengan CAPE dikoreksi dari negatif -> 0: {n_cape_corrected} dari {len(df_indices)} "
          f"({100*n_cape_corrected/len(df_indices):.2f}%)")
    print("(CAPE secara definisi tidak pernah negatif - nilai negatif adalah artefak numerik")
    print(" integrasi, dikoreksi ke 0, BUKAN dropped, karena sisa profil tetap valid.)")

    print(f"\nContoh 5 rows:\n{df_indices.head()}")

    out_path = f"{OUTPUT_DIR}/atmospheric_indices.csv"
    df_indices.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(df_indices)} rows)")
