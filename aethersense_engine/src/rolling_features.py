"""
PW_mean_3d, LI_min_3d untuk satu tanggal target - butuh CAPE/PW/LI dari D-1
dan D-2 juga.
Subpackage: aethersense_engine
"""

from datetime import timedelta

import pandas as pd

from aethersense_engine.src.atmospheric_indices import calculate_daily_indices
from aethersense_engine.src.ordinal_scoring import calculate_rolling_window
from aethersense_engine.src.wyoming import WyomingFetchError, fetch_wyoming_sounding

_OK_STATUSES = {"OK", "DICURIGAI"}


class RollingWindowError(Exception):
    """Window rolling 3 hari tidak lengkap/gagal dihitung."""


def _day_indices(target_date):
    try:
        profile = fetch_wyoming_sounding(target_date)
    except WyomingFetchError as exc:
        raise RollingWindowError(
            f"Data sounding {target_date.isoformat()} tidak tersedia: {exc}"
        ) from exc

    indices = calculate_daily_indices(profile)
    if indices["status"] not in _OK_STATUSES:
        raise RollingWindowError(
            f"Indeks atmosfer gagal dihitung untuk {target_date.isoformat()} "
            f"(status: {indices['status']})."
        )
    return indices


def compute_rolling_features(target_date, cape_today=None, pw_today=None, li_today=None) -> dict:
    """Hitung PW_mean_3d & LI_min_3d untuk target_date."""
    if pw_today is None or li_today is None or cape_today is None:
        idx_today = _day_indices(target_date)
        cape_today = idx_today["CAPE"]
        pw_today = idx_today["PW"]
        li_today = idx_today["LI"]

    day_m1 = target_date - timedelta(days=1)
    day_m2 = target_date - timedelta(days=2)
    idx_m1 = _day_indices(day_m1)
    idx_m2 = _day_indices(day_m2)

    df_window = pd.DataFrame({
        "date": [day_m2, day_m1, target_date],
        "CAPE": [idx_m2["CAPE"], idx_m1["CAPE"], cape_today],
        "PW": [idx_m2["PW"], idx_m1["PW"], pw_today],
        "LI": [idx_m2["LI"], idx_m1["LI"], li_today],
    })
    df_rolling = calculate_rolling_window(df_window, window=3)
    last = df_rolling.iloc[-1]

    if pd.isna(last["PW_mean_3d"]) or pd.isna(last["LI_min_3d"]):
        raise RollingWindowError(
            f"Window rolling 3 hari tidak lengkap untuk {target_date.isoformat()} "
            f"(kemungkinan tanggal tidak berurutan)."
        )

    return {"PW_mean_3d": float(last["PW_mean_3d"]), "LI_min_3d": float(last["LI_min_3d"])}
