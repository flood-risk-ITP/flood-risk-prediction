"""
Integrasi Stage 5: LEFT JOIN backbone_calendar -> Ogimet -> SounderPy, on
`date` (key kalender, BUKAN observation_datetime).

Identik urutan join dengan STAGE_DEPLOYMENT_MAP.md baris Stage 5: backbone
(kalender penuh) tidak pernah berkurang jumlah barisnya.
"""

from __future__ import annotations

import pandas as pd

from src.calendar_utils import build_master_calendar


# Menggabungkan data permukaan Ogimet dan data atmosfer Sounding ke dalam backbone kalender harian (LEFT JOIN pada kolom 'date') untuk membentuk dataset harian utuh.
def integrate(
    ogimet_df: pd.DataFrame,
    sounding_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Bangun dataset terintegrasi harian untuk rentang [start_date, end_date].

    backbone (kalender harian) LEFT JOIN ogimet_df (on date)
             LEFT JOIN sounding_df (on date)

    Parameters
    ----------
    ogimet_df : DataFrame dengan kolom date, rr, tavg, rh
    sounding_df : DataFrame dengan kolom date, selection_status,
        selected_hour, cin, kindex, li, tt, sweat
    """
    # Membentuk kerangka kalender harian lengkap (backbone) sebagai acuan tanggal berurutan.
    backbone = pd.DataFrame({
        "date": build_master_calendar(start_date, end_date),
    })

    ogimet_df = ogimet_df.copy()
    ogimet_df["date"] = pd.to_datetime(ogimet_df["date"]).dt.normalize()

    sounding_df = sounding_df.copy()
    sounding_df["date"] = pd.to_datetime(sounding_df["date"]).dt.normalize()

    # Menggabungkan data permukaan dan data atmosfer ke backbone kalender via LEFT JOIN agar tidak ada tanggal yang terlewat.
    merged = backbone.merge(ogimet_df, on="date", how="left")
    merged = merged.merge(sounding_df, on="date", how="left")

    # Memastikan jumlah baris hasil gabungan sama persis dengan jumlah hari pada backbone kalender.
    if len(merged) != len(backbone):
        raise AssertionError(
            "Integrasi Stage 5 menghasilkan jumlah baris berbeda dari backbone "
            f"({len(merged)} != {len(backbone)}) -- kemungkinan duplicate date "
            "pada ogimet_df/sounding_df."
        )

    return merged
