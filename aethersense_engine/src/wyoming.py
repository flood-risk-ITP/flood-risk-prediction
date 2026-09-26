"""
Live fetch profil sounding (12Z) untuk SATU tanggal dari Wyoming Upper Air
Archive (stasiun Padang, WMO 96163).
Subpackage: aethersense_engine
"""

import re
import time
from datetime import date as date_cls

import pandas as pd
import requests

from aethersense_engine.config.settings import SOUNDING_HOUR, STATION_ID

WYOMING_URL = "https://weather.arcc.uwyo.edu/wsgi/sounding"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://weather.arcc.uwyo.edu/",
}

COLUMN_NAMES = [
    "pres_hpa", "hght_m", "temp_c", "dwpt_c", "relh_pct",
    "mixr_gkg", "drct_deg", "sknt_knot", "thta_k", "thte_k", "thtv_k",
]
_COL_WIDTH = 7
_PRE_RE = re.compile(r"<PRE>(.*?)</PRE>", re.S)


class WyomingFetchError(Exception):
    """Profil sounding Wyoming tidak bisa diambil/tidak tersedia untuk tanggal diminta."""


def _fetch_html(target_date: date_cls, hour: int, timeout=20, retries=2):
    params = {
        "datetime": f"{target_date.isoformat()} {hour:02d}:00:00",
        "id": STATION_ID,
        "src": "UNKNOWN",
        "type": "TEXT:LIST",
    }
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(WYOMING_URL, params=params, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 200 and "PRES" in resp.text:
                return resp.text
            last_exc = WyomingFetchError(
                f"HTTP {resp.status_code} dari Wyoming (percobaan {attempt + 1})."
            )
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(2)
    raise WyomingFetchError(
        f"Gagal mengambil sounding Wyoming untuk {target_date.isoformat()} "
        f"{hour:02d}Z: {last_exc}"
    )


def _parse_row(line: str):
    values = []
    for i in range(len(COLUMN_NAMES)):
        start, end = i * _COL_WIDTH, i * _COL_WIDTH + _COL_WIDTH
        chunk = line[start:end].strip() if start < len(line) else ""
        values.append(float(chunk) if chunk else None)
    return values


def _parse_profile(html: str) -> pd.DataFrame:
    match = _PRE_RE.search(html)
    if not match:
        raise WyomingFetchError("Blok data sounding (<PRE>) tidak ditemukan pada halaman Wyoming.")

    rows = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.rstrip("\n").rstrip("\r")
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or "PRES" in line or "hPa" in line:
            continue
        values = _parse_row(line)
        if values[0] is None:  # PRES wajib ada untuk baris level yang valid
            continue
        rows.append(values)

    if not rows:
        raise WyomingFetchError("Tidak ada baris level tekanan yang berhasil di-parse dari profil Wyoming.")

    return pd.DataFrame(rows, columns=COLUMN_NAMES)


def fetch_wyoming_sounding(target_date: date_cls, hour: int = None) -> pd.DataFrame:
    """Ambil profil sounding untuk satu tanggal (live, dari Wyoming).

    Returns: DataFrame kolom [pres_hpa, hght_m, temp_c, dwpt_c, relh_pct,
             mixr_gkg, drct_deg, sknt_knot, thta_k, thte_k, thtv_k]
    Raises: WyomingFetchError kalau data tidak tersedia.
    """
    if hour is None:
        hour = SOUNDING_HOUR
    html = _fetch_html(target_date, hour)
    return _parse_profile(html)
