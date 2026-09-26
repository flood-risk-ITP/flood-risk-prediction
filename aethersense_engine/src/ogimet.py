"""
Live fetch RR, T_avg, RH untuk SATU tanggal dari Ogimet (stasiun Padang, WMO 96163).
Subpackage: aethersense_engine
"""

import re
import time
from datetime import date as date_cls

import requests
from bs4 import BeautifulSoup

from aethersense_engine.config.settings import STATION_ID

OGIMET_URL = "http://www.ogimet.com/cgi-bin/gsynres"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "http://www.ogimet.com/",
    "Accept-Language": "en-US,en;q=0.9",
}
_DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
_MISSING_TOKENS = {"", "-----", "----", "---", "--", "-"}


class OgimetFetchError(Exception):
    """Data Ogimet tidak bisa diambil/tidak tersedia untuk tanggal diminta."""


def _to_float(text):
    t = text.strip()
    if t in _MISSING_TOKENS:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _fetch_html(target_date: date_cls, timeout=20, retries=2):
    params = {
        "ind": STATION_ID,
        "ano": target_date.year,
        "mes": target_date.month,
        "day": target_date.day,
        "hora": 21,
        "min": 0,
        "ndays": 2,  # cukup untuk menangkap 8 laporan tanggal target + margin
        "lang": "en",
        "decoded": "yes",
    }
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(OGIMET_URL, params=params, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 200 and len(resp.text) > 500:
                return resp.text
            last_exc = OgimetFetchError(f"HTTP {resp.status_code} dari Ogimet (percobaan {attempt + 1}).")
        except requests.RequestException as exc:
            last_exc = exc
        if attempt < retries:
            time.sleep(2)
    raise OgimetFetchError(f"Gagal menghubungi Ogimet untuk {target_date.isoformat()}: {last_exc}")


def _parse_day_rows(html: str, target_date: date_cls):
    """Kembalikan list dict {time, T, Hr, prec_text} untuk semua laporan
    3-jaman pada tanggal target (waktu UTC)."""
    soup = BeautifulSoup(html, "html.parser")
    caption = next(
        (c for c in soup.find_all("caption") if "Decoded synop data" in c.get_text()),
        None,
    )
    table = caption.find_parent("table") if caption else None
    if table is None:
        raise OgimetFetchError("Format halaman Ogimet tidak dikenali (tabel data tidak ditemukan).")

    target_str = target_date.strftime("%m/%d/%Y")
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 14:
            continue
        date_text = cells[0].get_text(strip=True)
        if not _DATE_RE.match(date_text) or date_text != target_str:
            continue
        rows.append({
            "time": cells[1].get_text(strip=True),
            "T": _to_float(cells[2].get_text()),
            "Hr": _to_float(cells[4].get_text()),
            "prec_text": cells[13].get_text(strip=True),
        })
    return rows


def _parse_precip_mm(prec_text: str):
    """'101.0/24h' -> 101.0 ; 'Tr/24h' -> 0.0 ; '-----' / tidak ada -> None."""
    if not prec_text:
        return None
    head = prec_text.split("/")[0].strip()
    if head.lower() == "tr":
        return 0.0
    if head in _MISSING_TOKENS:
        return None
    try:
        return float(head)
    except ValueError:
        return None


def fetch_ogimet_daily(target_date: date_cls) -> dict:
    """Ambil RR, T_avg, RH untuk satu tanggal (live, dari Ogimet).

    Returns: {"RR": float, "T_avg": float, "RH": float}
    Raises: OgimetFetchError kalau data tidak lengkap/tidak tersedia.
    """
    html = _fetch_html(target_date)
    rows = _parse_day_rows(html, target_date)
    if not rows:
        raise OgimetFetchError(
            f"Tidak ada laporan SYNOP Ogimet untuk tanggal {target_date.isoformat()}."
        )

    t_values = [r["T"] for r in rows if r["T"] is not None]
    hr_values = [r["Hr"] for r in rows if r["Hr"] is not None]
    if not t_values or not hr_values:
        raise OgimetFetchError(
            f"Laporan Ogimet {target_date.isoformat()} ada tapi T/RH kosong semua."
        )
    t_avg = sum(t_values) / len(t_values)
    rh = sum(hr_values) / len(hr_values)

    row_00z = next((r for r in rows if r["time"] == "00:00"), None)
    rr = _parse_precip_mm(row_00z["prec_text"]) if row_00z else None
    if rr is None:
        raise OgimetFetchError(
            f"Grup presipitasi 24 jam (laporan 00 UTC) tidak tersedia untuk "
            f"{target_date.isoformat()} - curah hujan harian tidak bisa dihitung."
        )

    return {"RR": rr, "T_avg": t_avg, "RH": rh}
