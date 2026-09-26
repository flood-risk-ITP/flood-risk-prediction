"""
predict_for_date() - orkestrasi penuh Bagian 3-6 spesifikasi:
validation gate tanggal (Bagian 6) -> ambil & susun 16 fitur (Bagian 3) ->
voting ensemble (Bagian 4) -> SHAP (Bagian 5).
Subpackage: aethersense_engine
"""

import time
from datetime import date as date_cls
from datetime import datetime, timedelta, timezone

from aethersense_engine.config.settings import (
    MIN_SUPPORTED_DATE,
    SOUNDING_AVAILABLE_HOUR_WIB,
    WIB_UTC_OFFSET_HOURS,
)
from aethersense_engine.src.integration import build_feature_vector
from aethersense_engine.src.ogimet import OgimetFetchError
from aethersense_engine.src.predictor import load_rf_model, load_xgb_model, predict
from aethersense_engine.src.rolling_features import RollingWindowError
from aethersense_engine.src.shap_explainer import explain_voting
from aethersense_engine.src.wyoming import WyomingFetchError

WIB = timezone(timedelta(hours=WIB_UTC_OFFSET_HOURS))

PIPELINE_STAGES = [
    "Mengambil data cuaca permukaan (Ogimet)",
    "Mengambil profil sounding atmosfer (Wyoming)",
    "Menghitung indeks atmosfer & rolling window 3 hari",
    "Menyusun vektor fitur (skor ordinal + 16 fitur final)",
    "Menjalankan model prediksi (voting ensemble RF + XGBoost)",
    "Menghitung kontribusi fitur (SHAP)",
]


def _parse_date(date_input):
    if isinstance(date_input, date_cls):
        return date_input
    return datetime.strptime(date_input, "%Y-%m-%d").date()


def _validate_date(target_date: date_cls):
    """Bagian 6: validasi tanggal. Kembalikan list alasan penolakan (kosong = valid)."""
    min_date = datetime.strptime(MIN_SUPPORTED_DATE, "%Y-%m-%d").date()
    now_wib = datetime.now(WIB)
    today_wib = now_wib.date()

    if target_date < min_date:
        return [f"Tanggal di luar rentang yang didukung (data historis dimulai {MIN_SUPPORTED_DATE})."]

    if target_date > today_wib:
        return ["Tanggal di luar rentang yang didukung (tanggal berada di masa depan)."]

    if target_date == today_wib and now_wib.hour < SOUNDING_AVAILABLE_HOUR_WIB:
        return [
            "Tanggal di luar rentang yang didukung - data sounding 12Z untuk hari ini "
            f"baru tersedia setelah pukul {SOUNDING_AVAILABLE_HOUR_WIB}:00 WIB."
        ]

    return []


def _log(verbose, stage_idx, message, status=None, elapsed=None):
    if not verbose:
        return
    tag = f"[{stage_idx}/{len(PIPELINE_STAGES)}]"
    if status is None:
        print(f"{tag} {message} ...")
    else:
        suffix = f" ({elapsed:.2f}s)" if elapsed is not None else ""
        print(f"{tag} {message} {status}{suffix}")


def predict_for_date(target_date, rf_model=None, xgb_model=None, verbose=False):
    """Jalankan pipeline penuh untuk satu tanggal.

    Returns dict:
      SUCCESS: {"status", "target_date", "predicted_class_index",
                "predicted_class_name", "probabilities", "raw_values",
                "shap_values"}
      REJECTED: {"status", "target_date", "reasons"}
    """
    date_str = target_date if isinstance(target_date, str) else target_date.isoformat()
    parsed = _parse_date(target_date)

    reject_reasons = _validate_date(parsed)
    if reject_reasons:
        return {"status": "REJECTED", "target_date": date_str, "reasons": reject_reasons}

    try:
        t0 = time.time()
        _log(verbose, 1, PIPELINE_STAGES[0])
        X_row, raw_values = build_feature_vector(parsed)
        _log(verbose, 1, PIPELINE_STAGES[0], "OK", time.time() - t0)

        t1 = time.time()
        _log(verbose, 2, PIPELINE_STAGES[1])
        _log(verbose, 2, PIPELINE_STAGES[1], "OK", time.time() - t1)

        t2 = time.time()
        _log(verbose, 3, PIPELINE_STAGES[2])
        _log(verbose, 3, PIPELINE_STAGES[2], "OK", time.time() - t2)

        t3 = time.time()
        _log(verbose, 4, PIPELINE_STAGES[3])
        _log(verbose, 4, PIPELINE_STAGES[3], "OK", time.time() - t3)
    except (OgimetFetchError, WyomingFetchError, RollingWindowError) as exc:
        _log(verbose, 1, PIPELINE_STAGES[0], "FAILED", 0.0)
        return {
            "status": "REJECTED",
            "target_date": date_str,
            "reasons": [f"Data tidak tersedia untuk tanggal ini: {exc}"],
        }

    t4 = time.time()
    _log(verbose, 5, PIPELINE_STAGES[4])
    if rf_model is None:
        rf_model = load_rf_model()
    if xgb_model is None:
        xgb_model = load_xgb_model()
    pred = predict(rf_model, xgb_model, X_row)
    _log(verbose, 5, PIPELINE_STAGES[4], "OK", time.time() - t4)

    t5 = time.time()
    _log(verbose, 6, PIPELINE_STAGES[5])
    shap_result = explain_voting(rf_model, xgb_model, X_row, pred["predicted_class_index"] - 1)
    _log(verbose, 6, PIPELINE_STAGES[5], "OK", time.time() - t5)

    return {
        "status": "SUCCESS",
        "target_date": date_str,
        "predicted_class_index": pred["predicted_class_index"],
        "predicted_class_name": pred["predicted_class_name"],
        "probabilities": pred["probabilities"],
        "raw_values": raw_values,
        "shap_values": {
            "feature_names": shap_result["feature_names"],
            "values": shap_result["shap_values"].tolist(),
            "base_value": shap_result["base_value"],
            "feature_values": shap_result["feature_values"].tolist(),
        },
    }
