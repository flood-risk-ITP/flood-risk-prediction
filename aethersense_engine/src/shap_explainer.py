"""
SHAP untuk voting ensemble (Bagian 5 spesifikasi): hitung SHAP terpisah untuk
RF dan XGBoost, lalu rata-ratakan nilai SHAP kelas yang diprediksi - valid
karena voting ensemble kita adalah rata-rata linear (P_RF + P_XGB)/2 dan SHAP
memenuhi aksioma efisiensi.
Subpackage: aethersense_engine
"""

import numpy as np
import shap

from aethersense_engine.config.settings import FEATURE_COLUMNS


def _extract_class_shap(shap_values, class_index: int):
    """Ambil array SHAP (n_fitur,) untuk satu kelas, dari baris pertama (satu sampel)."""
    arr = np.asarray(shap_values, dtype=object) if isinstance(shap_values, list) else shap_values

    if isinstance(shap_values, list):
        # list sepanjang n_kelas, masing-masing (n_sampel, n_fitur)
        per_class = np.asarray(shap_values[class_index])
        return per_class[0]

    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        # (n_sampel, n_fitur, n_kelas)
        return arr[0, :, class_index]
    if arr.ndim == 2:
        # sudah single-class / single-sample bentuk (n_sampel, n_fitur)
        return arr[0]
    raise ValueError(f"Bentuk shap_values tidak dikenali: {arr.shape}")


def _extract_class_base_value(expected_value, class_index: int):
    if hasattr(expected_value, "__len__"):
        return float(np.asarray(expected_value)[class_index])
    return float(expected_value)


def explain_voting(rf_model, xgb_model, X_row, class_index: int) -> dict:
    """Hitung SHAP voting (rata-rata RF & XGBoost) untuk kelas yang diprediksi.

    class_index: index 0-based kelas (predicted_class_index - 1).

    Returns: {
        "feature_names": [...16 nama fitur...],
        "shap_values": np.ndarray (16,),
        "base_value": float,
        "feature_values": np.ndarray (16,),
    }
    """
    explainer_rf = shap.TreeExplainer(rf_model)
    explainer_xgb = shap.TreeExplainer(xgb_model)

    shap_rf = _extract_class_shap(explainer_rf.shap_values(X_row), class_index)
    shap_xgb = _extract_class_shap(explainer_xgb.shap_values(X_row), class_index)
    shap_voting = (np.asarray(shap_rf, dtype=float) + np.asarray(shap_xgb, dtype=float)) / 2

    base_rf = _extract_class_base_value(explainer_rf.expected_value, class_index)
    base_xgb = _extract_class_base_value(explainer_xgb.expected_value, class_index)
    base_voting = (base_rf + base_xgb) / 2

    return {
        "feature_names": list(FEATURE_COLUMNS),
        "shap_values": shap_voting,
        "base_value": base_voting,
        "feature_values": X_row.iloc[0].to_numpy(dtype=float),
    }
