"""
Muat rf_model.joblib + xgb_model.joblib dan jalankan voting_predict() dari
canvas/step9_model_training.py LANGSUNG (bukan ditulis ulang), supaya
probabilitas voting (P_RF + P_XGB)/2 dijamin identik dengan definisi yang
dipakai saat evaluasi model.
Subpackage: aethersense_engine
"""

import joblib

from aethersense_engine.canvas.step9_model_training import voting_predict  # noqa: F401
from aethersense_engine.config.settings import CLASS_NAMES, RF_MODEL_PATH, XGB_MODEL_PATH


def load_rf_model():
    return joblib.load(RF_MODEL_PATH)


def load_xgb_model():
    return joblib.load(XGB_MODEL_PATH)


def predict(rf_model, xgb_model, X_row):
    """Jalankan voting ensemble untuk satu baris fitur.

    Returns: dict {predicted_class_index (1-4), predicted_class_name,
                   probabilities: {nama_kelas: proba}}
    """
    label_pred, proba_voting = voting_predict(rf_model, xgb_model, X_row)
    class_index = int(label_pred[0])
    probs = {CLASS_NAMES[i + 1]: float(proba_voting[0][i]) for i in range(len(CLASS_NAMES))}
    return {
        "predicted_class_index": class_index,
        "predicted_class_name": CLASS_NAMES[class_index],
        "probabilities": probs,
    }
