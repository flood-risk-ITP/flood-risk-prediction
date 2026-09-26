"""
Load model & scaler, jalankan model.predict(), argmax, class mapping.

WAJIB `compile=False` (INFERENCE_CONTRACT.md Bagian 5): model tersimpan
dengan loss custom (`loss_fn`) yang tidak terdaftar sebagai custom_object,
`load_model(..., compile=True)` akan gagal dengan TypeError. Inference
hanya butuh `model.predict()`, sehingga compile/optimizer/loss tidak
relevan.

Dilarang (INFERENCE_CONTRACT.md Bagian 9): model.fit(...).
"""

from __future__ import annotations

import joblib
import numpy as np

from config.settings import CLASS_NAMES, MODEL_PATH, SCALER_PATH, LOOKBACK, FEATURE_COLUMNS


# Memuat model LSTM Keras yang telah dilatih. Menggunakan compile=False karena model hanya digunakan untuk inference, bukan training.
def load_model(path: str = MODEL_PATH):
    """keras.models.load_model(path, compile=False) -- WAJIB compile=False."""
    import keras

    return keras.models.load_model(path, compile=False)


# Memuat objek scaler (MinMaxScaler) yang disimpan saat training untuk menjaga konsistensi skala fitur.
def load_scaler(path: str = SCALER_PATH):
    return joblib.load(path)


# Menjalankan inference LSTM: menerima sequence (1, 7, 8), menghitung probabilitas 4 kelas, dan mengambil kelas dengan probabilitas tertinggi.
def predict(model, X_input: np.ndarray) -> dict:
    """Jalankan model.predict() pada X_input shape (1, 7, 8), argmax,
    dan class mapping. TIDAK memanggil model.fit() atau melatih ulang
    apa pun."""
    # Memastikan dimensi input sesuai dengan spesifikasi model LSTM (1 batch, 7 hari lookback, 8 fitur).
    expected_shape = (1, LOOKBACK, len(FEATURE_COLUMNS))
    if X_input.shape != expected_shape:
        raise ValueError(f"Shape input harus {expected_shape}, diterima {X_input.shape}")

    # Menjalankan prediksi probabilitas untuk 4 kategori risiko dari model LSTM.
    proba = model.predict(X_input, verbose=0)
    
    # Mengambil indeks kategori risiko dengan nilai probabilitas tertinggi (argmax).
    pred_class = int(np.argmax(proba, axis=1)[0])

    # Memetakan indeks hasil prediksi ke nama kategori risiko ("Rendah", "Sedang", "Tinggi", "Sangat Tinggi").
    return {
        "predicted_class_index": pred_class,
        "predicted_class_name": CLASS_NAMES[pred_class],
        "probabilities": {
            CLASS_NAMES[i]: float(proba[0][i]) for i in range(len(CLASS_NAMES))
        },
    }
