# AetherSense — Flood Risk Prediction System

An end-to-end meteorological inference pipeline and interactive web dashboard for real-time flood risk prediction using Deep Learning (LSTM).

---

## 🌟 Overview

**AetherSense** leverages historical surface meteorological observations and upper-air atmospheric soundings to predict daily flood risk categories (**Low**, **Moderate**, **High**, **Very High**). Built upon atmospheric instability indices and daily weather parameters, the system assists in early warning and climate hazard monitoring for the Minangkabau / Padang Meteorological Station (Station ID: `96163`).

---

## ⚡ Key Features

- **Sequential Deep Learning**: Employs an LSTM (Long Short-Term Memory) neural network designed to capture temporal patterns over a 7-day lookback window ($D-7$ to $D-1$).
- **Multi-Source Data Integration**: Seamlessly combines surface weather parameters (Ogimet) with radiosonde upper-air atmospheric stability indices (University of Wyoming / SounderPy / SHARPpy).
- **Automated Preprocessing & Validation**: Features a robust validation gate, time-based interpolation for missing surface metrics, and monthly median fallback for sounding missing values.
- **Interactive Streamlit Web Dashboard**: Built with a clean, modern aesthetic featuring real-time risk estimation, sequence visualizers, and custom date query interfaces.
- **Programmatic Python API**: Modular inference engine designed for easy integration into automated background pipelines or scheduled tasks.

---

## 🔬 Model & Technical Overview

- **Target Output**: 4 Flood Risk Classes (`Low`, `Moderate`, `High`, `Very High`)
- **Model Architecture**: Keras Sequential LSTM Deep Neural Network
- **Lookback Window**: 7 consecutive historical days ($D-7 \dots D-1$) preceding the target evaluation date ($D$)
- **Station Identifier**: BMKG Minangkabau Meteorological Station, Padang (WMO ID: `96163`)

### Input Features (8 Variables)

| Category | Variable | Description | Data Source |
| :--- | :--- | :--- | :--- |
| **Surface Weather** | `rr` | Daily Rainfall / Precipitation (mm) | Ogimet |
| | `tavg` | Average Daily Temperature (°C) | Ogimet |
| | `rh` | Relative Humidity (%) | Ogimet |
| **Upper-Air Atmospheric Instability** | `cin` | Surface-Based Convective Inhibition (SBCIN) | SounderPy / SHARPpy |
| | `kindex` | K-Index (Convective Potential) | SounderPy / SHARPpy |
| | `li` | Lifted Index (`LI_SB_500`) | SounderPy / SHARPpy |
| | `tt` | Total Totals Index | SounderPy / SHARPpy |
| | `sweat` | Severe Weather Threat Index | SounderPy / SHARPpy |

---

## 📁 Project Structure

```
inference_pipeline/
├── app.py                              # Streamlit Web Application & Interactive UI
├── config/
│   ├── settings.py                     # Pipeline configuration & feature schemas
│   └── stage7_monthly_medians.json     # Fallback medians for missing sounding features
├── data/
│   └── integrated/
│       ├── integrated_dataset_example.csv  # Reference schema & dataset template
│       └── integrated_dataset.csv          # [Local Only] Full integrated dataset
├── models/
│   ├── model_final_4_class.keras       # [Local Only] Trained Keras LSTM model
│   └── scaler.pkl                      # [Local Only] Scikit-Learn StandardScaler artifact
├── src/
│   ├── atmospheric_indices.py          # SounderPy & SHARPpy calculation wrapper
│   ├── calendar_utils.py               # Lookback window computation (D-7..D-1)
│   ├── integration.py                  # Multi-source data joining engine
│   ├── ogimet.py                       # Ogimet surface weather scraper & parser
│   ├── pipeline.py                     # End-to-end inference pipeline & validation gate
│   ├── predictor.py                    # Model loading and prediction logic
│   ├── preprocessing.py                # Missing data imputation & sequence scaler
│   ├── sequence.py                     # Feature scaling & sequence reshaping (1, 7, 8)
│   └── wyoming.py                      # University of Wyoming radiosonde downloader
├── tests/
│   └── test_inference.py               # Offline unit & integration tests
├── requirements.txt                    # Python runtime dependencies
└── README.md                           # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository & Navigate to Directory
```bash
cd inference_pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: `sounderpy` includes internal SHARPpy modules (`sounderpy.SHARPPYMAIN.*`) directly utilized by `src/atmospheric_indices.py` to calculate LI, TT, K-Index, and SWEAT indices.

### 3. Place Required Local Artifacts
For security and file-size constraints, trained model weights and full datasets are excluded from Git tracking. Please place the required files manually before running inference:
1. Copy `model_final_4_class.keras` into `models/`
2. Copy `scaler.pkl` into `models/`
3. Place `integrated_dataset.csv` into `data/integrated/` *(A sample structure is available at `data/integrated/integrated_dataset_example.csv`)*

---

## 🚀 Usage

### 1. Streamlit Web Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

### 2. Programmatic Python API
Run date-based risk inference directly within your Python script:

```python
from src.pipeline import predict_for_date

# Run prediction for a target date
result = predict_for_date("2024-06-15")
print(result)

# Example Output (SUCCESS):
# {
#   "status": "SUCCESS",
#   "target_date": "2024-06-15",
#   "predicted_class_index": 1,
#   "predicted_class_name": "Sedang",
#   "probabilities": {
#     "Rendah": 0.12,
#     "Sedang": 0.75,
#     "Tinggi": 0.10,
#     "Sangat Tinggi": 0.03
#   }
# }

# Example Output (REJECTED):
# {
#   "status": "REJECTED",
#   "target_date": "2024-06-15",
#   "reasons": ["Insufficient historical lookback window data"]
# }
```

For efficient batch execution across multiple dates, pre-load the model and scaler once:
```python
from src.predictor import load_model, load_scaler
from src.pipeline import predict_for_date

model = load_model()
scaler = load_scaler()

result = predict_for_date("2024-06-15", model=model, scaler=scaler)
```

---

## ⚠️ Excluded Artifacts & Notes

- **Model & Scaler Artifacts**: Neural network model (`models/model_final_4_class.keras`) and scaling weights (`models/scaler.pkl`) are managed locally.
- **Dataset Privacy**: The full integrated dataset (`data/integrated/integrated_dataset.csv`) is excluded from public tracking. Refer to `integrated_dataset_example.csv` for structural reference.

---
*© AetherSense — Flood Risk Prediction System*
