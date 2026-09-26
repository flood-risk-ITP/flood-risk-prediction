# AetherSense

AetherSense is a flood risk assessment system for Padang City. It uses local weather and atmospheric data to assess flood risk through two prediction methods: LSTM and a combined Random Forest + XGBoost model.

## Overview

AetherSense retrieves date-based weather observations for the Padang/Minangkabau station and prepares the atmospheric information needed for a flood risk assessment. The Streamlit application presents the selected model result, prediction confidence, contributing factors where available, and assessment history.

The application contains two separate prediction methods:

- **LSTM - 7-Day Weather Pattern Model**: uses weather observations from the previous seven days to assess flood risk.
- **RF + XGBoost - Combined Prediction Model**: combines Random Forest and XGBoost to assess flood risk using weather and atmospheric factors.

The methods are selectable independently; they are not merged into one model.

## Features

- Date-based flood risk assessment for Padang City.
- LSTM prediction using a seven-day weather window.
- RF + XGBoost prediction using the isolated AetherSense engine.
- Weather data retrieval from Ogimet and upper-air observations from the University of Wyoming service.
- Atmospheric condition processing and missing-data validation.
- Prediction confidence values in the result view.
- Factors influencing the prediction for RF + XGBoost through the implemented SHAP explanation.
- Assessment history for the current application session.
- User-friendly result presentation and dynamic processing stages.

## Project Structure

```text
inference_pipeline/
├── app.py
├── assets/
├── config/
├── data/
├── models/
│   ├── model_final_4_class.keras
│   └── scaler.pkl
├── src/
├── tests/
├── aethersense_engine/
│   ├── config/
│   ├── models/
│   │   ├── rf_model.joblib
│   │   └── xgb_model.joblib
│   ├── canvas/
│   └── src/
├── requirements.txt
└── README.md
```

## Installation

Create and activate a Python virtual environment from the repository root.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the application

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit in a browser.

## Requirements

Runtime dependencies are listed in `requirements.txt`. Important packages include:

- Streamlit for the web application.
- TensorFlow/Keras for the LSTM model.
- scikit-learn and joblib for preprocessing and model artifacts.
- XGBoost for the combined prediction method.
- SHAP for model-factor explanations.
- pandas and NumPy for data preparation.
- MetPy and SounderPy for atmospheric processing.
- Requests, lxml, html5lib, and BeautifulSoup for data retrieval and parsing.
- pytest for automated tests.

## Usage

1. Open the Streamlit application.
2. Select a **Prediction Method**.
3. Select the **Date to Assess**.
4. Select **Assess Flood Risk**.
5. Review the flood risk level and prediction confidence.
6. Open **View Factors Influencing Prediction** when available.
7. Open **View Weather Data Used** to inspect the input observations.
8. Review **Assessment History** for predictions made during the current session.

## Data Sources

The current implementation retrieves data from:

- **Ogimet** for daily surface weather observations.
- **University of Wyoming Upper Air** for atmospheric sounding observations.
- **SounderPy/SHARPpy calculations** for atmospheric condition processing used by the pipeline.

The station identifier used by the project is `96163` (Minangkabau/Padang).

## Model Artifacts

The application uses the following tracked artifacts:

- `models/model_final_4_class.keras` - LSTM model.
- `models/scaler.pkl` - scaler used by the LSTM input pipeline.
- `aethersense_engine/models/rf_model.joblib` - Random Forest model.
- `aethersense_engine/models/xgb_model.joblib` - XGBoost model.

## Testing

Run the test suite with:

```powershell
python -m pytest tests -q
```

The tests cover the inference contract, configuration, preprocessing, validation behavior, and model-related utilities. On Windows, pytest tests that use `tmp_path` require a writable temporary directory.

## Notes and Limitations

- Assessment results depend on the weather and atmospheric observations available for the selected date.
- Assessment history is stored for the current Streamlit session.
- The application is an assessment tool; it does not provide an official disaster warning or guarantee a flood outcome.
- The two prediction methods are presented independently. This project documentation does not claim that one method is superior to the other.

---

AetherSense - Flood Risk Assessment System