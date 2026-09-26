"""
Streamlit Presentation Layer — Padang City Flood Risk Assessment System
Multi-Engine Design (LSTM, RF + XGBoost Ensemble)
Core pipelines (src/, aethersense_engine/) are IMMUTABLE. This file is presentation layer only.
"""

# Import main Streamlit module and data/thread handling utilities
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import re
import time
import threading
import queue
import base64

# Add project root directory to sys.path to access src, config, and aethersense_engine modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reference: `src/pipeline.py` → LSTM predict_for_date()
from src.pipeline import predict_for_date as predict_lstm
from config.settings import CLASS_NAMES as LSTM_CLASS_NAMES

# Reference: `aethersense_engine/src/pipeline.py` → AetherSense predict_for_date()
from aethersense_engine.src.pipeline import predict_for_date as predict_aethersense
from aethersense_engine.config.settings import CLASS_NAMES as AETHERSENSE_CLASS_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG - Set layout configuration and Streamlit interface title
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AetherSense | Flood Risk Assessment System",
    page_icon="assets/favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://cdn.jsdelivr.net/npm/@fontsource/cal-sans/index.css');

@font-face {
    font-family: 'Cal Sans';
    font-style: normal;
    font-display: swap;
    font-weight: 400 800;
    src: url('https://cdn.jsdelivr.net/npm/@fontsource/cal-sans/files/cal-sans-latin-400-normal.woff2') format('woff2');
}

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }

html, body, .stApp {
    font-family: 'Cal Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: #f8f8f7 !important;
    color: #111 !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden !important; }
footer { visibility: hidden !important; }
header { visibility: hidden !important; }
section[data-testid="stSidebar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* Remove default block padding and force centering with max-width */
.block-container {
    max-width: 1100px !important;
    padding-top: 58px !important; /* height of navbar */
    padding-bottom: 80px !important;
    padding-left: 64px !important;
    padding-right: 64px !important;
    margin: 0 auto !important;
}

@media (max-width: 768px) {
    .block-container {
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
}

/* ── Fixed Navbar ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    height: 58px;
    background: rgba(248, 248, 247, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid #e4e4e0;
}
.navbar-inner {
    max-width: 1100px;
    height: 100%;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 64px;
}
@media (max-width: 768px) {
    .navbar-inner {
        padding: 0 24px;
    }
}
.nav-brand-wrap, .nav-brand-wrap:hover, .nav-brand-wrap:focus, .nav-brand-wrap:active, .nav-brand-wrap * {
    display: flex;
    align-items: center;
    gap: 10px;
    text-decoration: none !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}
.nav-logo-img {
    height: 26px;
    width: 26px;
    object-fit: contain;
    border-radius: 4px;
}
.nav-brand {
    font-size: 0.95rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    color: #111 !important;
    text-decoration: none !important;
}
.nav-links {
    display: flex;
    align-items: center;
    gap: 40px;
}
.nav-links a {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: capitalize;
    color: #666;
    text-decoration: none;
    transition: color 0.2s;
}
.nav-links a:hover { color: #111; }
.nav-spacer { height: 58px; }

/* ── Page Sections ── */
.page-section {
    width: 100%;
    padding: 60px 0 40px;
}
.page-section-inner {
    width: 100%;
}
.section-divider {
    width: 100%;
    border: none;
    border-top: 1px solid #e4e4e0;
    margin: 0;
}

/* ── Hero ── */
.hero-section {
    padding-top: 18px !important;
    padding-bottom: 24px !important;
}
.hero-wrap {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 0;
}
.hero-logo-img {
    width: 84px;
    height: 84px;
    object-fit: contain;
    margin-bottom: 20px;
    border-radius: 10px;
    display: block;
}
.hero-eyebrow {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #0284c7;
    margin-bottom: 12px;
    line-height: 1;
}
.hero-title {
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.08;
    letter-spacing: -0.035em;
    color: #0d0d0d;
    max-width: 800px;
    margin-bottom: 22px;
}
.prediction-note {
    margin-top: 5px;
    padding: 7px 7px;
    border-left: 2px solid #999;
    background: #f7f7f5;
    color: #666;
    font-size: 0.62rem;
    line-height: 1.6;
}
.hero-body {
    font-size: 1.05rem;
    line-height: 1.7;
    color: #4b5563;
    max-width: 680px;
    margin-bottom: 46px;
    font-weight: 400;
}
.hero-cta {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    width: fit-content;
    min-width: 240px;
    max-width: 280px;
    align-self: flex-start;
    background: #0d0d0d;
    color: #ffffff !important;
    text-decoration: none !important;
    padding: 14px 28px;
    border-radius: 4px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    transition: background 0.2s, transform 0.15s;
    box-sizing: border-box;
}
.hero-cta:hover {
    background: #2a2a2a;
    transform: translateY(-1px);
}
.hero-stats {
    width: 100%;
    align-self: stretch;
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    border-top: 1px solid #e4e4e0;
    margin-top: 56px;
    padding-top: 0;
}
.hero-stat {
    min-width: 0;
    min-height: 94px;
    padding: 24px 24px 22px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    border-right: 1px solid #e4e4e0;
}
.hero-stat:last-child { border-right: none; }
.hero-stat-val {
    font-size: clamp(1.05rem, 1.55vw, 1.4rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin-bottom: 7px;
}
.hero-stat-lbl {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa;
}

/* ── Section Typography ── */
.s-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 14px;
}
.s-title {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #0d0d0d;
    margin-bottom: 14px;
    line-height: 1.1;
}
.s-body {
    font-size: 0.9rem;
    line-height: 1.75;
    color: #666;
    max-width: 520px;
    margin-bottom: 14px;
}

/* ── Form Controls ── */
.stDateInput label, .stSelectbox label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 6px !important;
}
div[data-testid="stDateInput"] div[data-baseweb="input"],
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #ffffff !important;
    border: 1px solid #ddd !important;
    border-radius: 3px !important;
}
div[data-testid="stDateInput"] input {
    color: #111 !important;
}

/* ── Buttons ── */
div.stButton > button,
div.stDownloadButton > button {
    background: #0d0d0d !important;
    color: #f8f8f7 !important;
    border: 1px solid #0d0d0d !important;
    border-radius: 3px !important;
    padding: 13px 32px !important;
    font-family: 'Cal Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: background 0.2s, border-color 0.2s !important;
    cursor: pointer !important;
}
div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #2a2a2a !important;
    border-color: #2a2a2a !important;
    color: #ffffff !important;
}

/* ── Stage Tracker ── */
.stage-tracker {
    border: 1px solid #e4e4e0;
    border-radius: 6px;
    background: #fff;
    overflow: hidden;
    margin: 28px 0;
}
.stage-row {
    display: flex;
    align-items: center;
    padding: 13px 20px;
    border-bottom: 1px solid #f0f0ee;
    gap: 14px;
}
.stage-row:last-child { border-bottom: none; }
.stage-row.pending { opacity: 0.45; }
.stage-icon {
    width: 22px;
    height: 22px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.s-dot { width: 6px; height: 6px; border-radius: 50%; background: #ccc; }
.s-spinner {
    width: 14px; height: 14px;
    border: 2px solid #e4e4e0;
    border-top-color: #111;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
}
.s-check { font-size: 0.8rem; color: #111; font-weight: 700; }
.s-cross { font-size: 0.8rem; color: #c00; font-weight: 700; }
@keyframes spin { to { transform: rotate(360deg); } }
.stage-name {
    font-size: 0.825rem;
    font-weight: 500;
    color: #333;
    flex: 1;
}
.stage-time { font-size: 0.72rem; color: #bbb; }

/* ── Risk Display ── */
.result-card-box {
    border: 1px solid #e4e4e0;
    border-radius: 6px;
    background: #ffffff;
    padding: 24px;
    margin-top: 20px;
}
.result-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 10px;
}
.result-class {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1.1;
    margin-bottom: 24px;
}
.rc-rendah, .rc-ringan { color: #1a1a1a; }
.rc-sedang { color: #92400e; }
.rc-tinggi, .rc-lebat { color: #991b1b; }
.rc-sangat-tinggi, .rc-sangat-lebat { color: #6b21a8; }

/* ── Probability Grid ── */
.prob-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.prob-card {
    border: 1px solid #e4e4e0;
    border-radius: 4px;
    padding: 14px 12px;
    background: #fff;
}
.prob-card.active { border-color: #0d0d0d; background: #fafaf9; }
.prob-card-lbl {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
}
.prob-card-val {
    font-size: 1.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin-bottom: 6px;
}
.pbar-bg { height: 2px; background: #eee; border-radius: 1px; }
.pbar-fill { height: 2px; background: #0d0d0d; border-radius: 1px; }

/* ── Indices Grid ── */
.indices-label {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 16px;
}
.indices-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 24px;
}
.idx-card {
    border: 1px solid #e4e4e0;
    border-radius: 4px;
    padding: 14px 16px;
    background: #ffffff;
}
.idx-lbl {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #bbb;
    margin-bottom: 6px;
}
.idx-val {
    font-size: 1rem;
    font-weight: 700;
    color: #0d0d0d;
}
.idx-na {
    font-size: 0.78rem;
    color: #ccc;
    font-style: italic;
}

/* ── Expanders ── */
div[data-testid="stExpander"] {
    margin-top: 10px;
}

div[data-testid="stExpander"] details summary {
    font-size: 0.72rem !important;
    color: #111111 !important;
    background-color: #ffffff !important;
}

div[data-testid="stExpander"] details[open] summary {
    color: #111111 !important;
    background-color: #ffffff !important;
}

div[data-testid="stExpander"] details summary:hover,
div[data-testid="stExpander"] details summary:focus,
div[data-testid="stExpander"] details summary:active {
    color: #111111 !important;
    background-color: #f5f5f4 !important;
    outline: none !important;
}

div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span {
    color: #111111 !important;
}

div[data-testid="stExpander"] details summary svg {
    color: #111111 !important;
    fill: currentColor !important;
}

.about-desc {
    max-width: 780px;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #4b5563;
}
.about-divider {
    border: none;
    border-top: 1px solid #e8e8e4;
    margin: 32px 0;
}
.about-section-label {
    margin-bottom: 14px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #888;
}
.about-methods-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
    margin-bottom: 28px;
}
.about-step-card,
.about-explanation-card {
    border: 1px solid #e8e8e4;
    border-radius: 6px;
    padding: 22px;
    background: #ffffff;
}
.about-step-idx {
    margin-bottom: 10px;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    color: #0284c7;
}
.about-step-title {
    margin-bottom: 8px;
    font-size: 0.95rem;
    font-weight: 700;
    color: #111;
}
.about-step-desc {
    margin: 0;
    font-size: 0.85rem;
    line-height: 1.55;
    color: #555;
}
.about-explanation-card {
    max-width: 780px;
}
@media (max-width: 768px) {
    .about-methods-grid {
        grid-template-columns: 1fr;
        gap: 14px;
    }
}
/* ── History ── */
.empty-state {
    border: 1px dashed #ddd;
    border-radius: 6px;
    padding: 64px 40px;
    text-align: center;
    color: #bbb;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "running": False,
    "selected_engine": "LSTM",
    "result": None,
    "target_date": None,
    "history": [],
    "csv_path": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

LSTM_PIPELINE_STAGES = [
    {"key": "[1/7]", "label": "Retrieving daily weather observations"},
    {"key": "[2/7]", "label": "Retrieving upper-air weather observations"},
    {"key": "[3/7]", "label": "Analyzing atmospheric conditions"},
    {"key": "[4/7]", "label": "Combining weather observations"},
    {"key": "[5/7]", "label": "Completing missing weather data"},
    {"key": "[6/7]", "label": "Checking whether enough data is available"},
    {"key": "[7/7]", "label": "Assessing flood risk using the 7-day weather pattern"},
]

AETHERSENSE_PIPELINE_STAGES = [
    {"key": "[1/6]", "label": "Retrieving daily weather observations"},
    {"key": "[2/6]", "label": "Retrieving upper-air weather observations"},
    {"key": "[3/6]", "label": "Analyzing atmospheric conditions and recent weather patterns"},
    {"key": "[4/6]", "label": "Preparing weather factors for assessment"},
    {"key": "[5/6]", "label": "Assessing flood risk using combined models"},
    {"key": "[6/6]", "label": "Identifying factors influencing the assessment"},
]

class ThreadAwareWriter:
    def __init__(self, original, log_queue, inference_thread_id):
        self._orig = original
        self._queue = log_queue
        self._tid = inference_thread_id

    def write(self, text):
        if threading.current_thread().ident == self._tid:
            if text:
                self._queue.put(text)
        else:
            try:
                self._orig.write(text)
            except Exception:
                pass

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass


def _inference_worker(engine_choice, target_date_str, result_ref, log_queue):
    """Run prediction for selected engine in a background thread."""
    res_dict = {"engine_choice": engine_choice, "target_date": target_date_str}
    
    if engine_choice == "LSTM":
        try:
            res_lstm = predict_lstm(target_date_str, verbose=True, use_cache=True)
            res_dict["lstm_result"] = res_lstm
        except Exception as exc:
            res_dict["lstm_result"] = {
                "status": "REJECTED",
                "target_date": target_date_str,
                "reasons": [str(exc)],
            }
            
    elif engine_choice == "RF + XGBoost Ensemble":
        try:
            res_aether = predict_aethersense(target_date_str, verbose=True)
            res_dict["aethersense_result"] = res_aether
        except Exception as exc:
            res_dict["aethersense_result"] = {
                "status": "REJECTED",
                "target_date": target_date_str,
                "reasons": [str(exc)],
            }
            
    result_ref["result"] = res_dict
    log_queue.put(None)


def _render_tracker(statuses: list, stage_defs: list) -> str:
    rows = []
    for i, stage in enumerate(stage_defs):
        s = statuses[i]
        st_cls = s["status"]
        if st_cls == "pending":
            icon = '<div class="s-dot"></div>'
        elif st_cls == "running":
            icon = '<div class="s-spinner"></div>'
        elif st_cls == "done":
            icon = '<span class="s-check">✓</span>'
        else:
            icon = '<span class="s-cross">✕</span>'
        elapsed = f'<span class="stage-time">{s["elapsed"]}s</span>' if s.get("elapsed") else ""
        rows.append(
            f'<div class="stage-row {st_cls}">'
            f'<div class="stage-icon">{icon}</div>'
            f'<span class="stage-name">{stage["label"]}</span>'
            f'{elapsed}'
            f'</div>'
        )
    return f'<div class="stage-tracker">{"".join(rows)}</div>'


def _fv(val, unit="", decimals=2):
    if val is None:
        return None
    try:
        f = float(val)
        import math
        if math.isnan(f):
            return None
        return f"{f:.{decimals}f}{unit}"
    except (TypeError, ValueError):
        s = str(val).strip()
        return s if s else None


def _render_idx(val, unit="", decimals=2):
    v = _fv(val, unit, decimals)
    if v is None:
        return '<span class="idx-na">not available</span>'
    return f'<span class="idx-val">{v}</span>'


_RISK_CSS = {
    "Low": "rc-low",
    "Medium": "rc-medium",
    "High": "rc-high",
    "Very High": "rc-very-high",
    "Low": "rc-low",
    "High": "rc-high",
    "Very High": "rc-very-high",
}

def _get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "aethersense_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

_LOGO_B64 = _get_logo_base64()

# ── NAVBAR ──
st.markdown(f"""
<nav class="navbar">
    <div class="navbar-inner">
        <a href="#home" class="nav-brand-wrap">
            <img src="data:image/jpeg;base64,{_LOGO_B64}" class="nav-logo-img" alt="AetherSense Logo" />
            <span class="nav-brand">AetherSense</span>
        </a>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#prediction">Prediction</a>
            <a href="#history">History</a>
            <a href="#about">About</a>
        </div>
    </div>
</nav>
<div class="nav-spacer"></div>
""", unsafe_allow_html=True)

# ── SECTION: HOME ──
st.markdown('<a id="home" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown(f"""
<div class="page-section hero-section">
<div class="page-section-inner">
<div class="hero-wrap">
    <img src="data:image/png;base64,{_LOGO_B64}" class="hero-logo-img" alt="AetherSense Logo" />
    <div class="hero-eyebrow">AETHERSENSE</div>
    <h1 class="hero-title">Flood Risk Assessment</h1>
    <p class="hero-body">
       Assess flood risk in Padang City with data-driven insights from local weather and atmospheric conditions
    </p>
    <a href="#prediction" class="hero-cta">START NOW &nbsp;→</a>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">LSTM / RF + XGBoost</div>
            <div class="hero-stat-lbl">Prediction Method</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">7 Days</div>
            <div class="hero-stat-lbl">Weather Data Period Used</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">8 / 16 Features</div>
            <div class="hero-stat-lbl">Weather Factors Analyzed</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">4 Levels</div>
            <div class="hero-stat-lbl">Flood Risk Levels</div>
        </div>
        <div class="hero-stat">
            <div class="hero-stat-val">Ogimet / Wyoming</div>
            <div class="hero-stat-lbl">Weather Data Sources</div>
        </div>
    </div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── SECTION: PREDICTION ──
st.markdown('<a id="prediction" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section prediction-section">
<div class="page-section-inner">
    <div class="s-eyebrow">Prediction</div>
    <div class="s-title">Flood Risk Assessment</div>
    <div class="s-body">
       Select a prediction method and date to assess. The system will retrieve the weather observations needed for the assessment
    </div>
    <div class="prediction-note">
        <strong>Note:</strong> The assessment uses available weather observations from Ogimet and Wyoming Upper-Air Weather Data
    </div>
""", unsafe_allow_html=True)

col_input, col_pad = st.columns([2, 3])
with col_input:
    engine_choice_widget = st.selectbox(
        "Prediction Method",
        options=["LSTM", "RF + XGBoost Ensemble"],
        format_func=lambda value: (
            "LSTM"
            if value == "LSTM"
            else "RF + XGBoost"
        ),
        key="engine_picker",
        help="Select the model used to assess flood risk",
    )
    target_date_widget = st.date_input(
        "Date to Assess",
        value=None,
        key="date_picker",
        help="The system will retrieve weather data for this date",
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    submit_disabled = (target_date_widget is None) or st.session_state.running
    submit_clicked = st.button(
        "Assess Flood Risk",
        disabled=submit_disabled,
        key="btn_submit",
    )

if submit_clicked and target_date_widget is not None:
    st.session_state.target_date = target_date_widget.strftime("%Y-%m-%d")
    st.session_state.selected_engine = engine_choice_widget
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.csv_path = None
    st.rerun()

# Real-time progress tracking
if st.session_state.running and st.session_state.target_date:
    tgt = st.session_state.target_date
    eng = st.session_state.selected_engine
    
    current_stages = LSTM_PIPELINE_STAGES if eng == "LSTM" else AETHERSENSE_PIPELINE_STAGES

    st.markdown(
        f'<div style="margin-top:32px; font-size:0.7rem; font-weight:700; '
        f'letter-spacing:0.18em; text-transform:uppercase; color:#aaa;">'
        f'Preparing Assessment [{eng}] for Date {tgt}</div>',
        unsafe_allow_html=True,
    )
    tracker_ph = st.empty()

    statuses = [
        {"status": "running" if i == 0 else "pending", "elapsed": None}
        for i in range(len(current_stages))
    ]
    tracker_ph.markdown(_render_tracker(statuses, current_stages), unsafe_allow_html=True)

    log_q: queue.Queue = queue.Queue()
    result_ref: dict = {"result": None}
    orig_stdout = sys.stdout

    thread = threading.Thread(
        target=_inference_worker,
        args=(eng, tgt, result_ref, log_q),
        daemon=True,
    )

    thread.start()
    sys.stdout = ThreadAwareWriter(orig_stdout, log_q, thread.ident)

    done = False
    while not done:
        try:
            while True:
                line = log_q.get_nowait()
                if line is None:
                    done = True
                    break
                # Update status
                for i, stage in enumerate(current_stages):
                    if stage["key"] in line:
                        if "OK" in line or "FAILED" in line:
                            m = re.search(r"\((\d+\.\d+)s\)", line)
                            statuses[i]["elapsed"] = m.group(1) if m else None
                            statuses[i]["status"] = "done" if "OK" in line else "failed"
                        else:
                            statuses[i]["status"] = "running"
                tracker_ph.markdown(_render_tracker(statuses, current_stages), unsafe_allow_html=True)
        except queue.Empty:
            pass

        if not done:
            tracker_ph.markdown(_render_tracker(statuses, current_stages), unsafe_allow_html=True)
            time.sleep(0.2)

    sys.stdout = orig_stdout
    thread.join(timeout=10)

    for s in statuses:
        if s["status"] == "running":
            s["status"] = "done"
    tracker_ph.markdown(_render_tracker(statuses, current_stages), unsafe_allow_html=True)

    final_result = result_ref["result"]
    st.session_state.result = final_result
    st.session_state.running = False

    csv_candidate = f"inference_features_{tgt}.csv"
    if os.path.exists(csv_candidate):
        st.session_state.csv_path = csv_candidate

    # Update history
    if final_result:
        pred_class_str = "-"
        if "lstm_result" in final_result and final_result["lstm_result"].get("status") == "SUCCESS":
            pred_class_str = final_result["lstm_result"].get("predicted_class_name", "-")
        elif "aethersense_result" in final_result and final_result["aethersense_result"].get("status") == "SUCCESS":
            pred_class_str = final_result["aethersense_result"].get("predicted_class_name", "-")
            
        entry = {
            "Date": tgt,
            "Engine Selected": eng,
            "Predicted Risk Class": pred_class_str,
        }
        if not any(h["Date"] == tgt and h["Engine Selected"] == eng for h in st.session_state.history):
            st.session_state.history.insert(0, entry)

def _friendly_reason(reason):
    """Translate pipeline diagnostics into concise user-facing English."""
    text = str(reason)
    replacements = {
        "Data tidak tersedia untuk tanggal ini:": "Weather data is unavailable for this date:",
        "Kolom fitur tidak tersedia:": "Required weather factors are unavailable:",
        "Kolom fitur bukan numerik:": "Some weather data is not numeric:",
        "Gagal akuisisi data Ogimet:": "Could not retrieve daily weather observations:",
        "Gagal akuisisi data sounding:": "Could not retrieve upper-air weather observations:",
        "Validation gate gagal:": "The required weather data could not be verified:",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text
# Helper function to render a single engine result card
def render_engine_result_card(engine_title, res):
    if not res:
        st.info("No assessment result is available for this method.")
        return
        
    tgt_date = res.get("target_date", "")
    status = res.get("status", "")
    
    with st.container(border=True):
        st.markdown(f'<div class="result-eyebrow">{engine_title}</div>', unsafe_allow_html=True)
        
        if status == "SUCCESS":
            pred = res.get("predicted_class_name", "")
            rc = _RISK_CSS.get(pred, "rc-rendah")
            probs = res.get("probabilities", {})
            
            st.markdown(f'<div class="result-class {rc}">{pred}</div>', unsafe_allow_html=True)
            
            cards_html = ""
            for cls_name, p in probs.items():
                is_active = "active" if cls_name == pred else ""
                cards_html += f"""
                <div class="prob-card {is_active}">
                    <div class="prob-card-lbl">{cls_name}</div>
                    <div class="prob-card-val">{p*100:.1f}%</div>
                    <div class="pbar-bg"><div class="pbar-fill" style="width:{p*100:.1f}%"></div></div>
                </div>"""
            st.markdown(f'<div class="prob-grid">{cards_html}</div>', unsafe_allow_html=True)
            
            # Display SHAP if available
            if "shap_values" in res:
                shap_info = res["shap_values"]
                with st.expander("View Factors Influencing Prediction"):
                    df_shap = pd.DataFrame({
                        "Weather Factor": shap_info["feature_names"],
                        "Value": shap_info["feature_values"],
                        "Influence on Prediction": shap_info["values"],
                    }).sort_values(by="Influence on Prediction", key=abs, ascending=False)
                    st.dataframe(df_shap, use_container_width=True, hide_index=True)
                    
        else:
            reasons = res.get("reasons", ["The required weather data could not be verified."])
            st.error(f"Flood risk assessment could not be completed using {engine_title}:")
            for r in reasons:
                st.markdown(f"<small>• {_friendly_reason(r)}</small>", unsafe_allow_html=True)
                


# Render Prediction Results
if st.session_state.result and not st.session_state.running:
    res_dict = st.session_state.result
    eng_choice = res_dict.get("engine_choice", "LSTM")
    tgt_date = res_dict.get("target_date", "")

    st.markdown(f"""
    <div style="margin-top:20px; font-size:0.75rem; font-weight:700; letter-spacing:0.16em; text-transform:uppercase; color:#888;">
        Flood Risk Assessment — Date: {tgt_date}
    </div>
    """, unsafe_allow_html=True)

    if eng_choice == "LSTM":
        render_engine_result_card("LSTM — 7-Day Weather Pattern Model", res_dict.get("lstm_result"))
    elif eng_choice == "RF + XGBoost Ensemble":
        render_engine_result_card("RF + XGBoost — Combined Prediction Model", res_dict.get("aethersense_result"))

    # Feature table download if available
    csv_path = st.session_state.csv_path
    if csv_path and os.path.exists(csv_path):
        try:
            df_feat = pd.read_csv(csv_path)
            with st.expander("View Weather Data Used"):
                st.dataframe(df_feat, use_container_width=True, hide_index=True)
        except Exception:
            pass

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("Assess Another Date", key="btn_reset"):
            st.session_state.result = None
            st.session_state.target_date = None
            st.session_state.csv_path = None
            st.rerun()

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── SECTION: HISTORY ──
st.markdown('<a id="history" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section history-section">
<div class="page-section-inner">
    <div class="s-eyebrow">History</div>
    <div class="s-title">Assessment History</div>
    <div class="s-body">List of flood risk assessments performed during this session</div>
""", unsafe_allow_html=True)

hist = st.session_state.history
if hist:
    history_df = pd.DataFrame(hist).rename(columns={
        "Engine Selected": "Prediction Method",
        "Predicted Risk Class": "Flood Risk Level",
    })
    st.dataframe(history_df, use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div class="empty-state">No assessments performed yet. Select a date above to begin.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ── SECTION: ABOUT ──
st.markdown('<a id="about" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
<div class="s-eyebrow">About</div>
<div class="s-title">What is AetherSense?</div>
<p class="about-desc">
AetherSense is a flood risk assessment system for Padang City. It uses local weather and atmospheric data to assess the level of flood risk using two prediction methods.
</p>
<hr class="about-divider">
<div class="about-section-label">Prediction Methods</div>
<div class="about-methods-grid">
<div class="about-step-card">
<div class="about-step-idx">LSTM</div>
<div class="about-step-title">7-Day Weather Pattern Model</div>
<p class="about-step-desc">Uses weather observations from the previous 7 days to assess flood risk</p>
</div>
<div class="about-step-card">
<div class="about-step-idx">RF + XGBoost</div>
<div class="about-step-title">Combined Prediction Model</div>
<p class="about-step-desc">Combines Random Forest and XGBoost to assess weather and atmospheric factors</p>
</div>
</div>
<div class="about-explanation-card">
<div class="about-section-label">UNDERSTANDING THE ASSESSMENT</div>
<div class="about-step-title">Factors Influencing the Prediction</div>
<p class="about-step-desc">Shows which weather and atmospheric factors contributed most to the assessment.</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER ──
st.markdown("""
<div style="border-top:1px solid #e4e4e0; padding:32px 64px; display:flex; justify-content:space-between; align-items:center;">
    <span style="font-size:0.7rem; color:#bbb; letter-spacing:0.1em; text-transform:uppercase;">
        AetherSense — Flood Risk Assessment System
    </span>
    <span style="font-size:0.7rem; color:#bbb; letter-spacing:0.06em;">
        © 2026 Imam. All rights reserved.
    </span>
</div>
""", unsafe_allow_html=True)
