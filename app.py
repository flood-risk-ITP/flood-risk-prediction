"""
Streamlit Presentation Layer — Padang City Flood Risk Assessment System
Single-page design with live progress tracking.
Core pipeline (src/) is IMMUTABLE. This file is presentation layer only.
"""

# Import main Streamlit module and data/thread handling utilities
import streamlit as st
import pandas as pd
import sys
import os
import re
import time
import threading
import queue

# Add project root directory to sys.path to access src and config modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reference: `src/pipeline.py` → `predict_for_date()`
# Connect user interface (dashboard) to the main entry point of the inference pipeline orchestrating data retrieval to model call.
from src.pipeline import predict_for_date

# Reference: `config/settings.py` → `CLASS_NAMES`
# Import 4 flood risk classification category labels ("Rendah", "Sedang", "Tinggi", "Sangat Tinggi") per model contract.
from config.settings import CLASS_NAMES

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
.nav-brand-dot {
    font-size: 0.8rem;
    color: #ccc;
    margin: 0 2px;
}
.nav-brand-sub {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #888;
    text-transform: uppercase;
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
.hero-title em { font-style: normal; color: #666; }
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
    display: flex;
    gap: 0;
    border-top: 1px solid #e4e4e0;
    margin-top: 56px;
    padding-top: 0;
}
.hero-stat {
    
    flex: 1;
    padding: 26px 0;
    border-right: 1px solid #e4e4e0;
}
.hero-stat:last-child { border-right: none; }
.hero-stat-val {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin-bottom: 4px;
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
.stDateInput label, .stNumberInput label, .stTextInput label {
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
    color: #888 !important;
    margin-bottom: 6px !important;
}
div[data-testid="stDateInput"] div[data-baseweb="input"],
div[data-testid="stTextInput"] div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border: 1px solid #ddd !important;
    border-radius: 3px !important;
}
div[data-testid="stDateInput"] input,
div[data-testid="stTextInput"] input {
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
div.stButton > button:focus,
div.stDownloadButton > button:focus,
div.stButton > button:active,
div.stDownloadButton > button:active {
    background: #2a2a2a !important;
    border-color: #2a2a2a !important;
    color: #ffffff !important;
    outline: none !important;
    box-shadow: none !important;
}
div.stButton > button:disabled,
div.stDownloadButton > button:disabled {
    background: #e4e4e0 !important;
    color: #777 !important;
    border-color: #e4e4e0 !important;
    cursor: not-allowed !important;
}

/* ── Expanders ── */
div[data-testid="stExpander"] details summary {
    background-color: #ffffff !important;
    color: #111111 !important;
    border: 1px solid #e4e4e0 !important;
    border-radius: 3px !important;
    transition: background-color 0.15s, color 0.15s !important;
}
div[data-testid="stExpander"] details summary:hover {
    background-color: #f5f5f4 !important;
    color: #111111 !important;
}
div[data-testid="stExpander"] details summary:focus,
div[data-testid="stExpander"] details summary:active {
    background-color: #f5f5f4 !important;
    color: #111111 !important;
    outline: none !important;
}
div[data-testid="stExpander"] details summary p,
div[data-testid="stExpander"] details summary span,
div[data-testid="stExpander"] details summary svg {
    color: inherit !important;
    fill: currentColor !important;
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
    transition: background 0.1s;
}
.stage-row:last-child { border-bottom: none; }
.stage-row.running { background: #fafaf9; }
.stage-row.done { }
.stage-row.pending { opacity: 0.45; }
.stage-icon {
    width: 22px;
    height: 22px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.s-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #ccc;
}
.s-spinner {
    width: 14px;
    height: 14px;
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
    letter-spacing: 0.01em;
}
.stage-time {
    font-size: 0.72rem;
    color: #bbb;
    font-variant-numeric: tabular-nums;
}

/* ── Risk Display ── */
.result-wrap {
    border-top: 1px solid #e4e4e0;
    margin-top: 40px;
    padding-top: 40px;
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
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
    margin-bottom: 40px;
}
.rc-rendah { color: #1a1a1a; }
.rc-sedang { color: #92400e; }
.rc-tinggi { color: #991b1b; }
.rc-sangat-tinggi { color: #6b21a8; }

/* ── Probability Grid ── */
.prob-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 48px;
}
.prob-card {
    border: 1px solid #e4e4e0;
    border-radius: 4px;
    padding: 18px 16px;
    background: #fff;
    transition: border-color 0.2s;
}
.prob-card.active { border-color: #0d0d0d; }
.prob-card-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 10px;
}
.prob-card-val {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #0d0d0d;
    margin-bottom: 10px;
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
    margin-bottom: 40px;
}
.idx-card {
    border: 1px solid #e4e4e0;
    border-radius: 4px;
    padding: 14px 16px;
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
    letter-spacing: -0.01em;
}
.idx-na {
    font-size: 0.78rem;
    color: #ccc;
    font-style: italic;
}

/* ── History ── */
.empty-state {
    border: 1px dashed #ddd;
    border-radius: 6px;
    padding: 64px 40px;
    text-align: center;
    color: #bbb;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}

/* ── About Section ── */
.about-desc {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #4b5563;
    max-width: 760px;
    margin-bottom: 0;
}
.about-divider {
    border: none;
    border-top: 1px solid #e8e8e4;
    margin: 36px 0;
}
.about-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 18px;
}

/* 01 How It Works Grid */
.about-steps-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}
@media (max-width: 768px) {
    .about-steps-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}
.about-step-card {
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 24px 22px;
    display: flex;
    flex-direction: column;
}
.about-step-idx {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    color: #0284c7;
    margin-bottom: 8px;
}
.about-step-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #111;
    letter-spacing: -0.01em;
    margin-bottom: 8px;
}
.about-step-desc {
    font-size: 0.85rem;
    line-height: 1.55;
    color: #555;
    margin: 0;
}

/* 02 Model Inputs Grid */
.about-inputs-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
}
@media (max-width: 768px) {
    .about-inputs-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}
.about-input-card {
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 24px 24px;
}
.about-input-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    padding-bottom: 12px;
    border-bottom: 1px solid #f0f0ec;
    margin-bottom: 12px;
}
.about-var-list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.about-var-item {
    font-size: 0.86rem;
    line-height: 1.6;
    padding: 8px 0;
    border-bottom: 1px solid #fbfbfa;
    display: flex;
    gap: 10px;
    align-items: baseline;
}
.about-var-item:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.about-var-code {
    font-weight: 700;
    color: #111;
    min-width: 64px;
}
.about-var-sep {
    color: #ccc;
}
.about-var-name {
    color: #4b5563;
}

/* 03 Data Sources Grid */
.about-sources-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}
@media (max-width: 768px) {
    .about-sources-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}
.about-source-card {
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 22px 22px;
}
.about-source-name {
    font-size: 0.95rem;
    font-weight: 700;
    color: #111;
    margin-bottom: 6px;
    letter-spacing: -0.01em;
}
.about-source-desc {
    font-size: 0.85rem;
    line-height: 1.55;
    color: #555;
    margin: 0;
}

/* 04 Model Grid */
.about-model-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
}
@media (max-width: 768px) {
    .about-model-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}
.about-model-card {
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    padding: 22px 22px;
}
.about-spec-lbl {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}
.about-spec-val {
    font-size: 0.98rem;
    font-weight: 700;
    color: #111;
    letter-spacing: -0.01em;
}

/* 05 Technical Details */
.about-tech-details {
    margin-top: 36px;
    border: 1px solid #e8e8e4;
    border-radius: 8px;
    background: #ffffff;
    overflow: hidden;
}
.about-tech-summary {
    padding: 14px 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #666;
    cursor: pointer;
    user-select: none;
    transition: color 0.15s, background 0.15s;
    list-style: none;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.about-tech-summary::-webkit-details-marker {
    display: none;
}
.about-tech-summary::marker {
    display: none;
    content: "";
}
.about-tech-summary:hover {
    color: #111;
    background: #fafaf9;
}
.about-tech-toggle {
    font-size: 1.1rem;
    font-weight: 400;
    color: #888;
    transition: transform 0.2s ease, color 0.2s ease;
    line-height: 1;
}
.about-tech-details[open] .about-tech-toggle {
    transform: rotate(45deg);
    color: #111;
}
.about-tech-content {
    padding: 22px 24px;
    border-top: 1px solid #f0f0ec;
    background: #ffffff;
}
.about-tech-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 22px 32px;
}
@media (max-width: 768px) {
    .about-tech-grid {
        grid-template-columns: 1fr;
        gap: 16px;
    }
}
.about-tech-col {}
.about-tech-lbl {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 4px;
}
.about-tech-val {
    font-size: 0.83rem;
    line-height: 1.6;
    color: #555;
}
.about-tech-val code {
    font-size: 0.78rem;
    background: #f4f4f2;
    padding: 2px 5px;
    border-radius: 3px;
    color: #111;
}

/* ── Misc ── */
.stSpinner > div > div { border-top-color: #111 !important; }
.stAlert { border-radius: 4px !important; font-size: 0.875rem !important; }
div[data-testid="stDataFrame"] { border: 1px solid #e4e4e0 !important; border-radius: 4px !important; }
input, select, textarea, .stMarkdown, .stAlert, div[data-baseweb], div[data-testid="stDataFrame"] {
    font-family: 'Cal Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE - Persist application state across Streamlit reruns
# ─────────────────────────────────────────────────────────────────────────────
defaults = {
    "running": False,      # Flag indicating whether inference process is running in background thread
    "result": None,        # Store dict of inference results (SUCCESS/REJECTED status, predicted class name, probabilities)
    "target_date": None,   # Store target date string selected by user ("YYYY-MM-DD") as reference window D-7..D-1
    "history": [],         # Store search history with SUCCESS status during active session
    "csv_path": None,      # Store CSV path of processed 7-day features (`inference_features_{target_date}.csv`) for download
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGE DEFINITIONS - 7 stages of inference displayed on progress tracker UI
# Reference: `src/pipeline.py` → log stages `[1/7]` to `[7/7]` in `predict_for_date()`
# ─────────────────────────────────────────────────────────────────────────────
PIPELINE_STAGES = [
    {"key": "[1/7]", "label": "Fetching surface weather data"},       # Reference: `src/ogimet.py` → `get_ogimet_daily()`
    {"key": "[2/7]", "label": "Fetching atmospheric data"},           # Reference: `src/wyoming.py` & `src/sounding.py`
    {"key": "[3/7]", "label": "Computing atmospheric indices"},       # Reference: `src/atmospheric_indices.py` → `compute_indices_for_sounding()`
    {"key": "[4/7]", "label": "Integrating data"},                    # Reference: `src/integration.py` → `integrate()`
    {"key": "[5/7]", "label": "Preparing data for prediction"},       # Reference: `src/preprocessing.py` → `apply_stage7_missing_value_handling()`
    {"key": "[6/7]", "label": "Checking 7-day data completeness"},    # Reference: `src/pipeline.py` → `_run_validation_gate()` & `src/sequence.py`
    {"key": "[7/7]", "label": "Running prediction model"},           # Reference: `src/predictor.py` → `predict()`
]

# Display translation map for internal risk class names
RISK_DISPLAY_NAME = {
    "Rendah": "Low",
    "Sedang": "Moderate",
    "Tinggi": "High",
    "Sangat Tinggi": "Very High",
}

# ─────────────────────────────────────────────────────────────────────────────
# THREAD-SAFE STDOUT CAPTURE - Capture stdout logs from inference thread to update UI
# ─────────────────────────────────────────────────────────────────────────────
class ThreadAwareWriter:
    """Route writes from inference thread to queue; all others → original."""

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


# Worker Thread: Asynchronous connector between user interface (dashboard) and inference pipeline.
def _inference_worker(target_date_str, result_ref, log_queue):
    """Run predict_for_date in a background thread."""
    try:
        res = predict_for_date(target_date_str, verbose=True, use_cache=True)
        result_ref["result"] = res
    except Exception as exc:
        result_ref["result"] = {
            "status": "REJECTED",
            "target_date": target_date_str,
            "reasons": [str(exc)],
        }
    finally:
        log_queue.put(None)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE TRACKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _parse_stage(line: str, statuses: list):
    """Parse one pipeline print line; update statuses in-place."""
    for i, stage in enumerate(PIPELINE_STAGES):
        if stage["key"] not in line:
            continue
        if "OK" in line or "FAILED" in line:
            m = re.search(r"\((\d+\.\d+)s\)", line)
            statuses[i]["elapsed"] = m.group(1) if m else None
            statuses[i]["status"] = "done" if "OK" in line else "failed"
            # Advance next stage to 'running'
            if statuses[i]["status"] == "done" and i + 1 < len(PIPELINE_STAGES):
                if statuses[i + 1]["status"] == "pending":
                    statuses[i + 1]["status"] = "running"
        else:
            if statuses[i]["status"] == "pending":
                statuses[i]["status"] = "running"
        break


def _render_tracker(statuses: list) -> str:
    rows = []
    for i, stage in enumerate(PIPELINE_STAGES):
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


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _fv(val, unit="", decimals=2):
    """Format a numeric value; returns None if unavailable."""
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
    "Rendah": "rc-rendah",
    "Sedang": "rc-sedang",
    "Tinggi": "rc-tinggi",
    "Sangat Tinggi": "rc-sangat-tinggi",
}

# Encode logo as base64 for inline HTML rendering in navbar/hero
import base64
def _get_logo_base64():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "aethersense_logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

_LOGO_B64 = _get_logo_base64()

# ─────────────────────────────────────────────────────────────────────────────
# ── NAVBAR ──
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: HOME ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="home" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown(f"""
<div class="page-section hero-section">
<div class="page-section-inner">
<div class="hero-wrap">
    <img src="data:image/png;base64,{_LOGO_B64}" class="hero-logo-img" alt="AetherSense Logo" />
    <div class="hero-eyebrow">AETHERSENSE</div>
    <h1 class="hero-title">Flood Risk Assessment</h1>
    <p class="hero-body">
        Forecast flood risk levels in Padang City for your selected date, based on weather and atmospheric conditions over the past 7 days
    </p>
    <a href="#prediction" class="hero-cta">START PREDICTION &nbsp;→</a>
    <div class="hero-stats">
        <div class="hero-stat">
            <div class="hero-stat-val">LSTM</div>
            <div class="hero-stat-lbl">Prediction Method</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">7 Days</div>
            <div class="hero-stat-lbl">Data Range Used</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">8 Variables</div>
            <div class="hero-stat-lbl">Analyzed Variables</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">4 Levels</div>
            <div class="hero-stat-lbl">Risk Levels</div>
        </div>
        <div class="hero-stat" style="padding-left:32px;">
            <div class="hero-stat-val">Ogimet · Wyoming</div>
            <div class="hero-stat-lbl">Data Sources</div>
        </div>
    </div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: PREDICTION ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="prediction" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
    <div class="s-eyebrow">Prediction</div>
    <div class="s-title">Flood Risk Prediction</div>
    <div class="s-body">
       Select the target date to predict. The system will automatically fetch weather and atmospheric data for the preceding 7 days. No manual data entry is required
    </div>
    <div class="prediction-note">
        <strong>Note:</strong> The prediction process may take a few moments as the system retrieves atmospheric data from external servers
    </div>
""", unsafe_allow_html=True)

# Form Input Target Date
col_input, col_pad = st.columns([2, 3])
with col_input:
    target_date_widget = st.date_input(
        "Prediction Date",
        value=None,
        key="date_picker",
        help="The system will use data from the 7 days prior to this date to make predictions",
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    submit_disabled = (target_date_widget is None) or st.session_state.running
    submit_clicked = st.button(
        "Predict Now",
        disabled=submit_disabled,
        key="btn_submit",
    )

if submit_clicked and target_date_widget is not None:
    st.session_state.target_date = target_date_widget.strftime("%Y-%m-%d")
    st.session_state.running = True
    st.session_state.result = None
    st.session_state.csv_path = None
    st.rerun()

# Real-time progress tracking
if st.session_state.running and st.session_state.target_date:
    tgt = st.session_state.target_date
    st.markdown(
        f'<div style="margin-top:32px; font-size:0.7rem; font-weight:700; '
        f'letter-spacing:0.18em; text-transform:uppercase; color:#aaa;">'
        f'Processing Date {tgt}</div>',
        unsafe_allow_html=True,
    )
    tracker_ph = st.empty()

    statuses = [
        {"status": "running" if i == 0 else "pending", "elapsed": None}
        for i in range(len(PIPELINE_STAGES))
    ]
    tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)

    log_q: queue.Queue = queue.Queue()
    result_ref: dict = {"result": None}
    orig_stdout = sys.stdout

    thread = threading.Thread(
        target=_inference_worker,
        args=(tgt, result_ref, log_q),
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
                _parse_stage(line, statuses)
                tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)
        except queue.Empty:
            pass

        if not done:
            tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)
            time.sleep(0.2)

    sys.stdout = orig_stdout
    thread.join(timeout=10)

    for s in statuses:
        if s["status"] == "running":
            s["status"] = "done"
    tracker_ph.markdown(_render_tracker(statuses), unsafe_allow_html=True)

    final_result = result_ref["result"]
    st.session_state.result = final_result
    st.session_state.running = False

    csv_candidate = f"inference_features_{tgt}.csv"
    if os.path.exists(csv_candidate):
        st.session_state.csv_path = csv_candidate

    if final_result and final_result.get("status") == "SUCCESS":
        probs = final_result.get("probabilities", {})
        internal_pred = final_result.get("predicted_class_name", "-")
        disp_pred = RISK_DISPLAY_NAME.get(internal_pred, internal_pred)
        entry = {
            "Date": tgt,
            "Prediction": disp_pred,
            "Low %": f"{probs.get('Rendah', 0)*100:.1f}",
            "Moderate %": f"{probs.get('Sedang', 0)*100:.1f}",
            "High %": f"{probs.get('Tinggi', 0)*100:.1f}",
            "Very High %": f"{probs.get('Sangat Tinggi', 0)*100:.1f}",
        }
        if not any(h["Date"] == tgt for h in st.session_state.history):
            st.session_state.history.insert(0, entry)

# Render Prediction Results
if st.session_state.result and not st.session_state.running:
    res = st.session_state.result
    tgt_date = res.get("target_date", "")

    if res.get("status") == "SUCCESS":
        pred = res.get("predicted_class_name", "")
        disp_pred = RISK_DISPLAY_NAME.get(pred, pred)
        rc = _RISK_CSS.get(pred, "rc-rendah")
        probs = res.get("probabilities", {})

        st.markdown(f"""
        <div class="result-wrap">
            <div class="result-eyebrow">Prediction Result for {tgt_date}</div>
            <div class="result-class {rc}">{disp_pred}</div>
        </div>
        """, unsafe_allow_html=True)

        cards_html = ""
        for cls in CLASS_NAMES:
            p = probs.get(cls, 0.0)
            is_active = "active" if cls == pred else ""
            disp_cls = RISK_DISPLAY_NAME.get(cls, cls)
            cards_html += f"""
            <div class="prob-card {is_active}">
                <div class="prob-card-lbl">{disp_cls}</div>
                <div class="prob-card-val">{p*100:.1f}%</div>
                <div class="pbar-bg"><div class="pbar-fill" style="width:{p*100:.1f}%"></div></div>
            </div>"""
        st.markdown(f'<div class="prob-grid">{cards_html}</div>', unsafe_allow_html=True)

        csv_path = st.session_state.csv_path
        df_feat = None
        if csv_path and os.path.exists(csv_path):
            try:
                df_feat = pd.read_csv(csv_path)
            except Exception:
                pass

        if df_feat is not None and not df_feat.empty:
            last = df_feat.iloc[-1]
            d1 = str(last.get("date", "D-1"))

            st.markdown(
                f'<div class="indices-label">Latest Atmospheric Conditions Used ({d1})</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class="indices-grid">
                <div class="idx-card">
                    <div class="idx-lbl">Rainfall</div>
                    {_render_idx(last.get('rr'), ' mm')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Average Air Temperature</div>
                    {_render_idx(last.get('tavg'), ' °C')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Relative Humidity</div>
                    {_render_idx(last.get('rh'), ' %')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">CIN (Convective Inhibition)</div>
                    {_render_idx(last.get('cin'), ' J/kg')}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">K-Index</div>
                    {_render_idx(last.get('kindex'))}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Lifted Index (LI)</div>
                    {_render_idx(last.get('li'))}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">Total Totals (TT)</div>
                    {_render_idx(last.get('tt'))}
                </div>
                <div class="idx-card">
                    <div class="idx-lbl">SWEAT Index</div>
                    {_render_idx(last.get('sweat'))}
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View Data from Last 7 Days Used"):
                st.dataframe(df_feat, use_container_width=True, hide_index=True)

            col_dl, _ = st.columns([1, 3])
            with col_dl:
                with open(csv_path, "rb") as fh:
                    st.download_button(
                        "Download CSV",
                        data=fh,
                        file_name=f"features_{tgt_date}.csv",
                        mime="text/csv",
                    )
        else:
            st.info("Atmospheric parameter details are not available for this date")

    else:
        reasons = res.get("reasons", ["Unknown reason."])
        st.error("Prediction cannot be performed for this date:")
        for r in reasons:
            st.markdown(f"<small>• {r}</small>", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("Try Another Date", key="btn_reset"):
            st.session_state.result = None
            st.session_state.target_date = None
            st.session_state.csv_path = None
            st.rerun()

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: HISTORY ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="history" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
    <div class="s-eyebrow">History</div>
    <div class="s-title">Prediction History</div>
    <div class="s-body">List of predictions viewed during this session. History will reset if the page is closed or refreshed</div>
""", unsafe_allow_html=True)

hist = st.session_state.history
if hist:
    st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)
else:
    st.markdown(
        '<div class="empty-state">No predictions performed yet. Select a date above to start.</div>',
        unsafe_allow_html=True,
    )

st.markdown("</div></div>", unsafe_allow_html=True)
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ── SECTION: ABOUT ──
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<a id="about" style="display:block;position:relative;top:-58px;"></a>', unsafe_allow_html=True)
st.markdown("""
<div class="page-section">
<div class="page-section-inner">
<div class="s-eyebrow">About</div>
<div class="s-title">What is AetherSense?</div>
<p class="about-desc">
AetherSense is an atmospheric-based flood risk assessment system developed for Padang City. It evaluates localized flood vulnerability by analyzing multi-day surface meteorological observations alongside upper-air thermodynamic instability indices to provide automated, data-driven classifications
</p>
<hr class="about-divider">
<div class="about-section-label">How It Works</div>
<div class="about-steps-grid">
<div class="about-step-card">
<div class="about-step-idx">01</div>
<div class="about-step-title">Data Collection</div>
<p class="about-step-desc">Surface weather and upper-air observations</p>
</div>
<div class="about-step-card">
<div class="about-step-idx">02</div>
<div class="about-step-title">Atmospheric Processing</div>
<p class="about-step-desc">Atmospheric indices and the 7-day input sequence are prepared</p>
</div>
<div class="about-step-card">
<div class="about-step-idx">03</div>
<div class="about-step-title">Risk Assessment</div>
<p class="about-step-desc">The LSTM model generates the flood-risk classification</p>
</div>
</div>
<hr class="about-divider">
<div class="about-section-label">Model Inputs</div>
<div class="about-inputs-grid">
<div class="about-input-card">
<div class="about-input-header">Surface Weather</div>
<ul class="about-var-list">
<li class="about-var-item">
<span class="about-var-code">RR</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">Daily Rainfall</span>
</li>
<li class="about-var-item">
<span class="about-var-code">Tavg</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">Average Air Temperature</span>
</li>
<li class="about-var-item">
<span class="about-var-code">RH</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">Relative Humidity</span>
</li>
</ul>
</div>
<div class="about-input-card">
<div class="about-input-header">Atmospheric Indices</div>
<ul class="about-var-list">
<li class="about-var-item">
<span class="about-var-code">CIN</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">Convective Inhibition</span>
</li>
<li class="about-var-item">
<span class="about-var-code">K-Index</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">K-Index</span>
</li>
<li class="about-var-item">
<span class="about-var-code">LI</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">Lifted Index</span>
</li>
<li class="about-var-item">
<span class="about-var-code">TT</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">Total Totals Index</span>
</li>
<li class="about-var-item">
<span class="about-var-code">SWEAT</span>
<span class="about-var-sep">—</span>
<span class="about-var-name">SWEAT Index</span>
</li>
</ul>
</div>
</div>
<hr class="about-divider">
<div class="about-section-label">Data Sources</div>
<div class="about-sources-grid">
<div class="about-source-card">
<div class="about-source-name">Ogimet</div>
<p class="about-source-desc">Daily surface weather observations</p>
</div>
<div class="about-source-card">
<div class="about-source-name">Wyoming Upper Air</div>
<p class="about-source-desc">Upper-air sounding observations</p>
</div>
<div class="about-source-card">
<div class="about-source-name">SounderPy</div>
<p class="about-source-desc">Atmospheric index processing</p>
</div>
</div>
<hr class="about-divider">
<div class="about-section-label">Model</div>
<div class="about-model-grid">
<div class="about-model-card">
<div class="about-spec-lbl">Architecture</div>
<div class="about-spec-val">Multi-layer LSTM</div>
</div>
<div class="about-model-card">
<div class="about-spec-lbl">Input Window</div>
<div class="about-spec-val">7 Days</div>
</div>
<div class="about-model-card">
<div class="about-spec-lbl">Risk Categories</div>
<div class="about-spec-val">Low · Moderate · High · Very High</div>
</div>
</div>
<details class="about-tech-details">
<summary class="about-tech-summary"><span>Technical Details</span><span class="about-tech-toggle">+</span></summary>
<div class="about-tech-content">
<div class="about-tech-grid">
<div class="about-tech-col">
<div class="about-tech-lbl">Historical Dataset</div>
<div class="about-tech-val">Longitudinal meteorological observations spanning 2017–2024 from BMKG Minangkabau Station (WMO ID 96163)</div>
</div>
<div class="about-tech-col">
<div class="about-tech-lbl">SHARPpy Integration</div>
<div class="about-tech-val">SounderPy delegates upper-air stability parameters (LI, TT, K-Index, SWEAT) directly to internal SHARPpy calculation modules (<code>sounderpy.SHARPPYMAIN</code>)</div>
</div>
<div class="about-tech-col">
<div class="about-tech-lbl">Preprocessing & Imputation</div>
<div class="about-tech-val">Surface metrics apply calendar-aware continuity, missing sounding indices are imputed using 2017–2024 calendar-month medians</div>
</div>
<div class="about-tech-col">
<div class="about-tech-lbl">Scaling & Serialization</div>
<div class="about-tech-val">Trained <code>MinMaxScaler</code> applied to 8-variable sequences across the 7-day look-back window, executed via Keras (<code>model_final_4_class.keras</code>)</div>
</div>
</div>
</div>
</details>
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
