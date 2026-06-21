from __future__ import annotations

import base64
import html
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_ntl_pipeline
from src.data_ingestion import inspect_uploaded_dataset, prepare_manual_mapping_dataset
from src.forecasting import build_operational_audit, generate_forecasts
from src.gemini_assistant import ask_gemini
from src.assistant_context import build_platform_context
from src.building_risk_lab import generate_building_consumption_dataset, analyze_building_dataset, read_and_standardize_building_file, load_building_outputs
from src.risk_scoring import _effective_tariff_lek_per_kwh
from src import workspace

try:
    import folium
    from folium.plugins import Fullscreen, HeatMap, MarkerCluster, MiniMap
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "selected_smart_grid_theft_sample.csv"
MANUAL_STANDARDIZED_PATH = PROJECT_ROOT / "data" / "raw" / "manual_mapped_upload.csv"
CASE_FILE = OUTPUT_DIR / "inspection_cases.csv"
BUILDING_OUTPUT_DIR = OUTPUT_DIR / "building_lab"

st.set_page_config(
    page_title="EnergyShield AI | OSHEE NTL Platform",
    page_icon="ES",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# TailAdmin-inspired visual shell
# -----------------------------------------------------------------------------
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <style>
        :root{
            --font-main:'Outfit', sans-serif;
            --bg:#f9fafb;
            --panel:#ffffff;
            --panel-soft:#fcfcfd;
            --ink:#101828;
            --muted:#667085;
            --muted-2:#98a2b3;
            --line:#e4e7ec;
            --line-soft:#f2f4f7;
            --brand-25:#f2f7ff;
            --brand-50:#ecf3ff;
            --brand-100:#dde9ff;
            --brand-500:#465fff;
            --brand-600:#3641f5;
            --brand-700:#2a31d8;
            --gray-25:#fcfcfd;
            --gray-50:#f9fafb;
            --gray-100:#f2f4f7;
            --gray-200:#e4e7ec;
            --gray-300:#d0d5dd;
            --gray-500:#667085;
            --gray-700:#344054;
            --gray-900:#101828;
            --success-50:#ecfdf3;
            --success-500:#12b76a;
            --success-700:#027a48;
            --warning-50:#fffaeb;
            --warning-500:#f79009;
            --warning-700:#b54708;
            --orange-50:#fff6ed;
            --orange-500:#fb6514;
            --orange-700:#c4320a;
            --error-50:#fef3f2;
            --error-500:#f04438;
            --error-700:#b42318;
            --shadow-xs:0 1px 2px 0 rgba(16,24,40,.05);
            --shadow-sm:0 1px 3px 0 rgba(16,24,40,.10), 0 1px 2px -1px rgba(16,24,40,.10);
            --shadow-md:0 4px 8px -2px rgba(16,24,40,.10), 0 2px 4px -2px rgba(16,24,40,.06);
            --radius:16px;
            --radius-sm:10px;
            --radius-xs:8px;
        }

        /* Base */
        html, body, [data-testid="stAppViewContainer"]{
            background:var(--bg);
            color:var(--ink);
            font-family:var(--font-main) !important;
        }
        *{font-family:var(--font-main) !important;}
        .main .block-container{
            max-width:1480px;
            padding:1.1rem 1.9rem 2.6rem 1.9rem;
        }
        [data-testid="stDecoration"], .stDeployButton, [data-testid="stToolbarActions"]{
            visibility:hidden !important; height:0 !important;
        }
        [data-testid="stHeader"]{
            background:rgba(249,250,251,.82);
            backdrop-filter:blur(14px);
            border-bottom:1px solid rgba(228,231,236,.72);
        }
        [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]{
            visibility:visible !important;
            display:flex !important;
            background:var(--panel) !important;
            border:1px solid var(--line) !important;
            border-radius:12px !important;
            box-shadow:var(--shadow-xs) !important;
            z-index:999999 !important;
        }

        /* Sidebar - TailAdmin model */
        [data-testid="stSidebar"]{
            background:var(--panel);
            border-right:1px solid var(--line);
            box-shadow:var(--shadow-xs);
        }
        [data-testid="stSidebar"] .block-container{padding:1.05rem .85rem 1.4rem;}
        [data-testid="stSidebar"] *{color:var(--gray-700) !important;}
        .brand-card{
            display:flex; align-items:center; gap:12px;
            padding:4px 6px 18px 6px; margin-bottom:14px;
            border-bottom:1px solid var(--line-soft);
        }
        .brand-logo{
            width:36px; height:36px; border-radius:10px;
            display:flex; align-items:center; justify-content:center;
            background:linear-gradient(180deg,var(--brand-500),var(--brand-600));
            color:#fff !important; font-size:13px; font-weight:900;
            box-shadow:0 10px 20px rgba(70,95,255,.22);
        }
        .brand-title{font-size:1.02rem; line-height:1.1; font-weight:800; letter-spacing:-.02em; color:var(--gray-900) !important;}
        .brand-subtitle{font-size:.72rem; color:var(--gray-500) !important; margin-top:2px; font-weight:600;}
        .side-section{
            margin:14px 8px 8px;
            font-size:.70rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase;
            color:var(--muted-2) !important;
        }
        [data-testid="stSidebar"] .stButton{margin:0 0 3px 0;}
        [data-testid="stSidebar"] .stButton > button{
            width:100%; min-height:38px; justify-content:flex-start; text-align:left;
            background:transparent; color:var(--gray-700) !important;
            border:1px solid transparent; border-radius:10px;
            box-shadow:none; padding:.58rem .75rem; font-size:.91rem; font-weight:600;
            transition:background .15s ease, color .15s ease, border-color .15s ease;
        }
        [data-testid="stSidebar"] .stButton > button:hover{
            background:var(--gray-100); color:var(--gray-900) !important; border-color:var(--gray-100);
        }
        .active-nav{
            min-height:38px; display:flex; align-items:center; gap:10px;
            padding:.58rem .75rem; margin:0 0 3px 0; border-radius:10px;
            background:var(--brand-50); color:var(--brand-500) !important;
            font-weight:800; font-size:.91rem; border:1px solid var(--brand-100);
        }
        .workflow-panel{
            margin:18px 2px 0; padding:13px; border:1px solid var(--line);
            border-radius:16px; background:linear-gradient(180deg,#fff,var(--gray-25));
            box-shadow:var(--shadow-xs);
        }
        .workflow-panel .side-section{margin:0 0 9px;}
        .workflow-row{
            display:flex; align-items:center; gap:9px; margin:7px 0;
            font-size:.80rem; font-weight:600; color:var(--gray-700) !important;
        }
        .workflow-num{
            width:22px; height:22px; flex:0 0 22px; border-radius:8px;
            display:inline-flex; align-items:center; justify-content:center;
            background:var(--brand-50); color:var(--brand-500) !important;
            font-size:.74rem; font-weight:800;
        }

        /* TailAdmin header/topbar */
        .topbar{
            display:grid; grid-template-columns:minmax(0,1fr) auto; gap:18px; align-items:center;
            margin-bottom:18px;
        }
        .page-kicker{
            display:inline-flex; align-items:center; gap:6px;
            color:var(--brand-500); background:var(--brand-50); border:1px solid var(--brand-100);
            font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase;
            padding:5px 9px; border-radius:999px; margin-bottom:8px;
        }
        .page-title{
            margin:0; color:var(--gray-900); font-size:1.86rem; line-height:1.12;
            font-weight:800; letter-spacing:-.03em;
        }
        .page-subtitle{
            margin-top:6px; max-width:980px; color:var(--gray-500);
            font-size:.94rem; line-height:1.48; font-weight:400;
        }
        .topbar-actions{
            display:flex; align-items:center; gap:10px; justify-content:flex-end;
        }
        .user-chip{
            display:flex; align-items:center; gap:9px;
            background:var(--panel); border:1px solid var(--line);
            border-radius:999px; padding:6px 10px 6px 6px;
            box-shadow:var(--shadow-xs); color:var(--gray-700) !important;
            font-size:.82rem; font-weight:700; white-space:nowrap;
        }
        .avatar-dot{
            width:28px; height:28px; border-radius:999px;
            display:inline-flex; align-items:center; justify-content:center;
            background:var(--brand-500); color:#fff !important; font-weight:800; font-size:.74rem;
        }

        /* Cards */
        .hero-grid{display:grid; grid-template-columns:1.35fr .82fr; gap:18px; margin-bottom:18px;}
        .hero-panel, .soft-card, .field-board, .forecast-card, .assistant-box, .report-summary, .action-card{
            background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
            box-shadow:var(--shadow-xs);
        }
        .hero-panel{padding:22px;}
        .hero-panel h2{margin:0; color:var(--gray-900); font-size:1.25rem; font-weight:800; letter-spacing:-.02em;}
        .hero-panel p{margin:8px 0 0; color:var(--gray-500); line-height:1.55; font-size:.94rem;}
        .soft-card{padding:16px 18px; margin-bottom:14px;}
        .soft-title{font-size:1rem; font-weight:800; color:var(--gray-900); margin-bottom:4px;}
        .soft-text{font-size:.89rem; color:var(--gray-500); line-height:1.52;}

        /* Revenue loss indicator (special treatment) */
        .loss-hero{
            background:linear-gradient(135deg,#fff5f5 0%,#ffffff 62%);
            border:1px solid #fee2e2; border-left:5px solid #ef4444;
            border-radius:18px; padding:20px 22px; margin:4px 0 16px;
            box-shadow:0 1px 2px rgba(16,24,40,.04), 0 14px 32px rgba(239,68,68,.08);
        }
        .loss-hero-head{display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:14px;}
        .loss-flag{font-size:.72rem; font-weight:900; letter-spacing:.08em; color:#b91c1c; background:#fee2e2; padding:5px 11px; border-radius:999px;}
        .loss-window{font-size:.78rem; color:var(--gray-500); font-weight:600;}
        .loss-hero-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:14px;}
        .loss-cell{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:14px 16px;}
        .loss-cell-label{font-size:.78rem; color:var(--gray-500); font-weight:600; margin-bottom:6px;}
        .loss-cell-value{font-size:1.62rem; font-weight:800; color:var(--gray-900); letter-spacing:-.02em; line-height:1.15;}
        .loss-cell-sub{font-size:.76rem; color:var(--gray-500); font-weight:600; margin-top:5px;}
        .loss-cell.expected{border-left:3px solid #2563eb;}
        .loss-cell.actual{border-left:3px solid #22c55e;}
        .loss-cell.bad{border-color:#fecaca; background:#fef2f2; border-left:3px solid #ef4444;}
        .loss-cell.bad .loss-cell-value{color:#dc2626;}
        .loss-cell.bad .loss-cell-sub{color:#b91c1c; font-weight:700;}
        .loss-bar{height:11px; background:#eef2f7; border-radius:999px; overflow:hidden; margin:16px 0 8px;}
        .loss-bar-fill{height:100%; background:linear-gradient(90deg,#f87171,#dc2626); border-radius:999px;}
        .loss-bar-legend{display:flex; justify-content:space-between; flex-wrap:wrap; gap:6px; font-size:.77rem; color:var(--gray-500); font-weight:600;}
        @media(max-width:760px){ .loss-hero-grid{grid-template-columns:1fr;} }

        /* Metrics as TailAdmin cards */
        [data-testid="stMetric"]{
            background:var(--panel); border:1px solid var(--line); border-radius:16px;
            padding:16px 18px; box-shadow:var(--shadow-xs); min-height:104px;
        }
        [data-testid="stMetric"] label{
            color:var(--gray-500) !important; font-size:.82rem !important; font-weight:600 !important;
        }
        [data-testid="stMetricValue"]{
            color:var(--gray-900); font-weight:800; font-size:1.85rem !important; letter-spacing:-.025em;
        }
        [data-testid="stMetricDelta"]{font-weight:700 !important;}

        /* Native widgets */
        .stButton > button[kind="primary"], .stDownloadButton > button{
            background:var(--brand-500) !important; border:1px solid var(--brand-500) !important;
            color:#fff !important; border-radius:12px !important; font-weight:700 !important;
            box-shadow:0 8px 18px rgba(70,95,255,.18) !important;
        }
        .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover{
            background:var(--brand-600) !important; border-color:var(--brand-600) !important;
        }
        .stButton > button:not([kind="primary"]){border-radius:12px !important; border-color:var(--line) !important; box-shadow:var(--shadow-xs) !important;}
        [data-baseweb="input"], [data-baseweb="select"], [data-baseweb="textarea"], [data-testid="stFileUploaderDropzone"]{
            border-radius:12px !important; border-color:var(--line) !important; background:var(--panel) !important;
        }
        [data-testid="stFileUploaderDropzone"]{padding:18px !important;}
        .stTabs [data-baseweb="tab-list"]{gap:8px; flex-wrap:wrap; border-bottom:1px solid var(--line);}
        .stTabs [data-baseweb="tab"]{
            background:transparent; border-radius:10px 10px 0 0; padding:10px 14px; font-weight:700; color:var(--gray-500);
        }
        .stTabs [aria-selected="true"]{color:var(--brand-500) !important; background:var(--brand-50) !important;}
        div[data-testid="stDataFrame"]{
            border-radius:16px; overflow:hidden; border:1px solid var(--line); box-shadow:var(--shadow-xs); background:var(--panel);
        }
        div[data-testid="stExpander"]{
            border:1px solid var(--line) !important; border-radius:16px !important; background:var(--panel) !important;
            box-shadow:var(--shadow-xs) !important;
        }
        hr{border-color:var(--line-soft) !important;}
        h1,h2,h3,h4{letter-spacing:-.02em; color:var(--gray-900);}

        /* Status, risk, alert cards */
        .status-card{border-radius:16px; padding:14px 16px; border:1px solid var(--line); background:var(--panel); margin-bottom:14px; box-shadow:var(--shadow-xs);}
        .status-good{background:var(--success-50); border-color:#abefc6; color:#054f31 !important;}
        .status-warning{background:var(--warning-50); border-color:#fedf89; color:#7a2e0e !important;}
        .status-bad{background:var(--error-50); border-color:#fecdca; color:#7a271a !important;}
        .risk-pill{display:inline-flex; align-items:center; gap:7px; border-radius:999px; padding:5px 10px; font-weight:800; font-size:.80rem; border:1px solid transparent;}
        .risk-low{color:var(--success-700); background:var(--success-50); border-color:#abefc6;}
        .risk-medium{color:var(--warning-700); background:var(--warning-50); border-color:#fedf89;}
        .risk-high{color:var(--orange-700); background:var(--orange-50); border-color:#fddcab;}
        .risk-critical{color:var(--error-700); background:var(--error-50); border-color:#fecdca;}
        .alert-box{border-radius:16px; padding:16px; margin:14px 0; border:1px solid var(--line); border-left:5px solid var(--brand-500); background:var(--panel); box-shadow:var(--shadow-xs);}
        .alert-box.status-good{border-left-color:var(--success-500);}
        .alert-box.status-warning{border-left-color:var(--warning-500);}
        .alert-box.status-bad{border-left-color:var(--error-500);}
        .alert-box strong{display:block; font-size:.98rem; margin-bottom:5px; color:inherit;}

        /* Operational blocks */
        .action-list{display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px;}
        .action-card{padding:16px; min-height:96px;}
        .action-card strong{display:block; margin-bottom:6px; color:var(--gray-900); font-size:.94rem; font-weight:800;}
        .action-card span{font-size:.85rem; color:var(--gray-500); line-height:1.44;}
        .field-board{padding:16px;}
        .field-card{background:var(--gray-50); border:1px solid var(--line); border-radius:12px; padding:13px; margin:9px 0;}
        .forecast-card{padding:16px; margin-bottom:14px;}
        .assistant-box{padding:16px;}
        .assistant-suggestion{display:inline-block; margin:4px 5px 4px 0; padding:7px 11px; border-radius:999px; background:var(--brand-50); color:var(--brand-500); border:1px solid var(--brand-100); font-weight:700; font-size:.82rem;}
        .report-summary{padding:18px; white-space:pre-line; font-size:.91rem; line-height:1.55; color:var(--gray-700);}
        .small-muted{color:var(--gray-500); font-size:.86rem; line-height:1.48;}
        .block-card{background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:var(--shadow-xs); margin-bottom:14px;}

        /* Plotly containers */
        [data-testid="stPlotlyChart"]{
            background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:8px;
            box-shadow:var(--shadow-xs);
        }



        /* Professional dark operations sidebar, inspired by the supplied template */
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg,#140904 0%,#2a0f04 47%,#120805 100%) !important;
            border-right:0 !important;
            box-shadow:14px 0 34px rgba(16,24,40,.12) !important;
            min-width:252px !important;
            max-width:252px !important;
            overflow-x:hidden !important;
        }
        section[data-testid="stSidebar"] > div{background:transparent !important; overflow-x:hidden !important;}
        section[data-testid="stSidebar"] .block-container{padding:1.05rem .95rem 1.2rem .95rem !important; overflow-x:hidden !important;}
        [data-testid="stSidebar"] *{color:rgba(255,255,255,.68) !important;}
        [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"]{
            display:none !important; visibility:hidden !important; width:0 !important; height:0 !important;
        }
        .brand-card{
            border-bottom:1px solid rgba(255,255,255,.08) !important;
            padding:4px 4px 18px 4px !important;
            margin-bottom:18px !important;
        }
        .brand-logo{
            width:42px !important; height:42px !important; border-radius:13px !important;
            background:linear-gradient(180deg,#ffd166 0%,#f59e0b 100%) !important;
            color:#2b1307 !important; font-weight:900 !important;
            box-shadow:0 12px 28px rgba(245,158,11,.26) !important;
        }
        .brand-title{color:#fff7ed !important; font-size:1.06rem !important; font-weight:900 !important;}
        .brand-subtitle{color:rgba(255,255,255,.48) !important; font-size:.72rem !important; font-weight:600 !important;}
        .side-section{
            margin:16px 8px 9px !important;
            color:rgba(255,255,255,.33) !important;
            letter-spacing:.13em !important;
        }
        [data-testid="stSidebar"] .stButton{margin:0 0 5px 0 !important;}
        [data-testid="stSidebar"] .stButton > button{
            min-height:43px !important; width:100% !important;
            border-radius:13px !important; padding:.66rem .76rem !important;
            background:transparent !important; border:1px solid transparent !important;
            color:rgba(255,255,255,.62) !important; box-shadow:none !important;
            font-size:.91rem !important; font-weight:650 !important;
            letter-spacing:0 !important; white-space:nowrap !important; overflow:hidden !important;
            text-overflow:ellipsis !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover{
            background:rgba(255,255,255,.07) !important;
            color:#ffffff !important; border-color:rgba(255,255,255,.08) !important;
            transform:translateX(1px);
        }
        .active-nav{
            min-height:43px !important; display:flex !important; align-items:center !important; gap:11px !important;
            padding:.66rem .76rem !important; margin:0 0 5px 0 !important;
            border-radius:13px !important;
            background:rgba(255,255,255,.10) !important;
            color:#ffffff !important; border:1px solid rgba(255,255,255,.10) !important;
            box-shadow:inset 3px 0 0 #fbbf24, 0 12px 24px rgba(0,0,0,.14) !important;
            font-weight:800 !important; font-size:.91rem !important;
            white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
        }
        .nav-ico{width:21px; min-width:21px; display:inline-flex; align-items:center; justify-content:center; color:#fbbf24 !important; font-size:1rem;}
        .workflow-panel{
            margin:18px 0 0 !important; padding:14px !important;
            background:rgba(255,255,255,.055) !important;
            border:1px solid rgba(255,255,255,.085) !important;
            border-radius:18px !important; box-shadow:none !important;
        }
        .workflow-panel .side-section{margin:0 0 10px !important; color:rgba(255,255,255,.34) !important;}
        .workflow-row{color:rgba(255,255,255,.62) !important; font-size:.80rem !important;}
        .workflow-num{background:rgba(251,191,36,.16) !important; color:#fbbf24 !important; border:1px solid rgba(251,191,36,.20);}
        .main .block-container{padding-left:2.2rem !important; padding-right:2.2rem !important;}
        [data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#f8fafc 0%,#f3f4f6 100%) !important;}
        .topbar{padding-top:.15rem;}
        .soft-card, .hero-panel, .field-board, .forecast-card, .assistant-box, .report-summary, .action-card, [data-testid="stMetric"], [data-testid="stPlotlyChart"], div[data-testid="stDataFrame"]{
            border-color:#edf0f5 !important;
            box-shadow:0 1px 2px rgba(16,24,40,.04), 0 10px 28px rgba(16,24,40,.045) !important;
        }
        div[data-testid="stDataFrame"]{max-width:100%;}
        [data-testid="stHorizontalBlock"]{max-width:100%;}
        body, html, [data-testid="stAppViewContainer"]{overflow-x:hidden !important;}

        @media(max-width:1050px){
            .topbar{grid-template-columns:1fr; align-items:start;}
            .topbar-actions{justify-content:flex-start; flex-wrap:wrap;}
            .hero-grid{grid-template-columns:1fr;}
            .main .block-container{padding-left:1rem; padding-right:1rem;}
        }
        @media(max-width:640px){
            .page-title{font-size:1.48rem;}
            .page-subtitle{font-size:.88rem;}
            [data-testid="stMetricValue"]{font-size:1.45rem !important;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Plotly visual defaults only; model/data logic stays unchanged.
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#465fff", "#12b76a", "#f79009", "#f04438", "#7a5af8", "#0ba5ec"]


# Final professional UI overrides: clean sidebar navigation and file-uploader text.
st.markdown(
    """
    <style>
        /* Hide Streamlit sidebar collapse control safely. It was exposing the raw material icon text. */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"],
        button[title="Close sidebar"],
        button[title="Open sidebar"],
        section[data-testid="stSidebar"] [data-testid*="Collapse"],
        section[data-testid="stSidebar"] [data-testid*="collapse"],
        section[data-testid="stSidebar"] [data-testid*="collapsed"]{
            display:none !important;
            visibility:hidden !important;
            width:0 !important;
            height:0 !important;
            opacity:0 !important;
            pointer-events:none !important;
        }
        /* Some Streamlit versions render the collapse control as the first raw button inside the sidebar. */
        section[data-testid="stSidebar"] > div:first-child > button:first-child{
            display:none !important;
            visibility:hidden !important;
        }

        /* Sidebar model: plain vertical nav, no boxed buttons. */
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg,#160803 0%,#2b0d03 52%,#130703 100%) !important;
            min-width:258px !important;
            max-width:258px !important;
            overflow-x:hidden !important;
        }
        section[data-testid="stSidebar"] .block-container{
            padding:1.1rem 1rem 1.35rem 1rem !important;
            overflow-x:hidden !important;
        }
        .brand-card{
            padding:2px 8px 20px 8px !important;
            margin:0 0 18px 0 !important;
            border-bottom:1px solid rgba(255,255,255,.10) !important;
        }
        .brand-logo{
            width:42px !important;
            height:42px !important;
            border-radius:14px !important;
            background:linear-gradient(180deg,#ffd36a 0%,#f5a400 100%) !important;
            color:#271000 !important;
            box-shadow:0 16px 30px rgba(245,164,0,.22) !important;
        }
        .brand-title{color:#fffaf1 !important; font-weight:900 !important; letter-spacing:-.02em !important;}
        .brand-subtitle{color:rgba(255,255,255,.45) !important;}
        .side-section{
            margin:20px 14px 10px !important;
            color:rgba(255,255,255,.36) !important;
            font-size:.68rem !important;
            letter-spacing:.14em !important;
        }
        section[data-testid="stSidebar"] .stButton{
            margin:1px 0 !important;
        }
        section[data-testid="stSidebar"] .stButton > button{
            min-height:42px !important;
            width:100% !important;
            justify-content:flex-start !important;
            text-align:left !important;
            background:transparent !important;
            border:none !important;
            outline:none !important;
            box-shadow:none !important;
            border-radius:0 !important;
            padding:.62rem .86rem !important;
            color:rgba(255,255,255,.58) !important;
            font-size:.92rem !important;
            font-weight:600 !important;
            line-height:1.15 !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
            transition:color .16s ease, transform .16s ease, background .16s ease !important;
        }
        section[data-testid="stSidebar"] .stButton > button p,
        section[data-testid="stSidebar"] .stButton > button span{
            color:inherit !important;
            font-size:inherit !important;
            font-weight:inherit !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover,
        section[data-testid="stSidebar"] .stButton > button:focus,
        section[data-testid="stSidebar"] .stButton > button:active{
            background:transparent !important;
            border:none !important;
            outline:none !important;
            box-shadow:none !important;
            color:#fff8ed !important;
            transform:translateX(2px) !important;
        }
        .active-nav{
            position:relative !important;
            min-height:42px !important;
            display:flex !important;
            align-items:center !important;
            gap:12px !important;
            padding:.62rem .86rem .62rem 1.05rem !important;
            margin:1px 0 !important;
            border-radius:0 !important;
            background:transparent !important;
            border:none !important;
            box-shadow:none !important;
            color:#fffaf1 !important;
            font-size:.92rem !important;
            font-weight:800 !important;
            white-space:nowrap !important;
            overflow:hidden !important;
            text-overflow:ellipsis !important;
        }
        .active-nav::before{
            content:"";
            position:absolute;
            left:0;
            top:11px;
            bottom:11px;
            width:3px;
            border-radius:999px;
            background:#f8b800;
            box-shadow:0 0 18px rgba(248,184,0,.42);
        }
        .nav-ico{
            width:20px !important;
            min-width:20px !important;
            display:inline-flex !important;
            justify-content:center !important;
            align-items:center !important;
            color:#f8b800 !important;
            font-size:.95rem !important;
            line-height:1 !important;
        }

        /* Workflow: remove the card/box feeling; keep it as a subtle operational checklist. */
        .workflow-panel{
            margin:22px 0 0 !important;
            padding:0 8px !important;
            background:transparent !important;
            border:none !important;
            box-shadow:none !important;
            border-radius:0 !important;
        }
        .workflow-row{
            margin:8px 0 !important;
            gap:10px !important;
            color:rgba(255,255,255,.50) !important;
            font-size:.79rem !important;
            line-height:1.25 !important;
        }
        .workflow-num{
            width:18px !important;
            height:18px !important;
            min-width:18px !important;
            border-radius:999px !important;
            background:rgba(248,184,0,.13) !important;
            color:#f8b800 !important;
            border:1px solid rgba(248,184,0,.18) !important;
            font-size:.66rem !important;
        }

        /* Fix duplicated/overlapping text inside the native file uploader button. */
        [data-testid="stFileUploader"] button{
            position:relative !important;
            min-width:112px !important;
            height:42px !important;
            border-radius:12px !important;
            overflow:hidden !important;
        }
        [data-testid="stFileUploader"] button p,
        [data-testid="stFileUploader"] button span{
            visibility:hidden !important;
            opacity:0 !important;
        }
        [data-testid="stFileUploader"] button::after{
            content:"Choose file";
            position:absolute;
            inset:0;
            display:flex;
            align-items:center;
            justify-content:center;
            color:#101828 !important;
            font-weight:700 !important;
            font-size:.86rem !important;
            visibility:visible !important;
            opacity:1 !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] p{
            color:#667085 !important;
        }

        /* Keep all main-page secondary buttons readable and not duplicated. */
        [data-testid="stAppViewContainer"] main .stButton > button:not([kind="primary"]){
            background:#ffffff !important;
            border:1px solid #d0d5dd !important;
            box-shadow:0 1px 2px rgba(16,24,40,.05) !important;
            color:#344054 !important;
        }
        [data-testid="stAppViewContainer"] main .stButton > button:not([kind="primary"]):hover{
            background:#f9fafb !important;
            border-color:#98a2b3 !important;
        }

        /* Prevent any raw material icon text from wrapping into content areas. */
        .stApp{overflow-x:hidden !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Final fixes: hide raw material icon text and use professional custom chat/map UI.
st.markdown(
    """
    <style>
        /* Stronger sidebar-header hide: prevents raw 'keyboard_double_arrow' text from showing. */
        [data-testid="stSidebarHeader"],
        [data-testid="stSidebarHeader"] *,
        section[data-testid="stSidebar"] header,
        section[data-testid="stSidebar"] header *,
        section[data-testid="stSidebar"] button[aria-label*="sidebar" i],
        section[data-testid="stSidebar"] button[title*="sidebar" i],
        section[data-testid="stSidebar"] button[aria-label*="Collapse" i],
        section[data-testid="stSidebar"] button[title*="Collapse" i]{
            display:none !important;
            visibility:hidden !important;
            opacity:0 !important;
            width:0 !important;
            height:0 !important;
            max-height:0 !important;
            overflow:hidden !important;
            pointer-events:none !important;
        }

        /* Hide native chat avatars if any legacy chat messages render. */
        [data-testid="stChatMessageAvatar"],
        [data-testid="stChatMessageAvatar"] *,
        .stChatMessage [data-testid="stChatMessageAvatar"]{
            display:none !important;
            visibility:hidden !important;
        }
        [data-testid="stChatMessage"]{
            background:transparent !important;
            border:none !important;
        }

        /* Custom assistant conversation cards. */
        .chat-thread{display:flex; flex-direction:column; gap:14px; margin:14px 0 10px;}
        .chat-row{display:flex; gap:12px; align-items:flex-start; width:100%;}
        .chat-row.user{justify-content:flex-end;}
        .chat-row.assistant{justify-content:flex-start;}
        .chat-avatar{
            width:34px; height:34px; border-radius:12px; flex:0 0 34px;
            display:flex; align-items:center; justify-content:center;
            font-weight:900; font-size:.75rem; letter-spacing:.02em;
            box-shadow:0 8px 22px rgba(16,24,40,.08);
        }
        .chat-avatar.ai{background:linear-gradient(180deg,#fbbf24,#f97316); color:#281004 !important;}
        .chat-avatar.you{background:#465fff; color:#ffffff !important;}
        .chat-bubble{
            max-width:78%; padding:13px 15px; border-radius:16px;
            background:#ffffff; border:1px solid #e4e7ec;
            box-shadow:0 1px 2px rgba(16,24,40,.04), 0 10px 22px rgba(16,24,40,.035);
            color:#101828 !important; font-size:.93rem; line-height:1.55;
            overflow-wrap:anywhere;
        }
        .chat-row.user .chat-bubble{background:#eef4ff; border-color:#dbe7ff;}
        .chat-bubble ul{margin:.45rem 0 .2rem 1.1rem;}
        .chat-bubble li{margin:.18rem 0;}
        .map-note{
            padding:12px 14px; border-radius:14px; margin:0 0 12px;
            background:#fff7ed; border:1px solid #fed7aa; color:#7c2d12 !important;
            font-size:.88rem; line-height:1.45;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Final theme override: light-blue (Notion-style) sidebar + restore Material Symbols icon font.
st.markdown(
    """
    <style>
        /* Restore the Material Symbols icon font. The global font override turned ligature icons
           (e.g. the expander chevron) into raw text like "arrow_right" that overlapped labels. */
        span[data-testid="stIconMaterial"], [data-testid="stIconMaterial"],
        [data-testid="stExpanderToggleIcon"],
        .material-icons, .material-icons-outlined, .material-icons-round, .material-icons-sharp,
        .material-symbols-outlined, .material-symbols-rounded, .material-symbols-sharp{
            font-family:'Material Symbols Outlined','Material Symbols Rounded','Material Icons','Material Icons Outlined' !important;
            font-weight:normal !important; font-style:normal !important; letter-spacing:normal !important;
            text-transform:none !important; white-space:nowrap !important; word-wrap:normal !important;
            direction:ltr !important; -webkit-font-feature-settings:'liga'; font-feature-settings:'liga';
            -webkit-font-smoothing:antialiased;
        }
        /* Keep the expander summary label and chevron from overlapping. */
        details[data-testid="stExpander"] summary{display:flex !important; align-items:center !important; gap:8px !important;}
        /* Some Streamlit builds render the expander chevron as raw ligature text ("arrow_right")
           that overlapped the label (e.g. "Answered requests history"). Hide that toggle glyph;
           the expander header stays fully clickable. */
        details[data-testid="stExpander"] summary [data-testid="stIconMaterial"],
        [data-testid="stExpanderToggleIcon"]{display:none !important;}

        /* Light-blue Notion-style sidebar */
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg,#eef4fc 0%,#e6eefa 55%,#eaf3fd 100%) !important;
            border-right:1px solid #d7e3f4 !important;
            box-shadow:8px 0 28px rgba(37,99,235,.06) !important;
        }
        section[data-testid="stSidebar"] > div{background:transparent !important;}
        [data-testid="stSidebar"] *{color:#33415c !important;}
        section[data-testid="stSidebar"] .stButton > button{
            color:#43506b !important; font-weight:600 !important;
            border-radius:10px !important; padding:.62rem .82rem !important;
        }
        section[data-testid="stSidebar"] .stButton > button p,
        section[data-testid="stSidebar"] .stButton > button span{color:inherit !important;}
        section[data-testid="stSidebar"] .stButton > button:hover,
        section[data-testid="stSidebar"] .stButton > button:focus,
        section[data-testid="stSidebar"] .stButton > button:active{
            background:rgba(37,99,235,.09) !important; color:#1d4ed8 !important; transform:none !important;
        }
        .brand-card{border-bottom:1px solid #d7e3f4 !important;}
        .brand-logo{
            background:linear-gradient(180deg,#3b82f6 0%,#2563eb 100%) !important;
            color:#ffffff !important; box-shadow:0 12px 26px rgba(37,99,235,.28) !important;
        }
        .brand-title{color:#101828 !important;}
        .brand-subtitle{color:#64748b !important;}
        .side-section{color:#7c8aa5 !important;}
        .workflow-panel .side-section{color:#7c8aa5 !important;}
        .active-nav{
            background:rgba(37,99,235,.10) !important; color:#1d4ed8 !important;
            border-radius:10px !important; border:1px solid rgba(37,99,235,.16) !important;
        }
        .active-nav::before{background:#2563eb !important; box-shadow:0 0 14px rgba(37,99,235,.40) !important;}
        .nav-ico{color:#2563eb !important;}
        .workflow-row{color:#5b6b86 !important;}
        .workflow-num{
            background:rgba(37,99,235,.12) !important; color:#1d4ed8 !important;
            border:1px solid rgba(37,99,235,.20) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# Background photo (supplied by the user). Loaded at runtime so the source stays clean: the asset is
# bootstrapped into the repo on first run, then embedded as a data URI behind a light readable overlay.
_BG_ASSET = PROJECT_ROOT / "dashboard" / "assets" / "app_background.png"
_BG_SOURCE_NAME = "c__Users_mateo_AppData_Roaming_Cursor_User_workspaceStorage_empty-window_images_image-2aa77357-3fde-4384-a5f4-aaf7f298870a.png"


@st.cache_data(show_spinner=False)
def _app_background_uri() -> str | None:
    try:
        if not _BG_ASSET.exists():
            _BG_ASSET.parent.mkdir(parents=True, exist_ok=True)
            search_root = Path.home() / ".cursor" / "projects"
            if search_root.exists():
                match = next(search_root.glob(f"*/assets/{_BG_SOURCE_NAME}"), None)
                if match and match.exists():
                    shutil.copyfile(match, _BG_ASSET)
        if _BG_ASSET.exists():
            encoded = base64.b64encode(_BG_ASSET.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{encoded}"
    except Exception:
        return None
    return None


_bg_uri = _app_background_uri()
if _bg_uri:
    st.markdown(
        f"""
        <style>
            [data-testid="stAppViewContainer"]{{
                background-image:linear-gradient(rgba(244,249,255,.86), rgba(238,245,255,.93)), url("{_bg_uri}") !important;
                background-size:cover !important; background-position:center top !important;
                background-attachment:fixed !important; background-repeat:no-repeat !important;
            }}
            [data-testid="stHeader"]{{background:transparent !important; backdrop-filter:blur(7px);}}
            .main .block-container{{background:transparent !important;}}
        </style>
        """,
        unsafe_allow_html=True,
    )

COLUMN_LABELS = {
    "customer_id": "Customer ID", "area_id": "Area", "transformer_id": "Transformer",
    "risk_score": "Risk Score", "risk_level": "Risk Level", "ai_anomaly_score": "AI Anomaly Score",
    "ai_combined_score": "AI Combined Score", "fraud_probability": "Fraud Probability",
    "historical_deviation_score": "Historical Deviation", "peer_deviation_score": "Similar Profile Deviation",
    "geographic_risk_score": "Geographic Risk", "sudden_flags_score": "Sudden Behavior Score",
    "context_deviation_score": "Expected Consumption Gap", "estimated_loss_all_30d": "Estimated 30-Day Loss (Lek)",
    "estimated_missing_kwh_30d": "Estimated Missing Energy (kWh)", "main_reason": "Main Reason",
    "alert_explanation": "Alert Explanation", "recommended_action": "Recommended Action",
    "priority_rank": "Priority Rank", "inspection_priority_score": "Inspection Priority Score",
    "area_risk_score": "Area Risk Score", "area_risk_level": "Area Risk Level", "customers": "Customers",
    "high_risk_customers": "High-Risk Customers", "critical_customers": "Critical Customers",
    "anomaly_density": "Anomaly Density", "avg_risk_score": "Average Risk Score", "latitude": "Latitude",
    "longitude": "Longitude", "customer_type": "Customer Type", "last_30_mean": "Recent 30-Day Average",
    "previous_90_mean": "Previous 90-Day Average", "peer_avg_last_30": "Similar Profile Average",
    "recent_drop_pct": "Recent Drop Percentage", "recent_spike_pct": "Recent Spike Percentage",
    "zero_days": "Zero-Consumption Days", "low_days": "Low-Consumption Days", "flatline_days": "Flatline Days",
    "sudden_drop_count": "Sudden Drop Events", "sudden_spike_count": "Sudden Spike Events",
    "expected_last_30_consumption": "Expected 30-Day Consumption", "expected_deviation_pct": "Expected Consumption Deviation",
    "weather_context_score": "Weather Context Score", "weather_mismatch_ratio": "Weather Mismatch Ratio",
    "weather_mismatch_days": "Weather Mismatch Days", "avg_temp_mean": "Average Temperature",
    "avg_heating_degree_days": "Heating Degree Days", "avg_cooling_degree_days": "Cooling Degree Days",
    "weather_demand_pressure": "Weather Demand Pressure", "weather_class": "Weather Class",
    "temp_mean": "Temperature Mean",
    "district": "District", "district_risk_score": "District Risk Score", "district_risk_level": "District Risk Level",
    "building_id": "Building ID", "unit_id": "Unit ID", "building_type": "Building Type", "location_type": "Location Type",
    "floor": "Floor", "unit_type": "Unit Type", "risk_score": "Risk Score", "fraud_probability": "Fraud Probability",
    "estimated_loss_lek_30d": "Estimated 30-Day Loss (Lek)", "high_risk_units": "High-Risk Units", "critical_units": "Critical Units",
    "avg_consumption": "Average Consumption", "median_consumption": "Median Consumption",
    "std_consumption": "Consumption Standard Deviation", "total_consumption": "Total Consumption",
    "consumption_kwh": "Consumption (kWh)", "rolling_30d_mean": "30-Day Baseline", "date": "Date",
    "payment_risk_score": "Payment Risk Score", "payment_late_count_12m": "Late Payments (12m)",
    "unpaid_bills": "Unpaid Bills", "avg_payment_delay_days": "Avg Payment Delay (days)",
    "arrears_amount_lek": "Arrears (Lek)", "disconnections_12m": "Non-Payment Disconnections (12m)",
    "months_since_last_payment": "Months Since Last Payment", "payment_method": "Payment Method",
    "account_status": "Account Status", "payment_on_time_ratio": "On-Time Payment Ratio",
    "cumulative_predicted_loss": "Cumulative Predicted Loss (Lek)",
    "forecasted_customer_risk_next_30d": "Forecasted 30-Day Risk", "forecast_priority": "Forecast Priority",
    "forecasted_area_risk_next_30d": "Forecasted Area Risk (30d)",
}
RISK_ORDER = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {"Low": "#0f7a3d", "Medium": "#a8650a", "High": "#bd4a0e", "Critical": "#b3192c"}


def pretty_label(name: str) -> str:
    return COLUMN_LABELS.get(name, str(name).replace("_", " ").replace("pct", "percentage").title())


def clean_display_df(df: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    data = df.copy()
    if columns:
        data = data[[c for c in columns if c in data.columns]]
    for col in data.select_dtypes(include=["object", "string"]).columns:
        data[col] = data[col].astype(str).str.replace("_", " ", regex=False)
    return data.rename(columns={c: pretty_label(c) for c in data.columns})


def fmt(value, decimals=1, suffix=""):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def apply_chart_style(fig, height: int | None = None):
    """Consistent executive-dashboard styling for all Plotly visuals."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", size=13, color="#344054"),
        title=dict(font=dict(size=16, color="#101828"), x=0.01, xanchor="left"),
        margin=dict(l=24, r=18, t=54, b=38),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#101828", font_color="#ffffff", bordercolor="#101828"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf0f5", zeroline=False, linecolor="#e4e7ec")
    fig.update_yaxes(showgrid=True, gridcolor="#edf0f5", zeroline=False, linecolor="#e4e7ec")
    if height:
        fig.update_layout(height=height)
    return fig


def render_chart(fig, height: int | None = None):
    st.plotly_chart(apply_chart_style(fig, height=height), use_container_width=True, config={"displayModeBar": False, "responsive": True})


def fmt_lek(value) -> str:
    """Compact Albanian Lek formatter (M/k Lek) for revenue and loss figures."""
    try:
        v = float(value)
    except Exception:
        return "N/A"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1_000_000:
        return f"{sign}{v/1_000_000:,.2f}M Lek"
    if v >= 1_000:
        return f"{sign}{v/1_000:,.0f}k Lek"
    return f"{sign}{v:,.0f} Lek"


def compute_loss_economics(customer: pd.DataFrame) -> dict:
    """Reconstruct expected vs. actually-recorded billable value from the pipeline's own figures.

    The NTL model already computes, per customer, the suspicious missing energy and its monetary
    value at the ERE/OSHEE tariff bands (loss = missing_kwh * tariff). We value the energy that was
    actually metered at the same tariff and define what OSHEE *should* have billed as:

        expected_value = actual_value + estimated_loss

    so the gap between expected and actual is exactly the suspicious loss the model flagged. This
    keeps every number consistent with ``risk_scoring.compute_final_customer_risk`` — no new
    assumptions are introduced.
    """
    df = customer.copy()
    tariff = _effective_tariff_lek_per_kwh(df)
    actual_kwh = pd.to_numeric(df.get("last_30_mean", 0), errors="coerce").fillna(0) * 30.0
    loss_value = pd.to_numeric(df.get("estimated_loss_all_30d", 0), errors="coerce").fillna(0)
    missing_kwh = pd.to_numeric(df.get("estimated_missing_kwh_30d", 0), errors="coerce").fillna(0)
    actual_value = actual_kwh * tariff
    expected_value = actual_value + loss_value
    total_expected = float(expected_value.sum())
    total_actual = float(actual_value.sum())
    total_loss = float(loss_value.sum())
    return {
        "expected": total_expected,
        "actual": total_actual,
        "loss": total_loss,
        "missing_kwh": float(missing_kwh.sum()),
        "leakage_pct": (total_loss / total_expected * 100.0) if total_expected > 0 else 0.0,
        "customers_with_loss": int((loss_value > 0).sum()),
    }


def render_loss_indicator(customer: pd.DataFrame, area: pd.DataFrame | None = None, *, compact: bool = False) -> dict:
    """Special-treatment revenue-loss indicator: expected vs. actually-recorded money, and the gap.

    ``compact=True`` renders only the headline banner (for the Admin Console); the full version
    adds an expected-vs-actual waterfall and the top loss-contributing areas.
    """
    econ = compute_loss_economics(customer)
    expected, actual, loss = econ["expected"], econ["actual"], econ["loss"]
    leakage = econ["leakage_pct"]
    bar_w = max(2.0, min(100.0, leakage))
    st.markdown(
        f"""
        <div class="loss-hero">
          <div class="loss-hero-head">
            <span class="loss-flag">REVENUE LOSS INDICATOR</span>
            <span class="loss-window">Rolling 30-day estimate · valued at ERE/OSHEE tariff</span>
          </div>
          <div class="loss-hero-grid">
            <div class="loss-cell expected">
              <div class="loss-cell-label">Expected billable revenue</div>
              <div class="loss-cell-value">{fmt_lek(expected)}</div>
              <div class="loss-cell-sub">If all metered energy were legitimate</div>
            </div>
            <div class="loss-cell actual">
              <div class="loss-cell-label">Actually recorded revenue</div>
              <div class="loss-cell-value">{fmt_lek(actual)}</div>
              <div class="loss-cell-sub">Energy that was actually metered</div>
            </div>
            <div class="loss-cell bad">
              <div class="loss-cell-label">Suspicious revenue loss</div>
              <div class="loss-cell-value">{fmt_lek(loss)}</div>
              <div class="loss-cell-sub">{leakage:.1f}% of expected revenue at risk</div>
            </div>
          </div>
          <div class="loss-bar"><div class="loss-bar-fill" style="width:{bar_w:.1f}%"></div></div>
          <div class="loss-bar-legend">
            <span>Recoverable energy if confirmed: {econ['missing_kwh']:,.0f} kWh / 30 days</span>
            <span>{econ['customers_with_loss']:,} customers contributing to the gap</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if compact:
        return econ

    left, right = st.columns([1.25, 1])
    with left:
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Expected<br>billable", "Suspicious<br>loss", "Actually<br>recorded"],
            text=[fmt_lek(expected), f"-{fmt_lek(loss)}", fmt_lek(actual)],
            textposition="outside",
            y=[expected, -loss, actual],
            connector={"line": {"color": "#cbd5e1", "width": 1.5}},
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#22c55e"}},
            totals={"marker": {"color": "#2563eb"}},
        ))
        wf.update_layout(
            title="Expected vs. actually-recorded revenue (30 days)",
            showlegend=False,
            yaxis_title="Lek",
            margin=dict(t=54, b=10, l=10, r=10),
        )
        render_chart(wf, height=340)
    with right:
        area_loss = None
        if area is not None and not area.empty and "estimated_loss_all_30d" in area.columns:
            area_loss = (
                area[["area_id", "estimated_loss_all_30d"]]
                .dropna(subset=["area_id"])
                .sort_values("estimated_loss_all_30d", ascending=False)
                .head(8)
                .iloc[::-1]
            )
        if area_loss is not None and not area_loss.empty and float(area_loss["estimated_loss_all_30d"].sum()) > 0:
            area_loss = area_loss.assign(loss_label=area_loss["estimated_loss_all_30d"].map(fmt_lek))
            bar = px.bar(
                area_loss,
                x="estimated_loss_all_30d",
                y="area_id",
                orientation="h",
                text="loss_label",
            )
            bar.update_traces(marker_color="#ef4444", textposition="outside", cliponaxis=False)
            bar.update_layout(
                title="Where the loss concentrates (top areas)",
                xaxis_title="Estimated 30-day loss (Lek)",
                yaxis_title="",
                margin=dict(t=54, b=10, l=10, r=10),
            )
            render_chart(bar, height=340)
        else:
            st.markdown(
                f'<div class="soft-card"><div class="soft-title">Recovery focus</div><div class="soft-text">'
                f'Confirming and correcting the flagged accounts would recover an estimated '
                f'<b>{fmt_lek(loss)}</b> over 30 days — about <b>{leakage:.1f}%</b> of expected revenue. '
                f'Prioritise the highest-risk customers in the register and dispatch them to field teams.'
                f'</div></div>',
                unsafe_allow_html=True,
            )
    return econ


def page_header(kicker: str, title: str, subtitle: str, chip: str | None = None):
    if chip is None:
        role = st.session_state.get("user_role")
        chip = f"OSHEE · {role}" if role else "OSHEE operations workspace"
    st.markdown(
        f"""
        <div class="topbar">
            <div>
                <div class="page-kicker">{html.escape(kicker)}</div>
                <h1 class="page-title">{html.escape(title)}</h1>
                <div class="page-subtitle">{html.escape(subtitle)}</div>
            </div>
            <div class="topbar-actions">
                <div class="user-chip"><span class="avatar-dot">OS</span>{html.escape(chip)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fmt_count(value) -> str:
    try:
        if pd.isna(value):
            return "N/A"
        return f"{int(float(value)):,}"
    except Exception:
        return str(value)


def risk_pill(level: str) -> str:
    level = str(level or "Low")
    css = level.lower() if level in RISK_ORDER else "low"
    icon = {"Low":"●", "Medium":"●", "High":"●", "Critical":"●"}.get(level, "●")
    return f'<span class="risk-pill risk-{css}">{icon} {html.escape(level)}</span>'


def alert_card(level: str, explanation: str, action: str | None = None):
    level = str(level or "Low")
    css = {"Low":"status-good", "Medium":"status-warning", "High":"status-warning", "Critical":"status-bad"}.get(level, "status-good")
    title = {"Low":"No urgent irregularity detected", "Medium":"Monitoring recommended", "High":"Field verification recommended", "Critical":"Urgent inspection required"}.get(level, "Risk signal")
    st.markdown(
        f"""
        <div class="alert-box {css}">
            <strong>{html.escape(title)}</strong>
            <div>{html.escape(str(explanation or 'No explanation available.'))}</div>
            <div style="margin-top:8px;"><b>Recommended action:</b> {html.escape(str(action or 'No action defined.'))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _outputs_have_payment_model() -> bool:
    """True when the saved risk scores already include the billing/payment-behavior signal."""
    path = OUTPUT_DIR / "customer_risk_scores.csv"
    if not path.exists():
        return False
    try:
        return "payment_risk_score" in pd.read_csv(path, nrows=0).columns
    except Exception:
        return True  # don't block on a header read issue


def _outputs_use_city_area_names() -> bool:
    """True when saved areas already use readable city names instead of legacy AREA_xx codes."""
    path = OUTPUT_DIR / "area_risk_scores.csv"
    if not path.exists():
        return True  # first-run generation path handles this
    try:
        ids = pd.read_csv(path, usecols=["area_id"])["area_id"].astype(str)
        return not ids.str.match(r"^AREA_\d+$").any()
    except Exception:
        return True  # don't block on a read issue


def ensure_outputs():
    base = OUTPUT_DIR / "customer_risk_scores.csv"
    if not base.exists():
        if DEFAULT_INPUT_PATH.exists():
            with st.spinner("Preparing analytics for the first time..."):
                run_ntl_pipeline(DEFAULT_INPUT_PATH, output_dir=OUTPUT_DIR, model_dir=MODEL_DIR, max_customers=1200, max_days=730)
        else:
            st.warning("No processed outputs are available yet. Upload a dataset in Data Intake and run NTL Detection.")
            st.stop()
        return
    # One-time auto-upgrade: rebuild outputs that predate the billing/payment-behavior model or
    # still use legacy AREA_xx codes, so the new fraud triggers, improved forecast and readable
    # city-based area names appear automatically. Falls back to existing outputs if regeneration
    # fails, so the dashboard never breaks. Attempted at most once per session.
    needs_upgrade = DEFAULT_INPUT_PATH.exists() and (
        not _outputs_have_payment_model() or not _outputs_use_city_area_names()
    )
    if needs_upgrade and not st.session_state.get("_payment_upgrade_attempted"):
        st.session_state["_payment_upgrade_attempted"] = True
        try:
            with st.spinner("Upgrading analytics (billing-behavior model + readable city area names, one time)..."):
                run_ntl_pipeline(DEFAULT_INPUT_PATH, output_dir=OUTPUT_DIR, model_dir=MODEL_DIR, max_customers=1200, max_days=730)
                # Stale auto-generated cases reference old AREA_xx codes; rebuild from the
                # refreshed priority list so case areas match the new city-based names.
                if CASE_FILE.exists():
                    CASE_FILE.unlink()
                load_outputs_cached.clear()
        except Exception as exc:
            st.info(f"Showing existing analytics. Re-run from Data Intake to refresh the model and area names. ({exc})")


@st.cache_data(show_spinner=False)
def load_outputs_cached(mtime: float | None):
    customer = pd.read_csv(OUTPUT_DIR / "customer_risk_scores.csv")
    area = pd.read_csv(OUTPUT_DIR / "area_risk_scores.csv")
    priority = pd.read_csv(OUTPUT_DIR / "inspection_priority.csv")
    transformer_path = OUTPUT_DIR / "transformer_risk_scores.csv"
    transformer = pd.read_csv(transformer_path) if transformer_path.exists() else pd.DataFrame()
    daily_path = OUTPUT_DIR / "daily_dashboard.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"]) if daily_path.exists() else pd.DataFrame()
    summary = json.loads((OUTPUT_DIR / "pipeline_summary.json").read_text()) if (OUTPUT_DIR / "pipeline_summary.json").exists() else {}
    metrics = json.loads((OUTPUT_DIR / "model_metrics.json").read_text()) if (OUTPUT_DIR / "model_metrics.json").exists() else {}
    ingestion = json.loads((OUTPUT_DIR / "ingestion_report.json").read_text()) if (OUTPUT_DIR / "ingestion_report.json").exists() else {}
    return customer, area, transformer, priority, daily, summary, metrics, ingestion


def load_outputs():
    ensure_outputs()
    mtime = (OUTPUT_DIR / "customer_risk_scores.csv").stat().st_mtime
    return load_outputs_cached(mtime)


def r2_score_manual(actual: pd.Series, predicted: pd.Series) -> float | None:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    mask = actual.notna() & predicted.notna()
    if mask.sum() < 3:
        return None
    y = actual[mask]
    yhat = predicted[mask]
    denom = float(((y - y.mean()) ** 2).sum())
    if denom <= 1e-12:
        return None
    return float(1 - ((y - yhat) ** 2).sum() / denom)


# Free-text / categorical case columns must stay string-typed. When a freshly created
# case log is written with empty notes and re-read, pandas infers an all-NaN float64
# column, and writing a text value into it later raises "Invalid value ... for dtype
# 'float64'". Normalising these columns to strings on every load prevents that crash.
CASE_TEXT_COLS = ["case_id", "status", "assigned_team", "inspection_result", "field_notes",
                  "risk_level", "area_id", "transformer_id", "main_reason", "recommended_action"]


def _normalize_case_text(cases: pd.DataFrame) -> pd.DataFrame:
    for col in CASE_TEXT_COLS:
        if col in cases.columns:
            cases[col] = cases[col].fillna("").astype(str).replace({"nan": "", "None": ""})
    return cases


def create_cases_from_priority(priority: pd.DataFrame):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    if CASE_FILE.exists():
        return _normalize_case_text(pd.read_csv(CASE_FILE))
    cols = ["priority_rank", "customer_id", "risk_score", "risk_level", "area_id", "transformer_id", "estimated_loss_all_30d", "main_reason", "recommended_action"]
    cases = priority[[c for c in cols if c in priority.columns]].head(75).copy()
    cases.insert(0, "case_id", [f"NTL-{i+1:05d}" for i in range(len(cases))])
    cases["status"] = "New"
    teams = workspace.inspector_teams() or ["Inspection Team 1"]
    cases["assigned_team"] = [teams[i % len(teams)] for i in range(len(cases))]
    cases["inspection_result"] = "Pending"
    cases["field_notes"] = ""
    cases = _normalize_case_text(cases)
    cases.to_csv(CASE_FILE, index=False)
    return cases


def save_cases(cases: pd.DataFrame):
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    cases.to_csv(CASE_FILE, index=False)


def load_optional_csv(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = OUTPUT_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except Exception:
        return pd.DataFrame()


def _valid_albania_points(df: pd.DataFrame) -> pd.DataFrame:
    """Keep map coordinates inside a conservative Albania land-oriented display envelope.

    Synthetic prototype GIS is generated around inland city centers. This filter prevents
    accidental sea/off-country points from distracting operators during demos.
    """
    if df.empty or not {"latitude", "longitude"}.issubset(df.columns):
        return df.copy()
    out = df.copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    mask = out["latitude"].between(39.60, 42.75) & out["longitude"].between(18.80, 21.10)
    # Avoid points far west of the Albanian coastline where synthetic jitter can land in the Adriatic/Ionian sea.
    mask &= ~((out["longitude"] < 19.30) & (out["latitude"].between(39.70, 42.30)))
    return out[mask].copy()




def _chat_html_message(role: str, content: str) -> str:
    safe = html.escape(str(content or ""))
    # small markdown-ish conversion for readable fallback answers
    safe = re.sub(r"^- (.*)$", r"<li>\1</li>", safe, flags=re.MULTILINE)
    if "<li>" in safe:
        safe = re.sub(r"(<li>.*</li>(?:\n<li>.*</li>)*)", r"<ul>\1</ul>", safe, flags=re.DOTALL)
    safe = safe.replace("\n\n", "<br><br>").replace("\n", "<br>")
    if role == "user":
        return f'<div class="chat-row user"><div class="chat-bubble">{safe}</div><div class="chat-avatar you">YOU</div></div>'
    return f'<div class="chat-row assistant"><div class="chat-avatar ai">AI</div><div class="chat-bubble">{safe}</div></div>'

def ensure_forecast_outputs(customer: pd.DataFrame, area: pd.DataFrame, daily: pd.DataFrame):
    # Always regenerate lightweight forecasts so UI and methodology updates are reflected immediately.
    try:
        generate_forecasts(customer, area, daily, OUTPUT_DIR)
    except Exception:
        pass
    return (
        load_optional_csv("loss_forecast.csv", parse_dates=["date"]),
        load_optional_csv("area_forecast.csv"),
        load_optional_csv("customer_forecast.csv"),
    )


def make_report_summary(customer: pd.DataFrame, area: pd.DataFrame, cases: pd.DataFrame, loss_forecast: pd.DataFrame) -> str:
    top_area = area.sort_values("area_risk_score", ascending=False).iloc[0]["area_id"] if len(area) and "area_risk_score" in area.columns else "N/A"
    forecast_loss = pd.to_numeric(loss_forecast.get("predicted_loss_all", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if len(loss_forecast) else 0
    return f"""EnergyShield AI - NTL Investigation Summary

Customers analyzed: {customer['customer_id'].nunique():,}
High/Critical customers: {customer['risk_level'].isin(['High','Critical']).sum():,}
Critical customers: {(customer['risk_level']=='Critical').sum():,}
Open inspection cases: {len(cases[~cases['status'].isin(['Confirmed NTL','False Alarm'])]) if len(cases) and 'status' in cases.columns else 0:,}
Estimated current 30-day suspicious loss: {customer['estimated_loss_all_30d'].sum():,.0f} Lek
Forecasted next 30-day suspicious loss: {forecast_loss:,.0f} Lek
Highest-risk area: {top_area}

Recommended next action:
Start with high-risk cases in the highest-risk areas, verify meter status in the field, and save Confirmed NTL / False Alarm results for model feedback."""


def assistant_response(prompt: str, customer: pd.DataFrame, area: pd.DataFrame, priority: pd.DataFrame, daily: pd.DataFrame, cases: pd.DataFrame, metrics: dict, ingestion: dict) -> str:
    text = (prompt or "").lower()
    loss_forecast, area_forecast, customer_forecast = ensure_forecast_outputs(customer, area, daily)

    # Customer lookup by exact ID substring.
    if len(customer) and "customer_id" in customer.columns:
        ids = customer["customer_id"].astype(str).tolist()
        found = next((cid for cid in ids if cid.lower() in text), None)
        if found:
            r = customer[customer["customer_id"].astype(str) == found].iloc[0]
            return (
                f"Customer {found} has Risk Score {r.get('risk_score', 0):.1f}/100 and Risk Level {r.get('risk_level', 'N/A')}.\n\n"
                f"Main reason: {r.get('main_reason', 'No reason available.')}\n\n"
                f"Recommended action: {r.get('recommended_action', 'No action available.')}"
            )

    if any(k in text for k in ["highest", "top", "priority", "inspect first", "who should"]):
        top = priority.head(5) if len(priority) else customer.sort_values("risk_score", ascending=False).head(5)
        lines = ["Top field priorities:"]
        for _, r in top.iterrows():
            lines.append(f"- {r.get('customer_id')}: Risk {float(r.get('risk_score', 0)):.1f}, Area {r.get('area_id', 'N/A')}, reason: {r.get('main_reason', 'N/A')}")
        return "\n".join(lines)

    if any(k in text for k in ["area", "zone", "hotspot", "map"]):
        if area.empty:
            return "No area score file is available. Run the pipeline or connect GIS/area information."
        top = area.sort_values("area_risk_score", ascending=False).head(5)
        lines = ["Highest-risk geographic areas:"]
        for _, r in top.iterrows():
            lines.append(f"- {r.get('area_id')}: Area Risk {float(r.get('area_risk_score', 0)):.1f}, High-risk customers {int(r.get('high_risk_customers', 0))}, Anomaly density {float(r.get('anomaly_density', 0)):.2f}")
        return "\n".join(lines)

    if any(k in text for k in ["forecast", "predict", "next month", "next 30"]):
        forecast_loss = pd.to_numeric(loss_forecast.get("predicted_loss_all", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if len(loss_forecast) else 0
        top_area = area_forecast.iloc[0].get("area_id", "N/A") if len(area_forecast) else "N/A"
        top_customer = customer_forecast.iloc[0].get("customer_id", "N/A") if len(customer_forecast) else "N/A"
        return f"Next 30-day planning forecast: estimated suspicious loss {forecast_loss:,.0f} Lek. Highest forecast area: {top_area}. Highest forecast customer risk: {top_customer}. Use this to plan inspection capacity, not as legal proof."

    if any(k in text for k in ["quality", "schema", "upload", "missing", "dataset", "data"]):
        profile = ingestion.get("raw_profile", {}) if isinstance(ingestion, dict) else {}
        return (
            f"Data intake summary: detected format {str(profile.get('detected_format','N/A')).replace('_',' ')}, "
            f"rows {profile.get('rows','N/A')}, columns {profile.get('columns','N/A')}, "
            f"date columns {profile.get('detected_date_columns','N/A')}, missing cells {profile.get('missing_percent','N/A')}%. "
            f"If confidence is low, use Manual Column Mapping in Data Intake."
        )

    if any(k in text for k in ["improve", "missing feature", "gap", "workflow", "weakness", "what is missing"]):
        audit = build_operational_audit(customer, area, priority, daily, ingestion)
        lines = ["Main operational improvement findings:"]
        for _, r in audit.head(5).iterrows():
            lines.append(f"- {r['Priority']} | {r['Audit Area']}: {r['Recommended Improvement']}")
        return "\n".join(lines)

    if any(k in text for k in ["model", "r2", "roc", "auc", "precision"]):
        roc = metrics.get("models", {}).get("fraud_classifier_roc_auc", "N/A") if isinstance(metrics, dict) else "N/A"
        ap = metrics.get("models", {}).get("fraud_classifier_avg_precision", "N/A") if isinstance(metrics, dict) else "N/A"
        mae = metrics.get("expected_consumption", {}).get("expected_consumption_mae", "N/A") if isinstance(metrics, dict) else "N/A"
        return f"Model diagnostics: ROC AUC = {roc}, Average Precision = {ap}, Expected Consumption MAE = {mae}. Use Model Governance for correlation heatmap, top-k precision, and validation scope."

    high = int(customer.get("risk_level", pd.Series(dtype=str)).isin(["High", "Critical"]).sum()) if len(customer) else 0
    crit = int((customer.get("risk_level", pd.Series(dtype=str)) == "Critical").sum()) if len(customer) else 0
    loss = float(pd.to_numeric(customer.get("estimated_loss_all_30d", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(customer) else 0
    return f"Current NTL overview: {len(customer):,} customers analyzed, {high:,} High/Critical, {crit:,} Critical, estimated current 30-day suspicious loss {loss:,.0f} Lek. Ask me about top customers, hotspots, forecasts, data quality, model metrics, or workflow gaps."


# -----------------------------------------------------------------------------
# Authentication and role-based departments
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .login-hero{max-width:430px; margin:4vh auto 0; text-align:center;}
        .login-logo{
            width:62px; height:62px; border-radius:18px; margin:0 auto 14px;
            display:flex; align-items:center; justify-content:center;
            background:linear-gradient(180deg,#465fff,#3641f5); color:#fff !important;
            font-weight:900; font-size:1.25rem; box-shadow:0 18px 36px rgba(70,95,255,.28);
        }
        .login-hero h1{font-size:1.7rem; margin:0; font-weight:900; letter-spacing:-.03em;}
        .login-hero p{color:var(--gray-500); margin:8px 0 0; font-size:.94rem;}
        .login-card{
            max-width:430px; margin:18px auto 0; padding:24px 26px;
            background:var(--panel); border:1px solid var(--line); border-radius:18px;
            box-shadow:0 1px 2px rgba(16,24,40,.04), 0 16px 40px rgba(16,24,40,.06);
        }
        .demo-box{
            max-width:430px; margin:14px auto 0; padding:14px 16px; border-radius:14px;
            background:var(--brand-25); border:1px solid var(--brand-100);
            font-size:.84rem; color:var(--gray-700); line-height:1.7;
        }
        .demo-box b{color:var(--gray-900);}
        .live-dot{
            display:inline-block; width:9px; height:9px; border-radius:50%;
            background:#12b76a; margin-right:7px; box-shadow:0 0 0 rgba(18,183,106,.5);
            animation:livePulse 1.8s infinite;
        }
        @keyframes livePulse{
            0%{box-shadow:0 0 0 0 rgba(18,183,106,.45);}
            70%{box-shadow:0 0 0 8px rgba(18,183,106,0);}
            100%{box-shadow:0 0 0 0 rgba(18,183,106,0);}
        }
        .live-strip{
            display:flex; flex-wrap:wrap; gap:10px; align-items:center;
            padding:11px 15px; border-radius:14px; margin:0 0 14px;
            background:var(--panel); border:1px solid var(--line); box-shadow:var(--shadow-xs);
            font-size:.86rem; color:var(--gray-700); font-weight:600;
        }
        .live-strip .sep{color:var(--gray-300);}
        .feed-item{
            padding:10px 13px; border:1px solid var(--line); border-left:3px solid var(--brand-500);
            border-radius:10px; margin:7px 0; background:var(--gray-25);
        }
        .feed-item .meta{font-size:.74rem; color:var(--muted-2); font-weight:700; text-transform:uppercase; letter-spacing:.04em;}
        .feed-item .act{font-size:.9rem; color:var(--gray-900); font-weight:700; margin-top:2px;}
        .feed-item .det{font-size:.84rem; color:var(--gray-500); margin-top:2px;}
        .user-id-card{
            margin:2px 4px 14px; padding:12px 13px; border-radius:14px;
            background:rgba(37,99,235,.06); border:1px solid rgba(37,99,235,.14);
        }
        .user-id-card .uname{color:#101828 !important; font-weight:800; font-size:.92rem;}
        .user-id-card .urole{color:#2563eb !important; font-weight:700; font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; margin-top:3px;}
        .user-id-card .uteam{color:#64748b !important; font-size:.78rem; margin-top:3px;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "auth_user" not in st.session_state:
    st.session_state.auth_user = None

ALL_PAGES = {
    "Admin Console": ("◆", "Admin Console"),
    "Command Center": ("▦", "Command Center"),
    "Data Intake": ("⇧", "Data Intake"),
    "Risk Register": ("◇", "Risk Register"),
    "Customer 360": ("◎", "Customer 360"),
    "Geographic View": ("⌖", "Geographic View"),
    "Weather Context": ("☼", "Weather Context"),
    "Building Risk Lab": ("▣", "Building Risk Lab"),
    "Forecasting": ("↗", "Forecasting"),
    "Model Governance": ("○", "Model Governance"),
    "Analyst Workspace": ("∑", "Analyst Workspace"),
    "Assistant": ("✦", "Operations Assistant"),
    "Cases": ("□", "Case Management"),
    "Inspector Mobile": ("◈", "My Inspections"),
    "Reports": ("☰", "Reports"),
}
ROLE_PAGES = {
    "admin": ["Admin Console", "Command Center", "Data Intake", "Risk Register", "Customer 360",
              "Geographic View", "Weather Context", "Building Risk Lab", "Forecasting",
              "Cases", "Assistant", "Reports"],
    "analyst": ["Analyst Workspace", "Model Governance", "Risk Register", "Customer 360",
                "Weather Context", "Forecasting", "Building Risk Lab", "Reports", "Assistant"],
    "inspector": ["Inspector Mobile", "Customer 360", "Geographic View"],
}
HOME_BY_ROLE = {"admin": "Admin Console", "analyst": "Analyst Workspace", "inspector": "Inspector Mobile"}

if st.session_state.auth_user is None:
    st.markdown(
        """
        <div class="login-hero">
            <div class="login-logo">ES</div>
            <h1>EnergyShield AI</h1>
            <p>OSHEE Non-Technical Loss Detection Platform — department sign in</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    lc = st.columns([1, 1.3, 1])[1]
    with lc:
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin / analyst / inspector1")
            password = st.text_input("Password", type="password", placeholder="Password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            account = workspace.authenticate(username, password)
            if account:
                st.session_state.auth_user = account
                st.session_state.active_page = HOME_BY_ROLE.get(account["role"], "Command Center")
                workspace.log_activity(OUTPUT_DIR, account["name"], account["role"], "Signed in to the workspace")
                st.rerun()
            else:
                st.error("Invalid username or password. Use one of the demo accounts below.")
        st.markdown(
            """
            <div class="demo-box">
                <b>Demo accounts</b> (password <b>oshee123</b> for all)<br>
                <b>admin</b> — OSHEE operations: dashboards, upload, dispatch duties, request summaries<br>
                <b>analyst</b> — data analytics: statistics, model quality, answer admin requests<br>
                <b>inspector1</b> / <b>inspector2</b> / <b>inspector3</b> — field teams: assigned duties
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.stop()

user = st.session_state.auth_user
role = user["role"]
st.session_state.user_role = user["role_label"]

PAGES = {k: ALL_PAGES[k] for k in ROLE_PAGES.get(role, list(ALL_PAGES.keys())) if k in ALL_PAGES}
if "active_page" not in st.session_state or st.session_state.active_page not in PAGES:
    st.session_state.active_page = HOME_BY_ROLE.get(role, next(iter(PAGES)))

# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
st.sidebar.markdown(
    """
    <div class="brand-card">
      <div class="brand-logo">ES</div>
      <div><div class="brand-title">EnergyShield</div><div class="brand-subtitle">AI NTL Operations</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
_team_html = f'<div class="uteam">{html.escape(str(user.get("team")))}</div>' if user.get("team") else ""
st.sidebar.markdown(
    f"""
    <div class="user-id-card">
        <div class="uname">{html.escape(str(user.get('name','')))}</div>
        <div class="urole">{html.escape(str(user.get('role_label','')))}</div>
        {_team_html}
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown('<div class="side-section">Navigation</div>', unsafe_allow_html=True)
for key, item in PAGES.items():
    icon, label = item
    nav_label = f"{icon}  {label}"
    if st.session_state.active_page == key:
        st.sidebar.markdown(f'<div class="active-nav"><span class="nav-ico">{html.escape(icon)}</span><span>{html.escape(label)}</span></div>', unsafe_allow_html=True)
    else:
        if st.sidebar.button(nav_label, key=f"nav_{key}"):
            st.session_state.active_page = key
            st.session_state.show_onboarding = False
            st.rerun()

WORKFLOW_BY_ROLE = {
    "admin": ["Upload meter data", "Detect suspicious behavior", "Dispatch inspection duties", "Request analyst summary", "Track verification results"],
    "analyst": ["Open admin requests", "Analyze statistical fields", "Validate model quality", "Send summary to admin", "Monitor data quality"],
    "inspector": ["Receive dispatched duty", "Review reason and location", "Inspect meter in field", "Confirm NTL or false alarm", "Report outcome"],
}
workflow_rows = "".join(
    f'<div class="workflow-row"><span class="workflow-num">{i+1}</span>{html.escape(step)}</div>'
    for i, step in enumerate(WORKFLOW_BY_ROLE.get(role, []))
)
st.sidebar.markdown(
    f"""
    <div class="workflow-panel">
        <div class="side-section" style="margin-top:0;">Workflow</div>
        {workflow_rows}
    </div>
    """,
    unsafe_allow_html=True,
)

# Optional guided start ("quiz") — admin only, since login already routes each department.
if "show_onboarding" not in st.session_state:
    st.session_state.show_onboarding = False

st.sidebar.markdown('<div class="side-section">Session</div>', unsafe_allow_html=True)
_fresh = workspace.data_last_updated(OUTPUT_DIR)
st.sidebar.caption(f"Data updated: {_fresh}" if _fresh else "No analysis generated yet")
if role == "admin":
    if st.sidebar.button("⊕  Guided start", key="nav_onboarding"):
        st.session_state.show_onboarding = True
        st.rerun()
if st.sidebar.button("⎋  Sign out", key="nav_signout"):
    workspace.log_activity(OUTPUT_DIR, user.get("name", "User"), role, "Signed out")
    st.session_state.auth_user = None
    st.session_state.show_onboarding = False
    st.session_state.pop("active_page", None)
    st.rerun()

page = st.session_state.active_page

# -----------------------------------------------------------------------------
# Optional admin guided start ("quiz")
# -----------------------------------------------------------------------------
ONBOARDING_ROUTES = {
    "Upload our meter / billing data": "Data Intake",
    "See an executive risk overview": "Command Center",
    "Review the highest-risk customers": "Risk Register",
    "Open the geographic hotspot map": "Geographic View",
    "Dispatch field inspection duties": "Cases",
    "Analyze buildings / villages": "Building Risk Lab",
    "Forecast losses and inspection demand": "Forecasting",
}

if role == "admin" and st.session_state.get("show_onboarding"):
    page_header(
        "Welcome back",
        "Guided Start",
        "A short, optional shortcut. Pick what you want to do first and the platform opens that workspace — or skip and use the Admin Console.",
        chip="OSHEE operations workspace",
    )
    with st.container():
        st.markdown('<div class="soft-card"><div class="soft-title">Quick shortcut</div><div class="soft-text">This only changes which page opens — it never changes the data or the AI model.</div></div>', unsafe_allow_html=True)
        goal_options = list(ONBOARDING_ROUTES.keys())
        goal = st.selectbox("What do you want to do first?", goal_options, key="onboarding_goal")
        b1, b2 = st.columns([1, 1])
        with b1:
            if st.button("Open this workspace", type="primary", use_container_width=True):
                target = ONBOARDING_ROUTES.get(goal, "Admin Console")
                st.session_state.active_page = target if target in PAGES else "Admin Console"
                st.session_state.show_onboarding = False
                st.rerun()
        with b2:
            if st.button("Skip to Admin Console", use_container_width=True):
                st.session_state.active_page = "Admin Console"
                st.session_state.show_onboarding = False
                st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# Real-time collaboration helpers
# -----------------------------------------------------------------------------
@st.fragment(run_every=8)
def live_status_strip():
    fresh = workspace.data_last_updated(OUTPUT_DIR) or "not generated yet"
    pending = workspace.pending_request_count(OUTPUT_DIR)
    act = workspace.load_activity(OUTPUT_DIR, limit=1)
    last = ""
    if not act.empty:
        r0 = act.iloc[0]
        last = f'{r0.get("actor", "")}: {r0.get("action", "")}'
    html_parts = [
        '<div class="live-strip"><span class="live-dot"></span>Live',
        ' <span class="sep">|</span> Data updated: <b>' + html.escape(str(fresh)) + '</b>',
        ' <span class="sep">|</span> Pending analyst requests: <b>' + str(pending) + '</b>',
    ]
    if last:
        html_parts.append(' <span class="sep">|</span> Latest: ' + html.escape(last))
    html_parts.append('</div>')
    st.markdown("".join(html_parts), unsafe_allow_html=True)


@st.fragment(run_every=6)
def live_activity_panel(limit: int = 8):
    df = workspace.load_activity(OUTPUT_DIR, limit=limit)
    st.markdown('<div class="soft-title"><span class="live-dot"></span>Live activity feed</div>', unsafe_allow_html=True)
    if df.empty:
        st.caption("No cross-department activity recorded yet.")
        return
    parts = []
    for _, r in df.iterrows():
        meta = f'{r.get("time", "")} · {r.get("role", "")}'
        act = f'{r.get("actor", "")} — {r.get("action", "")}'
        detail = str(r.get("detail", "") or "")
        det_html = '<div class="det">' + html.escape(detail) + '</div>' if detail else ""
        parts.append(
            '<div class="feed-item"><div class="meta">' + html.escape(meta)
            + '</div><div class="act">' + html.escape(act) + '</div>' + det_html + '</div>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def analyst_draft(topic: str, customer: pd.DataFrame, area: pd.DataFrame, priority: pd.DataFrame, metrics: dict, ingestion: dict, loss_forecast: pd.DataFrame) -> str:
    t = (topic or "").lower()
    total = customer["customer_id"].nunique() if len(customer) else 0
    high = int(customer["risk_level"].isin(["High", "Critical"]).sum()) if len(customer) else 0
    crit = int((customer["risk_level"] == "Critical").sum()) if len(customer) else 0
    loss = float(pd.to_numeric(customer.get("estimated_loss_all_30d", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(customer) else 0
    if "area" in t or "hotspot" in t:
        if area.empty or "area_risk_score" not in area.columns:
            return "No area-level scores are available for the current dataset."
        top = area.sort_values("area_risk_score", ascending=False).head(5)
        lines = ["Highest-risk areas (for inspection planning):"]
        for _, r in top.iterrows():
            lines.append(f"- {r.get('area_id')}: area risk {float(r.get('area_risk_score', 0)):.1f}, high-risk customers {int(r.get('high_risk_customers', 0))}, anomaly density {float(r.get('anomaly_density', 0)):.2f}")
        return "\n".join(lines)
    if "customer" in t or "inspect first" in t:
        src = priority if len(priority) else customer.sort_values("risk_score", ascending=False)
        lines = ["Top customers to inspect first:"]
        for _, r in src.head(8).iterrows():
            lines.append(f"- {r.get('customer_id')}: risk {float(r.get('risk_score', 0)):.1f}, area {r.get('area_id', 'N/A')}, reason: {r.get('main_reason', 'N/A')}")
        return "\n".join(lines)
    if "loss" in t or "financial" in t:
        fl = pd.to_numeric(loss_forecast.get("predicted_loss_all", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if len(loss_forecast) else 0
        return (f"Estimated financial exposure:\n- Current 30-day suspicious loss: {loss:,.0f} Lek\n"
                f"- Forecast next 30-day loss: {fl:,.0f} Lek\n"
                f"- High/Critical customers: {high:,} ({high / max(total, 1) * 100:.1f}% of {total:,})\n"
                f"- Urgent (Critical): {crit:,}")
    if "data quality" in t or "upload" in t:
        profile = ingestion.get("raw_profile", {}) if isinstance(ingestion, dict) else {}
        return (f"Latest dataset quality:\n- Format: {str(profile.get('detected_format', 'N/A')).replace('_', ' ')}\n"
                f"- Rows: {profile.get('rows', 'N/A')}, Columns: {profile.get('columns', 'N/A')}\n"
                f"- Missing cells: {profile.get('missing_percent', 'N/A')}%\n"
                f"- Detected date columns: {profile.get('detected_date_columns', 'N/A')}")
    if "model" in t or "reliab" in t or "quality" in t:
        models = metrics.get("models", {}) if isinstance(metrics, dict) else {}
        roc = models.get("fraud_classifier_roc_auc")
        ap = models.get("fraud_classifier_avg_precision")
        p10 = models.get("precision_at_top_10_percent")
        mae = metrics.get("expected_consumption", {}).get("expected_consumption_mae") if isinstance(metrics, dict) else None
        return (f"Model quality (decision-support, not legal proof):\n- ROC AUC: {roc}\n- Average precision: {ap}\n"
                f"- Precision@Top10%: {p10}\n- Expected-consumption MAE: {mae}\n"
                f"Validate against confirmed field outcomes before operational use.")
    if "weather" in t:
        wc = int((pd.to_numeric(customer.get("weather_context_score", pd.Series(dtype=float)), errors="coerce").fillna(0) > 40).sum()) if len(customer) else 0
        return (f"Weather-driven anomaly summary:\n- Customers with notable weather-consumption mismatch: {wc:,}\n"
                f"- Weather is used as context for prioritization, not as proof of theft.")
    return (f"Overall NTL summary:\n- Customers analyzed: {total:,}\n"
            f"- High/Critical: {high:,} ({high / max(total, 1) * 100:.1f}%)\n"
            f"- Urgent (Critical): {crit:,}\n- Estimated 30-day suspicious loss: {loss:,.0f} Lek")


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
if page == "Admin Console":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    cases = create_cases_from_priority(priority)
    page_header("OSHEE administration", "Admin Console", "One clear place to understand today's electricity-loss risk, dispatch field inspection duties, and ask the data analytics office for a summary.")
    live_status_strip()

    total = customer["customer_id"].nunique()
    high = int(customer["risk_level"].isin(["High", "Critical"]).sum())
    critical = int((customer["risk_level"] == "Critical").sum())
    loss = float(pd.to_numeric(customer.get("estimated_loss_all_30d", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    pct_high = high / max(total, 1) * 100
    top_area = str(area.sort_values("area_risk_score", ascending=False).iloc[0]["area_id"]) if len(area) and "area_risk_score" in area.columns else "N/A"
    dispatched = int((cases["status"] != "New").sum()) if len(cases) else 0
    confirmed = int((cases["status"] == "Confirmed NTL").sum()) if len(cases) else 0

    a, b, c, d = st.columns(4)
    a.metric("Customers analyzed", f"{total:,}")
    b.metric("Need inspection", f"{high:,}", f"{pct_high:.1f}% of customers")
    c.metric("Urgent (Critical)", f"{critical:,}")
    d.metric("Money at risk / 30 days", f"{loss:,.0f} Lek")

    status_level = "Critical" if pct_high >= 15 else "High" if pct_high >= 7 else "Medium" if pct_high >= 3 else "Low"
    plain = (
        f"Out of <b>{total:,}</b> customers analyzed, <b>{high:,}</b> ({pct_high:.1f}%) show suspicious "
        f"consumption patterns and should be checked in the field. <b>{critical:,}</b> are urgent. "
        f"The estimated money at risk over the next 30 days is <b>{loss:,.0f} Lek</b>. "
        f"The area that needs the most attention right now is <b>{html.escape(top_area)}</b>."
    )
    alert_card(
        status_level,
        f"In plain words: {high:,} of {total:,} customers look suspicious and {critical:,} are urgent. About {loss:,.0f} Lek may be lost in 30 days. Focus first on area {top_area}.",
        "Dispatch the urgent cases to a field team below, then ask the analyst for a deeper summary if needed.",
    )
    st.markdown(f'<div class="soft-card"><div class="soft-title">Today\'s situation</div><div class="soft-text">{plain}</div></div>', unsafe_allow_html=True)

    render_loss_indicator(customer, compact=True)

    st.subheader("1. Dispatch inspection duties to field teams")
    st.caption("Pick a team and how many of the highest-risk, not-yet-dispatched cases to send. The team sees them in real time on their device.")
    d1, d2, d3 = st.columns([1.1, 1, 1])
    with d1:
        team = st.selectbox("Assign to inspection team", workspace.inspector_teams())
    with d2:
        n_dispatch = st.number_input("Number of top-risk duties", min_value=1, max_value=50, value=5, step=1)
    with d3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        dispatch_clicked = st.button("Dispatch duties", type="primary", use_container_width=True)
    if dispatch_clicked:
        pool = cases[cases["status"] == "New"].sort_values("risk_score", ascending=False).head(int(n_dispatch))
        if pool.empty:
            st.warning("No undispatched cases remain. Reassign existing cases from Case Management if needed.")
        else:
            cases.loc[pool.index, "assigned_team"] = team
            cases.loc[pool.index, "status"] = "Assigned"
            save_cases(cases)
            workspace.log_activity(OUTPUT_DIR, user["name"], role, f"Dispatched {len(pool)} inspection duties", f"to {team}")
            st.success(f"Dispatched {len(pool)} duties to {team}. The team can see them now.")
            st.rerun()

    team_summary = (
        cases.assign(is_dispatched=cases["status"] != "New")
        .groupby("assigned_team")
        .agg(total=("case_id", "count"),
             dispatched=("is_dispatched", "sum"),
             confirmed=("status", lambda s: int((s == "Confirmed NTL").sum())),
             open=("status", lambda s: int((~s.isin(["Confirmed NTL", "False Alarm", "New"])).sum())))
        .reset_index()
        if len(cases) else pd.DataFrame()
    )
    if not team_summary.empty:
        st.dataframe(clean_display_df(team_summary.rename(columns={"assigned_team": "Inspection Team", "total": "Total Cases", "dispatched": "Dispatched", "confirmed": "Confirmed NTL", "open": "Open"})), use_container_width=True, hide_index=True)

    st.subheader("2. Request a summary from the Data Analytics Office")
    st.caption("The analyst gets your request instantly and sends back a written statistical summary.")
    r1, r2 = st.columns([1, 1.4])
    with r1:
        req_topic = st.selectbox("Topic", workspace.SUMMARY_TOPICS)
    with r2:
        req_note = st.text_input("Note for the analyst (optional)", placeholder="e.g. focus on Tirana feeders this week")
    if st.button("Send request to analyst"):
        rid = workspace.add_summary_request(OUTPUT_DIR, user["name"], req_topic, req_note)
        workspace.log_activity(OUTPUT_DIR, user["name"], role, "Requested an analyst summary", req_topic)
        st.success(f"Request {rid} sent to the Data Analytics Office.")
        st.rerun()

    reqs = workspace.load_summary_requests(OUTPUT_DIR)
    answered = reqs[reqs["status"] == "Answered"] if not reqs.empty else reqs
    if not answered.empty:
        with st.expander(f"Latest analyst replies ({len(answered)})", expanded=False):
            for _, r in answered.head(5).iterrows():
                st.markdown(
                    f'<div class="feed-item"><div class="meta">{html.escape(str(r["answered_at"]))} · {html.escape(str(r["answered_by"]))}</div>'
                    f'<div class="act">{html.escape(str(r["topic"]))}</div>'
                    f'<div class="det">{html.escape(str(r["response"]))}</div></div>',
                    unsafe_allow_html=True,
                )

elif page == "Analyst Workspace":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    loss_forecast, area_forecast, customer_forecast = ensure_forecast_outputs(customer, area, daily)
    page_header("Data analytics", "Analyst Workspace", "Statistical results for the analytics office, and the place to answer summary requests from OSHEE administration in real time.")
    live_status_strip()

    models = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    total = customer["customer_id"].nunique()
    high_rate = customer["risk_level"].isin(["High", "Critical"]).mean() * 100 if len(customer) else 0
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Customers", f"{total:,}")
    s2.metric("High-risk rate", f"{high_rate:.1f}%")
    s3.metric("Mean risk score", fmt(customer["risk_score"].mean() if len(customer) else 0))
    s4.metric("ROC AUC", "N/A" if models.get("fraud_classifier_roc_auc") is None else f"{models.get('fraud_classifier_roc_auc'):.3f}")
    s5.metric("Precision@Top10%", "N/A" if models.get("precision_at_top_10_percent") is None else f"{models.get('precision_at_top_10_percent'):.3f}")
    st.caption("Open Model Governance for correlation heatmaps, top-k quality, and validation detail.")

    sr1, sr2 = st.columns([3, 1])
    sr1.subheader("Summary requests from administration")
    if sr2.button("↻ Refresh requests", use_container_width=True):
        st.rerun()
    reqs = workspace.load_summary_requests(OUTPUT_DIR)
    pending = reqs[reqs["status"] != "Answered"] if not reqs.empty else reqs
    if pending.empty:
        st.info("No pending requests. New requests from the admin appear here — refresh after the live indicator shows a new pending request.")
    else:
        for _, r in pending.iterrows():
            rid = str(r["request_id"])
            st.markdown(
                f'<div class="feed-item"><div class="meta">{html.escape(str(r["requested_at"]))} · {html.escape(str(r["requested_by"]))} · {html.escape(rid)}</div>'
                f'<div class="act">{html.escape(str(r["topic"]))}</div>'
                f'{("<div class=det>" + html.escape(str(r["note"])) + "</div>") if str(r.get("note", "")) else ""}</div>',
                unsafe_allow_html=True,
            )
            draft = analyst_draft(str(r["topic"]), customer, area, priority, metrics, ingestion, loss_forecast)
            resp = st.text_area("Your summary (auto-drafted from the data — edit before sending)", value=draft, key=f"resp_{rid}", height=170)
            if st.button("Send summary to admin", key=f"send_{rid}", type="primary"):
                workspace.answer_summary_request(OUTPUT_DIR, rid, user["name"], resp)
                workspace.log_activity(OUTPUT_DIR, user["name"], role, "Answered an admin summary request", str(r["topic"]))
                st.success("Summary sent to administration.")
                st.rerun()

    with st.expander("Answered requests history"):
        answered = reqs[reqs["status"] == "Answered"] if not reqs.empty else reqs
        if answered.empty:
            st.caption("No answered requests yet.")
        else:
            st.dataframe(clean_display_df(answered[["request_id", "topic", "requested_by", "answered_at", "response"]]), use_container_width=True, hide_index=True)

elif page == "Data Intake":
    page_header("Data operations", "Data Intake Center", "Upload SGCC wide smart-meter files, STEG client/invoice ZIPs, or long smart-meter datasets. The platform detects the structure, explains file quality, and lets the analyst manually map columns when automatic detection is not enough.")
    uploaded = st.file_uploader("Meter consumption dataset", type=["csv", "xlsx", "xls", "zip"], label_visibility="visible")
    tmp_path = None
    inspection = None
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name
        try:
            inspection = inspect_uploaded_dataset(tmp_path)
            profile = inspection["profile"]
            schema = inspection["schema"]
            confidence = round(float(profile.get("schema_confidence", 0)) * 100, 1)
            status_class = "status-good" if confidence >= 80 else "status-warning" if confidence >= 50 else "status-bad"
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Rows", fmt_count(profile.get("rows", 0)))
            c2.metric("Columns", fmt_count(profile.get("columns", 0)))
            c3.metric("Date Columns", fmt_count(profile.get("detected_date_columns", 0)))
            c4.metric("Missing Cells", f"{profile.get('missing_percent', 0):.2f}%")
            c5.metric("Confidence", f"{confidence}%")
            st.markdown(
                f"""
                <div class="status-card {status_class}">
                    <b>Detected format:</b> {html.escape(str(profile.get('detected_format')).replace('_',' ').title())}<br>
                    <b>Customer field:</b> {html.escape(str(profile.get('customer_column')))} &nbsp; | &nbsp;
                    <b>Label field:</b> {html.escape(str(profile.get('label_column')))}<br>
                    <b>Readiness:</b> {'Ready for automated analysis' if confidence >= 80 else 'Usable, but review manual mapping' if confidence >= 50 else 'Manual mapping recommended'}
                </div>
                """,
                unsafe_allow_html=True,
            )
            if schema.get("issues"):
                with st.expander("Data warnings", expanded=False):
                    for issue in schema.get("issues", []):
                        st.warning(issue)
            with st.expander("Preview uploaded file", expanded=False):
                st.dataframe(clean_display_df(inspection["preview"]), use_container_width=True, height=260)
        except Exception as exc:
            st.warning(f"Automatic detection needs help: {exc}")
            st.info("Use manual column mapping below to tell the platform how to interpret the file.")

    st.markdown('<div class="soft-card"><div class="soft-title">Processing controls</div><div class="soft-text">Use automatic mode for SGCC-like data. Use manual mapping when OSHEE exports have unusual headers or custom formats.</div></div>', unsafe_allow_html=True)
    mode = st.segmented_control("Processing mode", ["Automatic detection", "Manual column mapping"], default="Automatic detection")
    max_customers = st.number_input("Maximum customers to process", min_value=0, value=1500, step=100, help="0 means all customers. Use a limit for large operational files.")
    max_days = st.number_input("Maximum recent days", min_value=0, value=365, step=30, help="0 means all available days.")

    manual_kwargs = None
    if uploaded is not None and mode == "Manual column mapping":
        try:
            cols = list(inspection["preview"].columns) if inspection else list(pd.read_csv(tmp_path, nrows=2).columns)
        except Exception:
            cols = []
        st.subheader("Manual Mapping Wizard")
        schema_choice = st.radio("Dataset structure", ["Wide table with many date columns", "Long table with one date column"])
        customer_col = st.selectbox("Customer ID column", cols, index=cols.index("CONS_NO") if "CONS_NO" in cols else 0)
        label_options = ["No label"] + cols
        label_col = st.selectbox("Fraud / inspection label column", label_options, index=label_options.index("FLAG") if "FLAG" in label_options else 0)
        label_col = None if label_col == "No label" else label_col
        if schema_choice.startswith("Wide"):
            guessed_dates = [c for c in cols if any(ch.isdigit() for ch in str(c)) and c not in {customer_col, label_col}]
            date_cols = st.multiselect("Date reading columns", cols, default=guessed_dates[: min(len(guessed_dates), 120)])
            manual_kwargs = {"schema_type":"wide", "customer_col":customer_col, "label_col":label_col, "date_cols":date_cols}
        else:
            date_col = st.selectbox("Date column", cols)
            consumption_col = st.selectbox("Consumption column", cols)
            manual_kwargs = {"schema_type":"long", "customer_col":customer_col, "label_col":label_col, "date_col":date_col, "consumption_col":consumption_col}

    if st.button("Run NTL analysis", type="primary", use_container_width=True):
        if tmp_path is None:
            st.warning("Upload a dataset first.")
        else:
            try:
                input_for_pipeline = tmp_path
                if mode == "Manual column mapping":
                    if not manual_kwargs:
                        st.error("Complete manual mapping before running.")
                        st.stop()
                    MANUAL_STANDARDIZED_PATH.parent.mkdir(parents=True, exist_ok=True)
                    prepare_manual_mapping_dataset(tmp_path, MANUAL_STANDARDIZED_PATH, manual_kwargs, max_customers=None if max_customers == 0 else int(max_customers), max_days=None if max_days == 0 else int(max_days))
                    input_for_pipeline = MANUAL_STANDARDIZED_PATH
                with st.spinner("Cleaning readings, building customer behavior profiles, detecting anomalies, calculating risk, and preparing inspection cases..."):
                    summary = run_ntl_pipeline(input_for_pipeline, output_dir=OUTPUT_DIR, model_dir=MODEL_DIR, max_customers=None if max_customers == 0 else int(max_customers), max_days=None if max_days == 0 else int(max_days))
                    if CASE_FILE.exists():
                        CASE_FILE.unlink()
                    load_outputs_cached.clear()
                workspace.log_activity(OUTPUT_DIR, user["name"], role, "Ran NTL analysis on a new dataset", f"{summary.get('customers_analyzed', '?')} customers, {summary.get('high_risk_customers', '?')} high-risk")
                st.success("NTL analysis completed. Open Command Center or Risk Register.")
                st.dataframe(clean_display_df(pd.DataFrame([summary])), use_container_width=True)
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.info("Try Manual Column Mapping, reduce max customers/days, or check if the file contains valid customer/date/consumption readings.")

elif page == "Command Center":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    page_header("Executive monitoring", "Command Center", "Operational overview for suspicious electricity consumption, high-risk clusters, expected losses, and inspection readiness.")
    total = customer["customer_id"].nunique()
    high = int(customer["risk_level"].isin(["High","Critical"]).sum())
    critical = int((customer["risk_level"] == "Critical").sum())
    loss = customer.get("estimated_loss_all_30d", pd.Series(dtype=float)).sum()
    a,b,c,d,e = st.columns(5)
    a.metric("Customers", f"{total:,}")
    b.metric("High/Critical", f"{high:,}", f"{high/max(total,1)*100:.1f}%")
    c.metric("Critical", f"{critical:,}")
    d.metric("High-Risk Areas", f"{int(area['area_risk_level'].isin(['High','Critical']).sum()) if 'area_risk_level' in area.columns else 0:,}")
    e.metric("Estimated 30-Day Loss", f"{loss:,.0f} Lek")

    render_loss_indicator(customer, area)

    col1, col2 = st.columns([1.1,1])
    with col1:
        fig = px.histogram(customer, x="risk_score", nbins=35, title="Risk Score Distribution", labels={"risk_score":"Risk Score"})
        fig.update_layout(yaxis_title="Customers")
        render_chart(fig)
    with col2:
        risk_counts = customer["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
        risk_counts.columns = ["Risk Level", "Customers"]
        fig = px.bar(risk_counts, x="Risk Level", y="Customers", color="Risk Level", text="Customers", color_discrete_map=RISK_COLORS, title="Risk Level Breakdown")
        fig.update_layout(showlegend=False)
        render_chart(fig)

    col3, col4 = st.columns(2)
    with col3:
        top_area = area.sort_values("area_risk_score", ascending=False).head(10).copy()
        fig = px.bar(top_area, x="area_risk_score", y="area_id", orientation="h", color="area_risk_level", color_discrete_map=RISK_COLORS, title="Highest-Risk Areas", labels={"area_risk_score":"Area Risk Score", "area_id":"Area"})
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        render_chart(fig)
    with col4:
        if not daily.empty:
            trend = daily.groupby("date").agg(Average_Consumption=("consumption_kwh","mean"), Sudden_Drop_Events=("sudden_drop_flag","sum")).reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=trend["date"], y=trend["Average_Consumption"], mode="lines", name="Average Consumption"))
            fig.add_trace(go.Scatter(x=trend["date"], y=trend["Sudden_Drop_Events"], mode="lines", name="Sudden Drops", yaxis="y2"))
            fig.update_layout(title="Consumption Trend vs Sudden Drop Events", yaxis=dict(title="Average Consumption"), yaxis2=dict(title="Drop Events", overlaying="y", side="right"), legend=dict(orientation="h", y=-.18))
            render_chart(fig)

    st.subheader("Immediate Attention")
    st.dataframe(clean_display_df(customer.sort_values("risk_score", ascending=False).head(15), ["customer_id","risk_score","risk_level","area_id","transformer_id","estimated_loss_all_30d","main_reason","recommended_action"]), use_container_width=True, height=430)

elif page == "Risk Register":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    page_header("Customer intelligence", "Risk Register", "Filter, search, export, and investigate customers ranked by suspicious consumption behavior.")
    c1,c2,c3,c4 = st.columns([1.2,1,1,1])
    query = c1.text_input("Search customer", placeholder="Example: C100295 or CONS_NO")
    levels = c2.multiselect("Risk Level", RISK_ORDER, default=["High","Critical"])
    min_score = c3.slider("Minimum Risk Score", 0, 100, 60)
    area_options = sorted(customer["area_id"].dropna().unique().tolist()) if "area_id" in customer.columns else []
    areas = c4.multiselect("Area", area_options)
    filtered = customer[customer["risk_score"] >= min_score].copy()
    if query:
        filtered = filtered[filtered["customer_id"].astype(str).str.contains(query, case=False, na=False)]
    if levels:
        filtered = filtered[filtered["risk_level"].isin(levels)]
    if areas:
        filtered = filtered[filtered["area_id"].isin(areas)]
    k1,k2,k3 = st.columns(3)
    k1.metric("Customers in View", f"{len(filtered):,}")
    k2.metric("Average Risk", fmt(filtered["risk_score"].mean() if len(filtered) else 0))
    k3.metric("Estimated Loss", f"{filtered.get('estimated_loss_all_30d', pd.Series(dtype=float)).sum():,.0f} Lek")
    cols = ["customer_id","risk_score","risk_level","fraud_probability","ai_anomaly_score","historical_deviation_score","peer_deviation_score","weather_context_score","payment_risk_score","unpaid_bills","arrears_amount_lek","geographic_risk_score","area_id","transformer_id","estimated_loss_all_30d","main_reason","recommended_action"]
    st.dataframe(clean_display_df(filtered.sort_values("risk_score", ascending=False), cols), use_container_width=True, height=600)
    st.download_button("Download current register", filtered.to_csv(index=False), file_name="energyshield_risk_register.csv", use_container_width=True)

elif page == "Customer 360":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    page_header("Investigation workspace", "Customer 360", "Understand one customer: behavior history, peer comparison, geographic context, risk drivers, and recommended field action.")
    ordered = customer.sort_values("risk_score", ascending=False)
    options = ordered["customer_id"].tolist()
    risk_lookup = ordered.set_index("customer_id")["risk_score"].to_dict()
    level_lookup = ordered.set_index("customer_id")["risk_level"].to_dict()
    selected = st.selectbox("Select customer", options, format_func=lambda cid: f"{cid} · {level_lookup.get(cid,'N/A')} · Risk {risk_lookup.get(cid,0):.1f}")
    row = customer[customer["customer_id"] == selected].iloc[0]
    st.markdown(f"<h2 style='margin-bottom:8px;'>Customer {html.escape(str(selected))} &nbsp; {risk_pill(row.get('risk_level'))}</h2>", unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Risk Score", f"{row['risk_score']:.1f}/100")
    m2.metric("Fraud Probability", fmt(row.get("fraud_probability"),1,"%"))
    m3.metric("AI Anomaly", f"{row.get('ai_anomaly_score',0):.1f}/100")
    m4.metric("Peer Gap", f"{row.get('peer_deviation_score',0):.1f}/100")
    m5.metric("Estimated Loss", f"{row.get('estimated_loss_all_30d',0):,.0f} Lek")
    alert_card(row.get("risk_level"), row.get("alert_explanation"), row.get("recommended_action"))
    tab1, tab2, tab3 = st.tabs(["Consumption timeline", "Risk drivers", "Profile and action"])
    with tab1:
        cd = daily[daily["customer_id"] == selected].copy()
        if not cd.empty:
            fig = px.line(cd, x="date", y=["consumption_kwh","rolling_30d_mean"], title="Actual Consumption vs 30-Day Baseline", labels={"value":"Consumption (kWh)", "variable":"Signal"})
            fig.for_each_trace(lambda t: t.update(name=pretty_label(t.name)))
            render_chart(fig)
        else:
            st.info("Daily history is not in the dashboard sample for this customer. Rerun with fewer customers or select a high-risk customer.")
    with tab2:
        drivers = pd.DataFrame({
            "Driver": ["AI anomaly", "Historical deviation", "Similar profile deviation", "Geographic risk", "Sudden behavior", "Weather context", "Payment behavior"],
            "Score": [row.get("ai_anomaly_score",0), row.get("historical_deviation_score",0), row.get("peer_deviation_score",0), row.get("geographic_risk_score",0), row.get("sudden_flags_score",0), row.get("weather_context_score",0), row.get("payment_risk_score", row.get("payment_behavior_score",0))],
        })
        fig = px.bar(drivers, x="Score", y="Driver", orientation="h", title="Risk Driver Breakdown", range_x=[0,100])
        fig.update_layout(yaxis={"categoryorder":"total ascending"})
        render_chart(fig)
    with tab3:
        c1, c2 = st.columns([1,1])
        with c1:
            st.dataframe(clean_display_df(pd.DataFrame([row]), ["customer_id","customer_type","area_id","transformer_id","avg_consumption","last_30_mean","previous_90_mean","peer_avg_last_30","avg_temp_mean","weather_mismatch_ratio","weather_mismatch_days","zero_days","flatline_days","sudden_drop_count"]), use_container_width=True, hide_index=True)
        with c2:
            if "payment_risk_score" in customer.columns or "unpaid_bills" in customer.columns:
                pay_score = float(row.get("payment_risk_score", row.get("payment_behavior_score", 0)) or 0)
                late = int(float(row.get("payment_late_count_12m", 0) or 0))
                unpaid = int(float(row.get("unpaid_bills", 0) or 0))
                arrears = float(row.get("arrears_amount_lek", 0) or 0)
                disc = int(float(row.get("disconnections_12m", 0) or 0))
                delay = float(row.get("avg_payment_delay_days", 0) or 0)
                status = html.escape(str(row.get("account_status", "active")))
                method = html.escape(str(row.get("payment_method", "n/a")).replace("_", " "))
                st.markdown(
                    f'<div class="soft-card"><div class="soft-title">Billing &amp; payment behavior</div><div class="soft-text">'
                    f'Payment risk: <b>{pay_score:.0f}/100</b> &middot; Account: <b>{status}</b><br>'
                    f'Late payments (12m): <b>{late}</b> &middot; Unpaid bills: <b>{unpaid}</b><br>'
                    f'Avg delay: <b>{delay:.0f} days</b> &middot; Arrears: <b>{arrears:,.0f} Lek</b><br>'
                    f'Non-payment disconnections: <b>{disc}</b> &middot; Method: <b>{method}</b>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('<div class="soft-card"><div class="soft-title">Inspector checklist</div><div class="soft-text">1. Verify physical meter reading.<br>2. Inspect meter seal and wiring.<br>3. Check possible bypass connection.<br>4. Compare neighboring meters in same transformer zone.<br>5. Cross-check billing/payment history vs metered consumption.<br>6. Record confirmed/false alarm result for model feedback.</div></div>', unsafe_allow_html=True)

elif page == "Geographic View":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    if role == "inspector":
        # Field teams only see the locations of the duties dispatched to them, not the whole network.
        my_team = user.get("team")
        geo_cases = create_cases_from_priority(priority)
        my_ids = set()
        if my_team and not geo_cases.empty:
            mine = geo_cases[(geo_cases["assigned_team"] == my_team) & (geo_cases["status"] != "New")]
            my_ids = set(mine["customer_id"].astype(str))
        customer = customer[customer["customer_id"].astype(str).isin(my_ids)].copy()
        if "area_id" in customer.columns and not customer.empty:
            area = area[area["area_id"].astype(str).isin(customer["area_id"].astype(str).unique())].copy()
        else:
            area = area.iloc[0:0]
        if "transformer_id" in customer.columns and not customer.empty and not transformer.empty:
            transformer = transformer[transformer["transformer_id"].astype(str).isin(customer["transformer_id"].astype(str).unique())].copy()
        else:
            transformer = transformer.iloc[0:0]
        page_header("Field map", "My Assigned Map", f"Only the duties dispatched to {my_team or 'your team'}: your assigned customers, their locations, and the areas/transformers you need to inspect.")
        if customer.empty:
            st.info("No duties have been dispatched to your team yet. Your assigned locations appear here as soon as the administrator dispatches cases to you.")
            st.stop()
    else:
        page_header("Geospatial operations", "Geographic View", "High-risk customer locations, area concentration, transformer clusters, and heatmap signals for field planning.")

    c_filter_1, c_filter_2 = st.columns([1, 1])
    with c_filter_1:
        metric = st.selectbox("Heatmap signal", ["Risk Score", "Fraud Probability", "Estimated Loss"])
    with c_filter_2:
        point_limit = st.slider("Customer points rendered", min_value=150, max_value=1000, value=500, step=50, help="Lower values make the map faster while the heatmap still uses all valid customers.")

    weight_col = {"Risk Score":"risk_score", "Fraud Probability":"fraud_probability", "Estimated Loss":"estimated_loss_all_30d"}.get(metric, "risk_score")

    if not FOLIUM_AVAILABLE:
        st.warning("Install folium and streamlit-folium to view maps.")
    elif not {"latitude","longitude"}.issubset(customer.columns):
        st.warning("No coordinates available. Connect GIS, meter registry, or generated prototype coordinates.")
    else:
        map_customer = _valid_albania_points(customer)
        if map_customer.empty:
            st.warning("No valid Albania coordinates were available after GIS validation. Check meter registry coordinates or rerun the pipeline with generated Albania GIS context.")
        else:
            center = [float(map_customer["latitude"].median()), float(map_customer["longitude"].median())]
            m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron", control_scale=True, prefer_canvas=True)
            Fullscreen().add_to(m)
            MiniMap(toggle_display=True).add_to(m)

            heat = map_customer[["latitude","longitude",weight_col]].dropna().copy()
            if not heat.empty:
                heat[weight_col] = pd.to_numeric(heat[weight_col], errors="coerce").fillna(0)
                max_w = max(float(heat[weight_col].max()), 1.0)
                HeatMap(
                    [[float(row["latitude"]), float(row["longitude"]), float(row[weight_col]) / max_w] for _, row in heat.iterrows()],
                    radius=16,
                    blur=21,
                    min_opacity=0.18,
                    name=f"{metric} heatmap",
                ).add_to(m)

            cluster = MarkerCluster(name="Customer investigation points", overlay=True, control=True).add_to(m)
            marker_source = map_customer.sort_values("risk_score", ascending=False).head(int(point_limit))
            for _, r in marker_source.iterrows():
                level = str(r.get("risk_level", ""))
                color = {"Low":"green", "Medium":"orange", "High":"red", "Critical":"darkred"}.get(level, "blue")
                popup = (
                    f"<b>{html.escape(str(r.get('customer_id','N/A')))}</b><br>"
                    f"Risk: {float(r.get('risk_score', 0)):.1f}<br>"
                    f"Level: {html.escape(level)}<br>"
                    f"Area: {html.escape(str(r.get('area_id','N/A')))}<br>"
                    f"Transformer: {html.escape(str(r.get('transformer_id','N/A')))}<br>"
                    f"{html.escape(str(r.get('main_reason','')))}"
                )
                folium.CircleMarker(
                    [float(r["latitude"]), float(r["longitude"])],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=.78,
                    weight=1,
                    popup=popup,
                ).add_to(cluster)

            folium.LayerControl(collapsed=True).add_to(m)
            st_folium(m, use_container_width=True, height=650)
            dropped = len(customer) - len(map_customer)
            st.caption(f"Heatmap uses {len(map_customer):,} validated customer coordinates. Showing top {len(marker_source):,} customer markers for performance.")
            if dropped > 0:
                st.caption(f"GIS validation hidden {dropped:,} records with invalid/off-country coordinates from the map view.")

    tab_area, tab_trans = st.tabs(["Area ranking", "Transformer ranking"])
    with tab_area:
        st.dataframe(clean_display_df(area, ["area_id","area_risk_score","area_risk_level","customers","high_risk_customers","critical_customers","anomaly_density","estimated_loss_all_30d"]), use_container_width=True, height=390, hide_index=True)
    with tab_trans:
        st.dataframe(clean_display_df(transformer.head(80)), use_container_width=True, height=390, hide_index=True)

elif page == "Weather Context":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    page_header("External context", "Weather Context", "Temperature-aware NTL analysis: cold and hot days are treated as expected higher-demand periods, so alerts focus on consumption-weather mismatches.")
    if daily.empty or "temp_mean" not in daily.columns:
        st.warning("Weather features are not available. Rerun the pipeline with the included Albania weather dataset.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Temperature", f"{daily['temp_mean'].mean():.1f} °C")
        c2.metric("Extreme Weather Days", f"{int((daily.get('weather_demand_pressure', 0)==1).sum()):,}")
        c3.metric("Mismatch Events", f"{int(daily.get('weather_consumption_mismatch', pd.Series(dtype=float)).sum()):,}")
        c4.metric("Customers with Weather Risk", f"{int((customer.get('weather_context_score', pd.Series(dtype=float)) > 40).sum()):,}")
        trend = daily.groupby("date").agg(
            Average_Consumption=("consumption_kwh", "mean"),
            Average_Temperature=("temp_mean", "mean"),
            Weather_Mismatch_Events=("weather_consumption_mismatch", "sum"),
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["Average_Consumption"], mode="lines", name="Average Consumption"))
        fig.add_trace(go.Scatter(x=trend["date"], y=trend["Average_Temperature"], mode="lines", name="Average Temperature", yaxis="y2"))
        fig.update_layout(title="Consumption vs Temperature Context", yaxis=dict(title="kWh"), yaxis2=dict(title="°C", overlaying="y", side="right"), legend=dict(orientation="h", y=-.18))
        render_chart(fig)
        top_weather = customer.sort_values("weather_context_score", ascending=False).head(50) if "weather_context_score" in customer.columns else pd.DataFrame()
        st.dataframe(clean_display_df(top_weather, ["customer_id", "risk_score", "risk_level", "weather_context_score", "weather_mismatch_ratio", "weather_mismatch_days", "avg_temp_mean", "area_id", "main_reason"]), use_container_width=True, height=430)

elif page == "Building Risk Lab":
    page_header("Asset intelligence", "Building / Village Risk Lab", "Independent module for buildings, apartment units, villages, and city zones. Upload or generate building-level consumption data, score fraud probability, visualize risk, and export reports without changing the main NTL pipeline.")
    BUILDING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    building_results = load_building_outputs(BUILDING_OUTPUT_DIR)
    unit_scores = building_results.get("unit_scores", pd.DataFrame())
    building_scores = building_results.get("building_scores", pd.DataFrame())
    floor_scores = building_results.get("floor_scores", pd.DataFrame())
    daily_building = building_results.get("daily_summary", pd.DataFrame())
    feature_importance = building_results.get("feature_importance", pd.DataFrame())
    building_metrics = building_results.get("metrics", {}) if isinstance(building_results.get("metrics", {}), dict) else {}
    building_report = str(building_results.get("report", ""))

    btab_build, btab_overview, btab_explorer, btab_map, btab_reports = st.tabs(["Create / Upload", "Overview", "Risk Explorer", "Map", "Reports"])

    with btab_build:
        left, right = st.columns([1.05, .95])
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Generate Albania building dataset")
            st.caption("Use this when you want a realistic city/village demo dataset with floors, units, weather context, and hidden fraud patterns.")
            g1, g2, g3 = st.columns(3)
            with g1:
                n_buildings = st.number_input("Buildings", min_value=10, max_value=260, value=80, step=10)
            with g2:
                avg_units = st.number_input("Average units / building", min_value=3, max_value=40, value=10, step=1)
            with g3:
                days = st.number_input("Days", min_value=30, max_value=730, value=180, step=30)
            g4, g5, g6 = st.columns(3)
            with g4:
                fraud_rate = st.slider("Hidden fraud rate", min_value=0.02, max_value=0.25, value=0.08, step=0.01)
            with g5:
                start_date = st.date_input("Start date", value=pd.Timestamp("2024-01-01"))
            with g6:
                seed = st.number_input("Seed", min_value=1, max_value=9999, value=42, step=1)
            if st.button("Generate and analyze building risk", type="primary", use_container_width=True):
                with st.spinner("Generating buildings, scoring units, and preparing reports..."):
                    df_build = generate_building_consumption_dataset(
                        n_buildings=int(n_buildings),
                        days=int(days),
                        start_date=str(start_date),
                        avg_units_per_building=int(avg_units),
                        fraud_rate=float(fraud_rate),
                        seed=int(seed),
                    )
                    df_build.to_csv(BUILDING_OUTPUT_DIR / "generated_building_input.csv", index=False)
                    analyze_building_dataset(df_build, output_dir=BUILDING_OUTPUT_DIR, seed=int(seed))
                st.success("Building Risk Lab dataset generated and analyzed. Open Overview, Risk Explorer, Map, or Reports.")
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Upload building / village dataset")
            st.caption("Required logical columns: date, building_id, unit_id/customer_id, consumption_kwh. Other columns are optional and inferred when missing.")
            bfile = st.file_uploader("Building consumption file", type=["csv", "xlsx", "xls", "zip"], key="building_file")
            if bfile is not None:
                suffix = Path(bfile.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(bfile.getbuffer())
                    tmp_build_path = tmp.name
                try:
                    preview_df = read_and_standardize_building_file(tmp_build_path)
                    st.success(f"File recognized: {len(preview_df):,} readings, {preview_df['unit_id'].nunique():,} units, {preview_df['building_id'].nunique():,} buildings.")
                    st.dataframe(clean_display_df(preview_df.head(12)), use_container_width=True, hide_index=True)
                    if st.button("Analyze uploaded building dataset", type="primary", use_container_width=True):
                        with st.spinner("Scoring uploaded building dataset..."):
                            analyze_building_dataset(preview_df, output_dir=BUILDING_OUTPUT_DIR)
                        st.success("Uploaded building dataset analyzed.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not process uploaded building dataset: {exc}")
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("Recommended schema for uploaded files"):
            st.code("""date, city, location_type, building_id, building_type, floor, unit_id, unit_type,
household_size, area_sqm, meter_type, connection_type, contracted_power_kw,
latitude, longitude, expected_kwh, consumption_kwh, fraud_label""", language="text")
            st.caption("Only date, building_id, unit_id/customer_id, and consumption_kwh are mandatory. The rest improves scoring and visualization.")

    with btab_overview:
        if unit_scores.empty or building_scores.empty:
            st.info("No Building Risk Lab analysis exists yet. Generate or upload data in the Create / Upload tab.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Buildings", f"{int(building_metrics.get('buildings', building_scores['building_id'].nunique())):,}", "Analyzed assets")
            c2.metric("Units", f"{int(building_metrics.get('units', unit_scores['unit_id'].nunique())):,}", "Apartments / shops / offices")
            c3.metric("High/Critical Units", f"{int(building_metrics.get('high_risk_units', unit_scores['risk_level'].isin(['High','Critical']).sum())):,}", "Inspection candidates")
            c4.metric("30-Day Loss", fmt(float(building_metrics.get("estimated_loss_lek_30d", unit_scores.get("estimated_loss_lek_30d", pd.Series(dtype=float)).sum())), 0, " Lek"), "Estimated suspicious loss")

            a, b = st.columns([1.2, .8])
            with a:
                daily_plot = daily_building.copy()
                if not daily_plot.empty:
                    daily_plot["date"] = pd.to_datetime(daily_plot["date"])
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_plot["date"], y=daily_plot["total_consumption_kwh"], mode="lines", name="Actual consumption", line=dict(width=3)))
                    fig.add_trace(go.Scatter(x=daily_plot["date"], y=daily_plot["expected_consumption_kwh"], mode="lines", name="Expected consumption", line=dict(width=3, dash="dot")))
                    fig.add_bar(x=daily_plot["date"], y=daily_plot["weather_mismatch_events"], name="Weather mismatch events", yaxis="y2", opacity=.32)
                    fig.update_layout(title="Building Consumption vs Expected Demand", yaxis_title="kWh", yaxis2=dict(title="Mismatch events", overlaying="y", side="right", showgrid=False))
                    render_chart(fig, height=390)
            with b:
                rc = unit_scores["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
                rc.columns = ["Risk Level", "Units"]
                fig = px.pie(rc, names="Risk Level", values="Units", hole=.62, color="Risk Level", color_discrete_map=RISK_COLORS, title="Unit Risk Mix")
                fig.update_traces(textposition="inside", textinfo="percent+label")
                render_chart(fig, height=390)

            c, d = st.columns(2)
            with c:
                topb = building_scores.head(12).sort_values("risk_score")
                fig = px.bar(topb, x="risk_score", y="building_id", orientation="h", color="risk_level", color_discrete_map=RISK_COLORS, title="Highest-Risk Buildings", labels={"risk_score":"Risk Score", "building_id":"Building"})
                render_chart(fig, height=420)
            with d:
                city = building_scores.groupby(["city", "location_type"], as_index=False).agg(risk_score=("risk_score", "mean"), high_risk_buildings=("risk_level", lambda x: int(pd.Series(x).isin(["High","Critical"]).sum())), buildings=("building_id", "nunique"))
                fig = px.scatter(city, x="buildings", y="risk_score", size="high_risk_buildings", color="location_type", hover_name="city", title="City / Village Risk Exposure", labels={"buildings":"Buildings analyzed", "risk_score":"Average Risk Score"})
                render_chart(fig, height=420)

    with btab_explorer:
        if unit_scores.empty:
            st.info("Generate or upload building data first.")
        else:
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                selected_city = st.multiselect("City / area", sorted(unit_scores["city"].dropna().unique().tolist()))
            with f2:
                selected_loc = st.multiselect("Location type", sorted(unit_scores["location_type"].dropna().unique().tolist()))
            with f3:
                selected_level = st.multiselect("Risk level", RISK_ORDER, default=["High", "Critical"])
            with f4:
                min_score = st.slider("Minimum risk score", 0, 100, 40)
            view = unit_scores.copy()
            if selected_city:
                view = view[view["city"].isin(selected_city)]
            if selected_loc:
                view = view[view["location_type"].isin(selected_loc)]
            if selected_level:
                view = view[view["risk_level"].isin(selected_level)]
            view = view[pd.to_numeric(view["risk_score"], errors="coerce").fillna(0) >= min_score]
            st.dataframe(clean_display_df(view[["priority_rank","unit_id","building_id","city","location_type","floor","risk_score","risk_level","fraud_probability","estimated_loss_lek_30d","main_reason","recommended_action"]].head(300)), use_container_width=True, height=390, hide_index=True)
            x1, x2 = st.columns([.95, 1.05])
            with x1:
                if not view.empty:
                    selected_unit = st.selectbox("Inspect unit", view["unit_id"].head(100).tolist())
                    row = unit_scores[unit_scores["unit_id"] == selected_unit].iloc[0]
                    driver_df = pd.DataFrame({
                        "Driver": ["AI Outlier", "Expected Gap", "Peer Gap", "Meter Behavior", "Weather Mismatch", "Loss Impact"],
                        "Score": [row.get("building_ai_anomaly_score",0), row.get("consumption_deviation_score",0), row.get("peer_deviation_score",0), row.get("meter_behavior_score",0), row.get("weather_context_score",0), row.get("loss_impact_score",0)],
                    })
                    fig = px.bar(driver_df.sort_values("Score"), x="Score", y="Driver", orientation="h", title=f"Risk Driver Breakdown — {selected_unit}", range_x=[0,100])
                    render_chart(fig, height=360)
                    st.info(str(row.get("alert_explanation", "No explanation available.")))
            with x2:
                raw_building = building_results.get("raw", pd.DataFrame())
                if not view.empty and isinstance(raw_building, pd.DataFrame) and not raw_building.empty:
                    selected_unit = locals().get("selected_unit", view["unit_id"].iloc[0])
                    ts = raw_building[raw_building["unit_id"] == selected_unit].copy()
                    if not ts.empty:
                        ts["date"] = pd.to_datetime(ts["date"])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ts["date"], y=ts["consumption_kwh"], name="Actual", mode="lines"))
                        fig.add_trace(go.Scatter(x=ts["date"], y=ts["expected_kwh"], name="Expected", mode="lines", line=dict(dash="dot")))
                        fig.update_layout(title="Unit Consumption Timeline", yaxis_title="kWh")
                        render_chart(fig, height=360)
            st.subheader("Building ranking")
            st.dataframe(clean_display_df(building_scores[["priority_rank","building_id","city","location_type","building_type","units","high_risk_units","critical_units","risk_score","risk_level","avg_fraud_probability","estimated_loss_lek_30d","top_reason"]].head(200)), use_container_width=True, height=360, hide_index=True)

    with btab_map:
        if building_scores.empty:
            st.info("Generate or upload building data first.")
        elif not FOLIUM_AVAILABLE:
            st.warning("Install folium and streamlit-folium to view maps.")
        else:
            m1, m2 = st.columns([1, 1])
            with m1:
                heat_metric = st.selectbox("Heatmap signal", ["Risk Score", "Fraud Probability", "Estimated Loss"], key="building_heat_signal")
            with m2:
                b_limit = st.slider("Building markers rendered", 50, 500, 180, 25)
            metric_map = {"Risk Score":"risk_score", "Fraud Probability":"avg_fraud_probability", "Estimated Loss":"estimated_loss_lek_30d"}
            bmap = building_scores.copy()
            bmap["latitude"] = pd.to_numeric(bmap["latitude"], errors="coerce")
            bmap["longitude"] = pd.to_numeric(bmap["longitude"], errors="coerce")
            bmap = bmap[(bmap["latitude"].between(39.65,42.65)) & (bmap["longitude"].between(19.28,21.05))]
            center = [float(bmap["latitude"].median()), float(bmap["longitude"].median())] if not bmap.empty else [41.3275, 19.8189]
            bm = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron", control_scale=True, prefer_canvas=True)
            Fullscreen().add_to(bm)
            col = metric_map[heat_metric]
            heatdf = bmap[["latitude", "longitude", col]].dropna()
            if not heatdf.empty:
                max_w = max(float(pd.to_numeric(heatdf[col], errors="coerce").max()), 1.0)
                HeatMap([[float(r["latitude"]), float(r["longitude"]), float(r[col]) / max_w] for _, r in heatdf.iterrows()], radius=22, blur=25, min_opacity=.16, name=f"{heat_metric} heatmap").add_to(bm)
            cl = MarkerCluster(name="Building investigation points", overlay=True, control=True).add_to(bm)
            for _, r in bmap.sort_values("risk_score", ascending=False).head(int(b_limit)).iterrows():
                level = str(r.get("risk_level", ""))
                color = {"Low":"green", "Medium":"orange", "High":"red", "Critical":"darkred"}.get(level, "blue")
                popup = f"<b>{html.escape(str(r.get('building_id','N/A')))}</b><br>City: {html.escape(str(r.get('city','N/A')))}<br>Risk: {float(r.get('risk_score',0)):.1f}<br>High-risk units: {int(r.get('high_risk_units',0))}/{int(r.get('units',0))}<br>Reason: {html.escape(str(r.get('top_reason','')))}"
                folium.CircleMarker([float(r["latitude"]), float(r["longitude"])], radius=6, color=color, fill=True, fill_color=color, fill_opacity=.8, weight=1, popup=popup).add_to(cl)
            folium.LayerControl(collapsed=True).add_to(bm)
            st_folium(bm, use_container_width=True, height=640)
            st.caption(f"Heatmap uses {len(bmap):,} validated building coordinates. Showing top {min(int(b_limit), len(bmap)):,} markers for responsiveness.")

    with btab_reports:
        if unit_scores.empty:
            st.info("Generate or upload building data first.")
        else:
            st.text_area("Operational report", value=building_report, height=380)
            dl1, dl2, dl3, dl4 = st.columns(4)
            with dl1:
                st.download_button("Download unit scores", data=unit_scores.to_csv(index=False), file_name="building_unit_risk_scores.csv", mime="text/csv", use_container_width=True)
            with dl2:
                st.download_button("Download building scores", data=building_scores.to_csv(index=False), file_name="building_risk_scores.csv", mime="text/csv", use_container_width=True)
            with dl3:
                st.download_button("Download floor scores", data=floor_scores.to_csv(index=False), file_name="floor_risk_scores.csv", mime="text/csv", use_container_width=True)
            with dl4:
                st.download_button("Download report", data=building_report, file_name="building_risk_operational_report.txt", mime="text/plain", use_container_width=True)
            if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty and "importance" in feature_importance:
                fi = feature_importance.dropna().head(12)
                if not fi.empty:
                    fig = px.bar(fi.sort_values("importance"), x="importance", y="feature", orientation="h", title="Supervised Model Feature Importance", labels={"importance":"Importance", "feature":"Feature"})
                    render_chart(fig, height=390)

elif page == "Forecasting":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    loss_forecast, area_forecast, customer_forecast = ensure_forecast_outputs(customer, area, daily)
    page_header("Forward planning", "Forecasting", "Predict next-period suspicious loss pressure, area risk, and customer inspection demand using recent consumption signals and risk drivers.")
    total_forecast_loss = pd.to_numeric(loss_forecast.get("predicted_loss_all", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() if len(loss_forecast) else 0
    last_30_loss = float(pd.to_numeric(customer.get("estimated_loss_all_30d", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    delta_pct = ((total_forecast_loss - last_30_loss) / last_30_loss * 100) if last_30_loss > 0 else 0.0
    top_area = area_forecast.iloc[0].get("area_id", "N/A") if len(area_forecast) else "N/A"
    lf = loss_forecast.copy()
    if not lf.empty:
        peak_idx = pd.to_numeric(lf["predicted_loss_all"], errors="coerce").fillna(0).idxmax()
        peak_day = pd.to_datetime(lf.loc[peak_idx, "date"]).strftime("%d %b")
    else:
        peak_day = "N/A"
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Projected loss · next 30 days", f"{total_forecast_loss:,.0f} Lek", f"{delta_pct:+.1f}% vs last 30 days", delta_color="inverse")
    c2.metric("Last 30 days (estimated)", f"{last_30_loss:,.0f} Lek")
    c3.metric("Peak exposure day", peak_day)
    c4.metric("Top forecast area", str(top_area).replace("_", " "))

    direction = "rise" if delta_pct > 4 else "fall" if delta_pct < -4 else "stay about flat"
    st.markdown(
        f'<div class="soft-card"><div class="soft-title">What this means</div><div class="soft-text">'
        f'Based on the recent trajectory of suspicious consumption and payment behaviour, non-technical losses are projected to '
        f'<b>{direction}</b> over the next 30 days — about <b>{total_forecast_loss:,.0f} Lek</b> '
        f'(<b>{delta_pct:+.1f}%</b> vs the last 30 days). Use this to size inspection capacity; it is a planning signal, not proof of theft.'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Loss Forecast", "Area Forecast", "Customer Forecast"])
    with tab1:
        if not loss_forecast.empty:
            if "cumulative_predicted_loss" not in lf.columns:
                lf["cumulative_predicted_loss"] = pd.to_numeric(lf["predicted_loss_all"], errors="coerce").fillna(0).cumsum()
            fig = go.Figure()
            if "forecast_lower_loss" in lf.columns and "forecast_upper_loss" in lf.columns:
                fig.add_trace(go.Scatter(x=lf["date"], y=lf["forecast_upper_loss"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=lf["date"], y=lf["forecast_lower_loss"], mode="lines", fill="tonexty", line=dict(width=0), name="Planning range", fillcolor="rgba(37,99,235,.12)", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=lf["date"], y=lf["predicted_loss_all"], mode="lines+markers", name="Predicted daily loss (Lek)", line=dict(width=3, color="#2563eb")))
            fig.add_trace(go.Scatter(x=lf["date"], y=lf["cumulative_predicted_loss"], mode="lines", name="Cumulative projected loss (Lek)", yaxis="y2", line=dict(width=2.5, color="#ef4444", dash="dot")))
            fig.update_layout(
                title="Next 30-Day NTL Loss Forecast (daily + cumulative)",
                yaxis=dict(title="Daily predicted loss (Lek)"),
                yaxis2=dict(title="Cumulative loss (Lek)", overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=-.18),
                height=470,
            )
            render_chart(fig)
            st.dataframe(clean_display_df(lf.head(30), ["date", "predicted_loss_all", "cumulative_predicted_loss", "predicted_total_events", "forecast_lower_loss", "forecast_upper_loss"]), use_container_width=True, height=320, hide_index=True)
        else:
            st.info("Forecast output is not available yet. Run the pipeline again from Data Intake.")
    with tab2:
        st.dataframe(clean_display_df(area_forecast.head(30), ["forecast_priority","area_id","forecasted_area_risk_next_30d","area_risk_score","anomaly_density","high_risk_customers","estimated_loss_all_30d"]), use_container_width=True, height=520, hide_index=True)
    with tab3:
        st.dataframe(clean_display_df(customer_forecast.head(50)), use_container_width=True, height=520, hide_index=True)

elif page == "Assistant":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    cases = create_cases_from_priority(priority)
    loss_forecast, area_forecast, customer_forecast = ensure_forecast_outputs(customer, area, daily)
    page_header("Decision support", "Operations Assistant", "Ask operational questions about customers, hotspots, forecasts, data quality, model metrics, and missing workflow features.")

    def _secret(name: str, default: str = "") -> str:
        env_val = os.environ.get(name, "")
        if env_val:
            return env_val
        try:
            return str(st.secrets.get(name, "") or default)
        except Exception:
            return default

    GEMINI_API_KEY = _secret("GEMINI_API_KEY")
    GEMINI_MODEL = _secret("GEMINI_MODEL", "gemini-2.0-flash")

    if not GEMINI_API_KEY:
        st.info(
            "The AI key is not configured. The assistant will use the built-in operational analyst mode, "
            "so the workspace remains usable without external services."
        )

    st.markdown('<div class="assistant-box"><b>Suggested questions</b><br><span class="assistant-suggestion">Which customers should we inspect first?</span><span class="assistant-suggestion">What is the highest-risk area?</span><span class="assistant-suggestion">Forecast next 30 days</span><span class="assistant-suggestion">What workflow gaps should OSHEE improve?</span><span class="assistant-suggestion">Explain customer C100295</span></div>', unsafe_allow_html=True)

    if "assistant_messages" not in st.session_state:
        st.session_state.assistant_messages = [{"role": "assistant", "content": "I can help diagnose NTL risk, prioritize inspections, explain customers, summarize data quality, and identify workflow gaps."}]

    def _answer(question: str) -> str:
        if not GEMINI_API_KEY:
            return assistant_response(question, customer, area, priority, daily, cases, metrics, ingestion)
        context_json = build_platform_context(customer, area, priority, cases, metrics, ingestion, loss_forecast, area_forecast)
        try:
            history = [(("user" if m["role"] == "user" else "model"), m["content"]) for m in st.session_state.assistant_messages[-6:]]
            return ask_gemini(question, context_json, history, GEMINI_API_KEY, GEMINI_MODEL)
        except Exception as exc:
            fallback = assistant_response(question, customer, area, priority, daily, cases, metrics, ingestion)
            detail = str(exc).strip().replace("\n", " ")
            if len(detail) > 160:
                detail = detail[:160] + "..."
            return f"{fallback}\n\n*(The AI assistant was unavailable — {detail or type(exc).__name__} — showing built-in analysis instead.)*"

    @st.fragment
    def assistant_chat():
        quick_cols = st.columns(4)
        quick_prompts = ["Top inspection priorities", "Highest-risk area", "Forecast next 30 days", "Workflow gaps"]

        new_question = None
        for i, qp in enumerate(quick_prompts):
            if quick_cols[i].button(qp, use_container_width=True, key=f"qp_{i}"):
                new_question = qp

        prompt = st.chat_input("Ask about risk, customers, hotspots, forecasts, data quality, or operational gaps")
        if prompt:
            new_question = prompt

        if new_question:
            st.session_state.assistant_messages.append({"role": "user", "content": new_question})
            with st.spinner("Analyzing operations data..."):
                reply = _answer(new_question)
            st.session_state.assistant_messages.append({"role": "assistant", "content": reply})

        # Custom chat rendering prevents broken material-icon labels such as smart_toy/face.
        rendered = ['<div class="chat-thread">']
        for msg in st.session_state.assistant_messages[-12:]:
            rendered.append(_chat_html_message(msg.get("role", "assistant"), msg.get("content", "")))
        rendered.append('</div>')
        st.markdown("".join(rendered), unsafe_allow_html=True)


    assistant_chat()

elif page == "Model Governance":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    page_header("Analytical validation", "Model Governance", "Model validation, top-k inspection quality, correlation analysis, and statistical indicators for analytical teams.")
    r2_value = r2_score_manual(customer.get("last_30_mean", pd.Series(dtype=float)), customer.get("expected_last_30_consumption", pd.Series(dtype=float)))
    mae = metrics.get("expected_consumption", {}).get("expected_consumption_mae")
    models = metrics.get("models", {}) if isinstance(metrics, dict) else {}
    roc_auc = models.get("fraud_classifier_roc_auc")
    avg_precision = models.get("fraud_classifier_avg_precision")
    p10 = models.get("precision_at_top_10_percent")
    r10 = models.get("recall_at_top_10_percent")
    n_features = models.get("n_model_features")
    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Expected R²", "N/A" if r2_value is None else f"{r2_value:.3f}")
    c2.metric("Expected MAE", "N/A" if mae is None else f"{mae:.3f}")
    c3.metric("ROC AUC", "N/A" if roc_auc is None else f"{roc_auc:.3f}")
    c4.metric("Avg Precision", "N/A" if avg_precision is None else f"{avg_precision:.3f}")
    c5.metric("Precision@Top10%", "N/A" if p10 is None else f"{p10:.3f}")
    c6.metric("Recall@Top10%", "N/A" if r10 is None else f"{r10:.3f}")
    c7.metric("Features", "N/A" if n_features is None else f"{int(float(n_features))}")
    st.markdown(
        '<div class="status-card status-warning"><b>Validation scope:</b> these metrics are measured on the currently '
        'loaded dataset. Scores are decision-support metrics, not proof of theft. Validate against a '
        'held-out slice of real metered data and confirmed inspection outcomes before using these numbers operationally.</div>',
        unsafe_allow_html=True,
    )
    tab1, tab2, tab3 = st.tabs(["Indicators", "Correlation heatmap", "Probability analysis"])
    with tab1:
        stats = pd.DataFrame([
            ["Mean customer consumption", customer["avg_consumption"].mean(), "Average daily consumption across customers."],
            ["Median customer consumption", customer["median_consumption"].median(), "Typical customer consumption level."],
            ["Consumption volatility", customer["std_consumption"].mean(), "Average instability in consumption."],
            ["Mean risk score", customer["risk_score"].mean(), "Average suspicious behavior score."],
            ["High-risk customer rate", customer["risk_level"].isin(["High","Critical"]).mean()*100, "Share of customers classified as High or Critical."],
            ["Critical customer rate", (customer["risk_level"] == "Critical").mean()*100, "Share requiring urgent verification."],
            ["Total estimated 30-day loss", customer["estimated_loss_all_30d"].sum(), "Estimated suspicious under-recorded value."],
        ], columns=["Indicator", "Value", "Meaning"])
        stats["Value"] = stats["Value"].map(lambda x: f"{x:,.2f}")
        st.dataframe(stats, use_container_width=True, hide_index=True)
    with tab2:
        corr_cols = ["risk_score","fraud_probability","ai_anomaly_score","historical_deviation_score","peer_deviation_score","geographic_risk_score","sudden_flags_score","recent_drop_pct","zero_day_ratio","flatline_ratio","sudden_drop_count","expected_deviation_pct","payment_risk_score","unpaid_bills","arrears_amount_lek","estimated_loss_all_30d"]
        corr_cols = [c for c in corr_cols if c in customer.columns and pd.to_numeric(customer[c], errors="coerce").notna().sum() > 2]
        corr = customer[corr_cols].apply(pd.to_numeric, errors="coerce").corr().round(2)
        corr.index = [pretty_label(c) for c in corr.index]
        corr.columns = [pretty_label(c) for c in corr.columns]
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", title="Correlation Heatmap of Risk Drivers")
        fig.update_layout(height=760)
        render_chart(fig)
    with tab3:
        prob_col = "fraud_probability" if "fraud_probability" in customer.columns else "risk_score"
        fig = px.scatter(customer, x="risk_score", y=prob_col, color="risk_level", hover_data=["customer_id","area_id","main_reason"], title="Probability Signal Compared With Risk Score", color_discrete_map=RISK_COLORS, labels={"risk_score":"Risk Score", prob_col: pretty_label(prob_col), "risk_level":"Risk Level"})
        render_chart(fig)

elif page == "Cases":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    page_header("Inspection operations", "Case Management", "Turn high-risk alerts into assigned inspection cases, track status, and capture verification outcomes for future model feedback.")
    cases = create_cases_from_priority(priority)
    status_filter = st.multiselect("Status", ["New","Assigned","In Field","Inspected","Confirmed NTL","False Alarm"], default=["New","Assigned","In Field","Inspected","Confirmed NTL","False Alarm"])
    view = cases[cases["status"].isin(status_filter)].copy() if status_filter else cases.copy()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Open Cases", f"{len(cases[~cases['status'].isin(['Confirmed NTL','False Alarm'])]):,}")
    c2.metric("Confirmed NTL", f"{int((cases['status']=='Confirmed NTL').sum()):,}")
    c3.metric("False Alarms", f"{int((cases['status']=='False Alarm').sum()):,}")
    c4.metric("Potential Loss", f"{cases.get('estimated_loss_all_30d', pd.Series(dtype=float)).sum():,.0f} Lek")
    st.dataframe(clean_display_df(view, ["case_id","priority_rank","customer_id","risk_score","risk_level","area_id","transformer_id","assigned_team","status","inspection_result","estimated_loss_all_30d","main_reason","recommended_action"]), use_container_width=True, height=520)
    with st.expander("Update / dispatch a case", expanded=False):
        case_id = st.selectbox("Case", cases["case_id"].tolist())
        idx = cases.index[cases["case_id"] == case_id][0]
        new_status = st.selectbox("Status", ["New","Assigned","In Field","Inspected","Confirmed NTL","False Alarm"], index=["New","Assigned","In Field","Inspected","Confirmed NTL","False Alarm"].index(cases.loc[idx,"status"]) if cases.loc[idx,"status"] in ["New","Assigned","In Field","Inspected","Confirmed NTL","False Alarm"] else 0)
        team_options = workspace.inspector_teams()
        cur_team = str(cases.loc[idx, "assigned_team"])
        if cur_team not in team_options:
            team_options = team_options + [cur_team]
        team = st.selectbox("Assigned team", team_options, index=team_options.index(cur_team) if cur_team in team_options else 0)
        result = st.text_input("Inspection result", value=str(cases.loc[idx,"inspection_result"]))
        notes = st.text_area("Field notes", value=str(cases.loc[idx,"field_notes"]) if pd.notna(cases.loc[idx,"field_notes"]) else "")
        if st.button("Save case update", type="primary"):
            cases.loc[idx, "status"] = new_status
            cases.loc[idx, "assigned_team"] = team
            cases.loc[idx, "inspection_result"] = result
            cases.loc[idx, "field_notes"] = notes
            save_cases(cases)
            workspace.log_activity(OUTPUT_DIR, user["name"], role, f"Updated case {case_id}", f"{new_status} · {team}")
            st.success("Case updated.")
            st.rerun()

elif page == "Inspector Mobile":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    cases = create_cases_from_priority(priority)
    my_team = user.get("team")
    page_header("Field workflow", "My Inspections", f"Duties dispatched to {my_team or 'your team'} by OSHEE administration. Inspect the meter, then report the outcome — administration sees it live.")
    live_status_strip()

    if not my_team:
        st.info("Your account is not linked to an inspection team. Ask the administrator to assign one.")
    else:
        @st.fragment(run_every=6)
        def inspector_board():
            live = create_cases_from_priority(priority)
            board = live[(live["assigned_team"] == my_team) & (live["status"] != "New")].copy()
            board = board.sort_values("risk_score", ascending=False)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dispatched to me", f"{len(board):,}")
            c2.metric("High/Critical", f"{int(board['risk_level'].isin(['High','Critical']).sum()):,}" if 'risk_level' in board.columns else "0")
            c3.metric("Potential Loss", f"{pd.to_numeric(board.get('estimated_loss_all_30d', pd.Series(dtype=float)), errors='coerce').fillna(0).sum():,.0f} Lek")
            c4.metric("Still open", f"{int((~board['status'].isin(['Confirmed NTL','False Alarm'])).sum()):,}" if 'status' in board.columns else f"{len(board):,}")
            st.markdown('<div class="field-board"><div class="soft-title"><span class="live-dot"></span>Today\'s field queue</div><div class="soft-text">New duties from administration appear here automatically. Each card gives the customer, location, reason, and recommended check.</div>', unsafe_allow_html=True)
            if board.empty:
                st.caption("No duties dispatched to your team yet. They will appear here as soon as the administrator dispatches them.")
            else:
                card_html = ""
                for _, r in board.head(20).iterrows():
                    card_html += f"""
                    <div class="field-card">
                        <b>{html.escape(str(r.get('case_id')))} · {html.escape(str(r.get('customer_id')))}</b> &nbsp; {risk_pill(r.get('risk_level'))}
                        <div class="small-muted" style="margin-top:8px;"><b>Status:</b> {html.escape(str(r.get('status')))} · <b>Area:</b> {html.escape(str(r.get('area_id')))} · <b>Transformer:</b> {html.escape(str(r.get('transformer_id')))}</div>
                        <div class="small-muted"><b>Reason:</b> {html.escape(str(r.get('main_reason')))}</div>
                        <div class="small-muted"><b>Action:</b> {html.escape(str(r.get('recommended_action')))}</div>
                    </div>
                    """
                st.markdown(card_html + '</div>', unsafe_allow_html=True)
        inspector_board()

        st.subheader("Report inspection outcome")
        my_cases = cases[(cases["assigned_team"] == my_team) & (cases["status"] != "New")].copy()
        if my_cases.empty:
            st.caption("No dispatched duties to report on yet.")
        else:
            STATUS_OPTIONS = ["Assigned", "In Field", "Inspected", "Confirmed NTL", "False Alarm"]
            RESULT_OPTIONS = ["Pending", "Meter OK", "Tampering found", "Illegal connection", "Meter bypass", "Inconclusive"]
            case_id = st.selectbox("Case", my_cases.sort_values("risk_score", ascending=False)["case_id"].tolist())
            idx = cases.index[cases["case_id"] == case_id][0]
            cur_status = str(cases.loc[idx, "status"])
            o1, o2 = st.columns(2)
            with o1:
                new_status = st.selectbox("Status", STATUS_OPTIONS, index=STATUS_OPTIONS.index(cur_status) if cur_status in STATUS_OPTIONS else 0)
            with o2:
                cur_result = str(cases.loc[idx, "inspection_result"])
                new_result = st.selectbox("Result", RESULT_OPTIONS, index=RESULT_OPTIONS.index(cur_result) if cur_result in RESULT_OPTIONS else 0)
            notes = st.text_area("Field notes", value=str(cases.loc[idx, "field_notes"]) if pd.notna(cases.loc[idx, "field_notes"]) else "")
            if st.button("Save and report outcome", type="primary"):
                cases.loc[idx, "status"] = new_status
                cases.loc[idx, "inspection_result"] = new_result
                cases.loc[idx, "field_notes"] = notes
                save_cases(cases)
                workspace.log_activity(OUTPUT_DIR, user["name"], role, f"Reported {case_id}", f"{new_status} / {new_result}")
                st.success("Outcome saved and reported to administration.")
                st.rerun()


elif page == "Reports":
    customer, area, transformer, priority, daily, summary, metrics, ingestion = load_outputs()
    cases = create_cases_from_priority(priority)
    loss_forecast, area_forecast, customer_forecast = ensure_forecast_outputs(customer, area, daily)
    audit = build_operational_audit(customer, area, priority, daily, ingestion)
    page_header("Management outputs", "Reports", "Export management summaries, inspection registers, case logs, forecasts, and workflow improvement findings.")
    report_text = make_report_summary(customer, area, cases, loss_forecast)
    st.markdown(f'<div class="report-summary">{html.escape(report_text)}</div>', unsafe_allow_html=True)
    st.subheader("Operational workflow audit")
    st.dataframe(clean_display_df(audit), use_container_width=True, height=330, hide_index=True)
    st.subheader("Exports")
    c1,c2,c3 = st.columns(3)
    c1.download_button("Download risk register", customer.to_csv(index=False), file_name="energyshield_customer_risk_register.csv", use_container_width=True)
    c2.download_button("Download inspection queue", priority.to_csv(index=False), file_name="energyshield_inspection_queue.csv", use_container_width=True)
    c3.download_button("Download case log", cases.to_csv(index=False), file_name="energyshield_case_log.csv", use_container_width=True)
    c4,c5,c6 = st.columns(3)
    c4.download_button("Download loss forecast", loss_forecast.to_csv(index=False), file_name="energyshield_loss_forecast.csv", use_container_width=True)
    c5.download_button("Download area forecast", area_forecast.to_csv(index=False), file_name="energyshield_area_forecast.csv", use_container_width=True)
    c6.download_button("Download workflow audit", audit.to_csv(index=False), file_name="energyshield_workflow_audit.csv", use_container_width=True)
    st.download_button("Download executive summary", report_text, file_name="energyshield_executive_summary.txt", use_container_width=True)
