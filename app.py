"""
EDEM Dataset Creator — Streamlit Application
=============================================
A professional web app for collecting, processing, and organizing
Mill EDEM Simulation data into a structured dataset for ML/AI training.

Merges the functionality of:
  - analyze_liner.py     (STEP file → angular profile + plots)
  - extract_ml_features.py (angular profile → ML feature vector)
  - dem_rotation_converter.py (DEM CSV → 100-point rotation data)

All intermediate data is handled in-memory; only the final dataset
rows, analysis images, and angular profile CSV are written to disk.
"""

import streamlit as st
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import os
import traceback
import json
from PIL import Image
from analysis_engine import start_analysis_process, convert_dem_to_rotation_pct
import tempfile
import subprocess
import sys as _sys
import io
import zipfile
import threading
import contextlib

# ─────────────────────────────────────────────────────────────────────────────
# Page config – must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

@st.fragment(run_every=1)
def render_analysis_progress():
    if st.session_state.get("analysis_just_finished"):
        st.session_state["analysis_just_finished"] = False
        st.rerun()

    # Poll multiprocessing queue
    if "_analysis_queue" in st.session_state:
        q = st.session_state["_analysis_queue"]
        import queue
        while True:
            try:
                msg = q.get_nowait()
                if msg["type"] == "log":
                    st.session_state.setdefault("analysis_log", []).append((msg["level"], msg["msg"]))
                elif msg["type"] == "progress":
                    st.session_state["analysis_progress"] = msg["val"]
                    if "stage" in msg:
                        st.session_state["analysis_stage"] = msg["stage"]
                elif msg["type"] == "done":
                    res = msg["results"]
                    for k, v in res.items():
                        st.session_state[k] = v
                    st.session_state["is_analysing"] = False
                    st.session_state["analysis_just_finished"] = True
                    break
                elif msg["type"] == "error":
                    st.session_state.setdefault("analysis_log", []).append(("err", msg["error"]))
                    st.session_state["is_analysing"] = False
                    st.session_state["analysis_just_finished"] = True
                    break
            except queue.Empty:
                break
        
        # If the process died unexpectedly
        if "_analysis_process" in st.session_state:
            p = st.session_state["_analysis_process"]
            if not p.is_alive() and st.session_state.get("is_analysing"):
                st.session_state.setdefault("analysis_log", []).append(("err", "Analysis process crashed silently."))
                st.session_state["is_analysing"] = False
                st.session_state["analysis_just_finished"] = True

    if st.session_state.get("is_analysing"):
        stage = st.session_state.get("analysis_stage", "Initializing...")
        prog  = st.session_state.get("analysis_progress", 0)

        st.markdown(f'''
<div class="info-banner" style="margin-bottom:8px;">
<span class="spinner"></span> <b>{stage}</b><br>
<span style="font-size:0.8rem;color:#64748b;">You may continue filling out the form below.</span>
</div>
        ''', unsafe_allow_html=True)

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown('<style>div[data-testid="stProgressBar"]>div>div{background-color:#16a34a!important;}</style>', unsafe_allow_html=True)
            st.progress(int(prog))
        with col2:
            if st.button("⏹ Stop", type="primary"):
                if "_analysis_process" in st.session_state:
                    p = st.session_state["_analysis_process"]
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=1.0)
                
                # Instantly reset the UI to the fallback configuration
                st.session_state["is_analysing"] = False
                st.session_state["analysis_log"] = []
                st.session_state["analysis_stage"] = ""
                st.session_state["analysis_progress"] = 0
                st.session_state["analysis_just_finished"] = False
                # Clear the STEP file uploader and its data
                st.session_state["step_uploader_key"] = st.session_state.get("step_uploader_key", 0) + 1
                st.session_state["step_file_path"] = None
                st.session_state["step_uploaded_name"] = ""
                st.rerun()

                st.session_state["step_uploader_key"] = st.session_state.get("step_uploader_key", 0) + 1
                st.session_state["step_file_path"] = None
                st.session_state["step_uploaded_name"] = ""
                st.rerun()
                st.rerun()

    log_lines = st.session_state.get("analysis_log", [])
    if log_lines:
        with st.expander("Analysis Log", expanded=False):
            _render_log(log_lines)


@st.fragment
def render_dem_uploader():
    uploaded_csv = st.file_uploader(
        "DEM CSV", type=["csv"], key=f"dem_csv_upload_{st.session_state.get('dem_uploader_key', 0)}", label_visibility="collapsed")

    if uploaded_csv is not None:
        target_rpm = st.session_state.get("inp_mill_rpm", 0.0)
        if target_rpm <= 0.0:
            st.error("Please enter a valid Mill RPM in the 'Mill Operating Conditions' section above before uploading the DEM CSV.")
        else:
            try:
                dem_df, rpm, fraction = convert_dem_to_rotation_pct(uploaded_csv.read(), target_rpm)
                st.session_state["dem_rotation_df"] = dem_df
                st.session_state["rpm_computed"]    = rpm
                st.session_state["dem_rotation_fraction"] = fraction
                
                # Auto-populate the RPM field if it hasn't been set yet
                if st.session_state.get("mill_rpm", 0.0) == 0.0:
                    st.session_state["mill_rpm"] = rpm
                    st.session_state["inp_mill_rpm"] = rpm
                    
                action_text = "Last full rotation extracted." if fraction >= 1.0 else "Modulo time-wrapper synthesized 1 rotation."
                st.markdown(
                    f'<div class="success-banner">DEM CSV parsed &mdash; '
                    f'<b>{(fraction * 100):.1f}%</b> of a rotation detected '
                    f'(RPM = {rpm:.3f}). {action_text}</div>',
                    unsafe_allow_html=True)
                with st.expander("Preview DEM data (first 10 rows)"):
                    st.table(dem_df.head(10))
            except Exception as e:
                st.markdown(f'<div class="warn-banner">Failed to parse DEM CSV: {e}</div>', unsafe_allow_html=True)
    elif st.session_state.get("dem_rotation_df") is not None:
        frac = st.session_state.get("dem_rotation_fraction", 1.0)
        st.markdown(
            f'<div class="success-banner">DEM data loaded &mdash; '
            f'<b>{(frac * 100):.1f}%</b> of a rotation detected '
            f'(RPM = {st.session_state["rpm_computed"]:.3f})</div>',
            unsafe_allow_html=True)



def _render_log(log_lines):
    """Render verbose log lines in a dark terminal-style box."""
    if not log_lines:
        return
    level_class = {"ok": "log-ok", "inf": "log-inf", "wrn": "log-wrn", "err": "log-err"}
    html_lines = []
    for lvl, msg in log_lines:
        css = level_class.get(lvl, "")
        msg_escaped = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        html_lines.append(f'<span class="{css}">{msg_escaped}</span>')
    st.markdown(
        '<div class="log-box">' + "<br>".join(html_lines) + '</div>',
        unsafe_allow_html=True)

import base64

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

def draw_splash(placeholder, pct, status_text):
    splash_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .splash-overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: #f7f8fc;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a;
    }}
    .splash-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        max-width: 460px;
        width: 90%;
    }}
    .splash-title {{
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        letter-spacing: -0.025em;
        margin-bottom: 32px !important;
    }}
    .splash-spinner-wrapper {{
        position: relative;
        margin-bottom: 26px !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .splash-spinner {{
        width: 44px;
        height: 44px;
        border: 3.5px solid #e2e8f0;
        border-top: 3.5px solid #2563eb;
        border-radius: 50%;
        animation: splashSpin 0.75s linear infinite;
    }}
    @keyframes splashSpin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    .splash-progress-track {{
        width: 280px;
        height: 4px;
        background: #e2e8f0;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 18px !important;
    }}
    .splash-progress-fill {{
        height: 100%;
        width: {pct}%;
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%);
        border-radius: 999px;
        transition: width 0.15s ease-out;
    }}
    .splash-pct-badge {{
        font-weight: 600 !important;
        color: #2563eb !important;
        background: #eff6ff !important;
        border: 1px solid #dbeafe !important;
        padding: 3px 10px !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        min-width: 48px;
        text-align: center;
        margin-bottom: 8px !important;
        display: inline-block;
    }}
    .splash-status {{
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #64748b !important;
        min-height: 22px !important;
    }}
    </style>
    
    <div class="splash-overlay">
        <div class="splash-container">
            <div class="splash-title">EDEM Surrogate Model</div>
            <div class="splash-spinner-wrapper">
                <div class="splash-spinner"></div>
            </div>
            <div class="splash-pct-badge">{pct}%</div>
            <div class="splash-progress-track">
                <div class="splash-progress-fill"></div>
            </div>
            <div class="splash-status">{status_text}</div>
        </div>
    </div>
    """
    placeholder.markdown(splash_html, unsafe_allow_html=True)

SPLASH_MILESTONES = [
    (0, 15, "Initializing runtime environment & session state..."),
    (16, 30, "Importing core numerical & surrogate modeling libraries..."),
    (31, 45, "Loading pre-trained machine learning model weight files..."),
    (46, 60, "Validating backend preprocessing & data pipelines..."),
    (61, 75, "Initializing session state stores & data structures..."),
    (76, 92, "Building page components & layout stylesheet tokens..."),
    (93, 100, "System Ready — Webpage Loaded Successfully")
]

def get_splash_status_text(pct):
    for start, end, text in SPLASH_MILESTONES:
        if start <= pct <= end:
            return text
    return "Loading application components..."

def animate_splash_step(placeholder, from_pct, to_pct, delay=0.025):
    if not placeholder:
        return
    for p in range(from_pct, to_pct + 1):
        status_text = get_splash_status_text(p)
        draw_splash(placeholder, p, status_text)
        time.sleep(delay)

def main():
    show_splash = not st.session_state.get("splash_shown", False)
    splash_placeholder = st.empty() if show_splash else None

    if show_splash:
        animate_splash_step(splash_placeholder, 0, 25, delay=0.03)

    # ─────────────────────────────────────────────────────────────────────────────
    # Session state initialisation helpers
    # ─────────────────────────────────────────────────────────────────────────────
    def _defaults():
        return {
            "excel_path": None,
            "mode": "Add Data",
            # ── liner analysis ──────────────────────────────────
            "step_file_path": None,
            "angular_profile_df": None,
            "ml_features": None,
            "liner_profile_img": None,
            "face_analysis_img": None,
            "analysis_img": None,
            "n_lifters_detected": 0,
            "analysis_log": [],          # list of (level, message) strings
            # ── DEM CSV ─────────────────────────────────────────
            "dem_rotation_df": None,
            "rpm_computed": None,
            # ── form fields ─────────────────────────────────────
            "mill_type": "Ball Mill",
            "ore_density": 2700.0,
            "ore_poisson": 0.25,
            "ore_shear_m": 1e8,
            "ore_radius": 0.0,
            "ore_psd_s0": 0.5,  "ore_psd_p0": 20.0,
            "ore_psd_s1": 0.7,  "ore_psd_p1": 50.0,
            "ore_psd_s2": 1.0,  "ore_psd_p2": 30.0,
            "ore_mass": 0.0,
            "liner_density": 7800.0,
            "liner_poisson": 0.28,
            "liner_shear_m": 7e10,
            "liner_same_as_media": False,
            "media_density": 7800.0,
            "media_poisson": 0.28,
            "media_shear_m": 7e10,
            "media_radius": 0.0,
            "media_psd_s0": 0.5,  "media_psd_p0": 20.0,
            "media_psd_s1": 0.7,  "media_psd_p1": 50.0,
            "media_psd_s2": 1.0,  "media_psd_p2": 30.0,
            "media_mass": 0.0,
            "mill_rpm": 0.0,
            "eff_mill_dia": 0.0,
            # interaction properties  (ol = Ore-Liner)
            "mm_rf": 0.001, "mo_rf": 0.002, "oo_rf": 0.005, "ml_rf": 0.001, "ol_rf": 0.002,
            "mm_sf": 0.3,   "mo_sf": 0.5,   "oo_sf": 0.8,   "ml_sf": 0.3,   "ol_sf": 0.5,
            "mm_res": 0.3,  "mo_res": 0.5,  "oo_res": 0.5,  "ml_res": 0.3,  "ol_res": 0.5,
            # update mode
            "selected_sim_id": None,
            "n_harmonics": 28,
        }

    SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    CONFIG_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

    def _save_settings(excel_path=None, last_page=None, n_harmonics=None):
        try:
            data = {}
            if os.path.exists(SETTINGS_PATH):
                try:
                    with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}
            elif os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    data = {}

            if excel_path is not None:
                data["excel_path"] = excel_path
            elif "excel_path" not in data and st.session_state.get("excel_path"):
                data["excel_path"] = st.session_state.get("excel_path")

            if last_page is not None:
                data["last_page"] = last_page
            elif "last_page" not in data and st.session_state.get("mode"):
                data["last_page"] = st.session_state.get("mode")

            if n_harmonics is not None:
                data["n_harmonics"] = n_harmonics
            elif "n_harmonics" not in data and st.session_state.get("n_harmonics"):
                data["n_harmonics"] = st.session_state.get("n_harmonics")

            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
        st.session_state["suggested_mill_dia"] = None

    def _resolve_default_excel():
        cand_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp1_DataSet.xlsx"),
            os.path.join(os.getcwd(), "temp1_DataSet.xlsx"),
            os.path.abspath("temp1_DataSet.xlsx"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "DEM_Surrogate_temp1_DataSet", "PRE PROCESSED", "data_v1.xlsx"),
        ]
        for p in cand_paths:
            if os.path.exists(p):
                return os.path.abspath(p)
        return None

    _save_config = _save_settings

    def init_state():
        for k, v in _defaults().items():
            if k not in st.session_state:
                st.session_state[k] = v

        if not st.session_state.get("_config_loaded"):
            st.session_state["_config_loaded"] = True
            
            valid_pages = ["Add Data", "View / Update Record", "Data Report", "Prediction Model"]
            loaded_page = "Add Data"
            
            target_path = SETTINGS_PATH if os.path.exists(SETTINGS_PATH) else (CONFIG_PATH if os.path.exists(CONFIG_PATH) else None)
            if target_path:
                try:
                    with open(target_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data.get("last_page") in valid_pages:
                            loaded_page = data["last_page"]
                        raw_excel = data.get("excel_path")
                        if raw_excel:
                            if os.path.exists(raw_excel):
                                st.session_state["excel_path"] = os.path.abspath(raw_excel)
                            elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), raw_excel)):
                                st.session_state["excel_path"] = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), raw_excel))
                        if data.get("n_harmonics"):
                            st.session_state["n_harmonics"] = int(data["n_harmonics"])
                except Exception:
                    pass

            st.session_state["mode"] = loaded_page
            st.session_state["mode_radio"] = loaded_page

        curr_excel = st.session_state.get("excel_path")
        if not curr_excel or not os.path.exists(curr_excel):
            def_excel = _resolve_default_excel()
            if def_excel:
                st.session_state["excel_path"] = def_excel

    init_state()

    PAGE_TITLE_MAP = {
        "Add Data": "Data Creator",
        "View / Update Record": "Update/View Data",
        "Data Report": "Data Report",
        "Prediction Model": "Prediction Model"
    }

    current_mode_name = st.session_state.get("mode_radio") or st.session_state.get("mode") or "Add Data"
    page_title_suffix = PAGE_TITLE_MAP.get(current_mode_name, current_mode_name)
    tab_title = f"EDEM Surrogate - {page_title_suffix}"

    st.set_page_config(
        page_title=tab_title,
        layout="centered",
        initial_sidebar_state="expanded",
    )

    if show_splash:
        animate_splash_step(splash_placeholder, 26, 50, delay=0.03)

    # ─────────────────────────────────────────────────────────────────────────────
    # CSS — Professional, clean, Inter-based design
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html {
        font-size: 17px !important;
    }
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── App background & layout ─────────────────────── */
    .stApp { background: #f7f8fc; color: #1e293b; }

    .main .block-container {
        max-width: 900px !important;
        padding: 0 2rem 6rem 2rem !important;
        margin: 0 auto !important;
    }

    /* ── App header banner ───────────────────────────── */
    .app-hero {
        background: transparent;
        padding: 2rem 0rem 1.6rem 0rem;
        margin: 0 0 2rem 0;
        color: #0f172a;
    }
    .app-hero h1 {
        font-size: 2.9rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        margin: 0 0 4px 0 !important;
        letter-spacing: -1px;
    }
    .app-hero p {
        font-size: 0.9rem;
        color: #475569;
        margin: 0;
    }
    .hero-badges {
        margin-top: 1rem;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(15, 23, 42, 0.04);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(15, 23, 42, 0.1);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.76rem;
        color: #1e293b;
        font-weight: 600;
    }

    /* ── Sidebar ─────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* ── Section header (colored bar + title) ───────── */
    .scroll-anchor { scroll-margin-top: 90px; display: block; }
    .sec-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 2.5rem 0 0.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    .sec-bar {
        width: 4px;
        height: 28px;
        border-radius: 3px;
        flex-shrink: 0;
    }
    .sec-title-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0f172a;
        margin: 0;
    }
    .sec-num {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94a3b8;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-left: auto;
    }
    .sec-desc-text {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0 0 1rem 16px;
    }

    /* ── Subsection labels ───────────────────────────── */
    .sub-label {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: #94a3b8;
        margin: 1.2rem 0 0.4rem 0;
        display: block;
    }

    /* ── Cards ───────────────────────────────────────── */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Status banners ──────────────────────────────── */
    .info-banner {
        background: #eff6ff;
        border-left: 3px solid #60a5fa;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #1d4ed8;
        font-size: 0.875rem;
    }
    .warn-banner {
        background: #fffbeb;
        border-left: 3px solid #fbbf24;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #92400e;
        font-size: 0.875rem;
    }
    .success-banner {
        background: #f0fdf4;
        border-left: 3px solid #4ade80;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #166534;
        font-size: 0.875rem;
    }
    .error-banner {
        background: #fef2f2;
        border-left: 3px solid #f87171;
        border-radius: 6px;
        padding: 0.7rem 1rem;
        margin: 0.5rem 0;
        color: #991b1b;
        font-size: 0.875rem;
    }

    /* ── Sim ID badge ────────────────────────────────── */
    .sim-badge {
        display: inline-flex;
        align-items: center;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        padding: 0.3rem 1rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #1d4ed8;
        letter-spacing: 0.04em;
        gap: 6px;
    }
    .sim-badge-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #60a5fa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* ── Metric chips ────────────────────────────────── */
    .metric-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 0.25rem 0.85rem;
        font-size: 0.82rem;
        color: #475569;
        margin: 0.15rem;
    }
    .metric-chip strong { color: #0f172a; font-weight: 600; }

    /* ── Interaction table ───────────────────────────── */
    .int-header-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 8px;
        padding: 6px 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        margin-bottom: -1px;
    }
    .int-col-head {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        text-align: center;
    }
    .int-col-head:first-child { text-align: left; }

    /* ── Buttons ─────────────────────────────────────── */
    .stButton > button {
        background: #1f77b4;
        color: #ffffff;
        border: none;
        border-radius: 7px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.45rem 1.1rem;
        transition: background 0.15s, box-shadow 0.15s;
        box-shadow: 0 1px 3px rgba(31,119,180,0.2);
    }
    .stButton > button:hover {
        background: #1565a0;
        box-shadow: 0 3px 10px rgba(31,119,180,0.3);
    }
    .stButton > button[kind="secondary"] {
        background: #ffffff;
        color: #374151;
        border: 1px solid #d1d5db;
        box-shadow: none;
    }
    .stButton > button[kind="secondary"]:hover { background: #f3f4f6; }

    /* ── Inputs ──────────────────────────────────────── */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        color: #111827 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus {
        border-color: #1f77b4 !important;
        box-shadow: 0 0 0 3px rgba(31,119,180,0.1) !important;
    }
    [data-testid="stSelectbox"] > div {
        background: #ffffff !important;
        border-color: #d1d5db !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] > div {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 10px !important;
        background: #f8fafc !important;
    }
    [data-testid="stCheckbox"] label { color: #374151 !important; }
    [data-testid="stExpander"] summary {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid #e2e8f0; border-radius: 8px; max-width: 100%; overflow-x: auto; }
    [data-testid="stTable"] th, [data-testid="stTable"] td { padding: 0.5rem 0.7rem; font-size: 0.85rem; border-bottom: 1px solid #e2e8f0; }
    hr { border-color: #e2e8f0; margin: 0.75rem 0; }
    [data-testid="stRadio"] > div { gap: 0.5rem; }

    /* ── Verbose log ─────────────────────────────────── */
    .log-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        font-family: 'Courier New', monospace;
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.7;
        max-height: 320px;
        overflow-y: auto;
    }
    .log-ok  { color: #16a34a; font-weight: 600; }
    .log-inf { color: #2563eb; }
    .log-wrn { color: #d97706; font-weight: 600; }
    .log-err { color: #dc2626; font-weight: 600; }

    </style>
    """, unsafe_allow_html=True)


    # Floating navigation removed from here





    def reset_form_fields():
        """Reset all transient / per-simulation session state back to defaults."""
        d = _defaults()
        keys_to_reset = [k for k in d if k not in ("excel_path", "mode", "selected_sim_id", "n_harmonics")]
        for k in keys_to_reset:
            st.session_state[k] = d[k]
            
        # Explicitly overwrite widget UI keys to force frontend sync on rerun
        st.session_state["inp_mill_rpm"] = 0.0
        st.session_state["inp_eff_dia"] = 0.0
        st.session_state["mill_type_sel"] = "Ball Mill"
        
        st.session_state["inp_ore_dens"] = d["ore_density"]
        st.session_state["inp_ore_pois"] = d["ore_poisson"]
        st.session_state["inp_ore_shear"] = d["ore_shear_m"]
        st.session_state["inp_ore_rad"] = d["ore_radius"]
        st.session_state["inp_ore_mass"] = d["ore_mass"]
        
        st.session_state["inp_lin_dens"] = d["liner_density"]
        st.session_state["inp_lin_pois"] = d["liner_poisson"]
        st.session_state["inp_lin_shear"] = d["liner_shear_m"]
        
        st.session_state["liner_same_as_media"] = False
        st.session_state["inp_med_dens"] = d["media_density"]
        st.session_state["inp_med_pois"] = d["media_poisson"]
        st.session_state["inp_med_shear"] = d["media_shear_m"]
        st.session_state["inp_med_rad"] = d["media_radius"]
        st.session_state["inp_med_mass"] = d["media_mass"]
        

        st.session_state["rf_0"] = d["mm_rf"]; st.session_state["sf_0"] = d["mm_sf"]; st.session_state["res_0"] = d["mm_res"]
        st.session_state["rf_1"] = d["mo_rf"]; st.session_state["sf_1"] = d["mo_sf"]; st.session_state["res_1"] = d["mo_res"]
        st.session_state["rf_2"] = d["oo_rf"]; st.session_state["sf_2"] = d["oo_sf"]; st.session_state["res_2"] = d["oo_res"]
        st.session_state["rf_4"] = d["ml_rf"]; st.session_state["sf_4"] = d["ml_sf"]; st.session_state["res_4"] = d["ml_res"]
        st.session_state["rf_5"] = d["ol_rf"]; st.session_state["sf_5"] = d["ol_sf"]; st.session_state["res_5"] = d["ol_res"]

        for p in ("ore", "media"):
            st.session_state[f"inp_{p}_psd_s0"] = d[f"{p}_psd_s0"]; st.session_state[f"inp_{p}_psd_p0"] = d[f"{p}_psd_p0"]
            st.session_state[f"inp_{p}_psd_s1"] = d[f"{p}_psd_s1"]; st.session_state[f"inp_{p}_psd_p1"] = d[f"{p}_psd_p1"]
            st.session_state[f"inp_{p}_psd_s2"] = d[f"{p}_psd_s2"]; st.session_state[f"inp_{p}_psd_p2"] = d[f"{p}_psd_p2"]
        # Clear STEP-derived hint so it doesn't persist to the next simulation entry
        st.session_state["suggested_mill_dia"] = None


    init_state()

    if st.session_state.get("_needs_reset"):
        reset_form_fields()
        st.session_state["_needs_reset"] = False

    # ─────────────────────────────────────────────────────────────────────────────
    # PSD helper — read from individual session-state keys
    # ─────────────────────────────────────────────────────────────────────────────
    def get_ore_psd():
        return [
            [st.session_state["ore_psd_s0"], st.session_state["ore_psd_p0"]],
            [st.session_state["ore_psd_s1"], st.session_state["ore_psd_p1"]],
            [st.session_state["ore_psd_s2"], st.session_state["ore_psd_p2"]],
        ]


    def get_media_psd():
        return [
            [st.session_state["media_psd_s0"], st.session_state["media_psd_p0"]],
            [st.session_state["media_psd_s1"], st.session_state["media_psd_p1"]],
            [st.session_state["media_psd_s2"], st.session_state["media_psd_p2"]],
        ]


    # ─────────────────────────────────────────────────────────────────────────────
    # Thread-safe file dialogs via subprocess
    # ─────────────────────────────────────────────────────────────────────────────
    def _run_file_dialog(script: str) -> str:
        import textwrap
        clean_script = textwrap.dedent(script).strip()
        result = subprocess.run(
            [_sys.executable, "-c", clean_script],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()


    def pick_open_file(title, filetypes):
        ft_repr = repr(filetypes)
        script = f"""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', True)
    result = filedialog.askopenfilename(title={repr(title)}, filetypes={ft_repr})
    print(result)
    """
        return _run_file_dialog(script)


    def pick_save_file(title, default_ext, filetypes):
        ft_repr = repr(filetypes)
        script = f"""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', True)
    result = filedialog.asksaveasfilename(
        title={repr(title)},
        defaultextension={repr(default_ext)},
        filetypes={ft_repr}
    )
    print(result)
    """
        return _run_file_dialog(script)


    # ─────────────────────────────────────────────────────────────────────────────
    # Excel helpers
    # ─────────────────────────────────────────────────────────────────────────────
    def get_excel_cols():
        n_harm = int(50)
        base = [
            "simulation_id",
            "is_AG", "is_SAG", "is_PM", "is_BM",
            "ore_density", "ore_poisson", "ore_shear_m", "ore_radius",
            "ore_psd_s0", "ore_psd_p0", "ore_psd_s1", "ore_psd_p1", "ore_psd_s2", "ore_psd_p2",
            "D10_ore", "D50_ore", "D90_ore", "ore_mass",
            "liner_density", "liner_poisson", "liner_shear_m",
            "media_density", "media_poisson", "media_shear_m", "media_radius",
            "media_psd_s0", "media_psd_p0", "media_psd_s1", "media_psd_p1", "media_psd_s2", "media_psd_p2",
            "D10_media", "D50_media", "D90_media", "media_mass",
            "mill_rpm", "eff_mill_dia",
            "mm_rf", "mo_rf", "oo_rf", "ml_rf", "ol_rf",
            "mm_sf", "mo_sf", "oo_sf", "ml_sf", "ol_sf",
            "mm_res","mo_res","oo_res","ml_res","ol_res",
            "n_total_lifters", "n_repeat_units", "n_lifters_per_unit",
            "leading_face_angle", "trailing_face_angle",
            "short_leading_face_angle", "short_trailing_face_angle",
            "pct_rotation",
        ]
        shapes = [f"shape_k{k}" for k in range(n_harm)]
        tail = [
            "cf_max_particle", "ke_max_particle", "power_total_geometry_kw"
        ]
        return base + shapes + tail


    def _compute_percentile(psd_rows, p):
        valid_rows = [r for r in psd_rows if r[1] > 0]
        if not valid_rows:
            return 0.0
        
        valid_rows.sort(key=lambda x: x[0])  # ensure sorted by scale
        scales = [r[0] for r in valid_rows]
        pcts   = [r[1] for r in valid_rows]
        
        cumulative = np.cumsum(pcts)
        total  = cumulative[-1]
        target = p / 100.0 * total
        
        if target <= 0:
            return 0.0
            
        cums = [0.0] + list(cumulative)
        scls = [0.0] + list(scales)
        
        for i in range(1, len(cums)):
            if cums[i] >= target:
                if cums[i] == cums[i-1]:
                    return scls[i]
                frac = (target - cums[i-1]) / (cums[i] - cums[i-1])
                return scls[i-1] + frac * (scls[i] - scls[i-1])
                
        return scls[-1]


    def load_excel(path):
        if os.path.exists(path):
            df = pd.read_excel(path)
            # Drop legacy columns if they exist
            legacy_cols = ['media_factory_vel', 'ore_factory_vel', 'media_factory_velocity', 'ore_factory_velocity', 'local_name']
            existing_legacy = [c for c in legacy_cols if c in df.columns]
            if existing_legacy:
                df.drop(columns=existing_legacy, inplace=True)
            for col in get_excel_cols():
                if col not in df.columns:
                    df[col] = np.nan
            return df[get_excel_cols()]
        return pd.DataFrame(columns=get_excel_cols())


    def save_excel(path, df):
        df.to_excel(path, index=False)


    def next_sim_id(df):
        if df.empty or "simulation_id" not in df.columns or df["simulation_id"].dropna().empty:
            return 1
        return int(df["simulation_id"].dropna().max()) + 1


    def find_simulation_images(sim_id, search_dirs):
        """
        Finds all geometry and performance plot images for a given simulation ID.
        Supports single underscore (sim10_), double underscore (sim10__), and alternate naming patterns.
        Searches across excel_dir, project_dir, PRE PROCESSED, model_evaluation, etc.
        """
        prefixes = [
            f"sim{sim_id}_",
            f"sim{sim_id}__",
            f"sim_{sim_id}_",
            f"sim_{sim_id}__",
        ]
        
        found = {
            "liner_profile": None,
            "face_detection": None,
            "profile_analysis": None,
            "evaluation": None,
            "angular_csv": None,
        }
        
        all_dirs = []
        for d in search_dirs:
            if d and os.path.exists(d) and d not in all_dirs:
                all_dirs.append(d)
                
        for d in all_dirs:
            try:
                for root, _, files in os.walk(d):
                    for f in files:
                        fl = f.lower()
                        if not found["angular_csv"]:
                            for pref in prefixes:
                                if fl.startswith(pref) and "angular_profile" in fl and fl.endswith(".csv"):
                                    found["angular_csv"] = os.path.join(root, f)
                                    break

                        if not fl.endswith(('.png', '.jpg', '.jpeg')):
                            continue

                        if not found["liner_profile"]:
                            for pref in prefixes:
                                if fl.startswith(pref) and "liner_profile" in fl:
                                    found["liner_profile"] = os.path.join(root, f)
                                    break

                        if not found["face_detection"]:
                            for pref in prefixes:
                                if fl.startswith(pref) and ("face_detection" in fl or "face_analysis" in fl):
                                    found["face_detection"] = os.path.join(root, f)
                                    break

                        if not found["profile_analysis"]:
                            for pref in prefixes:
                                if fl.startswith(pref) and "profile_analysis" in fl:
                                    found["profile_analysis"] = os.path.join(root, f)
                                    break

                        if not found["evaluation"]:
                            if f"sim{sim_id}.png" in fl and "eval_" in fl:
                                found["evaluation"] = os.path.join(root, f)
            except Exception:
                pass

        return found


    # ─────────────────────────────────────────────────────────────────────────────
    # STEP file analysis – merged from analyze_liner.py
    # ─────────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────────
    # MULTIPROCESSING IMPORT
    # ─────────────────────────────────────────────────────────────────────────────

    def build_record_rows(sim_id):
        ss = st.session_state
        mt = ss["mill_type"]
        is_AG  = 1 if mt == "AG Mill"     else 0
        is_SAG = 1 if mt == "SAG Mill"    else 0
        is_PM  = 1 if mt == "Pebble Mill" else 0
        is_BM  = 1 if mt == "Ball Mill"   else 0

        ml   = ss.get("ml_features") or {}

        ore_psd   = get_ore_psd()
        media_psd = get_media_psd()
        ore_rad_val = ss.get("ore_radius", 0.0)
        med_rad_val = ss.get("media_radius") or 0.0

        D10_ore = _compute_percentile(ore_psd,   10) * ore_rad_val
        D50_ore = _compute_percentile(ore_psd,   50) * ore_rad_val
        D90_ore = _compute_percentile(ore_psd,   90) * ore_rad_val
        D10_med = _compute_percentile(media_psd, 10) * med_rad_val
        D50_med = _compute_percentile(media_psd, 50) * med_rad_val
        D90_med = _compute_percentile(media_psd, 90) * med_rad_val

        no_media = (ss["media_radius"] == 0 or ss["media_radius"] is None)
        no_ore = is_BM or is_PM
        mm_rf  = np.nan if no_media else ss["mm_rf"]
        mo_rf  = np.nan if (no_media or no_ore) else ss["mo_rf"]
        ml_rf  = np.nan if no_media else ss["ml_rf"]
        mm_sf  = np.nan if no_media else ss["mm_sf"]
        mo_sf  = np.nan if (no_media or no_ore) else ss["mo_sf"]
        ml_sf  = np.nan if no_media else ss["ml_sf"]
        mm_res = np.nan if no_media else ss["mm_res"]
        mo_res = np.nan if (no_media or no_ore) else ss["mo_res"]
        ml_res = np.nan if no_media else ss["ml_res"]
        oo_rf  = np.nan if no_ore else ss["oo_rf"]
        oo_sf  = np.nan if no_ore else ss["oo_sf"]
        oo_res = np.nan if no_ore else ss["oo_res"]
        ol_rf  = np.nan if no_ore else ss["ol_rf"]
        ol_sf  = np.nan if no_ore else ss["ol_sf"]
        ol_res = np.nan if no_ore else ss["ol_res"]

        ml   = ss.get("ml_features") or {}
        dem  = ss.get("dem_rotation_df")

        static = dict(
            simulation_id  = sim_id,
            is_AG=is_AG, is_SAG=is_SAG, is_PM=is_PM, is_BM=is_BM,
            ore_density    = np.nan if no_ore else ss["ore_density"],
            ore_poisson    = np.nan if no_ore else ss["ore_poisson"],
            ore_shear_m    = np.nan if no_ore else ss["ore_shear_m"],
            ore_radius     = np.nan if no_ore else ss["ore_radius"],
            ore_psd_s0     = np.nan if no_ore else ore_psd[0][0],
            ore_psd_p0     = np.nan if no_ore else ore_psd[0][1],
            ore_psd_s1     = np.nan if no_ore else ore_psd[1][0],
            ore_psd_p1     = np.nan if no_ore else ore_psd[1][1],
            ore_psd_s2     = np.nan if no_ore else ore_psd[2][0],
            ore_psd_p2     = np.nan if no_ore else ore_psd[2][1],
            D10_ore=np.nan if no_ore else D10_ore, D50_ore=np.nan if no_ore else D50_ore, D90_ore=np.nan if no_ore else D90_ore,
            ore_mass       = np.nan if no_ore else ss["ore_mass"],
            liner_density  = ss["liner_density"],
            liner_poisson  = ss["liner_poisson"],
            liner_shear_m  = ss["liner_shear_m"],
            media_density  = np.nan if no_media else ss["media_density"],
            media_poisson  = np.nan if no_media else ss["media_poisson"],
            media_shear_m  = np.nan if no_media else ss["media_shear_m"],
            media_radius   = np.nan if no_media else ss["media_radius"],
            media_psd_s0   = np.nan if no_media else media_psd[0][0],
            media_psd_p0   = np.nan if no_media else media_psd[0][1],
            media_psd_s1   = np.nan if no_media else media_psd[1][0],
            media_psd_p1   = np.nan if no_media else media_psd[1][1],
            media_psd_s2   = np.nan if no_media else media_psd[2][0],
            media_psd_p2   = np.nan if no_media else media_psd[2][1],
            D10_media=np.nan if no_media else D10_med, D50_media=np.nan if no_media else D50_med, D90_media=np.nan if no_media else D90_med,
            media_mass     = np.nan if no_media else ss["media_mass"],
            mill_rpm       = ss["mill_rpm"],
            eff_mill_dia   = ss["eff_mill_dia"],
            mm_rf=mm_rf,  mo_rf=mo_rf,  oo_rf=oo_rf,  ml_rf=ml_rf,  ol_rf=ol_rf,
            mm_sf=mm_sf,  mo_sf=mo_sf,  oo_sf=oo_sf,  ml_sf=ml_sf,  ol_sf=ol_sf,
            mm_res=mm_res, mo_res=mo_res, oo_res=oo_res, ml_res=ml_res, ol_res=ol_res,
            n_total_lifters    = ml.get("n_total_lifters",    np.nan),
            n_repeat_units     = ml.get("n_repeat_units",     np.nan),
            n_lifters_per_unit = ml.get("n_lifters_per_unit", np.nan),
            pct_rotation       = 0.0,
        )
        n_harm = int(50)
        for k in range(n_harm):
            static[f"shape_k{k}"] = ml.get(f"shape_k{k}", np.nan)
            
        for key, val in ml.items():
            if "angle" in key.lower() or "pattern" in key.lower():
                if not key.startswith("_"):
                    static[key] = val
        # Explicitly ensure these are caught
        static["leading_face_angle"]       = ml.get("leading_face_angle",       float('nan'))
        static["trailing_face_angle"]      = ml.get("trailing_face_angle",      float('nan'))
        static["short_leading_face_angle"] = ml.get("short_leading_face_angle", float('nan'))
        static["short_trailing_face_angle"]= ml.get("short_trailing_face_angle",float('nan'))

        rows = []
        for p in range(1, 101):
            row = dict(static)
            if dem is not None and len(dem) >= p:
                drow = dem.iloc[p - 1]
                row["pct_rotation"]             = drow["pct_rotation"]
                row["cf_max_particle"]          = drow["cf_max_particle"]
                row["ke_max_particle"]          = drow["ke_max_particle"]
                row["power_total_geometry_kw"]  = drow["power_total_geometry_kw"]
            else:
                row["pct_rotation"]             = p
                row["cf_max_particle"]          = np.nan
                row["ke_max_particle"]          = np.nan
                row["power_total_geometry_kw"]  = np.nan
            rows.append(row)
        return rows


    # ─────────────────────────────────────────────────────────────────────────────
    # UI helpers
    # ─────────────────────────────────────────────────────────────────────────────
    def _sec(anchor_id, num, title, desc, color):
        st.markdown(
            f'<div id="{anchor_id}" class="scroll-anchor"></div>'
            f'<div class="sec-header">'
            f'  <div class="sec-bar" style="background:{color}"></div>'
            f'  <span class="sec-title-text">{title}</span>'
            f'  <span class="sec-num">{num}</span>'
            f'</div>'
            f'<p class="sec-desc-text">{desc}</p>',
            unsafe_allow_html=True)


    def _psd_editor(prefix, label):
        """
        Renders three rows of (Scale, % of Mass) number inputs.
        Reads/writes from session_state keys: {prefix}_psd_s0..s2, {prefix}_psd_p0..p2.
        Returns list of [[scale, pct], ...].
        """
        st.markdown(f'<span class="sub-label">{label}</span>', unsafe_allow_html=True)
        st.caption("Enter scale multipliers and corresponding mass percentages (3 fractions).")

        col_h1, col_h2, col_h3 = st.columns([1, 1.2, 1.2])
        col_h1.markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em">Fraction</span>', unsafe_allow_html=True)
        col_h2.markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em">Scale Multiplier</span>', unsafe_allow_html=True)
        col_h3.markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.08em">% of Mass</span>', unsafe_allow_html=True)

        rows = []
        for i in range(3):
            sk = f"{prefix}_psd_s{i}"
            pk = f"{prefix}_psd_p{i}"
            c1, c2, c3 = st.columns([1, 1.2, 1.2])
            c1.markdown(f'<span style="font-size:0.85rem;color:#475569;padding-top:0.5rem;display:block">Fraction {i+1}</span>', unsafe_allow_html=True)
            sv = c2.number_input(f"Scale {i+1}", min_value=0.0, value=float(st.session_state[sk]),
                                 step=0.1, format="%.3f", key=f"inp_{sk}", label_visibility="collapsed")
            pv = c3.number_input(f"% of Mass {i+1}", min_value=0.0, max_value=100.0,
                                 value=float(st.session_state[pk]),
                                 step=1.0, format="%.3f", key=f"inp_{pk}", label_visibility="collapsed")
            st.session_state[sk] = sv
            st.session_state[pk] = pv
            rows.append([sv, pv])
        return rows


    # ─────────────────────────────────────────────────────────────────────────────
    # Sidebar – Excel file selection
    # ─────────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## Settings")
        st.markdown("---")
        st.markdown("**Dataset Excel File**")
        
        curr_path = st.session_state.get("excel_path")
        if not curr_path or not os.path.exists(curr_path):
            def_p = _resolve_default_excel()
            if def_p:
                st.session_state["excel_path"] = def_p
                curr_path = def_p

        if curr_path and os.path.exists(curr_path):
            st.markdown(
                f'<div class="success-banner" style="word-break:break-all;font-size:0.8rem;padding:8px 12px;margin-bottom:8px;">'
                f'📁 <b>Active Dataset:</b><br>'
                f'<span style="font-weight:600;color:#0f766e;">{os.path.basename(curr_path)}</span><br>'
                f'<span style="color:#64748b;font-size:0.72rem;">{curr_path}</span>'
                f'</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="info-banner" style="word-break:break-all;font-size:0.8rem;padding:8px 12px;margin-bottom:8px;">'
                f'⚠️ <b>No dataset selected</b>'
                f'</div>',
                unsafe_allow_html=True)

        if st.button("Select / Change Excel File", key="btn_pick_excel"):
            fpath = pick_open_file(
                title="Select Excel dataset file",
                filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
            )
            if fpath and os.path.exists(fpath):
                st.session_state["excel_path"] = fpath
                _save_config(fpath)
                st.rerun()

        # Browser upload for cloud/server deployments
        sb_file = st.file_uploader(
            "Upload Custom Excel (.xlsx)",
            type=["xlsx", "xls"],
            key="sb_excel_upload",
            help="Upload an Excel dataset directly from your browser"
        )
        if sb_file is not None:
            save_dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), sb_file.name)
            with open(save_dest, "wb") as f:
                f.write(sb_file.getbuffer())
            if st.session_state.get("excel_path") != save_dest:
                st.session_state["excel_path"] = save_dest
                _save_config(save_dest)
                st.success(f"Loaded: {sb_file.name}")
                time.sleep(0.3)
                st.rerun()

        if st.button("Create New Excel File", key="btn_new_excel"):
            fpath = pick_save_file(
                title="Create new Excel dataset file",
                default_ext=".xlsx",
                filetypes=[("Excel files", "*.xlsx")],
            )
            if fpath:
                if not fpath.endswith(".xlsx"):
                    fpath += ".xlsx"
                save_excel(fpath, pd.DataFrame(columns=get_excel_cols()))
                st.session_state["excel_path"] = fpath
                _save_config(fpath)
                st.rerun()
                
        st.markdown("---")
        st.session_state["n_harmonics"] = 80

        st.divider()
        st.markdown(
            '<p style="color:#64748b;font-size:0.78rem;line-height:1.7">'
            '<b style="color:#374151">EDEM Dataset Creator</b><br>'
            'Collects simulation parameters and<br>'
            'processed outputs into a structured<br>'
            'Excel dataset for ML/AI training.<br><br>'
            '<span style="color:#94a3b8">100 rows per simulation<br>'
            '1 row = 1 rotation % point</span>'
            '</p>', unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    # App hero header (Dynamic)
    # ─────────────────────────────────────────────────────────────────────────────
    current_mode = st.session_state.get("mode_radio", "Add Data")
    
    if current_mode == "Add Data":
        hero_title = "EDEM Dataset Creator"
        hero_desc = "Mill Simulation Data Collection &amp; Processing for ML/AI Training"
    elif current_mode == "View / Update Record":
        hero_title = "EDEM Dataset Viewer/Updator"
        hero_desc = "Review, Update, and Delete Existing Simulation Records"
    elif current_mode == "Data Report":
        hero_title = "Data Preprocessing & Report Viewer"
        hero_desc = "Analyze dataset distributions and prepare machine learning features"
    else:
        hero_title = "AI Prediction Model"
        hero_desc = "Run simulations instantly with Hybrid Surrogate ML Models"

    st.markdown(f"""
    <div class="app-hero">
      <h1>{hero_title}</h1>
      <p>{hero_desc}</p>
      <div class="hero-badges">
        <span class="hero-badge">Liner Geometry Analysis</span>
        <span class="hero-badge">DEM Data Processing</span>
        <span class="hero-badge">ML Feature Extraction</span>
        <span class="hero-badge">Excel Dataset Builder</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # Mode selector
    # ─────────────────────────────────────────────────────────────────────────────
    def _on_mode_change():
        selected_mode = st.session_state.get("mode_radio", "Add Data")
        st.session_state["mode"] = selected_mode
        _save_settings(last_page=selected_mode)

    mode = st.radio(
        "Mode",
        ["Add Data", "View / Update Record", "Data Report", "Prediction Model"],
        horizontal=True,
        key="mode_radio",
        on_change=_on_mode_change,
        label_visibility="collapsed",
    )
    st.session_state["mode"] = mode

    page_title_suffix = PAGE_TITLE_MAP.get(mode, mode)
    tab_title = f"EDEM Surrogate - {page_title_suffix}"
    title_script = f"<script>window.parent.document.title = '{tab_title}';</script>"
    if hasattr(st, "html"):
        st.html(title_script)
    else:
        st.markdown(title_script, unsafe_allow_html=True)

    st.markdown("---")
    if show_splash:
        animate_splash_step(splash_placeholder, 51, 75, delay=0.03)

    # ─────────────────────────────────────────────────────────────────────────────
    # Independent Modes: Prediction Model & Data Report do not require Excel dataset
    # ─────────────────────────────────────────────────────────────────────────────
    if mode == "Prediction Model":
        if show_splash:
            animate_splash_step(splash_placeholder, 76, 100, delay=0.015)
            time.sleep(0.1)
            splash_placeholder.empty()
            st.session_state["splash_shown"] = True
        try:
            from predictive_dashboard_page import render_predictive_dashboard
            render_predictive_dashboard()
        except Exception as e:
            st.error(f"Predictive Dashboard error: {e}")
        return

    elif mode == "Data Report":
        if show_splash:
            animate_splash_step(splash_placeholder, 76, 100, delay=0.015)
            time.sleep(0.1)
            splash_placeholder.empty()
            st.session_state["splash_shown"] = True
        from data_report_page import render_data_report_page
        render_data_report_page(CONFIG_PATH)
        return

    # ─────────────────────────────────────────────────────────────────────────────
    # Guard: no Excel file selected (required only for Add Data & View / Update)
    # ─────────────────────────────────────────────────────────────────────────────
    if not st.session_state.get("excel_path"):
        if show_splash:
            animate_splash_step(splash_placeholder, 76, 100, delay=0.015)
            time.sleep(0.1)
            splash_placeholder.empty()
            st.session_state["splash_shown"] = True
        st.markdown(
            '<div class="info-banner">'
            '<b>No dataset file selected.</b>&nbsp;&nbsp;'
            'Use <b>Select / Change Excel File</b> or <b>Create New Excel File</b> in the sidebar to get started.'
            '</div>', unsafe_allow_html=True)
        st.stop()

    excel_path = st.session_state["excel_path"]
    master_df  = load_excel(excel_path)


    # ═════════════════════════════════════════════════════════════════════════════
    # MODE 1 – ADD DATA
    # ═════════════════════════════════════════════════════════════════════════════
    if mode == "Add Data":
        sim_id = next_sim_id(master_df)

        # Sim ID row
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem">'
            f'  <span class="sim-badge">'
            f'    <span class="sim-badge-label">SIM ID</span> {sim_id}'
            f'  </span>'
            f'  <span style="color:#64748b;font-size:0.85rem">Next simulation to be recorded</span>'
            f'</div>',
            unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 1 — Files & Geometry
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-files", "01", "Files &amp; Geometry",
             "Upload the liner STEP geometry file and DEM results CSV", "#818cf8")

        # ── STEP file via st.file_uploader ────────────────────────────────────
        st.markdown('<span class="sub-label">Liner Geometry File (STEP / STP)</span>', unsafe_allow_html=True)

        step_uploader = st.file_uploader(
            "STEP file", type=["step", "stp"],
            key=f"step_file_upload_{st.session_state.get('step_uploader_key', 0)}", label_visibility="collapsed")

        if step_uploader is not None:
            # Write to a temp file so cadquery can open it by path
            suf = ".stp" if step_uploader.name.lower().endswith(".stp") else ".step"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
                tmp.write(step_uploader.read())
                tmp_path = tmp.name

            # Only re-analyse if a different file was uploaded
            prev_name = st.session_state.get("step_uploaded_name", "")
            if step_uploader.name != prev_name:
                st.session_state["step_file_path"]      = tmp_path
                st.session_state["step_uploaded_name"]  = step_uploader.name
                st.session_state["angular_profile_df"]  = None
                st.session_state["ml_features"]         = None
                st.session_state["liner_profile_img"]   = None
                st.session_state["analysis_img"]        = None
                st.session_state["face_analysis_img"]   = None

        if st.session_state.get("liner_profile_img"):
            st.markdown('<span class="sub-label">Liner Profile Preview</span>', unsafe_allow_html=True)
            st.image(st.session_state["liner_profile_img"], use_container_width=True)

        if st.session_state.get("face_analysis_img"):
            st.markdown('<span class="sub-label">Face Angle Detection Regions</span>', unsafe_allow_html=True)
            st.image(st.session_state["face_analysis_img"], use_container_width=True)

        if st.session_state.get("analysis_img"):
            with st.expander("Full Diagnostic Analysis (9-panel)"):
                st.image(st.session_state["analysis_img"], use_container_width=True)

        if st.session_state.get("ml_features"):
            ml = st.session_state["ml_features"]
            st.markdown('<span class="sub-label">Extracted ML Geometry Features</span>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="metric-chip">Lifters <strong>{int(ml["n_total_lifters"])}</strong></div>'
                f'<div class="metric-chip">Repeat units <strong>{int(ml["n_repeat_units"])}</strong></div>'
                f'<div class="metric-chip">Lifters / unit <strong>{ml["n_lifters_per_unit"]:.2f}</strong></div>',
                unsafe_allow_html=True)
                
            st.write("")
            st.markdown('<div id="green_btn_marker"></div>', unsafe_allow_html=True)
            st.markdown('''
            <style>
            div.element-container:has(#green_btn_marker) + div.element-container button {
                background-color: #16a34a !important;
                color: white !important;
                border-color: #16a34a !important;
            }
            div.element-container:has(#green_btn_marker) + div.element-container button:hover {
                background-color: #15803d !important;
                border-color: #15803d !important;
            }
            </style>
            ''', unsafe_allow_html=True)
            prefix = "liner"
            if st.session_state.get("step_file_path"):
                prefix = os.path.splitext(os.path.basename(st.session_state["step_file_path"]))[0]
                
            # Create a Zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                df_ang = st.session_state.get("angular_profile_df")
                if df_ang is not None:
                    zf.writestr(f"{prefix}_angular_profile.csv", df_ang.to_csv(index=False))
                
                if st.session_state.get("liner_profile_img"):
                    zf.writestr(f"{prefix}_liner_profile.png", st.session_state["liner_profile_img"])
                    
                if st.session_state.get("analysis_img"):
                    zf.writestr(f"{prefix}_profile_analysis.png", st.session_state["analysis_img"])
                    
                if st.session_state.get("face_analysis_img"):
                    zf.writestr(f"{prefix}_face_detection.png", st.session_state["face_analysis_img"])
            
            st.download_button(
                label="Want just liner files ? (Download ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{prefix}_liner_files.zip",
                mime="application/zip",
                key="btn_save_liner"
            )


        # ── Analysis controls ─────────────────────────────────────────────────
        if st.session_state.get("step_file_path"):
            if st.session_state.get("angular_profile_df") is None:
                if not st.session_state.get("is_analysing"):
                    if st.button("▶  Run Liner Analysis", key="btn_analyse"):
                        start_analysis_process(st.session_state["step_file_path"])
                        st.rerun()
            else:
                if not st.session_state.get("is_analysing"):
                    st.markdown(
                        f'<div class="success-banner">✅ Liner analysis complete — '
                        f'{st.session_state.get("n_lifters_detected", "?")} lifters detected.</div>',
                        unsafe_allow_html=True)
                    if st.button("↺  Re-analyse", key="btn_reanalyse"):
                        st.session_state["angular_profile_df"] = None
                        start_analysis_process(st.session_state["step_file_path"])
                        st.rerun()
        elif step_uploader is None and st.session_state.get("step_file_path") is None:
            st.markdown(
                '<div class="info-banner">Upload a STEP / STP file above to analyse the liner geometry.</div>',
                unsafe_allow_html=True)

        # ── Live progress fragment (polls every second while analysis runs) ────

        render_analysis_progress()

        st.divider()

        # ── DEM Results CSV ───────────────────────────────────────────────────
        st.markdown('<span class="sub-label">DEM Results CSV</span>', unsafe_allow_html=True)

        render_dem_uploader()

        st.divider()



        # ── Mill type ─────────────────────────────────────────────────────────
        st.markdown('<span class="sub-label">Mill Type</span>', unsafe_allow_html=True)
        mill_opts = ["AG Mill", "SAG Mill", "Pebble Mill", "Ball Mill"]
        st.session_state["mill_type"] = st.selectbox(
            "Mill Type", mill_opts,
            index=mill_opts.index(st.session_state["mill_type"]),
            key="mill_type_sel", label_visibility="collapsed")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 2 — Mill Parameters
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-mill", "02", "Mill Parameters",
             "Operating speed and effective inner diameter of the mill", "#34d399")

        c1, c2 = st.columns(2)
        with c1:
            if "inp_mill_rpm" not in st.session_state:
                st.session_state["inp_mill_rpm"] = float(st.session_state.get("mill_rpm", 0.0))
            st.session_state["mill_rpm"] = st.number_input(
                "Mill RPM", min_value=0.0, step=0.1, format="%.3f", key="inp_mill_rpm")
        with c2:
            st.session_state["eff_mill_dia"] = st.number_input(
                "Effective Mill Diameter (m)", min_value=0.0,
                value=float(st.session_state["eff_mill_dia"]),
                step=0.01, format="%.3f", key="inp_eff_dia")

        # Auto-hint from STEP file analysis
        suggested_dia = st.session_state.get("suggested_mill_dia")
        if suggested_dia and st.session_state["eff_mill_dia"] == 0.0:
            st.markdown(
                f'<div class="info-banner">💡 Based on your STEP file geometry: '
                f'exact cross-sectional area analysis (total shell area minus liner profile area) suggests an effective mill diameter of '
                f'<b>{suggested_dia:.4f} m</b>. '
                f'Please verify against your mill engineering drawings and enter the correct value above.</div>',
                unsafe_allow_html=True)

        if st.session_state["rpm_computed"] is not None and st.session_state["mill_rpm"] == 0.0:
            st.markdown(
                f'<div class="info-banner">💡 The DEM CSV suggests RPM &asymp; '
                f'<b>{st.session_state["rpm_computed"]:.3f}</b>. '
                f'You may use this value in the Mill RPM field above.</div>',
                unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 3 — Ore Properties
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-ore", "03", "Ore Properties",
             "Material properties, particle size distribution, and total mass", "#fb923c")

        curr_mill_type = st.session_state.get("mill_type", "Ball Mill")
        no_ore = curr_mill_type in ["Ball Mill", "Pebble Mill"]
        if no_ore:
            st.info(f"Ore properties are not required for {curr_mill_type}. Data will be recorded as NaN.")
        else:
            st.markdown('<span class="sub-label">Material Properties</span>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.session_state["ore_density"] = st.number_input(
                    "Density (kg/m³)", min_value=0.0, value=float(st.session_state["ore_density"]),
                    step=10.0, format="%.3f", key="inp_ore_dens")
            with c2:
                st.session_state["ore_poisson"] = st.number_input(
                    "Poisson's Ratio", min_value=0.0, max_value=0.5,
                    value=float(st.session_state["ore_poisson"]),
                    step=0.01, format="%.3f", key="inp_ore_pois")
            with c3:
                st.session_state["ore_shear_m"] = st.number_input(
                    "Shear Modulus (N/m²)", min_value=0.0, value=float(st.session_state["ore_shear_m"]),
                    step=1e6, format="%.3e", key="inp_ore_shear")
            with c4:
                st.session_state["ore_radius"] = st.number_input(
                    "Radius (mm)", min_value=0.0, value=float(st.session_state["ore_radius"]),
                    step=1.0, format="%.3f", key="inp_ore_rad")
        
            ore_psd = _psd_editor("ore", "Particle Size Distribution")
            D10 = _compute_percentile(ore_psd, 10)
            D50 = _compute_percentile(ore_psd, 50)
            D90 = _compute_percentile(ore_psd, 90)
            st.markdown(
                f'<div class="metric-chip">D10 <strong>{D10:.3f}</strong></div>'
                f'<div class="metric-chip">D50 <strong>{D50:.3f}</strong></div>'
                f'<div class="metric-chip">D90 <strong>{D90:.3f}</strong></div>',
                unsafe_allow_html=True)
        
            st.markdown('<span class="sub-label" style="margin-top:1.2rem">Total Ore Mass</span>', unsafe_allow_html=True)
            st.session_state["ore_mass"] = st.number_input(
                "Total Ore Mass (kg)", min_value=0.0, value=float(st.session_state["ore_mass"]),
                step=100.0, format="%.3f", key="inp_ore_mass")

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 4 — Liner Properties
        # ══════════════════════════════════════════════════════════════════════
        def sync_liner_to_media():
            if st.session_state.get("liner_same_as_media"):
                if "inp_med_dens" in st.session_state:
                    st.session_state["media_density"] = st.session_state.get("inp_lin_dens", st.session_state["liner_density"])
                    st.session_state["inp_med_dens"] = st.session_state["media_density"]
                    st.session_state["media_poisson"] = st.session_state.get("inp_lin_pois", st.session_state["liner_poisson"])
                    st.session_state["inp_med_pois"] = st.session_state["media_poisson"]
                    st.session_state["media_shear_m"] = st.session_state.get("inp_lin_shear", st.session_state["liner_shear_m"])
                    st.session_state["inp_med_shear"] = st.session_state["media_shear_m"]

        def break_media_link():
            st.session_state["liner_same_as_media"] = False

        _sec("sec-liner", "04", "Liner Properties",
             "Liner shell material mechanical properties", "#a78bfa")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.session_state["liner_density"] = st.number_input(
                "Density (kg/m³)", min_value=0.0, value=float(st.session_state["liner_density"]),
                step=10.0, format="%.3f", key="inp_lin_dens", on_change=sync_liner_to_media)
        with c2:
            st.session_state["liner_poisson"] = st.number_input(
                "Poisson's Ratio", min_value=0.0, max_value=0.5,
                value=float(st.session_state["liner_poisson"]),
                step=0.01, format="%.3f", key="inp_lin_pois", on_change=sync_liner_to_media)
        with c3:
            st.session_state["liner_shear_m"] = st.number_input(
                "Shear Modulus (N/m²)", min_value=0.0, value=float(st.session_state["liner_shear_m"]),
                step=1e9, format="%.3e", key="inp_lin_shear", on_change=sync_liner_to_media)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 5 — Media Properties
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-media", "05", "Media Properties",
             "Grinding media material properties, particle size distribution, and total mass", "#38bdf8")

        curr_mill_type = st.session_state.get("mill_type", "Ball Mill")
        no_media = curr_mill_type == "AG Mill"
        if no_media:
            st.info(f"Media properties are not required for {curr_mill_type}. Data will be recorded as NaN.")
            st.session_state["media_radius"] = 0.0
        else:
            st.session_state["media_radius"] = st.number_input(
                "Media Radius (mm)", min_value=0.0, value=float(st.session_state["media_radius"]),
                step=1.0, format="%.3f", key="inp_med_rad")

            has_media = st.session_state["media_radius"] > 0
            if has_media:
                chk_media_same = st.checkbox(
                    "Media Properties are same as liner ?",
                    value=st.session_state.get("liner_same_as_media", False),
                    key="liner_same_as_media",
                    on_change=sync_liner_to_media)

                if chk_media_same:
                    st.markdown(
                        '<div class="info-banner">Media material properties are linked to liner values. '
                        'If you edit them independently below, they will automatically be unlinked.</div>',
                        unsafe_allow_html=True)
            else:
                st.session_state["liner_same_as_media"] = False
                chk_media_same = False

            st.markdown('<span class="sub-label" style="margin-top:0.8rem">Material Properties</span>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.session_state["media_density"] = st.number_input("Density (kg/m³)", min_value=0.0,
                    value=float(st.session_state["media_density"]),
                    step=10.0, format="%.3f", key="inp_med_dens", on_change=break_media_link)
            with c2:
                st.session_state["media_poisson"] = st.number_input("Poisson's Ratio", min_value=0.0, max_value=0.5,
                    value=float(st.session_state["media_poisson"]),
                    step=0.01, format="%.3f", key="inp_med_pois", on_change=break_media_link)
            with c3:
                st.session_state["media_shear_m"] = st.number_input("Shear Modulus (N/m²)", min_value=0.0,
                    value=float(st.session_state["media_shear_m"]),
                    step=1e9, format="%.3e", key="inp_med_shear", on_change=break_media_link)

            media_psd = _psd_editor("media", "Media Particle Size Distribution")
            Dm10 = _compute_percentile(media_psd, 10)
            Dm50 = _compute_percentile(media_psd, 50)
            Dm90 = _compute_percentile(media_psd, 90)
            st.markdown(
                f'<div class="metric-chip">D10 <strong>{Dm10:.3f}</strong></div>'
                f'<div class="metric-chip">D50 <strong>{Dm50:.3f}</strong></div>'
                f'<div class="metric-chip">D90 <strong>{Dm90:.3f}</strong></div>',
                unsafe_allow_html=True)

            st.markdown('<span class="sub-label" style="margin-top:1.2rem">Total Media Mass</span>', unsafe_allow_html=True)
            st.session_state["media_mass"] = st.number_input(
                "Total Media Mass (kg)", min_value=0.0, value=float(st.session_state["media_mass"]),
                step=100.0, format="%.3f", key="inp_med_mass")
            total_charge = st.session_state["ore_mass"] + st.session_state["media_mass"]
            st.markdown(
                f'<div class="metric-chip" style="margin-top:0.5rem">'
                f'Total Charge Mass <strong>{total_charge:.2f} kg</strong></div>',
                unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 6 — Factory & Velocities
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-factory", "06", "Factory &amp; Velocities",
             "Particle factory injection velocities for ore and grinding media", "#4ade80")
             
        st.markdown(
            '<div class="info-banner">Injection velocities have been removed from the dataset requirements.</div>',
            unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 7 — Interaction Properties
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-interact", "07", "Interaction Properties",
             "Contact mechanics coefficients for all particle-pair combinations", "#f472b6")

        # Info banners for hidden pairs
        if not has_media:
            st.markdown(
                '<div class="info-banner">Media-related interactions are hidden (Media Radius = 0 or AG Mill). '
                'These values will be saved as NaN in the dataset.</div>',
                unsafe_allow_html=True)
        if no_ore:
            st.markdown(
                f'<div class="info-banner">Ore-related interactions are hidden for {curr_mill_type} — '
                'no ore particles in the simulation. These values will be saved as NaN.</div>',
                unsafe_allow_html=True)

        # Header row
        hc = st.columns([2.2, 1.6, 1.6, 1.6])
        hc[0].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Pair</span>', unsafe_allow_html=True)
        hc[1].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Rolling Friction</span>', unsafe_allow_html=True)
        hc[2].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Sliding Friction</span>', unsafe_allow_html=True)
        hc[3].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Restitution</span>', unsafe_allow_html=True)

        # Interaction pair definitions:
        #   (display name, rf_key, sf_key, res_key, needs_media, needs_ore)
        interactions = [
            ("Media–Media",  "mm_rf", "mm_sf", "mm_res", True,  False),
            ("Media–Ore",    "mo_rf", "mo_sf", "mo_res", True,  True),
            ("Ore–Ore",      "oo_rf", "oo_sf", "oo_res", False, True),
            ("Media–Liner",  "ml_rf", "ml_sf", "ml_res", True,  False),
            ("Ore–Liner",    "ol_rf", "ol_sf", "ol_res", False, True),
        ]

        for i, (iname, krf, ksf, kres, is_media, is_ore) in enumerate(interactions):
            hidden = (is_media and not has_media) or (is_ore and no_ore)
            if hidden:
                # Store NaN for hidden pairs — written correctly to dataset
                st.session_state[krf]  = float('nan')
                st.session_state[ksf]  = float('nan')
                st.session_state[kres] = float('nan')
                continue

            # Restore to 0 default if previously NaN (switching mill types)
            for k in [krf, ksf, kres]:
                if st.session_state.get(k) != st.session_state.get(k):  # NaN check
                    st.session_state[k] = 0.0

            cols = st.columns([2.2, 1.6, 1.6, 1.6])
            cols[0].markdown(f'<span style="font-size:0.9rem;font-weight:500;color:#1e293b">{iname}</span>', unsafe_allow_html=True)
            val_rf  = cols[1].number_input(f"RF {iname}",  min_value=0.0, max_value=1.0,
                value=float(st.session_state[krf] if st.session_state[krf] == st.session_state[krf] else 0.0),
                step=0.001, format="%.3f", key=f"rf_{i}", label_visibility="collapsed")
            val_sf  = cols[2].number_input(f"SF {iname}",  min_value=0.0, max_value=2.0,
                value=float(st.session_state[ksf] if st.session_state[ksf] == st.session_state[ksf] else 0.0),
                step=0.01,  format="%.3f", key=f"sf_{i}", label_visibility="collapsed")
            val_res = cols[3].number_input(f"Res {iname}", min_value=0.0, max_value=1.0,
                value=float(st.session_state[kres] if st.session_state[kres] == st.session_state[kres] else 0.0),
                step=0.01,  format="%.3f", key=f"res_{i}", label_visibility="collapsed")

            st.session_state[krf]  = val_rf
            st.session_state[ksf]  = val_sf
            st.session_state[kres] = val_res

        # ══════════════════════════════════════════════════════════════════════
        # SECTION 8 — Record Data
        # ══════════════════════════════════════════════════════════════════════
        _sec("sec-record", "08", "Record Data",
             "Review completeness and write the simulation record to the dataset", "#1f77b4")

        # Completeness check
        warnings_list = []
        mill_type_val  = st.session_state.get("mill_type", "")
        _no_ore_warn   = mill_type_val in ["Ball Mill", "Pebble Mill"]
        media_rad_val  = st.session_state.get("media_radius", 0.0)
        media_mass_val = st.session_state.get("media_mass", 0.0)

        if st.session_state.get("angular_profile_df") is None:
            warnings_list.append("No STEP file analysed — liner geometry features will be saved as NaN.")
        if st.session_state.get("dem_rotation_df") is None:
            warnings_list.append("No DEM CSV uploaded — power / CF / KE columns will be saved as NaN.")
        if st.session_state.get("eff_mill_dia", 0.0) == 0.0:
            warnings_list.append("Effective Mill Diameter is zero — please enter the inner working diameter of the mill (in metres).")

        # Ore-property warnings — only for mill types that actually use ore
        if not _no_ore_warn:
            if st.session_state.get("ore_mass", 0.0) == 0.0:
                warnings_list.append("Total Ore Mass is zero — please provide a valid mass for the ore charge.")
            if st.session_state.get("ore_radius", 0.0) == 0.0:
                warnings_list.append("Ore Radius is zero — please provide a valid ore radius.")

        # Media-property warnings
        if mill_type_val not in ["AG Mill"]:
            if media_rad_val == 0.0:
                warnings_list.append(f"Mill Type is '{mill_type_val}' but Media Radius is 0. Please provide a valid media radius.")
            if media_mass_val == 0.0:
                warnings_list.append(f"Mill Type is '{mill_type_val}' but Media Mass is 0. Please provide a valid media mass.")

        if mill_type_val == "AG Mill":
            if media_rad_val != 0.0:
                warnings_list.append("Mill Type is 'AG Mill' but Media Radius is not 0. AG Mills should not have grinding media.")
            if media_mass_val != 0.0:
                warnings_list.append("Mill Type is 'AG Mill' but Media Mass is not 0. AG Mills should not have grinding media.")

        # Check visible interaction properties — skip hidden pairs
        _has_media_warn = (mill_type_val != "AG Mill") and media_rad_val > 0
        for (iname, krf, ksf, kres, is_media, is_ore) in interactions:
            if is_media and not _has_media_warn:
                continue
            if is_ore and _no_ore_warn:
                continue
            val = st.session_state.get(krf, 0.0)
            # NaN means hidden — don't warn
            if val != val:
                continue
            if st.session_state.get(krf, 0.0) == 0.0 or st.session_state.get(ksf, 0.0) == 0.0 or st.session_state.get(kres, 0.0) == 0.0:
                warnings_list.append(f"Interaction properties for '{iname}' cannot be zero. Please provide non-zero values for RF, SF, and Res.")

        if warnings_list:
            st.markdown(
                '<div class="error-banner"><b>Incomplete inputs:</b><br>' +
                "<br>".join(f"&bull;&nbsp;{w}" for w in warnings_list) +
                '<br><br><b>Please complete the input fields above and then click on Record Data again.</b></div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="success-banner">All inputs are complete. Ready to record.</div>',
                unsafe_allow_html=True)

        # Summary chips
        st.markdown(
            f'<div style="margin:0.75rem 0">'
            f'<div class="metric-chip">Mill: <strong>{st.session_state["mill_type"]}</strong></div>'
            f'<div class="metric-chip">RPM: <strong>{st.session_state["mill_rpm"]:.2f}</strong></div>'
            f'<div class="metric-chip">Dia: <strong>{st.session_state["eff_mill_dia"]:.4f} m</strong></div>'
            f'<div class="metric-chip">Ore: <strong>{st.session_state["ore_density"]:.0f} kg/m³</strong></div>'
            f'<div class="metric-chip">Liner: <strong>{st.session_state["liner_density"]:.0f} kg/m³</strong></div>'
            f'</div>',
            unsafe_allow_html=True)

        # ── Live dataset row preview ──────────────────────────────────────────
        st.markdown('<span class="sub-label" style="margin-top:1rem">Dataset Row Preview</span>', unsafe_allow_html=True)
        st.markdown(
            '<div class="info-banner" style="margin-bottom:0.5rem">This preview shows the first row of what will be written to the dataset. '
            'Liner geometry columns show NaN until the STEP file is analysed. DEM columns show NaN until a CSV is uploaded. '
            'The full record has 100 rows (one per % rotation).</div>',
            unsafe_allow_html=True)

        try:
            preview_rows = build_record_rows(sim_id)
            preview_df   = pd.DataFrame(preview_rows, columns=get_excel_cols())

            # Show a curated 10-column view
            preview_cols = [
                "simulation_id", "mill_rpm", "eff_mill_dia",
                "n_total_lifters", "ore_density", "media_radius",
                "pct_rotation", "cf_max_particle", "power_total_geometry_kw"
            ]
            show_cols = [c for c in preview_cols if c in preview_df.columns]
            st.dataframe(preview_df[show_cols].head(5))

            # Full CSV download
            csv_bytes = preview_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Full Preview CSV (100 rows)",
                data=csv_bytes,
                file_name=f"preview_sim{sim_id}.csv",
                mime="text/csv",
                key="btn_preview_dl"
            )
        except Exception as prev_e:
            st.markdown(
                f'<div class="warn-banner">Preview not available yet: {prev_e}</div>',
                unsafe_allow_html=True)

        st.write("")
        record_clicked = st.button("Record Data", key="btn_record", type="primary")

        if record_clicked:
            if warnings_list:
                st.markdown('<div class="error-banner"><b>Action Blocked:</b> Cannot record data because some inputs are missing. Please complete them and try again.</div>', unsafe_allow_html=True)
                st.stop()
                
            try:
                new_rows = build_record_rows(sim_id)
                new_df   = pd.DataFrame(new_rows, columns=get_excel_cols())
                master_df = pd.concat([master_df, new_df], ignore_index=True)
                save_excel(excel_path, master_df)

                file_prefix = f"sim{sim_id}"

                if st.session_state.get("angular_profile_df") is not None:
                    ang_csv_path = os.path.join(
                        os.path.dirname(excel_path), f"{file_prefix}_angular_profile.csv")
                    st.session_state["angular_profile_df"].to_csv(ang_csv_path, index=False)

                if st.session_state.get("liner_profile_img"):
                    img_path = os.path.join(
                        os.path.dirname(excel_path), f"{file_prefix}_liner_profile.png")
                    with open(img_path, "wb") as f:
                        f.write(st.session_state["liner_profile_img"])

                if st.session_state.get("analysis_img"):
                    img_path = os.path.join(
                        os.path.dirname(excel_path), f"{file_prefix}_profile_analysis.png")
                    with open(img_path, "wb") as f:
                        f.write(st.session_state["analysis_img"])
                
                if st.session_state.get("face_analysis_img"):
                    img_path = os.path.join(
                        os.path.dirname(excel_path), f"{file_prefix}_face_detection.png")
                    with open(img_path, "wb") as f:
                        f.write(st.session_state["face_analysis_img"])

                st.markdown(
                    f'<div class="success-banner"><b>Simulation {sim_id} saved successfully.</b> '
                    f'100 rows written to the dataset. The form has been reset for the next entry.</div>',
                    unsafe_allow_html=True)
                    
                # Waiting for sometime for the success message to be seen by the user.
                time.sleep(1)

                # ── Full reset ────────────────────────────────────────────────
                st.session_state["_needs_reset"] = True
                # Also clear file uploader state by clearing their session keys
                st.session_state["step_uploader_key"] = st.session_state.get("step_uploader_key", 0) + 1
                st.session_state["dem_uploader_key"]  = st.session_state.get("dem_uploader_key", 0) + 1
                for k in ["step_uploaded_name"]:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

            except Exception as e:
                st.markdown(
                    f'<div class="error-banner">Failed to record data: {e}<br>'
                    f'<pre style="font-size:0.75rem;margin-top:0.5rem">{traceback.format_exc()}</pre></div>',
                    unsafe_allow_html=True)

        # ── Floating FAB nav ──────────────────────────────────────────────────
        


    # ═════════════════════════════════════════════════════════════════════════════
    # MODE 2 – VIEW / UPDATE
    # ═════════════════════════════════════════════════════════════════════════════
    elif mode == "View / Update Record":
        # ── Handle pending deletion (runs BEFORE the rest of the page) ────────
        if st.session_state.get("_delete_confirmed"):
            _del_id = st.session_state.pop("_delete_confirmed")
            
            prefix = f"sim{_del_id}"
                    
            master_df = master_df[master_df["simulation_id"] != _del_id]
            try:
                save_excel(excel_path, master_df)
                # Clean up associated files using the correct prefix
                edir = os.path.dirname(excel_path)
                for suffix in ["_liner_profile.png", "_profile_analysis.png", "_angular_profile.csv"]:
                    fp = os.path.join(edir, f"{prefix}{suffix}")
                    if os.path.exists(fp):
                        try: os.remove(fp)
                        except: pass
                st.session_state.pop("_delete_pending", None)
                st.session_state["selected_sim_id"] = None
                st.session_state["_flash_msg"] = f"Simulation {_del_id} has been deleted successfully."
                st.rerun()
            except PermissionError:
                st.session_state.pop("_delete_pending", None)
                # Reload from disk since in-memory df was already filtered
                master_df = load_excel(excel_path)
                st.markdown(
                    '<div class="error-banner">Could not save — the Excel file is open in another application. '
                    'Please close it and try again.</div>',
                    unsafe_allow_html=True)

        # ── Flash message (shown once after rerun) ────────────────────────────
        if st.session_state.get("_flash_msg"):
            st.markdown(
                f'<div class="success-banner">{st.session_state.pop("_flash_msg")}</div>',
                unsafe_allow_html=True)

        st.markdown(
            '<h2 style="font-size:1.2rem;font-weight:700;color:#0f172a;margin-bottom:0.25rem">'
            'Dataset Records</h2>'
            '<p style="color:#64748b;font-size:0.85rem;margin-top:0">'
            'Browse and update previously recorded simulation entries.</p>',
            unsafe_allow_html=True)

        if master_df.empty:
            st.markdown(
                '<div class="info-banner">No records found. Switch to <b>Add Data</b> to create the first record.</div>',
                unsafe_allow_html=True)
            st.stop()

        # ── Summary table ─────────────────────────────────────────────────────
        summary_cols = ["simulation_id", "local_name", "is_AG", "is_SAG", "is_PM", "is_BM",
                        "mill_rpm", "eff_mill_dia", "n_total_lifters",
                        "ore_density", "liner_density", "media_radius"]
        show_cols = [c for c in summary_cols if c in master_df.columns]
        sim_summary = (master_df.drop_duplicates("simulation_id")[show_cols]
                       .reset_index(drop=True))
        st.dataframe(sim_summary)

        st.divider()
        sim_ids = sorted(master_df["simulation_id"].dropna().unique().astype(int).tolist())

        sel_c1, sel_c2 = st.columns([3, 1])
        with sel_c1:
            selected = st.selectbox("Select Simulation ID to View / Edit", sim_ids, key="view_sim_sel")
            st.session_state["selected_sim_id"] = selected
        with sel_c2:
            st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
            if st.button("Delete Data", width="stretch", key="btn_delete_sim"):
                st.session_state["_delete_pending"] = selected

        # ── Confirmation banner (shown only while pending) ────────────────────
        if st.session_state.get("_delete_pending") == selected:
            st.markdown(
                f'<div class="warn-banner" style="margin-bottom:0.5rem">'
                f'Are you sure you want to <b>permanently delete</b> all records for '
                f'<b>Simulation ID {selected}</b>? This action cannot be undone.</div>',
                unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns([1, 1, 2])
            if cc1.button("Yes, Delete", type="primary", width="stretch", key="btn_confirm_del"):
                st.session_state["_delete_confirmed"] = selected
                st.session_state.pop("_delete_pending", None)
                st.rerun()
            if cc2.button("Cancel", width="stretch", key="btn_cancel_del"):
                st.session_state.pop("_delete_pending", None)
                st.rerun()

        sim_rows = master_df[master_df["simulation_id"] == selected].copy()
        first = sim_rows.iloc[0]

        mill_opts = ["AG Mill", "SAG Mill", "Pebble Mill", "Ball Mill"]
        cur_mill = (
            "AG Mill"     if first.get("is_AG",  0) == 1 else
            "SAG Mill"    if first.get("is_SAG", 0) == 1 else
            "Pebble Mill" if first.get("is_PM",  0) == 1 else "Ball Mill"
        )

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:1rem">'
            f'<span class="sim-badge"><span class="sim-badge-label">SIM</span> {selected}</span>'
            f'<span style="color:#64748b;font-size:0.85rem">{len(sim_rows)} rows in dataset</span>'
            f'</div>',
            unsafe_allow_html=True)

        # ── Look for saved liner profile images ───────────────────────────────
        excel_dir = os.path.dirname(excel_path)
        project_dir = os.path.dirname(os.path.abspath(__file__))
        
        search_dirs = [excel_dir, project_dir]
        sim_imgs = find_simulation_images(selected, search_dirs)

        liner_img_path    = sim_imgs.get("liner_profile")
        face_img_path     = sim_imgs.get("face_detection")
        analysis_img_path = sim_imgs.get("profile_analysis")
        eval_img_path     = sim_imgs.get("evaluation")
        ang_csv_path_v    = sim_imgs.get("angular_csv")

        # ── All sections in a form ────────────────────────────────────────────
        with st.form("update_form"):

            # ── 1. Liner Geometry (read-only display) ─────────────────────────
            st.markdown(
                '<div class="sec-header" style="margin-top:0.5rem">'
                '  <div class="sec-bar" style="background:#818cf8"></div>'
                '  <span class="sec-title-text">Liner Geometry</span>'
                '  <span class="sec-num">01</span>'
                '</div>',
                unsafe_allow_html=True)

            ml_n = first.get("n_total_lifters", None)
            if pd.notna(ml_n) and str(ml_n).strip() not in ("", "nan", "—"):
                ml_n_val = int(float(ml_n))
                ml_ru = first.get("n_repeat_units", None)
                ml_ru_str = str(int(float(ml_ru))) if (pd.notna(ml_ru) and str(ml_ru).strip() not in ("", "nan", "—")) else "—"
                ml_lpu = first.get("n_lifters_per_unit", None)
                ml_lpu_str = f"{float(ml_lpu):.2f}" if (pd.notna(ml_lpu) and str(ml_lpu).strip() not in ("", "nan", "—")) else "—"

                st.markdown(
                    f'<div class="metric-chip">Lifters <strong>{ml_n_val}</strong></div>'
                    f'<div class="metric-chip">Repeat units <strong>{ml_ru_str}</strong></div>'
                    f'<div class="metric-chip">Per unit <strong>{ml_lpu_str}</strong></div>',
                    unsafe_allow_html=True)
                lead_v  = first.get("leading_face_angle", None)
                trail_v = first.get("trailing_face_angle", None)
                s_lead_v  = first.get("short_leading_face_angle", None)
                s_trail_v = first.get("short_trailing_face_angle", None)

                angle_chips = ""
                if lead_v is not None and pd.notna(lead_v):
                    angle_chips += f'<div class="metric-chip">Tall Lead <strong>{float(lead_v):.1f}°</strong></div>'
                if trail_v is not None and pd.notna(trail_v):
                    angle_chips += f'<div class="metric-chip">Tall Trail <strong>{float(trail_v):.1f}°</strong></div>'
                if s_lead_v is not None and pd.notna(s_lead_v):
                    angle_chips += f'<div class="metric-chip">Short Lead <strong>{float(s_lead_v):.1f}°</strong></div>'
                if s_trail_v is not None and pd.notna(s_trail_v):
                    angle_chips += f'<div class="metric-chip">Short Trail <strong>{float(s_trail_v):.1f}°</strong></div>'
                if angle_chips:
                    st.markdown(angle_chips, unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-banner">No liner geometry data recorded for this simulation.</div>', unsafe_allow_html=True)

            if liner_img_path and os.path.exists(liner_img_path):
                st.markdown('<span class="sub-label">Liner Profile (saved)</span>', unsafe_allow_html=True)
                st.image(liner_img_path, use_container_width=True)
            if face_img_path and os.path.exists(face_img_path):
                st.markdown('<span class="sub-label">Face Angle Detection (saved)</span>', unsafe_allow_html=True)
                st.image(face_img_path, use_container_width=True)
            if analysis_img_path and os.path.exists(analysis_img_path):
                with st.expander("Full Diagnostic Analysis (9-panel)", expanded=False):
                    st.image(analysis_img_path, use_container_width=True)
            if eval_img_path and os.path.exists(eval_img_path):
                with st.expander("Model Performance Evaluation Plot", expanded=False):
                    st.image(eval_img_path, use_container_width=True)
            if ang_csv_path_v and os.path.exists(ang_csv_path_v):
                st.markdown(
                    f'<div class="success-banner">Angular profile CSV found: '
                    f'<code>{os.path.basename(ang_csv_path_v)}</code></div>',
                    unsafe_allow_html=True)

            st.markdown("---")

            # ── 2. Mill Configuration ─────────────────────────────────────────
            st.markdown(
                '<div class="sec-header">'
                '  <div class="sec-bar" style="background:#34d399"></div>'
                '  <span class="sec-title-text">Mill Configuration</span>'
                '  <span class="sec-num">02</span>'
                '</div>',
                unsafe_allow_html=True)

            new_mill = st.selectbox("Mill Type", mill_opts,
                                    index=mill_opts.index(cur_mill), key="upd_mill")
            c1, c2 = st.columns(2)
            with c1:
                new_rpm = st.number_input("Mill RPM",
                    value=float(first.get("mill_rpm", 0) or 0),
                    min_value=0.0, step=0.1, format="%.3f", key="upd_rpm")
            with c2:
                new_dia = st.number_input("Effective Mill Diameter (m)",
                    value=float(first.get("eff_mill_dia", 0) or 0),
                    min_value=0.0, step=0.01, format="%.3f", key="upd_dia")

            st.markdown("---")

            # ── 3. Ore Properties ─────────────────────────────────────────────
            st.markdown(
                '<div class="sec-header">'
                '  <div class="sec-bar" style="background:#fb923c"></div>'
                '  <span class="sec-title-text">Ore Properties</span>'
                '  <span class="sec-num">03</span>'
                '</div>',
                unsafe_allow_html=True)

            upd_no_ore = new_mill in ["Ball Mill", "Pebble Mill"]
            if upd_no_ore:
                st.info(f"Ore properties are not applicable for {new_mill}. Values will be saved as NaN.")
                new_ore_dens = new_ore_pois = new_ore_shear = new_ore_rad = new_ore_mass = 0.0
                upd_ore_psd = [[0.5, 33.33], [0.5, 33.33], [0.5, 33.34]]
            else:
                oc1, oc2, oc3, oc4 = st.columns(4)
                new_ore_dens  = oc1.number_input("Density (kg/m³)", value=float(first.get("ore_density", 2700) or 2700), min_value=0.0, format="%.3f", key="upd_od")
                new_ore_pois  = oc2.number_input("Poisson's Ratio", value=float(first.get("ore_poisson", 0.25) or 0.25), min_value=0.0, max_value=0.5, format="%.3f", key="upd_op")
                new_ore_shear = oc3.number_input("Shear Modulus (N/m²)", value=float(first.get("ore_shear_m", 1e8) or 1e8), min_value=0.0, format="%.3e", key="upd_os")
                new_ore_rad   = oc4.number_input("Radius (mm)", value=float(first.get("ore_radius", 25) or 25), min_value=0.0, format="%.3f", key="upd_or")

                st.markdown('<span class="sub-label">Particle Size Distribution</span>', unsafe_allow_html=True)
                psd_h = st.columns([1, 1.2, 1.2])
                psd_h[0].markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8">Fraction</span>', unsafe_allow_html=True)
                psd_h[1].markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8">Scale</span>', unsafe_allow_html=True)
                psd_h[2].markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8">% of Mass</span>', unsafe_allow_html=True)
                upd_ore_psd = []
                for i in range(3):
                    pc1, pc2, pc3 = st.columns([1, 1.2, 1.2])
                    pc1.markdown(f'<span style="font-size:0.85rem;color:#475569">Fraction {i+1}</span>', unsafe_allow_html=True)
                    raw_s = first.get(f"ore_psd_s{i}")
                    raw_p = first.get(f"ore_psd_p{i}")
                    if pd.notna(raw_s) and pd.notna(raw_p):
                        scale_val = float(raw_s)
                        mass_pct = float(raw_p)
                    else:
                        d_val = float(first.get(f"D{['10','50','90'][i]}_ore", 0.5) or 0.5)
                        scale_val = d_val / new_ore_rad if new_ore_rad > 0 else 0.0
                        mass_pct = [10.0, 40.0, 50.0][i]
                    
                    sv = pc2.number_input(f"Scale o{i}", value=scale_val,
                                          min_value=0.0, step=0.1, format="%.3f", key=f"upd_ors_{i}", label_visibility="collapsed")
                    pv = pc3.number_input(f"% o{i}", value=mass_pct, min_value=0.0, max_value=100.0,
                                          step=1.0, format="%.3f", key=f"upd_orp_{i}", label_visibility="collapsed")
                    upd_ore_psd.append([sv, pv])

                new_ore_mass = st.number_input("Total Ore Mass (kg)", value=float(first.get("ore_mass", 0) or 0), min_value=0.0, format="%.3f", key="upd_om")

            st.markdown("---")

            # ── 4. Liner Properties ───────────────────────────────────────────
            st.markdown(
                '<div class="sec-header">'
                '  <div class="sec-bar" style="background:#a78bfa"></div>'
                '  <span class="sec-title-text">Liner Properties</span>'
                '  <span class="sec-num">04</span>'
                '</div>',
                unsafe_allow_html=True)

            lc1, lc2, lc3 = st.columns(3)
            new_lin_dens  = lc1.number_input("Density (kg/m³)", value=float(first.get("liner_density", 7800) or 7800), min_value=0.0, format="%.3f", key="upd_ld")
            new_lin_pois  = lc2.number_input("Poisson's Ratio", value=float(first.get("liner_poisson", 0.28) or 0.28), min_value=0.0, max_value=0.5, format="%.3f", key="upd_lp")
            new_lin_shear = lc3.number_input("Shear Modulus (N/m²)", value=float(first.get("liner_shear_m", 7e10) or 7e10), min_value=0.0, format="%.3e", key="upd_ls")

            st.markdown("---")

            # ── 5. Media Properties ───────────────────────────────────────────
            st.markdown(
                '<div class="sec-header">'
                '  <div class="sec-bar" style="background:#38bdf8"></div>'
                '  <span class="sec-title-text">Media Properties</span>'
                '  <span class="sec-num">05</span>'
                '</div>',
                unsafe_allow_html=True)

            new_med_rad = st.number_input("Media Radius (mm)", value=float(first.get("media_radius", 0) or 0), min_value=0.0, format="%.3f", key="upd_mr")
            
            upd_med_psd = []
            if new_med_rad > 0:
                mc1, mc2, mc3 = st.columns(3)
                new_med_dens  = mc1.number_input("Density (kg/m³)", value=float(first.get("media_density", 7800) or 7800), min_value=0.0, format="%.3f", key="upd_mden")
                new_med_pois  = mc2.number_input("Poisson's Ratio", value=float(first.get("media_poisson", 0.28) or 0.28), min_value=0.0, max_value=0.5, format="%.3f", key="upd_mpoi")
                new_med_shear = mc3.number_input("Shear Modulus (N/m²)", value=float(first.get("media_shear_m", 7e10) or 7e10), min_value=0.0, format="%.3e", key="upd_msh")
                
                st.markdown('<span class="sub-label">Media Particle Size Distribution</span>', unsafe_allow_html=True)
                mpsd_h = st.columns([1, 1.2, 1.2])
                mpsd_h[0].markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8">Fraction</span>', unsafe_allow_html=True)
                mpsd_h[1].markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8">Scale</span>', unsafe_allow_html=True)
                mpsd_h[2].markdown('<span style="font-size:0.72rem;font-weight:700;color:#94a3b8">% of Mass</span>', unsafe_allow_html=True)
                for i in range(3):
                    pc1, pc2, pc3 = st.columns([1, 1.2, 1.2])
                    pc1.markdown(f'<span style="font-size:0.85rem;color:#475569">Fraction {i+1}</span>', unsafe_allow_html=True)
                    raw_s = first.get(f"media_psd_s{i}")
                    raw_p = first.get(f"media_psd_p{i}")
                    if pd.notna(raw_s) and pd.notna(raw_p):
                        scale_val = float(raw_s)
                        mass_pct = float(raw_p)
                    else:
                        d_val = float(first.get(f"D{['10','50','90'][i]}_media", 0.5) or 0.5)
                        scale_val = d_val / new_med_rad if new_med_rad > 0 else 0.0
                        mass_pct = [10.0, 40.0, 50.0][i]
                    
                    sv = pc2.number_input(f"Scale m{i}", value=scale_val,
                                          min_value=0.0, step=0.1, format="%.3f", key=f"upd_mds_{i}", label_visibility="collapsed")
                    pv = pc3.number_input(f"% m{i}", value=mass_pct, min_value=0.0, max_value=100.0,
                                          step=1.0, format="%.3f", key=f"upd_mdp_{i}", label_visibility="collapsed")
                    upd_med_psd.append([sv, pv])
                    
                new_med_mass  = st.number_input("Total Media Mass (kg)", value=float(first.get("media_mass", 0) or 0), min_value=0.0, format="%.3f", key="upd_mm")
            else:
                st.markdown('<div class="info-banner">Media Radius is 0. Media properties are hidden and will be saved as 0.</div>', unsafe_allow_html=True)
                new_med_dens = 0.0
                new_med_pois = 0.0
                new_med_shear = 0.0
                new_med_mass = 0.0

            st.markdown("---")

            st.markdown(
                '<div class="sec-header">'
                '  <div class="sec-bar" style="background:#4ade80"></div>'
                '  <span class="sec-title-text">Factory &amp; Velocities</span>'
                '  <span class="sec-num">06</span>'
                '</div>',
                unsafe_allow_html=True)

            st.markdown(
                '<div class="info-banner">Injection velocities have been removed from the dataset requirements.</div>',
                unsafe_allow_html=True)

            st.markdown("---")

            # ── 7. Interaction Properties ─────────────────────────────────────
            st.markdown(
                '<div class="sec-header">'
                '  <div class="sec-bar" style="background:#f472b6"></div>'
                '  <span class="sec-title-text">Interaction Properties</span>'
                '  <span class="sec-num">07</span>'
                '</div>',
                unsafe_allow_html=True)

            int_hc = st.columns([2.2, 1.6, 1.6, 1.6])
            for j, lbl in enumerate(["Pair", "Rolling Friction", "Sliding Friction", "Restitution"]):
                int_hc[j].markdown(f'<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">{lbl}</span>', unsafe_allow_html=True)

            upd_interactions = []
            int_defs = [
                ("Media–Media", "mm_rf", "mm_sf", "mm_res", True,  False),
                ("Media–Ore",   "mo_rf", "mo_sf", "mo_res", True,  True),
                ("Ore–Ore",     "oo_rf", "oo_sf", "oo_res", False, True),
                ("Media–Liner", "ml_rf", "ml_sf", "ml_res", True,  False),
                ("Ore–Liner",   "ol_rf", "ol_sf", "ol_res", False, True),
            ]

            has_media  = new_med_rad > 0
            upd_no_ore = new_mill in ["Ball Mill", "Pebble Mill"]

            if not has_media:
                st.markdown('<div class="info-banner">Media-related interactions are hidden (Media Radius = 0 or AG Mill). Saved as NaN.</div>', unsafe_allow_html=True)
            if upd_no_ore:
                st.markdown(f'<div class="info-banner">Ore-related interactions are hidden for {new_mill}. Saved as NaN.</div>', unsafe_allow_html=True)

            for ii, (iname, krf, ksf, kres, is_media, is_ore) in enumerate(int_defs):
                hidden = (is_media and not has_media) or (is_ore and upd_no_ore)
                if hidden:
                    upd_interactions.append((krf, ksf, kres, np.nan, np.nan, np.nan))
                    continue

                def _safe_val(v, default):
                    try:
                        f = float(v)
                        return f if f == f else default  # NaN check
                    except (TypeError, ValueError):
                        return default

                ic = st.columns([2.2, 1.6, 1.6, 1.6])
                ic[0].markdown(f'<span style="font-size:0.9rem;font-weight:500;color:#1e293b">{iname}</span>', unsafe_allow_html=True)
                vrf  = ic[1].number_input(f"RF {iname}",  value=_safe_val(first.get(krf), 0.001), min_value=0.0, max_value=1.0, step=0.001, format="%.3f", key=f"upd_rf_{ii}", label_visibility="collapsed")
                vsf  = ic[2].number_input(f"SF {iname}",  value=_safe_val(first.get(ksf), 0.3),   min_value=0.0, max_value=2.0, step=0.01,  format="%.3f", key=f"upd_sf_{ii}", label_visibility="collapsed")
                vres = ic[3].number_input(f"Res {iname}", value=_safe_val(first.get(kres), 0.3),   min_value=0.0, max_value=1.0, step=0.01,  format="%.3f", key=f"upd_res_{ii}", label_visibility="collapsed")
                upd_interactions.append((krf, ksf, kres, vrf, vsf, vres))

            st.markdown("---")

            # ── DEM time-series preview ───────────────────────────────────────
            dem_cols_show = ["pct_rotation", "cf_max_particle", "ke_max_particle", "power_total_geometry_kw"]
            avail_dem = [c for c in dem_cols_show if c in sim_rows.columns]
            if avail_dem:
                st.markdown('<span class="sub-label">DEM Time-Series (first 10 rows)</span>', unsafe_allow_html=True)
                st.table(sim_rows[avail_dem].head(10))

            # ── Submit ────────────────────────────────────────────────────────
            submitted = st.form_submit_button("Update Record", type="primary", width="stretch")

        if submitted:
            try:
                idx = master_df.index[master_df["simulation_id"] == selected]
                mill_oh = {
                    "is_AG":  1 if new_mill == "AG Mill"     else 0,
                    "is_SAG": 1 if new_mill == "SAG Mill"    else 0,
                    "is_PM":  1 if new_mill == "Pebble Mill" else 0,
                    "is_BM":  1 if new_mill == "Ball Mill"   else 0,
                }
                for col, val in mill_oh.items():
                    master_df.loc[idx, col] = val

                master_df.loc[idx, "mill_rpm"]      = new_rpm
                no_ore = (new_mill in ["Ball Mill", "Pebble Mill"])
                master_df.loc[idx, "eff_mill_dia"]  = new_dia
                master_df.loc[idx, "ore_density"]   = np.nan if no_ore else new_ore_dens
                master_df.loc[idx, "ore_poisson"]   = np.nan if no_ore else new_ore_pois
                master_df.loc[idx, "ore_shear_m"]   = np.nan if no_ore else new_ore_shear
                master_df.loc[idx, "ore_radius"]    = np.nan if no_ore else new_ore_rad
                master_df.loc[idx, "ore_mass"]      = np.nan if no_ore else new_ore_mass
                master_df.loc[idx, "liner_density"] = new_lin_dens
                master_df.loc[idx, "liner_poisson"] = new_lin_pois
                master_df.loc[idx, "liner_shear_m"] = new_lin_shear
                master_df.loc[idx, "media_density"] = np.nan if new_med_rad == 0 else new_med_dens
                master_df.loc[idx, "media_poisson"] = np.nan if new_med_rad == 0 else new_med_pois
                master_df.loc[idx, "media_shear_m"] = np.nan if new_med_rad == 0 else new_med_shear
                master_df.loc[idx, "media_radius"]  = np.nan if new_med_rad == 0 else new_med_rad
                master_df.loc[idx, "media_mass"]    = np.nan if new_med_rad == 0 else new_med_mass

                # PSD D-percentiles from updated inputs
                if no_ore:
                    upd_ore_d10, upd_ore_d50, upd_ore_d90 = np.nan, np.nan, np.nan
                    for i in range(3):
                        master_df.loc[idx, f"ore_psd_s{i}"] = np.nan
                        master_df.loc[idx, f"ore_psd_p{i}"] = np.nan
                else:
                    upd_ore_d10 = _compute_percentile(upd_ore_psd, 10) * new_ore_rad
                    upd_ore_d50 = _compute_percentile(upd_ore_psd, 50) * new_ore_rad
                    upd_ore_d90 = _compute_percentile(upd_ore_psd, 90) * new_ore_rad
                    for i in range(3):
                        master_df.loc[idx, f"ore_psd_s{i}"] = upd_ore_psd[i][0]
                        master_df.loc[idx, f"ore_psd_p{i}"] = upd_ore_psd[i][1]
                master_df.loc[idx, "D10_ore"] = upd_ore_d10
                master_df.loc[idx, "D50_ore"] = upd_ore_d50
                master_df.loc[idx, "D90_ore"] = upd_ore_d90
                
                if new_med_rad > 0:
                    upd_med_d10 = _compute_percentile(upd_med_psd, 10) * new_med_rad
                    upd_med_d50 = _compute_percentile(upd_med_psd, 50) * new_med_rad
                    upd_med_d90 = _compute_percentile(upd_med_psd, 90) * new_med_rad
                    for i in range(3):
                        master_df.loc[idx, f"media_psd_s{i}"] = upd_med_psd[i][0]
                        master_df.loc[idx, f"media_psd_p{i}"] = upd_med_psd[i][1]
                else:
                    upd_med_d10, upd_med_d50, upd_med_d90 = np.nan, np.nan, np.nan
                    for i in range(3):
                        master_df.loc[idx, f"media_psd_s{i}"] = np.nan
                        master_df.loc[idx, f"media_psd_p{i}"] = np.nan
                    
                master_df.loc[idx, "D10_media"] = upd_med_d10
                master_df.loc[idx, "D50_media"] = upd_med_d50
                master_df.loc[idx, "D90_media"] = upd_med_d90
                # NaN-out media physical properties and media-related interactions when no media
                if new_med_rad == 0:
                    for mc in ["media_density", "media_poisson", "media_shear_m", "media_radius", "media_mass",
                               "mm_rf", "mm_sf", "mm_res", "ml_rf", "ml_sf", "ml_res"]:
                        if mc in master_df.columns:
                            master_df.loc[idx, mc] = np.nan
                            
                # NaN-out ore physical properties and ore-related interactions when no ore
                if no_ore:
                    for mc in ["ore_density", "ore_poisson", "ore_shear_m", "ore_radius", "ore_mass",
                               "oo_rf", "oo_sf", "oo_res", "ol_rf", "ol_sf", "ol_res"]:
                        if mc in master_df.columns:
                            master_df.loc[idx, mc] = np.nan
                            
                # Media-Ore interactions are NaN if EITHER media or ore is missing
                if new_med_rad == 0 or no_ore:
                    for mc in ["mo_rf", "mo_sf", "mo_res"]:
                        if mc in master_df.columns:
                            master_df.loc[idx, mc] = np.nan

                for krf, ksf, kres, vrf, vsf, vres in upd_interactions:
                    master_df.loc[idx, krf]  = vrf
                    master_df.loc[idx, ksf]  = vsf
                    master_df.loc[idx, kres] = vres

                save_excel(excel_path, master_df)
                st.markdown(
                    f'<div class="success-banner">Simulation {selected} updated successfully.</div>',
                    unsafe_allow_html=True)
                st.rerun()
            except Exception as e:
                st.markdown(
                    f'<div class="error-banner">Update failed: {e}</div>',
                    unsafe_allow_html=True)



    # Finalize splash screen once the very last content element of the page has loaded
    if show_splash:
        animate_splash_step(splash_placeholder, 76, 100, delay=0.02)
        time.sleep(0.15)
        splash_placeholder.empty()
        st.session_state["splash_shown"] = True

if __name__ == '__main__':
    main()
