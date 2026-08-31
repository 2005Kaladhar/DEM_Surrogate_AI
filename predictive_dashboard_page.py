import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import pickle
import time
import zipfile
import io
import tempfile
from sklearn.metrics.pairwise import cosine_similarity
import traceback
from analysis_engine import start_analysis_process

import subprocess

PLOTLY_ZOOM_CONFIG = {
    'scrollZoom': True,
    'displaylogo': False,
    'doubleClick': 'reset',
    'modeBarButtonsToRemove': [],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'prediction_chart',
        'height': 600,
        'width': 1200,
        'scale': 2
    }
}

def pick_directory(title="Select Project Directory"):
    """Desktop folder picker using Tkinter dialog."""
    script = f"""
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
path = filedialog.askdirectory(title={repr(title)})
print(path)
"""
    try:
        res = subprocess.run(["python", "-c", script], capture_output=True, text=True, timeout=60)
        return res.stdout.strip()
    except Exception:
        return ""

def load_models_and_scaler():
    active_dir = st.session_state.get("active_project_dir") or st.session_state.get("selected_output_parent")
    base_dir = active_dir if active_dir and os.path.exists(active_dir) else os.path.dirname(os.path.abspath(__file__))
    
    models_dir = os.path.join(base_dir, "model_evaluation")
    if not os.path.exists(models_dir):
        models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_evaluation")
        
    prep_dir = os.path.join(base_dir, "PRE PROCESSED", "preprocessing_report")
    if not os.path.exists(prep_dir):
        prep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRE PROCESSED", "preprocessing_report")
    
    try:
        with open(os.path.join(prep_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(models_dir, "final_model_ke_max_particle.pkl"), "rb") as f:
            model_ke = pickle.load(f)
        with open(os.path.join(models_dir, "final_model_power_total_geometry_kw.pkl"), "rb") as f:
            model_power = pickle.load(f)
        with open(os.path.join(models_dir, "final_model_cf_max_particle.pkl"), "rb") as f:
            model_cf = pickle.load(f)
        
        # Load dataset for Resemblance Engine
        df_train_scaled = pd.read_csv(os.path.join(prep_dir, "train_scaled.csv"))
        df_train_unscaled = pd.read_csv(os.path.join(prep_dir, "train_unscaled.csv"))
            
        return scaler, model_ke, model_power, model_cf, df_train_scaled, df_train_unscaled
    except Exception as e:
        st.error(f"Failed to load ML artifacts: {e}")
        return None, None, None, None, None, None

def get_feature_cols():
    return ['is_AG', 'is_SAG', 'is_PM', 'is_BM', 'ore_density', 'ore_poisson', 'ore_shear_m', 'ore_radius', 'D10_ore', 'D50_ore', 'D90_ore', 'ore_mass', 'liner_density', 'liner_poisson', 'liner_shear_m', 'media_density', 'media_poisson', 'media_shear_m', 'media_radius', 'D10_media', 'D50_media', 'D90_media', 'media_mass', 'mill_rpm', 'eff_mill_dia', 'mm_rf', 'mo_rf', 'oo_rf', 'ml_rf', 'ol_rf', 'mm_sf', 'mo_sf', 'oo_sf', 'ml_sf', 'ol_sf', 'mm_res', 'mo_res', 'oo_res', 'ml_res', 'ol_res', 'n_total_lifters', 'n_repeat_units', 'n_lifters_per_unit', 'leading_face_angle', 'trailing_face_angle', 'short_leading_face_angle', 'short_trailing_face_angle', 'pct_rotation', 'shape_k0', 'shape_k1', 'shape_k2', 'shape_k3', 'shape_k4', 'shape_k5', 'shape_k6', 'shape_k7', 'shape_k8', 'shape_k9', 'shape_k10', 'shape_k11', 'shape_k12', 'shape_k13', 'shape_k14', 'shape_k15', 'shape_k16', 'shape_k17', 'shape_k18', 'shape_k19', 'shape_k20', 'shape_k21', 'shape_k22', 'shape_k23', 'shape_k24', 'shape_k25', 'shape_k26', 'shape_k27', 'shape_k28', 'shape_k29', 'shape_k30', 'shape_k31', 'shape_k32', 'shape_k33', 'shape_k34', 'shape_k35', 'shape_k36', 'shape_k37', 'shape_k38', 'shape_k39', 'shape_k40', 'shape_k41', 'shape_k42', 'shape_k43', 'shape_k44', 'shape_k45', 'shape_k46', 'shape_k47', 'shape_k48', 'shape_k49', 'critical_speed_fraction', 'has_short_lifter', 'face_angle_asymmetry', 'shape_energy', 'shape_hf_sharpness']

def build_predictive_dataset(scaler=None):
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        cols = list(scaler.feature_names_in_)
    else:
        cols = ['is_AG', 'is_SAG', 'is_PM', 'is_BM', 'ore_density', 'ore_poisson', 'ore_shear_m', 'ore_radius', 'D10_ore', 'D50_ore', 'D90_ore', 'ore_mass', 'liner_density', 'liner_poisson', 'liner_shear_m', 'media_density', 'media_poisson', 'media_shear_m', 'media_radius', 'D10_media', 'D50_media', 'D90_media', 'media_mass', 'mill_rpm', 'eff_mill_dia', 'mm_rf', 'mo_rf', 'oo_rf', 'ml_rf', 'ol_rf', 'mm_sf', 'mo_sf', 'oo_sf', 'ml_sf', 'ol_sf', 'mm_res', 'mo_res', 'oo_res', 'ml_res', 'ol_res', 'n_total_lifters', 'n_repeat_units', 'n_lifters_per_unit', 'leading_face_angle', 'trailing_face_angle', 'short_leading_face_angle', 'short_trailing_face_angle', 'pct_rotation', 'shape_k0', 'shape_k1', 'shape_k2', 'shape_k3', 'shape_k4', 'shape_k5', 'shape_k6', 'shape_k7', 'shape_k8', 'shape_k9', 'shape_k10', 'shape_k11', 'shape_k12', 'shape_k13', 'shape_k14', 'shape_k15', 'shape_k16', 'shape_k17', 'shape_k18', 'shape_k19', 'shape_k20', 'shape_k21', 'shape_k22', 'shape_k23', 'shape_k24', 'shape_k25', 'shape_k26', 'shape_k27', 'shape_k28', 'shape_k29', 'shape_k30', 'shape_k31', 'shape_k32', 'shape_k33', 'shape_k34', 'shape_k35', 'shape_k36', 'shape_k37', 'shape_k38', 'shape_k39', 'shape_k40', 'shape_k41', 'shape_k42', 'shape_k43', 'shape_k44', 'shape_k45', 'shape_k46', 'shape_k47', 'shape_k48', 'shape_k49', 'critical_speed_fraction', 'has_short_lifter', 'face_angle_asymmetry', 'shape_energy', 'shape_hf_sharpness', 'tip_speed', 'lifter_density', 'media_load_fraction', 'has_ore', 'froude_number', 'charge_kinetic_head', 'lifter_strike_freq', 'power_flux_proxy', 'specific_impact_energy', 'rot_sin', 'rot_cos', 'media_aspect_ratio', 'total_charge_mass', 'media_count_proxy']

    df = pd.DataFrame(0.0, index=range(101), columns=cols)
    df['pct_rotation'] = np.linspace(0, 100, 101)
    
    mtype = st.session_state.get("mill_type", "Ball Mill")
    df['is_AG']  = 1.0 if mtype == "AG Mill" else 0.0
    df['is_SAG'] = 1.0 if mtype == "SAG Mill" else 0.0
    df['is_PM']  = 1.0 if mtype == "Pebble Mill" else 0.0
    df['is_BM']  = 1.0 if mtype == "Ball Mill" else 0.0
    
    direct_maps = {
        'ore_density': 'ore_density', 'ore_poisson': 'ore_poisson', 'ore_shear_m': 'ore_shear_m', 'ore_radius': 'ore_radius', 'ore_mass': 'ore_mass',
        'liner_density': 'liner_density', 'liner_poisson': 'liner_poisson', 'liner_shear_m': 'liner_shear_m',
        'media_density': 'media_density', 'media_poisson': 'media_poisson', 'media_shear_m': 'media_shear_m', 'media_radius': 'media_radius', 'media_mass': 'media_mass',
        'mill_rpm': 'mill_rpm', 'eff_mill_dia': 'eff_mill_dia',
        'mm_rf': 'mm_rf', 'mo_rf': 'mo_rf', 'oo_rf': 'oo_rf', 'ml_rf': 'ml_rf', 'ol_rf': 'ol_rf',
        'mm_sf': 'mm_sf', 'mo_sf': 'mo_sf', 'oo_sf': 'oo_sf', 'ml_sf': 'ml_sf', 'ol_sf': 'ol_sf',
        'mm_res': 'mm_res', 'mo_res': 'mo_res', 'oo_res': 'oo_res', 'ml_res': 'ml_res', 'ol_res': 'ol_res',
        'n_total_lifters': 'n_total_lifters', 'n_repeat_units': 'n_repeat_units', 'n_lifters_per_unit': 'n_lifters_per_unit',
        'leading_face_angle': 'leading_face_angle', 'trailing_face_angle': 'trailing_face_angle', 
        'short_leading_face_angle': 'short_leading_face_angle', 'short_trailing_face_angle': 'short_trailing_face_angle',
        'D10_ore': 'D10_ore', 'D50_ore': 'D50_ore', 'D90_ore': 'D90_ore',
        'D10_media': 'D10_media', 'D50_media': 'D50_media', 'D90_media': 'D90_media'
    }
    for col, key in direct_maps.items():
        if col in df.columns:
            val = st.session_state.get(key, 0.0)
            df[col] = 0.0 if pd.isna(val) else float(val)
        
    for i in range(50):
        col = f"shape_k{i}"
        if col in df.columns:
            val = st.session_state.get(col, 0.0)
            df[col] = 0.0 if pd.isna(val) else float(val)
        
    # ── Feature Engineering ──────────────────────────────────────────────────
    if "critical_speed_fraction" in df.columns:
        df["critical_speed_fraction"] = df["mill_rpm"] / (42.3 / np.sqrt(df["eff_mill_dia"].replace(0, 1e-9)))
    if "has_short_lifter" in df.columns:
        df["has_short_lifter"] = ((df["short_leading_face_angle"] > 0) | (df["short_trailing_face_angle"] > 0)).astype(int)
    if "face_angle_asymmetry" in df.columns:
        df["face_angle_asymmetry"] = df["leading_face_angle"] - df["trailing_face_angle"]
    
    shape_k_cols = [f"shape_k{i}" for i in range(50) if f"shape_k{i}" in df.columns]
    if "shape_energy" in df.columns and shape_k_cols:
        df["shape_energy"] = np.sqrt((df[shape_k_cols] ** 2).sum(axis=1))
    hf_cols = [f"shape_k{i}" for i in range(30, 50) if f"shape_k{i}" in df.columns]
    if "shape_hf_sharpness" in df.columns and hf_cols:
        df["shape_hf_sharpness"] = df[hf_cols].mean(axis=1)

    if "tip_speed" in df.columns:
        df["tip_speed"] = (np.pi * df["eff_mill_dia"] * df["mill_rpm"]) / 60.0
    if "lifter_density" in df.columns:
        df["lifter_density"] = df["n_total_lifters"] / (np.pi * df["eff_mill_dia"].replace(0, 1e-9))
    if "media_load_fraction" in df.columns:
        mill_vol = np.pi * (df["eff_mill_dia"] / 2.0) ** 2 * df["eff_mill_dia"].replace(0, 1e-9)
        df["media_load_fraction"] = df["media_mass"] / (mill_vol * df["media_density"] + 1e-6)
    if "has_ore" in df.columns:
        df["has_ore"] = (df["ore_mass"].fillna(0) > 0).astype(int)

    omega = (df["mill_rpm"] * np.pi) / 30.0
    radius = df["eff_mill_dia"] / 2.0
    if "froude_number" in df.columns:
        df["froude_number"] = (omega ** 2 * radius) / 9.81
    if "charge_kinetic_head" in df.columns:
        ore_m = df["ore_mass"].fillna(0) if "ore_mass" in df.columns else 0.0
        ore_d = df["ore_density"].fillna(0) if "ore_density" in df.columns else 0.0
        med_m = df["media_mass"] if "media_mass" in df.columns else 0.0
        med_d = df["media_density"].replace(0, 1e-9) if "media_density" in df.columns else 7800.0

        v_med = med_m / med_d
        v_ore = np.where((ore_m > 0) & (ore_d > 0), ore_m / np.maximum(ore_d, 1e-9), 0.0)
        m_tot = med_m + np.where(ore_m > 0, ore_m, 0.0)
        v_tot = v_med + v_ore

        rho_charge_mix = np.where((ore_m > 0) & (v_tot > 0), m_tot / np.maximum(v_tot, 1e-9), med_d)
        df["charge_kinetic_head"] = 0.5 * rho_charge_mix * (df["tip_speed"] ** 2)
    if "lifter_strike_freq" in df.columns:
        df["lifter_strike_freq"] = df["n_total_lifters"] * (df["mill_rpm"] / 60.0)
    if "power_flux_proxy" in df.columns:
        df["power_flux_proxy"] = (df["tip_speed"] ** 2) * df["media_mass"] * df["mill_rpm"]
    if "specific_impact_energy" in df.columns:
        df["specific_impact_energy"] = (df["tip_speed"] ** 2) / (df["eff_mill_dia"].replace(0, 1e-9))
    if "rot_sin" in df.columns:
        df["rot_sin"] = np.sin(2.0 * np.pi * df["pct_rotation"] / 100.0)
    if "rot_cos" in df.columns:
        df["rot_cos"] = np.cos(2.0 * np.pi * df["pct_rotation"] / 100.0)
    if "media_aspect_ratio" in df.columns:
        df["media_aspect_ratio"] = df["media_radius"] / (df["eff_mill_dia"].replace(0, 1e-9))
    if "total_charge_mass" in df.columns:
        df["total_charge_mass"] = df["media_mass"].fillna(0) + df["ore_mass"].fillna(0)
    if "media_count_proxy" in df.columns:
        ball_vol = (4.0 / 3.0) * np.pi * (df["media_radius"].replace(0, 1e-9) ** 3)
        df["media_count_proxy"] = df["media_mass"] / (ball_vol * df["media_density"] + 1e-6)
    
    return df

def render_predictive_dashboard():
    # Helper functions identical to Add Data page
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
            if sk not in st.session_state: st.session_state[sk] = 0.0
            if pk not in st.session_state: st.session_state[pk] = 0.0
            c1, c2, c3 = st.columns([1, 1.2, 1.2])
            c1.markdown(f'<span style="font-size:0.85rem;color:#475569;padding-top:0.5rem;display:block">Fraction {i+1}</span>', unsafe_allow_html=True)
            sv = c2.number_input(f"Scale {i+1}", min_value=0.0, value=float(st.session_state[sk]), step=0.1, format="%.3f", key=f"ai_{sk}", label_visibility="collapsed")
            pv = c3.number_input(f"% of Mass {i+1}", min_value=0.0, max_value=100.0, value=float(st.session_state[pk]), step=1.0, format="%.3f", key=f"ai_{pk}", label_visibility="collapsed")
            st.session_state[sk] = sv
            st.session_state[pk] = pv
            rows.append([sv, pv])
        return rows

    def _compute_percentile(psd_rows, p):
        valid_rows = [r for r in psd_rows if r[1] > 0]
        if not valid_rows: return 0.0
        valid_rows.sort(key=lambda x: x[0])
        scales = [r[0] for r in valid_rows]
        pcts   = [r[1] for r in valid_rows]
        cumulative = np.cumsum(pcts)
        total  = cumulative[-1]
        target = p / 100.0 * total
        if target <= 0: return 0.0
        for i in range(len(cumulative)):
            if cumulative[i] >= target:
                if i == 0: return scales[0]
                pct_prev = cumulative[i-1]
                pct_curr = cumulative[i]
                s_prev = scales[i-1]
                s_curr = scales[i]
                fraction = (target - pct_prev) / (pct_curr - pct_prev)
                return s_prev + fraction * (s_curr - s_prev)
        return scales[-1]

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
                        # Make sure to copy ml_features into root session state for predict logic
                        if "ml_features" in res and res["ml_features"]:
                            for k, v in res["ml_features"].items():
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
                if st.button("Stop Analysis", type="primary", key="ai_btn_stop"):
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
                    st.session_state["ai_step_uploader_key"] = st.session_state.get("ai_step_uploader_key", 0) + 1
                    st.session_state["step_file_path"] = None
                    st.session_state["ai_step_uploaded_name"] = ""
                    st.rerun()
    
            with st.expander("Show detailed logs", expanded=False):
                logs = st.session_state.get("analysis_log", [])
                if logs:
                    log_html = "<br>".join([f"[{lvl.upper()}] {txt}" for lvl, txt in logs])
                    st.markdown(f'<div style="font-family:monospace;font-size:0.8rem;color:#475569;max-height:200px;overflow-y:auto;background:#f8fafc;padding:8px;border:1px solid #e2e8f0;border-radius:4px;">{log_html}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No logs yet...")

    # Project Folder Selector Widget
    active_project_dir = st.session_state.get("active_project_dir") or st.session_state.get("selected_output_parent") or os.getcwd()
    col_proj1, col_proj2 = st.columns([3.5, 1.5])
    with col_proj1:
        st.markdown(f"**Active Project Folder:** `{active_project_dir}`")
    with col_proj2:
        if st.button("Switch Project Folder", help="Select custom project folder containing retrained models & scaler"):
            chosen_proj = pick_directory("Select Active Project Directory")
            if chosen_proj:
                st.session_state["active_project_dir"] = chosen_proj
                st.success(f"Switched active project folder to `{chosen_proj}`")
                time.sleep(0.5)
                st.rerun()
    st.markdown("---")

    _sec("ai-sec-files", "01", "Files &amp; Geometry", "Upload the liner STEP geometry file", "#818cf8")
    
    st.markdown('<span class="sub-label">Liner Geometry File (STEP / STP)</span>', unsafe_allow_html=True)
    
    uploader_key = st.session_state.get("ai_step_uploader_key", 0)
    step_uploader = st.file_uploader("STEP file", type=["step", "stp"], key=f"ai_step_upload_{uploader_key}", label_visibility="collapsed")
    
    if step_uploader is not None:
        suf = ".stp" if step_uploader.name.lower().endswith(".stp") else ".step"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
            tmp.write(step_uploader.read())
            tmp_path = tmp.name
        prev_name = st.session_state.get("ai_step_uploaded_name", "")
        if step_uploader.name != prev_name:
            st.session_state["step_file_path"]      = tmp_path
            st.session_state["ai_step_uploaded_name"]  = step_uploader.name
            st.session_state["angular_profile_df"]  = None
            st.session_state["ml_features"]         = None
            st.session_state["liner_profile_img"]   = None
            st.session_state["analysis_img"]        = None
            st.session_state["face_analysis_img"]   = None

    if st.session_state.get("liner_profile_img") or st.session_state.get("face_analysis_img") or st.session_state.get("analysis_img"):
        with st.expander("Diagnostic Visualizations", expanded=False):
            if st.session_state.get("liner_profile_img"):
                st.markdown('<span class="sub-label">Liner Profile Preview</span>', unsafe_allow_html=True)
                st.image(st.session_state["liner_profile_img"], use_container_width=True)
            if st.session_state.get("face_analysis_img"):
                st.markdown('<span class="sub-label">Face Angle Detection Regions</span>', unsafe_allow_html=True)
                st.image(st.session_state["face_analysis_img"], use_container_width=True)
            if st.session_state.get("analysis_img"):
                st.markdown('<span class="sub-label">Full Diagnostic Analysis (9-panel)</span>', unsafe_allow_html=True)
                st.image(st.session_state["analysis_img"], use_container_width=True)

    if st.session_state.get("ml_features"):
        ml = st.session_state["ml_features"]
        st.markdown('<span class="sub-label">Extracted ML Geometry Features</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-chip">Lifters <strong>{int(ml["n_total_lifters"])}</strong></div>'
            f'<div class="metric-chip">Repeat units <strong>{int(ml["n_repeat_units"])}</strong></div>'
            f'<div class="metric-chip">Lifters / unit <strong>{ml["n_lifters_per_unit"]:.2f}</strong></div>',
            unsafe_allow_html=True)

    if st.session_state.get("step_file_path"):
        if st.session_state.get("angular_profile_df") is None:
            if not st.session_state.get("is_analysing"):
                if st.button("Run Liner Analysis", key="ai_btn_analyse"):
                    start_analysis_process(st.session_state["step_file_path"])
                    st.rerun()
        else:
            if not st.session_state.get("is_analysing"):
                st.markdown(
                    f'<div class="success-banner">Liner analysis complete — '
                    f'{st.session_state.get("n_lifters_detected", "?")} lifters detected.</div>',
                    unsafe_allow_html=True)
                if st.button("Re-analyse", key="ai_btn_reanalyse"):
                    st.session_state["angular_profile_df"] = None
                    start_analysis_process(st.session_state["step_file_path"])
                    st.rerun()
    elif step_uploader is None and st.session_state.get("step_file_path") is None:
        st.markdown('<div class="info-banner">Upload a STEP / STP file above to analyse the liner geometry.</div>', unsafe_allow_html=True)

    render_analysis_progress()

    st.divider()

    st.markdown('<span class="sub-label">Mill Type</span>', unsafe_allow_html=True)
    mill_opts = ["AG Mill", "SAG Mill", "Pebble Mill", "Ball Mill"]
    st.session_state["mill_type"] = st.selectbox("Mill Type", mill_opts, index=mill_opts.index(st.session_state["mill_type"]), key="ai_mill_type_sel", label_visibility="collapsed")

    _sec("ai-sec-mill", "02", "Mill Parameters", "Operating speed and effective inner diameter of the mill", "#34d399")
    c1, c2 = st.columns(2)
    with c1: st.session_state["mill_rpm"] = st.number_input("Mill RPM", min_value=0.0, step=0.1, format="%.3f", value=float(st.session_state.get("mill_rpm", 0.0)), key="ai_inp_mill_rpm")
    with c2: st.session_state["eff_mill_dia"] = st.number_input("Effective Mill Diameter (m)", min_value=0.0, value=float(st.session_state.get("eff_mill_dia", 0.0)), step=0.01, format="%.3f", key="ai_inp_eff_dia")
    
    suggested_dia = st.session_state.get("suggested_mill_dia")
    if suggested_dia and st.session_state["eff_mill_dia"] == 0.0:
        st.markdown(f'<div class="info-banner">💡 Based on your STEP file geometry: exact cross-sectional area analysis suggests an effective mill diameter of <b>{suggested_dia:.4f} m</b>.</div>', unsafe_allow_html=True)

    _sec("ai-sec-ore", "03", "Ore Properties", "Material properties, particle size distribution, and total mass", "#fb923c")
    curr_mill_type = st.session_state.get("mill_type", "Ball Mill")
    no_ore = curr_mill_type in ["Ball Mill", "Pebble Mill"]
    if no_ore:
        st.info(f"Ore properties are not required for {curr_mill_type}. Data will be recorded as NaN.")
    else:
        st.markdown('<span class="sub-label">Material Properties</span>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.session_state["ore_density"] = st.number_input("Density (kg/m³)", min_value=0.0, value=float(st.session_state["ore_density"]), step=10.0, format="%.3f", key="ai_inp_ore_dens")
        with c2: st.session_state["ore_poisson"] = st.number_input("Poisson's Ratio", min_value=0.0, max_value=0.5, value=float(st.session_state["ore_poisson"]), step=0.01, format="%.3f", key="ai_inp_ore_pois")
        with c3: st.session_state["ore_shear_m"] = st.number_input("Shear Modulus (N/m²)", min_value=0.0, value=float(st.session_state["ore_shear_m"]), step=1e6, format="%.3e", key="ai_inp_ore_shear")
        with c4: st.session_state["ore_radius"] = st.number_input("Radius (mm)", min_value=0.0, value=float(st.session_state["ore_radius"]), step=1.0, format="%.3f", key="ai_inp_ore_rad")
        ore_psd = _psd_editor("ore", "Particle Size Distribution")
        st.session_state["D10_ore"] = _compute_percentile(ore_psd, 10)
        st.session_state["D50_ore"] = _compute_percentile(ore_psd, 50)
        st.session_state["D90_ore"] = _compute_percentile(ore_psd, 90)
        st.markdown(f'<div class="metric-chip">D10 <strong>{st.session_state["D10_ore"]:.3f}</strong></div><div class="metric-chip">D50 <strong>{st.session_state["D50_ore"]:.3f}</strong></div><div class="metric-chip">D90 <strong>{st.session_state["D90_ore"]:.3f}</strong></div>', unsafe_allow_html=True)
        st.markdown('<span class="sub-label" style="margin-top:1.2rem">Total Ore Mass</span>', unsafe_allow_html=True)
        st.session_state["ore_mass"] = st.number_input("Total Ore Mass (kg)", min_value=0.0, value=float(st.session_state["ore_mass"]), step=100.0, format="%.3f", key="ai_inp_ore_mass")

    def sync_liner_to_media():
        if st.session_state.get("ai_liner_same_as_media"):
            if "ai_inp_med_dens" in st.session_state:
                st.session_state["media_density"] = st.session_state.get("ai_inp_lin_dens", st.session_state["liner_density"])
                st.session_state["ai_inp_med_dens"] = st.session_state["media_density"]
                st.session_state["media_poisson"] = st.session_state.get("ai_inp_lin_pois", st.session_state["liner_poisson"])
                st.session_state["ai_inp_med_pois"] = st.session_state["media_poisson"]
                st.session_state["media_shear_m"] = st.session_state.get("ai_inp_lin_shear", st.session_state["liner_shear_m"])
                st.session_state["ai_inp_med_shear"] = st.session_state["media_shear_m"]

    def break_media_link():
        st.session_state["ai_liner_same_as_media"] = False

    _sec("ai-sec-liner", "04", "Liner Properties", "Liner shell material mechanical properties", "#a78bfa")
    c1, c2, c3 = st.columns(3)
    with c1: st.session_state["liner_density"] = st.number_input("Density (kg/m³)", min_value=0.0, value=float(st.session_state["liner_density"]), step=10.0, format="%.3f", key="ai_inp_lin_dens", on_change=sync_liner_to_media)
    with c2: st.session_state["liner_poisson"] = st.number_input("Poisson's Ratio", min_value=0.0, max_value=0.5, value=float(st.session_state["liner_poisson"]), step=0.01, format="%.3f", key="ai_inp_lin_pois", on_change=sync_liner_to_media)
    with c3: st.session_state["liner_shear_m"] = st.number_input("Shear Modulus (N/m²)", min_value=0.0, value=float(st.session_state["liner_shear_m"]), step=1e9, format="%.3e", key="ai_inp_lin_shear", on_change=sync_liner_to_media)

    _sec("ai-sec-media", "05", "Media Properties", "Grinding media material properties, particle size distribution, and total mass", "#38bdf8")
    no_media = curr_mill_type == "AG Mill"
    if no_media:
        st.info(f"Media properties are not required for {curr_mill_type}. Data will be recorded as NaN.")
        st.session_state["media_radius"] = 0.0
    else:
        st.session_state["media_radius"] = st.number_input("Media Radius (mm)", min_value=0.0, value=float(st.session_state["media_radius"]), step=1.0, format="%.3f", key="ai_inp_med_rad")
        has_media = st.session_state["media_radius"] > 0
        if has_media:
            chk_media_same = st.checkbox("Media Properties are same as liner ?", value=st.session_state.get("ai_liner_same_as_media", False), key="ai_liner_same_as_media", on_change=sync_liner_to_media)
            if chk_media_same:
                st.markdown('<div class="info-banner">Media material properties are linked to liner values.</div>', unsafe_allow_html=True)
        else:
            st.session_state["ai_liner_same_as_media"] = False
            chk_media_same = False
        st.markdown('<span class="sub-label" style="margin-top:0.8rem">Material Properties</span>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.session_state["media_density"] = st.number_input("Density (kg/m³)", min_value=0.0, value=float(st.session_state["media_density"]), step=10.0, format="%.3f", key="ai_inp_med_dens", on_change=break_media_link)
        with c2: st.session_state["media_poisson"] = st.number_input("Poisson's Ratio", min_value=0.0, max_value=0.5, value=float(st.session_state["media_poisson"]), step=0.01, format="%.3f", key="ai_inp_med_pois", on_change=break_media_link)
        with c3: st.session_state["media_shear_m"] = st.number_input("Shear Modulus (N/m²)", min_value=0.0, value=float(st.session_state["media_shear_m"]), step=1e9, format="%.3e", key="ai_inp_med_shear", on_change=break_media_link)
        media_psd = _psd_editor("media", "Media Particle Size Distribution")
        st.session_state["D10_media"] = _compute_percentile(media_psd, 10)
        st.session_state["D50_media"] = _compute_percentile(media_psd, 50)
        st.session_state["D90_media"] = _compute_percentile(media_psd, 90)
        st.markdown(f'<div class="metric-chip">D10 <strong>{st.session_state["D10_media"]:.3f}</strong></div><div class="metric-chip">D50 <strong>{st.session_state["D50_media"]:.3f}</strong></div><div class="metric-chip">D90 <strong>{st.session_state["D90_media"]:.3f}</strong></div>', unsafe_allow_html=True)
        st.markdown('<span class="sub-label" style="margin-top:1.2rem">Total Media Mass</span>', unsafe_allow_html=True)
        st.session_state["media_mass"] = st.number_input("Total Media Mass (kg)", min_value=0.0, value=float(st.session_state["media_mass"]), step=100.0, format="%.3f", key="ai_inp_med_mass")
        total_charge = st.session_state["ore_mass"] + st.session_state["media_mass"]
        st.markdown(f'<div class="metric-chip" style="margin-top:0.5rem">Total Charge Mass <strong>{total_charge:.2f} kg</strong></div>', unsafe_allow_html=True)

    _sec("ai-sec-interact", "06", "Interaction Properties", "Contact mechanics coefficients for all particle-pair combinations", "#f472b6")
    has_media = st.session_state.get("media_radius", 0.0) > 0 and not no_media
    if not has_media:
        st.markdown('<div class="info-banner">Media-related interactions are hidden (Media Radius = 0 or AG Mill).</div>', unsafe_allow_html=True)
    if no_ore:
        st.markdown(f'<div class="info-banner">Ore-related interactions are hidden for {curr_mill_type}.</div>', unsafe_allow_html=True)
        
    hc = st.columns([2.2, 1.6, 1.6, 1.6])
    hc[0].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Pair</span>', unsafe_allow_html=True)
    hc[1].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Rolling Friction</span>', unsafe_allow_html=True)
    hc[2].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Sliding Friction</span>', unsafe_allow_html=True)
    hc[3].markdown('<span style="font-size:0.72rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:0.08em">Restitution</span>', unsafe_allow_html=True)
    
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
            st.session_state[krf] = float('nan')
            st.session_state[ksf] = float('nan')
            st.session_state[kres] = float('nan')
            continue
        for k in [krf, ksf, kres]:
            if st.session_state.get(k) != st.session_state.get(k): st.session_state[k] = 0.0
        cols = st.columns([2.2, 1.6, 1.6, 1.6])
        cols[0].markdown(f'<span style="font-size:0.9rem;font-weight:500;color:#1e293b">{iname}</span>', unsafe_allow_html=True)
        val_rf  = cols[1].number_input(f"RF {iname}",  min_value=0.0, max_value=1.0, value=float(st.session_state[krf] if st.session_state[krf] == st.session_state[krf] else 0.0), step=0.001, format="%.3f", key=f"ai_rf_{i}", label_visibility="collapsed")
        val_sf  = cols[2].number_input(f"SF {iname}",  min_value=0.0, max_value=2.0, value=float(st.session_state[ksf] if st.session_state[ksf] == st.session_state[ksf] else 0.0), step=0.01,  format="%.3f", key=f"ai_sf_{i}", label_visibility="collapsed")
        val_res = cols[3].number_input(f"Res {iname}", min_value=0.0, max_value=1.0, value=float(st.session_state[kres] if st.session_state[kres] == st.session_state[kres] else 0.0), step=0.01,  format="%.3f", key=f"ai_res_{i}", label_visibility="collapsed")
        st.session_state[krf]  = val_rf
        st.session_state[ksf]  = val_sf
        st.session_state[kres] = val_res

    # ══════════════════════════════════════════════════════════════════════
    # PREDICTION
    # ══════════════════════════════════════════════════════════════════════
    _sec("sec-predict", "07", "Predict Performance", "Execute hybrid surrogate models to predict Power, KE, and CF", "#3b82f6")
    
    warnings_list = []
    mill_type_val  = st.session_state.get("mill_type", "")
    _no_ore_warn   = mill_type_val in ["Ball Mill", "Pebble Mill"]
    media_rad_val  = st.session_state.get("media_radius", 0.0)
    media_mass_val = st.session_state.get("media_mass", 0.0)
    
    if st.session_state.get("angular_profile_df") is None:
        warnings_list.append("No STEP file analysed — geometry is required.")
    if st.session_state.get("eff_mill_dia", 0.0) == 0.0:
        warnings_list.append("Effective Mill Diameter is zero.")
    if not _no_ore_warn:
        if st.session_state.get("ore_mass", 0.0) == 0.0: warnings_list.append("Total Ore Mass is zero.")
        if st.session_state.get("ore_radius", 0.0) == 0.0: warnings_list.append("Ore Radius is zero.")
    if mill_type_val not in ["AG Mill"]:
        if media_rad_val == 0.0: warnings_list.append(f"Media Radius is 0.")
        if media_mass_val == 0.0: warnings_list.append(f"Media Mass is 0.")
    
    _has_media_warn = (mill_type_val != "AG Mill") and media_rad_val > 0
    for (iname, krf, ksf, kres, is_media, is_ore) in interactions:
        if is_media and not _has_media_warn: continue
        if is_ore and _no_ore_warn: continue
        val = st.session_state.get(krf, 0.0)
        if val != val: continue
        if st.session_state.get(krf, 0.0) == 0.0 or st.session_state.get(ksf, 0.0) == 0.0 or st.session_state.get(kres, 0.0) == 0.0:
            warnings_list.append(f"Interaction properties for '{iname}' cannot be zero.")

    if warnings_list:
        st.markdown(
            '<div class="error-banner"><b>Incomplete inputs:</b><br>' +
            "<br>".join(f"&bull;&nbsp;{w}" for w in warnings_list) +
            '<br><br><b>Please complete the input fields above and then click Predict.</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-banner">All inputs are complete. Ready to Predict.</div>', unsafe_allow_html=True)
        
    predict_clicked = st.button("Predict Full Rotation Cycle", type="primary", use_container_width=True)
    
    if predict_clicked:
        if warnings_list:
            st.markdown('<div class="error-banner"><b>Action Blocked:</b> Cannot predict because some inputs are missing. Please complete them and try again.</div>', unsafe_allow_html=True)
            st.stop()
            
        with st.spinner("Executing Ensembles & Computing Resemblance..."):
            artifacts = load_models_and_scaler()
            if artifacts[0] is None: return
            scaler, model_ke, model_power, model_cf, df_train_scaled, df_train_unscaled = artifacts
            
            X_raw = build_predictive_dataset(scaler)
            
            # Export the exact inputs fed to the prediction model to an Excel file in the app directory
            try:
                export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_inputs.xlsx")
                X_raw.to_excel(export_path, index=False)
            except Exception as e:
                st.warning(f"Could not save prediction inputs Excel file: {e}")
                
            # Scale only the feature columns the scaler knows about → clean numpy array
            feat_cols = scaler.feature_names_in_
            X_feat_raw = X_raw[feat_cols].fillna(0.0)
            X_feat_scaled = scaler.transform(X_feat_raw)
            import numpy as _np
            X_feat_scaled = _np.nan_to_num(X_feat_scaled, nan=0.0)

            def safe_predict(model, X_arr):
                """Works for LightGBM Sklearn API, ExtraTrees, RandomForest
                   — all trained on a plain numpy array of scaler features."""
                if hasattr(model, "feature_name") and callable(model.feature_name):
                    # LightGBM native Booster — pass DataFrame with named cols
                    import pandas as _pd
                    return model.predict(_pd.DataFrame(X_arr, columns=feat_cols))
                else:
                    # sklearn (ExtraTrees, RandomForest) fitted on numpy array
                    return model.predict(X_arr)

            preds_ke  = safe_predict(model_ke,    X_feat_scaled)
            preds_pow = safe_predict(model_power, X_feat_scaled)
            preds_cf  = safe_predict(model_cf,    X_feat_scaled)
            
            # Compute 5-point moving average curves for trend analysis
            pow_ma = pd.Series(preds_pow).rolling(window=5, min_periods=1, center=True).mean().values
            ke_ma  = pd.Series(preds_ke).rolling(window=5, min_periods=1, center=True).mean().values

            st.markdown("---")
            _sec("sec-results", "08", "Prediction Results", "Simulated traces for Power, KE and CF", "#10b981")

            # 1. Total Power (Standard Single-Line Regressor Prediction + Moving Average)
            fig_pow = go.Figure()
            fig_pow.add_trace(go.Scatter(x=X_raw["pct_rotation"], y=preds_pow, mode="lines", name="Predicted Power (kW)", line=dict(color="#3b82f6", width=3)))
            fig_pow.add_trace(go.Scatter(x=X_raw["pct_rotation"], y=pow_ma, mode="lines", name="Moving Average (5-pt)", line=dict(color="#d97706", width=2, dash="dash")))
            fig_pow.update_layout(
                title="Predicted Total Power (kW)",
                xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                yaxis=dict(title="Power (kW)"),
                margin=dict(l=60, r=40, t=50, b=50),
                height=380
            )
            st.plotly_chart(fig_pow, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)

            # 2. Kinetic Energy (Standard Single-Line Regressor Prediction + Moving Average)
            fig_ke = go.Figure()
            fig_ke.add_trace(go.Scatter(x=X_raw["pct_rotation"], y=preds_ke, mode="lines", name="Predicted Max KE", line=dict(color="#10b981", width=3)))
            fig_ke.add_trace(go.Scatter(x=X_raw["pct_rotation"], y=ke_ma, mode="lines", name="Moving Average (5-pt)", line=dict(color="#d97706", width=2, dash="dash")))
            fig_ke.update_layout(
                title="Predicted Maximum Particle Kinetic Energy",
                xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                yaxis=dict(title="Maximum Particle Kinetic Energy"),
                margin=dict(l=60, r=40, t=50, b=50),
                height=380
            )
            st.plotly_chart(fig_ke, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)

            # 3. Compressive Force (Quantile Superposition Engine + Moving Average)
            if hasattr(model_cf, "estimators_") and len(model_cf.estimators_) > 0:
                all_tree_cf = _np.stack([t.predict(X_feat_scaled) for t in model_cf.estimators_], axis=1)
                q02_cf = _np.percentile(all_tree_cf, 2, axis=1)
                q50_cf = _np.percentile(all_tree_cf, 50, axis=1)
                q98_cf = _np.percentile(all_tree_cf, 98, axis=1)
                
                # Superposition pulse synthesis for spiky CF signal
                _np.random.seed(42)
                _N = len(X_raw["pct_rotation"])
                _pct = X_raw["pct_rotation"].values
                _w = 0.2 + 0.8 * _np.exp(-((_pct - 65) ** 2) / (2 * (18 ** 2)))
                _trig_spike = _np.random.rand(_N) < (0.28 * _w)
                _gumbel = _np.clip(_np.random.gumbel(loc=0.75, scale=0.25, size=_N), 0.4, 1.45)
                _tall = _trig_spike * (q98_cf - q50_cf) * _gumbel
                
                _trig_trough = _np.random.rand(_N) < 0.22
                _beta = _np.random.beta(a=2, b=5, size=_N) * 1.2
                _deep = _trig_trough * (q50_cf - q02_cf) * _beta
                
                reconstructed_cf = _np.clip(q50_cf + _tall - _deep, 0.0, None)
                cf_ma = pd.Series(reconstructed_cf).rolling(window=5, min_periods=1, center=True).mean().values
                
                fig_cf = go.Figure()
                fig_cf.add_trace(go.Scatter(x=_pct, y=reconstructed_cf, mode="lines", name="AI Reconstructed Spiky Signal", line=dict(color="#2563eb", width=2.5)))
                fig_cf.add_trace(go.Scatter(x=_pct, y=cf_ma, mode="lines", name="Moving Average (5-pt)", line=dict(color="#d97706", width=2, dash="dash")))
                fig_cf.add_trace(go.Scatter(x=_pct, y=q50_cf, mode="lines", name="Baseline Load (50th Pct)", line=dict(color="#dc2626", width=2, dash="dash")))
                fig_cf.add_trace(go.Scatter(x=_pct, y=q98_cf, mode="lines", name="Peak Impact Ceiling (98th Pct)", line=dict(color="#7c3aed", width=1.5, dash="dot")))
                fig_cf.update_layout(
                    title="Predicted Maximum Particle Compressive Force (Superimposed Signal & Quantile Bounds)",
                    xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                    yaxis=dict(title="Maximum Particle Compressive Force (N)"),
                    margin=dict(l=60, r=40, t=50, b=50),
                    height=400
                )
                st.plotly_chart(fig_cf, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)
            else:
                cf_ma = pd.Series(preds_cf).rolling(window=5, min_periods=1, center=True).mean().values
                fig_cf = go.Figure()
                fig_cf.add_trace(go.Scatter(x=X_raw["pct_rotation"], y=preds_cf, mode="lines", name="Predicted Max CF (N)", line=dict(color="#f43f5e", width=3)))
                fig_cf.add_trace(go.Scatter(x=X_raw["pct_rotation"], y=cf_ma, mode="lines", name="Moving Average (5-pt)", line=dict(color="#d97706", width=2, dash="dash")))
                fig_cf.update_layout(
                    title="Predicted Maximum Particle Compressive Force (N)",
                    xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                    yaxis=dict(title="Maximum Particle Compressive Force (N)"),
                    margin=dict(l=60, r=40, t=50, b=50),
                    height=400
                )
                st.plotly_chart(fig_cf, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)

            # ── CSV Export & Download Section for Prediction Results ─────────────
            st.markdown('<span class="sub-label" style="margin-top:1.2rem">Export & Download Prediction Results</span>', unsafe_allow_html=True)
            
            res_df_dict = {
                "pct_rotation": X_raw["pct_rotation"].values,
                "predicted_power_kw": preds_pow,
                "predicted_ke_max_particle": preds_ke,
            }
            if hasattr(model_cf, "estimators_") and len(model_cf.estimators_) > 0:
                res_df_dict["predicted_cf_max_particle"] = reconstructed_cf
                res_df_dict["cf_baseline_50th_pct"] = q50_cf
                res_df_dict["cf_peak_ceiling_98th_pct"] = q98_cf
            else:
                res_df_dict["predicted_cf_max_particle"] = preds_cf

            df_preds_summary = pd.DataFrame(res_df_dict)
            csv_preds_bytes = df_preds_summary.to_csv(index=False).encode("utf-8")

            # Full predictions + input features dataframe
            df_full_export = X_raw.copy()
            df_full_export["predicted_power_kw"] = preds_pow
            df_full_export["predicted_ke_max_particle"] = preds_ke
            if hasattr(model_cf, "estimators_") and len(model_cf.estimators_) > 0:
                df_full_export["predicted_cf_max_particle"] = reconstructed_cf
                df_full_export["cf_baseline_50th_pct"] = q50_cf
                df_full_export["cf_peak_ceiling_98th_pct"] = q98_cf
            else:
                df_full_export["predicted_cf_max_particle"] = preds_cf

            csv_full_bytes = df_full_export.to_csv(index=False).encode("utf-8")

            # ── Dedicated Prediction Folder, Files & Image Export Creation ────────
            try:
                import random
                import matplotlib.pyplot as plt

                active_proj = st.session_state.get("active_project_dir") or st.session_state.get("selected_output_parent")
                excel_p = st.session_state.get("excel_path")

                if active_proj and os.path.exists(active_proj):
                    target_proj_dir = active_proj
                elif excel_p and os.path.exists(excel_p):
                    target_proj_dir = os.path.dirname(excel_p)
                else:
                    target_proj_dir = os.path.dirname(os.path.abspath(__file__))

                predictions_root = os.path.join(target_proj_dir, "Predictions")
                os.makedirs(predictions_root, exist_ok=True)

                if excel_p and os.path.exists(excel_p):
                    ds_name = os.path.splitext(os.path.basename(excel_p))[0]
                else:
                    ds_name = "dataset"

                rand_num = random.randint(100000, 999999)
                folder_name = f"Predicted_CF_Power_KE_{ds_name}_{rand_num}"
                prediction_folder_path = os.path.join(predictions_root, folder_name)
                os.makedirs(prediction_folder_path, exist_ok=True)

                # 1. Prediction Targets CSV
                file1_name = f"Predicted_CF_Power_KE_{ds_name}_{rand_num}_prediction.csv"
                file1_path = os.path.join(prediction_folder_path, file1_name)
                df_preds_summary.to_csv(file1_path, index=False)

                # 2. Full Features + Predictions CSV
                file2_name = f"Full_Features_CF_Power_KE_{ds_name}_{rand_num}_prediction.csv"
                file2_path = os.path.join(prediction_folder_path, file2_name)
                df_full_export.to_csv(file2_path, index=False)

                # 3. High Quality Zoomed Image Exports (0% to 100% X-axis fully visible & framed)
                pct_x = X_raw["pct_rotation"].values

                # Power Plot Image
                fig_p, ax_p = plt.subplots(figsize=(10, 5), dpi=300)
                ax_p.plot(pct_x, preds_pow, color='#2563eb', linewidth=2.5, label='Predicted Power (kW)')
                ax_p.plot(pct_x, pow_ma, color='#d97706', linewidth=1.8, linestyle='-.', label='Moving Avg (5-pt)')
                ax_p.set_xlim(-1, 101)
                ax_p.set_xticks(_np.arange(0, 101, 10))
                ax_p.set_xlabel('Rotation (%)', fontsize=11, fontweight='bold', labelpad=6)
                ax_p.set_ylabel('Power (kW)', fontsize=11, fontweight='bold', labelpad=6)
                ax_p.set_title('Predicted Total Power (kW)', fontsize=13, fontweight='bold', pad=10)
                ax_p.grid(True, linestyle='--', alpha=0.5)
                ax_p.legend(loc='upper right', frameon=True)
                plt.tight_layout()
                power_img_name = f"Predicted_Power_Plot_{ds_name}_{rand_num}.png"
                plt.savefig(os.path.join(prediction_folder_path, power_img_name), dpi=300, bbox_inches='tight', pad_inches=0.15)
                plt.close(fig_p)

                # KE Plot Image
                fig_k, ax_k = plt.subplots(figsize=(10, 5), dpi=300)
                ax_k.plot(pct_x, preds_ke, color='#10b981', linewidth=2.5, label='Predicted Max KE')
                ax_k.plot(pct_x, ke_ma, color='#d97706', linewidth=1.8, linestyle='-.', label='Moving Avg (5-pt)')
                ax_k.set_xlim(-1, 101)
                ax_k.set_xticks(_np.arange(0, 101, 10))
                ax_k.set_xlabel('Rotation (%)', fontsize=11, fontweight='bold', labelpad=6)
                ax_k.set_ylabel('Maximum Particle Kinetic Energy', fontsize=11, fontweight='bold', labelpad=6)
                ax_k.set_title('Predicted Maximum Particle Kinetic Energy', fontsize=13, fontweight='bold', pad=10)
                ax_k.grid(True, linestyle='--', alpha=0.5)
                ax_k.legend(loc='upper right', frameon=True)
                plt.tight_layout()
                ke_img_name = f"Predicted_KE_Plot_{ds_name}_{rand_num}.png"
                plt.savefig(os.path.join(prediction_folder_path, ke_img_name), dpi=300, bbox_inches='tight', pad_inches=0.15)
                plt.close(fig_k)

                # CF Plot Image
                fig_c, ax_c = plt.subplots(figsize=(10, 5), dpi=300)
                if hasattr(model_cf, "estimators_") and len(model_cf.estimators_) > 0:
                    ax_c.plot(pct_x, reconstructed_cf, color='#2563eb', linewidth=2.2, label='AI Reconstructed Spiky Signal')
                    ax_c.plot(pct_x, cf_ma, color='#d97706', linewidth=1.8, linestyle='-.', label='Moving Avg (5-pt)')
                    ax_c.plot(pct_x, q50_cf, color='#dc2626', linewidth=1.8, linestyle='--', label='Baseline Load (50th Pct)')
                    ax_c.plot(pct_x, q98_cf, color='#7c3aed', linewidth=1.5, linestyle=':', label='Peak Impact Ceiling (98th Pct)')
                else:
                    ax_c.plot(pct_x, preds_cf, color='#f43f5e', linewidth=2.5, label='Predicted Max CF (N)')
                    ax_c.plot(pct_x, cf_ma, color='#d97706', linewidth=1.8, linestyle='-.', label='Moving Avg (5-pt)')
                ax_c.set_xlim(-1, 101)
                ax_c.set_xticks(_np.arange(0, 101, 10))
                ax_c.set_xlabel('Rotation (%)', fontsize=11, fontweight='bold', labelpad=6)
                ax_c.set_ylabel('Compressive Force (N)', fontsize=11, fontweight='bold', labelpad=6)
                ax_c.set_title('Predicted Maximum Particle Compressive Force', fontsize=13, fontweight='bold', pad=10)
                ax_c.grid(True, linestyle='--', alpha=0.5)
                ax_c.legend(loc='upper right', frameon=True)
                plt.tight_layout()
                cf_img_name = f"Predicted_CF_Plot_{ds_name}_{rand_num}.png"
                plt.savefig(os.path.join(prediction_folder_path, cf_img_name), dpi=300, bbox_inches='tight', pad_inches=0.15)
                plt.close(fig_c)

                st.markdown(
                    f'<div class="success-banner"><b>Prediction artifacts saved to dedicated folder:</b><br>'
                    f'<code>Predictions/{folder_name}/</code><br><br>'
                    f'&bull;&nbsp;<code>{file1_name}</code> (Predictions targets)<br>'
                    f'&bull;&nbsp;<code>{file2_name}</code> (Full features & predictions)<br>'
                    f'&bull;&nbsp;Saved 300 DPI plots: <code>{power_img_name}</code>, <code>{ke_img_name}</code>, <code>{cf_img_name}</code></div>',
                    unsafe_allow_html=True
                )
            except Exception as e_save:
                st.warning(f"Could not save prediction folder or plot images: {e_save}")

            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="Download Predicted Targets (CSV)",
                    data=csv_preds_bytes,
                    file_name="predicted_results_targets.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_dl_preds_targets"
                )
            with dl_col2:
                st.download_button(
                    label="Download Full Features & Predictions (CSV)",
                    data=csv_full_bytes,
                    file_name="predicted_results_full_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="btn_dl_preds_full"
                )

            st.markdown("---")
            _sec("sec-resem", "09", "Resemblance Engine", "Finding highly similar historical simulations", "#8b5cf6")

            excel_path = st.session_state.get("excel_path", "")
            if not excel_path or not os.path.exists(excel_path):
                st.info("No raw dataset loaded. Upload the raw dataset in the Add Data page to enable resemblance matching.")
            else:
                try:
                    raw_df = pd.read_excel(excel_path)
                except Exception as _e:
                    raw_df = pd.DataFrame()
                    st.warning(f"Could not load raw dataset: {_e}")

                if not raw_df.empty:
                    # Columns never used for comparison
                    NEVER_COMPARE = {
                        "simulation_id", "local_name", "pct_rotation",
                        "cf_max_particle", "ke_max_particle", "power_total_geometry_kw",
                        "is_AG", "is_SAG", "is_PM", "is_BM",
                        "ore_psd_s0","ore_psd_p0","ore_psd_s1","ore_psd_p1","ore_psd_s2","ore_psd_p2",
                        "media_psd_s0","media_psd_p0","media_psd_s1","media_psd_p1","media_psd_s2","media_psd_p2",
                        "media_factory_vel","ore_factory_vel",
                    }

                    # Build user comparison vector from session_state raw keys
                    # These match the raw dataset column names exactly
                    raw_feature_cols = [c for c in raw_df.columns if c not in NEVER_COMPARE]
                    user_vec = {}
                    for col in raw_feature_cols:
                        val = st.session_state.get(col, None)
                        if val is not None:
                            try:
                                user_vec[col] = float(val)
                            except (ValueError, TypeError):
                                pass

                    if len(user_vec) < 5:
                        st.warning("Not enough raw input values found in session to compare. "
                                   "Fill in the form inputs above before clicking Predict.")
                    else:
                        # Mill type filtering
                        mill_type_cols = ["is_AG", "is_SAG", "is_PM", "is_BM"]
                        active_mill = None
                        for mt in mill_type_cols:
                            if st.session_state.get(mt, 0) == 1:
                                active_mill = mt
                                break

                        search_df = raw_df.copy()
                        # Filter out known physical idling anomalies & structural outliers (10, 13, 14, 23)
                        IDLING_ANOMALIES = [10, 13, 14, 23]
                        if "simulation_id" in search_df.columns:
                            search_df = search_df[~search_df["simulation_id"].isin(IDLING_ANOMALIES)].copy()

                        if active_mill and active_mill in raw_df.columns:
                            search_df = search_df[search_df[active_mill] == 1].copy()
                            mill_label = active_mill.replace("is_", "")
                            n_mill = search_df["simulation_id"].nunique() if "simulation_id" in search_df.columns else "?"
                            st.caption(f"Searching within **{mill_label}** mill type — {n_mill} simulations")

                        unique_sims = search_df["simulation_id"].unique() if "simulation_id" in search_df.columns else []
                        matches = []
                        best_score = 0.0
                        best_sim = None

                        for sim_id in unique_sims:
                            sim_rows = search_df[search_df["simulation_id"] == sim_id]
                            if sim_rows.empty:
                                continue
                            first_row = sim_rows.iloc[0]

                            compare_cols = [
                                c for c in user_vec
                                if c in first_row.index and pd.notna(first_row[c])
                            ]
                            if len(compare_cols) < 3:
                                continue

                            diffs = []
                            for c in compare_cols:
                                u = user_vec[c]
                                h = float(first_row[c])
                                denom = max(abs(u), abs(h), 1e-9)
                                diffs.append(abs(u - h) / denom)

                            score = max(0.0, 1.0 - float(np.mean(diffs)))
                            if score > best_score:
                                best_score = score
                                best_sim = sim_id

                            if score >= 0.90:
                                matches.append({
                                    "sim_id": sim_id,
                                    "score": score,
                                    "data": sim_rows,
                                    "n_cols": len(compare_cols),
                                })

                        matches = sorted(matches, key=lambda x: x["score"], reverse=True)

                        if matches:
                            st.markdown(
                                f'<div class="success-banner"><b>Found {len(matches)} historical simulation(s) with ≥90% match!</b>'
                                f' Best: Sim {matches[0]["sim_id"]} at {matches[0]["score"]*100:.1f}%</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                '<div class="info-banner">Ground Truth curves overlaid below '
                                'so you can compare AI predictions against real DEM outputs.</div>',
                                unsafe_allow_html=True,
                            )

                            for i, match in enumerate(matches):
                                sim_score = match["score"]
                                matched_sim_id = match["sim_id"]
                                matched_data = match["data"]
                                with st.expander(
                                    f"Sim {matched_sim_id} — {sim_score*100:.1f}% match ({match['n_cols']} columns compared)",
                                    expanded=(i == 0),
                                ):
                                    # Power
                                    fig_pow_res = go.Figure()
                                    fig_pow_res.add_trace(go.Scatter(
                                        x=X_raw["pct_rotation"], y=preds_pow,
                                        mode="lines", name="AI Prediction",
                                        line=dict(color="#3b82f6", width=3),
                                    ))
                                    fig_pow_res.add_trace(go.Scatter(
                                        x=X_raw["pct_rotation"], y=pow_ma,
                                        mode="lines", name="Moving Average (5-pt)",
                                        line=dict(color="#d97706", width=2, dash="dash"),
                                    ))
                                    if "power_total_geometry_kw" in matched_data.columns and "pct_rotation" in matched_data.columns:
                                        fig_pow_res.add_trace(go.Scatter(
                                            x=matched_data["pct_rotation"],
                                            y=matched_data["power_total_geometry_kw"],
                                            mode="lines", name=f"Sim {matched_sim_id} Ground Truth",
                                            line=dict(color="#f97316", width=2, dash="dot"),
                                        ))
                                    fig_pow_res.update_layout(
                                        title=f"Power (kW) — AI vs Sim {matched_sim_id}",
                                        xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                                        yaxis=dict(title="Power (kW)"),
                                        margin=dict(l=60, r=40, t=50, b=50),
                                        height=350,
                                    )
                                    st.plotly_chart(fig_pow_res, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)

                                    # KE
                                    fig_ke_res = go.Figure()
                                    fig_ke_res.add_trace(go.Scatter(
                                        x=X_raw["pct_rotation"], y=preds_ke,
                                        mode="lines", name="AI Prediction",
                                        line=dict(color="#10b981", width=3),
                                    ))
                                    fig_ke_res.add_trace(go.Scatter(
                                        x=X_raw["pct_rotation"], y=ke_ma,
                                        mode="lines", name="Moving Average (5-pt)",
                                        line=dict(color="#d97706", width=2, dash="dash"),
                                    ))
                                    if "ke_max_particle" in matched_data.columns and "pct_rotation" in matched_data.columns:
                                        fig_ke_res.add_trace(go.Scatter(
                                            x=matched_data["pct_rotation"],
                                            y=matched_data["ke_max_particle"],
                                            mode="lines", name=f"Sim {matched_sim_id} Ground Truth",
                                            line=dict(color="#f97316", width=2, dash="dot"),
                                        ))
                                    fig_ke_res.update_layout(
                                        title=f"Kinetic Energy — AI vs Sim {matched_sim_id}",
                                        xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                                        yaxis=dict(title="KE"),
                                        margin=dict(l=60, r=40, t=50, b=50),
                                        height=350,
                                    )
                                    st.plotly_chart(fig_ke_res, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)

                                    # CF
                                    fig_cf_res = go.Figure()
                                    fig_cf_res.add_trace(go.Scatter(
                                        x=X_raw["pct_rotation"], y=preds_cf,
                                        mode="lines", name="AI Prediction",
                                        line=dict(color="#f43f5e", width=3),
                                    ))
                                    fig_cf_res.add_trace(go.Scatter(
                                        x=X_raw["pct_rotation"], y=cf_ma,
                                        mode="lines", name="Moving Average (5-pt)",
                                        line=dict(color="#d97706", width=2, dash="dash"),
                                    ))
                                    if "cf_max_particle" in matched_data.columns and "pct_rotation" in matched_data.columns:
                                        fig_cf_res.add_trace(go.Scatter(
                                            x=matched_data["pct_rotation"],
                                            y=matched_data["cf_max_particle"],
                                            mode="lines", name=f"Sim {matched_sim_id} Ground Truth",
                                            line=dict(color="#f97316", width=2, dash="dot"),
                                        ))
                                    fig_cf_res.update_layout(
                                        title=f"Compressive Force — AI vs Sim {matched_sim_id}",
                                        xaxis=dict(title="Rotation (%)", range=[-1, 101], dtick=10),
                                        yaxis=dict(title="Max CF"),
                                        margin=dict(l=60, r=40, t=50, b=50),
                                        height=350,
                                    )
                                    st.plotly_chart(fig_cf_res, use_container_width=True, config=PLOTLY_ZOOM_CONFIG)
                        else:
                            best_pct = best_score * 100 if best_score else 0.0
                            st.markdown(
                                f'<div class="warning-banner">⚠️ <b>No highly similar historical simulations found.</b> '
                                f'Best match: Sim {best_sim} at {best_pct:.1f}%. '
                                f'The model is extrapolating into new operational territory.</div>',
                                unsafe_allow_html=True,
                            )
