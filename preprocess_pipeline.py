"""
=============================================================================
DEM Surrogate Model — Professional Data Preprocessing Pipeline
=============================================================================
Author  : Preprocessing Script (auto-generated)
Purpose : Full data audit, EDA, feature engineering, NaN handling,
          outlier detection, scaling, and train/val/test split.
Outputs : All plots saved to ./preprocessing_report/
          Cleaned dataset saved to cleaned_dataset.csv
          Scaler saved to scaler.pkl
          Train/Val/Test splits saved to train.csv, val.csv, test.csv
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import pickle

warnings.filterwarnings("ignore")

import sys

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
if len(sys.argv) >= 3:
    INPUT_FILE = sys.argv[1]
    OUTPUT_DIR = sys.argv[2]
else:
    INPUT_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRE PROCESSED", "data_v1.xlsx")
    OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRE PROCESSED", "preprocessing_report")

TARGETS      = ["cf_max_particle", "ke_max_particle", "power_total_geometry_kw"]
ID_COL       = "simulation_id"
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# TEST_RATIO   = 0.15 (remainder)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  [SAVED] {name}")

def main():
    print("=" * 70)
    print("  DEM SURROGATE — PROFESSIONAL PREPROCESSING PIPELINE")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 1 : LOAD & AUDIT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 1] Loading & Auditing raw data ...")
    df = pd.read_excel(INPUT_FILE)

    # Drop legacy columns if they exist
    legacy_cols = ['media_factory_vel', 'ore_factory_vel', 'media_factory_velocity', 'ore_factory_velocity', 'local_name']
    existing_legacy = [c for c in legacy_cols if c in df.columns]
    if existing_legacy:
        df.drop(columns=existing_legacy, inplace=True)
        print(f"  [CLEANUP] Dropped legacy columns: {existing_legacy}")

    print(f"  Raw shape      : {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Simulations    : {df[ID_COL].nunique()} unique IDs")
    print(f"  Rows per sim   : {df.groupby(ID_COL).size().unique().tolist()}")

    # ── Exclude simulations flagged as anomalous / invalid ────────────────────
    # 1, 2 = Corrupt target anomalies (KE ~ 3.4, CF ~ 50-100 N on 8.2m mills drawing 450-528 kW power — 100x lower than real physics)
    # 10 = Isolation Forest structural anomaly
    # 13, 14 = Physical idling / zero-load anomalies (CF < 600 N, Power < 25 kW)
    # 23 = Extreme trailing angle outlier (65.5 deg)
    EXCLUDE_SIMS = [1, 2, 10, 13, 14, 23]
    if EXCLUDE_SIMS:
        before = len(df)
        df = df[~df[ID_COL].isin(EXCLUDE_SIMS)].copy()
        print(f"\n  [EXCLUDED] Simulations {EXCLUDE_SIMS} removed — {before - len(df)} rows dropped.")
        print(f"  Remaining: {df[ID_COL].nunique()} simulations, {len(df)} rows.")

    # ── Fix incorrect media density (78500 -> 7850) for sims 1,16,17,36 ──────────
    # NOTE: Sim 15 was already corrected directly in data_v1.xlsx (source fixed)
    FIX_MEDIA_DENSITY_SIMS = [1, 16, 17, 36]
    wrong_mask = (df[ID_COL].isin(FIX_MEDIA_DENSITY_SIMS)) & (df['media_density'] == 78500)
    n_fixed = wrong_mask.sum()
    if n_fixed > 0:
        df.loc[wrong_mask, 'media_density'] = 7850.0
        print(f"\n  [FIXED] media_density corrected from 78500 -> 7850 for sims {FIX_MEDIA_DENSITY_SIMS} ({n_fixed} rows).")
    else:
        print(f"\n  [OK] media_density already correct for sims {FIX_MEDIA_DENSITY_SIMS}.")

    # ── Force-assign specific simulations to training regardless of stratified split ─
    # Sims 3, 4, 5, 24 have unique geometries/angles far outside normal distribution
    # that must be in training to prevent severe extrapolation errors.
    FORCE_TRAIN_SIMS = [3, 4, 5, 24]

    # ── 1a. Per-simulation row count bar chart ────────────────────────────────
    sim_counts = df.groupby(ID_COL).size()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(sim_counts.index, sim_counts.values, color="#3b82f6", edgecolor="white")
    ax.axhline(100, color="red", linestyle="--", linewidth=1, label="Expected = 100")
    ax.set_xlabel("Simulation ID"); ax.set_ylabel("Row Count")
    ax.set_title("Row Count per Simulation (Expected = 100)")
    ax.legend(); plt.tight_layout()
    save_fig("1a_sim_row_counts.png")

    # ── 1b. NaN audit heatmap ────────────────────────────────────────────────
    print("\n[PHASE 1b] NaN audit per column ...")
    nan_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    nan_pct = nan_pct[nan_pct > 0]
    print(nan_pct.to_string())

    fig, ax = plt.subplots(figsize=(10, max(4, len(nan_pct) * 0.35)))
    colors = ["#ef4444" if v > 50 else "#f97316" if v > 20 else "#eab308" for v in nan_pct.values]
    ax.barh(nan_pct.index[::-1], nan_pct.values[::-1], color=colors[::-1])
    ax.set_xlabel("% Missing"); ax.set_title("NaN Percentage per Column")
    plt.tight_layout(); save_fig("1b_nan_audit.png")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 2 : SMART NaN HANDLING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 2] Smart NaN Handling ...")

    # Ore columns: NaN because Ball Mills have no ore — fill with 0 (physically correct)
    ore_cols = [c for c in df.columns if c.startswith("ore_") or c in ["D10_ore","D50_ore","D90_ore"]]
    ore_interaction_cols = ["mo_rf","oo_rf","ol_rf","mo_sf","oo_sf","ol_sf","mo_res","oo_res","ol_res"]
    for col in ore_cols + ore_interaction_cols:
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        if filled > 0:
            print(f"  [NaN->0] {col}: {filled} values filled (Ball Mill / no ore — physically 0)")

    # short_leading_face_angle / short_trailing_face_angle: NaN = single lifter pattern
    # Fill with 0 (no short lifter exists)
    for col in ["short_leading_face_angle", "short_trailing_face_angle"]:
        filled = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        print(f"  [NaN->0] {col}: {filled} values filled (single lifter pattern — 0)")

    # trailing_face_angle: Only 1 simulation (sim 38) has NaN — inspect it
    sim38_trail = df[df[ID_COL] == 38]["trailing_face_angle"]
    print(f"\n  Simulation 38 trailing_face_angle: all NaN = {sim38_trail.isna().all()}")
    # Fill with median of other simulations (safe fallback for just 1 simulation)
    median_trail = df[df["trailing_face_angle"].notna()]["trailing_face_angle"].median()
    df["trailing_face_angle"] = df["trailing_face_angle"].fillna(median_trail)
    print(f"  [NaN->median={median_trail:.2f}] trailing_face_angle: sim 38 filled with dataset median")

    print(f"\n  Remaining NaN count: {df.isnull().sum().sum()}")
    assert df.isnull().sum().sum() == 0, "ERROR: Unexpected NaN values remain!"
    print("  [OK] All NaN values resolved.")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 3 : PHYSICAL VALIDITY CHECK
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 3] Physical validity checks ...")

    checks = {
        "mill_rpm"                  : (0, 1000),
        "eff_mill_dia"              : (0.5, 15),
        "ore_density"               : (0, 10000),
        "leading_face_angle"        : (0, 90),
        "trailing_face_angle"       : (0, 90),
        "cf_max_particle"           : (0, None),
        "ke_max_particle"           : (0, None),
        "power_total_geometry_kw"   : (0, None),
    }
    invalid_flags = []
    for col, (lo, hi) in checks.items():
        if lo is not None:
            mask = df[col] < lo
            if mask.any():
                print(f"  [WARN]  {col}: {mask.sum()} rows < {lo}")
                invalid_flags.append(col)
        if hi is not None:
            mask = df[col] > hi
            if mask.any():
                print(f"  [WARN]  {col}: {mask.sum()} rows > {hi}")
                invalid_flags.append(col)

    if not invalid_flags:
        print("  [OK] All physical validity checks passed.")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 4 : FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 4] Feature engineering ...")

    # 4a. Critical Speed Fraction (the most important physics feature for tumbling mills)
    # Formula: Nc = 42.3 / sqrt(D)  — where D is mill diameter in meters
    df["critical_speed_fraction"] = df["mill_rpm"] / (42.3 / np.sqrt(df["eff_mill_dia"]))
    print("  [NEW] critical_speed_fraction = RPM / (42.3 / sqrt(eff_mill_dia))")

    # 4b. Has short lifter (derived flag from short angles)
    df["has_short_lifter"] = ((df["short_leading_face_angle"] > 0) | (df["short_trailing_face_angle"] > 0)).astype(int)
    print("  [NEW] has_short_lifter (binary flag)")

    # 4c. Face angle asymmetry
    df["face_angle_asymmetry"] = df["leading_face_angle"] - df["trailing_face_angle"]
    print("  [NEW] face_angle_asymmetry = leading - trailing angle")

    # 4d. shape_k total energy (L2 norm of all harmonics = total shape energy)
    shape_k_cols = [f"shape_k{i}" for i in range(50)]
    df["shape_energy"] = np.sqrt((df[shape_k_cols] ** 2).sum(axis=1))
    print("  [NEW] shape_energy = L2 norm of shape_k0..49")

    # 4e. High-frequency sharpness (mean of k30..k49)
    hf_cols = [f"shape_k{i}" for i in range(30, 50)]
    df["shape_hf_sharpness"] = df[hf_cols].mean(axis=1)
    print("  [NEW] shape_hf_sharpness = mean(shape_k30..49)")

    # ── NEW PHYSICS FEATURES (Fix for failing sims 3,4,15,27,30,33,38) ────────────

    # 4f. Shell Tip Speed (m/s) — absolute impact velocity at shell surface
    # Tip speed = pi * D * RPM / 60. Key discriminator between similar-RPM but different-diameter mills.
    df["tip_speed"] = (np.pi * df["eff_mill_dia"] * df["mill_rpm"]) / 60.0
    print("  [NEW] tip_speed = pi * eff_mill_dia * mill_rpm / 60  (m/s)")

    # 4g. Lifter Circumferential Density (lifters per meter of shell circumference)
    # Discriminates 96-lifter large mills (sim 3) from 48-lifter mills of similar diameter.
    df["lifter_density"] = df["n_total_lifters"] / (np.pi * df["eff_mill_dia"])
    print("  [NEW] lifter_density = n_total_lifters / (pi * eff_mill_dia)  (lifters/m)")

    # 4h. Media Load Fraction — normalised grinding media mass by mill volume
    # Captures how full the mill is relative to its size (key for power draw & KE).
    # Using mill_length if available, otherwise approximate from diameter (D/L ~ 1 for BM)
    if "mill_length" in df.columns:
        mill_vol = np.pi * (df["eff_mill_dia"] / 2) ** 2 * df["mill_length"]
    else:
        # Approximate: Ball Mills typically have L/D ~ 1.0-1.5; use L = D as conservative estimate
        mill_vol = np.pi * (df["eff_mill_dia"] / 2) ** 2 * df["eff_mill_dia"]
    df["media_load_fraction"] = df["media_mass"] / (mill_vol * df["media_density"] + 1e-6)
    print("  [NEW] media_load_fraction = media_mass / (mill_vol * media_density)")

    # 4i. Ore presence flag (explicit binary — helps SAG mill KE differentiation)
    df["has_ore"] = (df["ore_mass"] > 0).astype(int)
    print("  [NEW] has_ore = (ore_mass > 0) binary flag")

    # ── DIMENSIONLESS & INTERACTION PHYSICS FEATURES ─────────────────────────────

    # 4j. Froude Number (fundamental dimensionless number governing cataract vs cascade charge motion)
    # Fr = (omega^2 * R) / g = ((RPM * pi / 30)^2 * (eff_mill_dia / 2)) / 9.81
    omega = (df["mill_rpm"] * np.pi) / 30.0
    radius = df["eff_mill_dia"] / 2.0
    df["froude_number"] = (omega ** 2 * radius) / 9.81
    print("  [NEW] froude_number = (omega^2 * R) / 9.81")

    # 4k. Charge Kinetic Head (dynamic pressure of tumbling charge acting on shell & lifters)
    # q_head = 0.5 * rho_charge_mix * tip_speed^2
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
    print("  [NEW] charge_kinetic_head = 0.5 * rho_charge_mix * tip_speed^2  (Pa)")

    # 4l. Lifter Strike Frequency (total lifter impacts per second)
    # f_strike = n_total_lifters * (mill_rpm / 60)
    df["lifter_strike_freq"] = df["n_total_lifters"] * (df["mill_rpm"] / 60.0)
    print("  [NEW] lifter_strike_freq = n_total_lifters * (mill_rpm / 60)")

    # 4m. Power Flux Proxy (energy transfer rate into charge — strongly correlates with Total Power)
    # P_proxy = tip_speed^2 * media_mass * mill_rpm
    df["power_flux_proxy"] = (df["tip_speed"] ** 2) * df["media_mass"] * df["mill_rpm"]
    print("  [NEW] power_flux_proxy = tip_speed^2 * media_mass * mill_rpm")

    # 4n. Specific Impact Energy (impact intensity per unit diameter)
    # E_spec = tip_speed^2 / eff_mill_dia
    df["specific_impact_energy"] = (df["tip_speed"] ** 2) / (df["eff_mill_dia"].replace(0, 1e-9))
    print("  [NEW] specific_impact_energy = tip_speed^2 / eff_mill_dia")

    # 4o. Cyclical Rotation Encoding (smooth periodic wave instead of linear step index)
    # Rot_sin = sin(2*pi * pct_rotation/100), Rot_cos = cos(2*pi * pct_rotation/100)
    df["rot_sin"] = np.sin(2.0 * np.pi * df["pct_rotation"] / 100.0)
    df["rot_cos"] = np.cos(2.0 * np.pi * df["pct_rotation"] / 100.0)
    print("  [NEW] rot_sin & rot_cos = sin/cos(2*pi * pct_rotation / 100)")

    # 4p. Media-to-Mill Radius Ratio (ball size relative to mill diameter)
    df["media_aspect_ratio"] = df["media_radius"] / (df["eff_mill_dia"].replace(0, 1e-9))
    print("  [NEW] media_aspect_ratio = media_radius / eff_mill_dia")

    # 4q. Total Charge Mass (combined mass of ore + grinding media)
    df["total_charge_mass"] = df["media_mass"].fillna(0) + df["ore_mass"].fillna(0)
    print("  [NEW] total_charge_mass = media_mass + ore_mass")

    # 4r. Media Ball Count Proxy (estimated total number of media balls in charge)
    # Discriminates dense small-media beds (e.g. 32mm balls in sim 24) vs larger ball beds.
    ball_vol = (4.0 / 3.0) * np.pi * (df["media_radius"].replace(0, 1e-9) ** 3)
    df["media_count_proxy"] = df["media_mass"] / (ball_vol * df["media_density"] + 1e-6)
    print("  [NEW] media_count_proxy = media_mass / (V_ball * media_density)")

    print(f"\n  Total features after engineering: {df.shape[1]} columns")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 5 : EXPLORATORY DATA ANALYSIS (EDA)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 5] Exploratory Data Analysis ...")

    # ── 5a. Target distributions ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, TARGETS):
        ax.hist(df[col], bins=40, color="#6366f1", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
        ax.axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean={df[col].mean():.1f}")
        ax.axvline(df[col].median(), color="orange", linestyle="--", linewidth=1.5, label=f"Median={df[col].median():.1f}")
        ax.legend(fontsize=8)
    fig.suptitle("Target Variable Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout(); save_fig("5a_target_distributions.png")

    # ── 5b. Target box plots by simulation ────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    for ax, col in zip(axes, TARGETS):
        sim_groups = [df[df[ID_COL] == sid][col].values for sid in sorted(df[ID_COL].unique())]
        ax.boxplot(sim_groups, patch_artist=True,
                   boxprops=dict(facecolor="#bfdbfe"),
                   medianprops=dict(color="#1d4ed8", linewidth=2))
        ax.set_xticklabels(sorted(df[ID_COL].unique()), fontsize=7, rotation=45)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("Simulation ID"); ax.set_ylabel("Value")
    fig.suptitle("Target Distribution per Simulation", fontsize=13, fontweight="bold")
    plt.tight_layout(); save_fig("5b_target_boxplots_per_sim.png")

    # ── 5c. Key feature distributions ─────────────────────────────────────────
    key_features = ["mill_rpm","eff_mill_dia","leading_face_angle","trailing_face_angle",
                    "critical_speed_fraction","shape_k0","shape_energy","n_total_lifters"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    axes = axes.flatten()
    for ax, col in zip(axes, key_features):
        ax.hist(df.drop_duplicates(ID_COL)[col], bins=20, color="#10b981", edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("Value"); ax.set_ylabel("Count")
    fig.suptitle("Key Feature Distributions (1 point per simulation)", fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig("5c_feature_distributions.png")

    # ── 5d. Correlation heatmap (key features + targets) ──────────────────────
    print("  Generating correlation heatmap ...")
    corr_cols = key_features + ["face_angle_asymmetry","shape_hf_sharpness",
                                "has_short_lifter"] + TARGETS
    corr_df = df[corr_cols].copy()
    corr = corr_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 7}, linewidths=0.5)
    ax.set_title("Correlation Matrix — Key Features & Targets", fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig("5d_correlation_heatmap.png")

    # ── 5e. Scatter plots: targets vs critical features ─────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    x_vars = ["critical_speed_fraction", "leading_face_angle", "shape_k0"]
    for row_i, target in enumerate(TARGETS):
        for col_i, xvar in enumerate(x_vars):
            ax = axes[row_i][col_i]
            ax.scatter(df[xvar], df[target], alpha=0.15, s=8, color="#6366f1")
            ax.set_xlabel(xvar, fontsize=8)
            ax.set_ylabel(target, fontsize=8)
            ax.set_title(f"{xvar} vs {target}", fontsize=8)
    fig.suptitle("Scatter: Key Features vs Targets", fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig("5e_scatter_features_vs_targets.png")

    # ── 5f. shape_k magnitude spectrum ──────────────────────────────────────
    mean_k = df[shape_k_cols].mean()
    std_k  = df[shape_k_cols].std()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(range(50), mean_k.values, color="#3b82f6", alpha=0.7, label="Mean")
    ax.errorbar(range(50), mean_k.values, yerr=std_k.values, fmt='none',
                ecolor='#ef4444', elinewidth=0.8, capsize=2, label="±1 std")
    ax.set_xlabel("Harmonic Index (k)"); ax.set_ylabel("Magnitude")
    ax.set_title("shape_k Spectrum (Mean ± 1 Std) — across all simulations")
    ax.legend(); plt.tight_layout(); save_fig("5f_shape_k_spectrum.png")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 6 : OUTLIER DETECTION
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 6] Outlier Detection ...")

    # ── 6a. IQR outlier flags on targets ────────────────────────────────────
    print("  IQR-based outlier check on targets ...")
    outlier_rows = pd.Series(False, index=df.index)
    for col in TARGETS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR  # Using 3xIQR (strict) for engineering data
        flagged = (df[col] < lo) | (df[col] > hi)
        print(f"  {col}: {flagged.sum()} rows flagged (bounds: [{lo:.2f}, {hi:.2f}])")
        outlier_rows |= flagged

    # ── 6b. Box plot of outliers ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, TARGETS):
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
        ax.boxplot(df[col], vert=True, patch_artist=True,
                   boxprops=dict(facecolor="#bfdbfe"),
                   medianprops=dict(color="#1d4ed8", linewidth=2),
                   flierprops=dict(marker='o', color='red', markersize=4))
        ax.axhline(hi, color='red', linestyle='--', linewidth=1, label=f'Upper fence={hi:.1f}')
        ax.axhline(lo, color='orange', linestyle='--', linewidth=1, label=f'Lower fence={lo:.1f}')
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Target Outlier Detection (3×IQR Fences)", fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig("6b_target_outlier_boxplots.png")

    # ── 6c. Isolation Forest on feature space ────────────────────────────────
    print("  Running Isolation Forest on key numeric features ...")
    feat_for_iso = ["mill_rpm","eff_mill_dia","leading_face_angle","trailing_face_angle",
                    "critical_speed_fraction","shape_k0","shape_k1","shape_k2","shape_energy",
                    "cf_max_particle","ke_max_particle","power_total_geometry_kw"]
    iso_data = df[feat_for_iso].fillna(0)
    clf = IsolationForest(contamination=0.03, random_state=RANDOM_SEED, n_estimators=200)
    iso_labels = clf.fit_predict(iso_data)
    df["iso_outlier"] = (iso_labels == -1).astype(int)
    iso_sims = df[df["iso_outlier"] == 1][ID_COL].value_counts()
    print(f"  Isolation Forest flagged {df['iso_outlier'].sum()} rows as anomalous.")
    print("  Flagged rows per simulation:")
    print(iso_sims.to_string())

    fig, ax = plt.subplots(figsize=(10, 4))
    normal = df[df["iso_outlier"] == 0]
    anomaly = df[df["iso_outlier"] == 1]
    ax.scatter(normal["critical_speed_fraction"], normal["power_total_geometry_kw"],
               alpha=0.15, s=8, color="#3b82f6", label="Normal")
    ax.scatter(anomaly["critical_speed_fraction"], anomaly["power_total_geometry_kw"],
               alpha=0.9, s=30, color="#ef4444", marker="x", label="Anomaly (Isolation Forest)")
    ax.set_xlabel("Critical Speed Fraction"); ax.set_ylabel("Power (kW)")
    ax.set_title("Isolation Forest: Anomaly Detection")
    ax.legend(); plt.tight_layout(); save_fig("6c_isolation_forest.png")

    print(f"\n  NOTE: Isolation Forest flags are advisory. Data is NOT dropped automatically.")
    print(f"  Review the flagged simulations and decide manually whether to exclude them.")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 7 : DROP REDUNDANT FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 7] Redundant feature removal ...")

    # Drop raw PSD knot columns (D10/D50/D90 already capture this information)
    psd_raw_cols = [c for c in df.columns if "_psd_s" in c or "_psd_p" in c]
    print(f"  Dropping {len(psd_raw_cols)} raw PSD knot columns: {psd_raw_cols}")
    df.drop(columns=psd_raw_cols, inplace=True)

    # Check for near-zero variance shape_k columns (carry no signal)
    shape_k_vars = df[shape_k_cols].var()
    zero_var_k = shape_k_vars[shape_k_vars < 0.01].index.tolist()
    if zero_var_k:
        print(f"  Near-zero variance shape_k columns: {zero_var_k} -> dropping")
        df.drop(columns=zero_var_k, inplace=True)
    else:
        print("  [OK] All shape_k columns have meaningful variance — keeping all 50.")

    # Drop the iso_outlier flag column (advisory only, not a training feature)
    df.drop(columns=["iso_outlier"], inplace=True)

    print(f"  Final dataset shape: {df.shape}")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 8 : TRAIN / VALIDATION / TEST SPLIT (CF-STRATIFIED BY SIMULATION)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 8] Train / Validation / Test split (CF-stratified by simulation ID) ...")

    np.random.seed(RANDOM_SEED)

    # Compute mean CF per simulation — used for stratification
    sim_cf_mean = df.groupby(ID_COL)["cf_max_particle"].mean().sort_values()

    # Bin simulations into 4 CF quartile strata
    sim_cf_mean_df = sim_cf_mean.reset_index()
    sim_cf_mean_df.columns = [ID_COL, "mean_cf"]
    sim_cf_mean_df["cf_stratum"] = pd.qcut(sim_cf_mean_df["mean_cf"], q=4, labels=False, duplicates="drop")

    print("  CF strata (quartile bins) per simulation:")
    for stratum, grp in sim_cf_mean_df.groupby("cf_stratum"):
        sims_in = sorted(grp[ID_COL].tolist())
        cf_range = f"[{grp['mean_cf'].min():.0f} – {grp['mean_cf'].max():.0f}]"
        print(f"    Stratum {stratum}: {sims_in}  CF range {cf_range}")

    train_sims, val_sims, test_sims = [], [], []

    for stratum, grp in sim_cf_mean_df.groupby("cf_stratum"):
        sids = grp[ID_COL].tolist()
        np.random.shuffle(sids)
        n_s = len(sids)
        n_tr = max(1, int(np.floor(TRAIN_RATIO * n_s)))
        n_va = max(1, int(np.floor(VAL_RATIO   * n_s)))
        # Ensure at least 1 sim in test per stratum if possible
        if n_tr + n_va >= n_s and n_s > 2:
            n_va = n_s - n_tr - 1
        elif n_tr + n_va >= n_s:
            n_va = 0
        train_sims.extend(sids[:n_tr])
        val_sims.extend(sids[n_tr:n_tr + n_va])
        test_sims.extend(sids[n_tr + n_va:])

    # ── Apply FORCE_TRAIN overrides ───────────────────────────────────────────────
    # Move any FORCE_TRAIN_SIMS from val/test into training (they're far outside
    # normal training distribution and MUST be learned from, not predicted on).
    for fsim in FORCE_TRAIN_SIMS:
        if fsim in val_sims:
            val_sims.remove(fsim)
            train_sims.append(fsim)
            print(f"  [FORCE_TRAIN] Sim {fsim} moved: VAL -> TRAIN")
        elif fsim in test_sims:
            test_sims.remove(fsim)
            train_sims.append(fsim)
            print(f"  [FORCE_TRAIN] Sim {fsim} moved: TEST -> TRAIN")
        elif fsim not in train_sims:
            print(f"  [FORCE_TRAIN] Sim {fsim} not found in any split (may be excluded).")
        else:
            print(f"  [FORCE_TRAIN] Sim {fsim} already in TRAIN — no action needed.")

    df_train = df[df[ID_COL].isin(train_sims)].copy()
    df_val   = df[df[ID_COL].isin(val_sims)].copy()
    df_test  = df[df[ID_COL].isin(test_sims)].copy()

    print(f"\n  Train : {len(train_sims)} sims -> {len(df_train)} rows  {sorted(train_sims)}")
    print(f"  Val   : {len(val_sims)} sims -> {len(df_val)} rows  {sorted(val_sims)}")
    print(f"  Test  : {len(test_sims)} sims -> {len(df_test)} rows  {sorted(test_sims)}")

    # Report CF distribution across splits to confirm balance
    for label, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        cf_m = split_df["cf_max_particle"].mean()
        print(f"  {label} CF mean: {cf_m:.0f}")

    # ── Split visualization ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 3))
    all_sorted = sorted(df[ID_COL].unique())
    colors_map = {sid: ("#3b82f6" if sid in train_sims else "#10b981" if sid in val_sims else "#ef4444")
                  for sid in all_sorted}
    for i, sid in enumerate(all_sorted):
        ax.bar(i, 1, color=colors_map[sid], edgecolor="white")
        ax.text(i, 0.5, str(sid), ha="center", va="center", fontsize=7, color="white", fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#3b82f6", label=f"Train ({len(train_sims)} sims)"),
                       Patch(color="#10b981", label=f"Val ({len(val_sims)} sims)"),
                       Patch(color="#ef4444", label=f"Test ({len(test_sims)} sims)")])
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("Train / Validation / Test Split — CF-Stratified (by Simulation)", fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig("8a_train_val_test_split.png")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 9 : FEATURE SCALING
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 9] Feature scaling (StandardScaler) ...")

    feature_cols = [c for c in df.columns if c not in [ID_COL] + TARGETS]
    print(f"  Scaling {len(feature_cols)} feature columns ...")
    print(f"  CRITICAL: Scaler fit ONLY on Train set.")

    scaler = StandardScaler()
    df_train_scaled = df_train.copy()
    df_val_scaled   = df_val.copy()
    df_test_scaled  = df_test.copy()

    df_train_scaled[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val_scaled[feature_cols]   = scaler.transform(df_val[feature_cols])
    df_test_scaled[feature_cols]  = scaler.transform(df_test[feature_cols])

    # Save scaler for later use during inference
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved -> {scaler_path}")

    # ── Before/After scaling distributions for 3 key features ───────────────
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    demo_features = ["mill_rpm", "leading_face_angle", "shape_k0"]
    for col_i, col in enumerate(demo_features):
        axes[0][col_i].hist(df_train[col], bins=30, color="#3b82f6", edgecolor="white")
        axes[0][col_i].set_title(f"BEFORE: {col}", fontsize=9)
        axes[1][col_i].hist(df_train_scaled[col], bins=30, color="#10b981", edgecolor="white")
        axes[1][col_i].set_title(f"AFTER scaling: {col}", fontsize=9)
    fig.suptitle("Feature Scaling — Before vs After (Train set)", fontsize=12, fontweight="bold")
    plt.tight_layout(); save_fig("9a_scaling_before_after.png")

    # ─────────────────────────────────────────────────────────────────────────────
    # PHASE 10 : SAVE OUTPUTS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[PHASE 10] Saving cleaned datasets ...")

    base_out = OUTPUT_DIR

    # Save unscaled (for interpretability / ensemble models)
    df_train.to_csv(os.path.join(base_out, "train_unscaled.csv"), index=False)
    df_val.to_csv(os.path.join(base_out,   "val_unscaled.csv"),   index=False)
    df_test.to_csv(os.path.join(base_out,  "test_unscaled.csv"),  index=False)
    print("  Saved: train_unscaled.csv, val_unscaled.csv, test_unscaled.csv")

    # Save scaled (for Neural Nets / distance-based models)
    df_train_scaled.to_csv(os.path.join(base_out, "train_scaled.csv"), index=False)
    df_val_scaled.to_csv(os.path.join(base_out,   "val_scaled.csv"),   index=False)
    df_test_scaled.to_csv(os.path.join(base_out,  "test_scaled.csv"),  index=False)
    print("  Saved: train_scaled.csv, val_scaled.csv, test_scaled.csv")

    # Save full cleaned dataset
    df.to_csv(os.path.join(base_out, "cleaned_dataset.csv"), index=False)
    print("  Saved: cleaned_dataset.csv")

    # ─────────────────────────────────────────────────────────────────────────────
    # FINAL REPORT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PREPROCESSING COMPLETE — SUMMARY REPORT")
    print("=" * 70)
    print(f"  Raw rows              : 3800 (Excluded sims {EXCLUDE_SIMS} -> {len(df)} rows used)")
    print(f"  Final feature count   : {len(feature_cols)}")
    print(f"  Train simulations     : {len(train_sims)} ({len(df_train)} rows)")
    print(f"  Val simulations       : {len(val_sims)}  ({len(df_val)} rows)")
    print(f"  Test simulations      : {len(test_sims)}  ({len(df_test)} rows)")
    print(f"  NaN handling          : Ore cols -> 0 (physics), Short angle -> 0 (single lifter)")
    print(f"  New features created  : critical_speed_fraction, face_angle_asymmetry,")
    print(f"                          has_short_lifter, shape_energy, shape_hf_sharpness,")
    print(f"                          tip_speed, lifter_density, media_load_fraction, has_ore")
    print(f"  Force-trained sims    : {FORCE_TRAIN_SIMS} (moved from val/test into training)")
    print(f"  Scaler                : StandardScaler (fit on Train ONLY)")
    print(f"  All plots             : {OUTPUT_DIR}")
    print("=" * 70)
    print("\n[OK] Ready to train the ExtraTrees & Quantile Random Forest surrogate models!")


if __name__ == "__main__":
    main()
