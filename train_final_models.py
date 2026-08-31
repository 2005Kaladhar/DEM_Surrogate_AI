"""
=============================================================================
DEM Surrogate Model — Automated Model Retraining & Evaluation Engine
=============================================================================
Author  : Machine Learning Engine
Purpose : Retrain ExtraTrees & Quantile Random Forest model suite on cleaned dataset,
          silence feature name warnings, and output model evaluation plots into ./model_evaluation/
Outputs : Trained model pickles saved to ./model_evaluation/
          Evaluation plot images saved to ./model_evaluation/
=============================================================================
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

def plot_model_parity(y_train, p_train, y_val, p_val, y_test, p_test, target_name, unit, out_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    
    # Parity line 1:1
    all_y = np.concatenate([y_train, y_val, y_test]) if len(y_val) > 0 or len(y_test) > 0 else y_train
    all_p = np.concatenate([p_train, p_val, p_test]) if len(p_val) > 0 or len(p_test) > 0 else p_train
    min_v = min(all_y.min(), all_p.min())
    max_v = max(all_y.max(), all_p.max())
    
    ax.plot([min_v, max_v], [min_v, max_v], color='#94a3b8', linestyle='--', linewidth=1.8, label='Ideal 1:1 Match')
    
    # Scatter points with distinct professional colors
    ax.scatter(y_train, p_train, color='#2563eb', alpha=0.4, s=25, label='Train Set')
    if len(y_val) > 0:
        ax.scatter(y_val, p_val, color='#d97706', alpha=0.8, s=35, marker='s', label='Validation Set')
    if len(y_test) > 0:
        ax.scatter(y_test, p_test, color='#16a34a', alpha=0.8, s=35, marker='^', label='Test Set')
        
    # Moving average trend line
    sort_idx = np.argsort(all_y)
    sorted_y = all_y[sort_idx]
    sorted_p = all_p[sort_idx]
    ma_p = pd.Series(sorted_p).rolling(window=20, min_periods=1, center=True).mean().values
    ax.plot(sorted_y, ma_p, color='#dc2626', linewidth=2, linestyle='-.', label='Moving Avg Trend')

    ax.set_xlabel(f'Actual {target_name} ({unit})', fontweight='bold', fontsize=11)
    ax.set_ylabel(f'Predicted {target_name} ({unit})', fontweight='bold', fontsize=11)
    ax.set_title(f'Model Evaluation Parity Plot — {target_name}', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    if len(sys.argv) >= 2:
        target_project_dir = sys.argv[1]
    else:
        target_project_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 70)
    print("  DEM SURROGATE — AUTOMATED MODEL RETRAINING & EVALUATION")
    print("=" * 70)
    print(f"Target Project Directory: {target_project_dir}")

    prep_dir = os.path.join(target_project_dir, "PRE PROCESSED", "preprocessing_report")
    if not os.path.exists(prep_dir):
        prep_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PRE PROCESSED", "preprocessing_report")

    train_unscaled_path = os.path.join(prep_dir, "train_unscaled.csv")
    train_scaled_path   = os.path.join(prep_dir, "train_scaled.csv")
    val_unscaled_path   = os.path.join(prep_dir, "val_unscaled.csv")
    val_scaled_path     = os.path.join(prep_dir, "val_scaled.csv")
    test_unscaled_path  = os.path.join(prep_dir, "test_unscaled.csv")
    test_scaled_path    = os.path.join(prep_dir, "test_scaled.csv")

    if not os.path.exists(train_unscaled_path) or not os.path.exists(train_scaled_path):
        print(f"ERROR: Preprocessing artifacts not found in {prep_dir}!")
        sys.exit(1)

    df_train_u = pd.read_csv(train_unscaled_path)
    df_train_s = pd.read_csv(train_scaled_path)
    df_val_u   = pd.read_csv(val_unscaled_path) if os.path.exists(val_unscaled_path) else pd.DataFrame()
    df_val_s   = pd.read_csv(val_scaled_path) if os.path.exists(val_scaled_path) else pd.DataFrame()
    df_test_u  = pd.read_csv(test_unscaled_path) if os.path.exists(test_unscaled_path) else pd.DataFrame()
    df_test_s  = pd.read_csv(test_scaled_path) if os.path.exists(test_scaled_path) else pd.DataFrame()

    non_feat_cols = ["simulation_id", "cf_max_particle", "ke_max_particle", "power_total_geometry_kw"]
    feature_cols  = [c for c in df_train_u.columns if c not in non_feat_cols]

    X_tr_u = df_train_u[feature_cols].values
    X_tr_s = df_train_s[feature_cols].values
    X_va_u = df_val_u[feature_cols].values if not df_val_u.empty else np.empty((0, len(feature_cols)))
    X_va_s = df_val_s[feature_cols].values if not df_val_s.empty else np.empty((0, len(feature_cols)))
    X_te_u = df_test_u[feature_cols].values if not df_test_u.empty else np.empty((0, len(feature_cols)))
    X_te_s = df_test_s[feature_cols].values if not df_test_s.empty else np.empty((0, len(feature_cols)))

    y_tr_pow = df_train_u["power_total_geometry_kw"].values
    y_va_pow = df_val_u["power_total_geometry_kw"].values if not df_val_u.empty else np.array([])
    y_te_pow = df_test_u["power_total_geometry_kw"].values if not df_test_u.empty else np.array([])

    y_tr_ke = df_train_u["ke_max_particle"].values
    y_va_ke = df_val_u["ke_max_particle"].values if not df_val_u.empty else np.array([])
    y_te_ke = df_test_u["ke_max_particle"].values if not df_test_u.empty else np.array([])

    y_tr_cf = df_train_s["cf_max_particle"].values
    y_va_cf = df_val_s["cf_max_particle"].values if not df_val_s.empty else np.array([])
    y_te_cf = df_test_s["cf_max_particle"].values if not df_test_s.empty else np.array([])

    models_dir = os.path.join(target_project_dir, "model_evaluation")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Total Power Model
    print("\n[1/3] Training Total Power Draw model (ExtraTreesRegressor)...")
    model_power = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model_power.fit(X_tr_u, y_tr_pow)
    pow_path = os.path.join(models_dir, "final_model_power_total_geometry_kw.pkl")
    with open(pow_path, "wb") as f:
        pickle.dump(model_power, f)
    print(f"  [SAVED] {pow_path}")

    # 2. Kinetic Energy Model
    print("\n[2/3] Training Max Particle Kinetic Energy model (ExtraTreesRegressor)...")
    model_ke = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model_ke.fit(X_tr_u, y_tr_ke)
    ke_path = os.path.join(models_dir, "final_model_ke_max_particle.pkl")
    with open(ke_path, "wb") as f:
        pickle.dump(model_ke, f)
    print(f"  [SAVED] {ke_path}")

    # 3. Compressive Force Model
    print("\n[3/3] Training Max Particle Compressive Force model (RandomForestRegressor)...")
    model_cf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    model_cf.fit(X_tr_s, y_tr_cf)
    cf_path = os.path.join(models_dir, "final_model_cf_max_particle.pkl")
    with open(cf_path, "wb") as f:
        pickle.dump(model_cf, f)
    print(f"  [SAVED] {cf_path}")

    # 4. Generate & Save Model Evaluation Plots inside model_evaluation/
    print("\n[PLOTS] Generating model evaluation parity & performance plots into model_evaluation/ ...")
    
    p_tr_pow = model_power.predict(X_tr_u)
    p_va_pow = model_power.predict(X_va_u) if len(X_va_u) > 0 else np.array([])
    p_te_pow = model_power.predict(X_te_u) if len(X_te_u) > 0 else np.array([])
    plot_model_parity(y_tr_pow, p_tr_pow, y_va_pow, p_va_pow, y_te_pow, p_te_pow, "Total Power Draw", "kW", os.path.join(models_dir, "power_model_evaluation.png"))
    print(f"  [SAVED] {os.path.join(models_dir, 'power_model_evaluation.png')}")

    p_tr_ke = model_ke.predict(X_tr_u)
    p_va_ke = model_ke.predict(X_va_u) if len(X_va_u) > 0 else np.array([])
    p_te_ke = model_ke.predict(X_te_u) if len(X_te_u) > 0 else np.array([])
    plot_model_parity(y_tr_ke, p_tr_ke, y_va_ke, p_va_ke, y_te_ke, p_te_ke, "Max Particle Kinetic Energy", "J", os.path.join(models_dir, "ke_model_evaluation.png"))
    print(f"  [SAVED] {os.path.join(models_dir, 'ke_model_evaluation.png')}")

    p_tr_cf = model_cf.predict(X_tr_s)
    p_va_cf = model_cf.predict(X_va_s) if len(X_va_s) > 0 else np.array([])
    p_te_cf = model_cf.predict(X_te_s) if len(X_te_s) > 0 else np.array([])
    plot_model_parity(y_tr_cf, p_tr_cf, y_va_cf, p_va_cf, y_te_cf, p_te_cf, "Max Particle Compressive Force", "N", os.path.join(models_dir, "cf_model_evaluation.png"))
    print(f"  [SAVED] {os.path.join(models_dir, 'cf_model_evaluation.png')}")

    # Overall Summary Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)
    for idx, (y_tr, p_tr, y_te, p_te, name, unit, color) in enumerate([
        (y_tr_pow, p_tr_pow, y_te_pow, p_te_pow, "Total Power", "kW", "#2563eb"),
        (y_tr_ke, p_tr_ke, y_te_ke, p_te_ke, "Max Kinetic Energy", "J", "#10b981"),
        (y_tr_cf, p_tr_cf, y_te_cf, p_te_cf, "Compressive Force", "N", "#f43f5e")
    ]):
        min_val = min(y_tr.min(), p_tr.min())
        max_val = max(y_tr.max(), p_tr.max())
        axes[idx].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='1:1')
        axes[idx].scatter(y_tr, p_tr, color=color, alpha=0.4, s=20, label='Train')
        if len(y_te) > 0:
            axes[idx].scatter(y_te, p_te, color='#16a34a', alpha=0.8, s=25, marker='^', label='Test')
        
        # Add moving average line
        all_y_s = np.concatenate([y_tr, y_te]) if len(y_te) > 0 else y_tr
        all_p_s = np.concatenate([p_tr, p_te]) if len(p_te) > 0 else p_tr
        sort_i = np.argsort(all_y_s)
        ma_s = pd.Series(all_p_s[sort_i]).rolling(window=15, min_periods=1, center=True).mean().values
        axes[idx].plot(all_y_s[sort_i], ma_s, color='#d97706', linewidth=1.8, linestyle='-.', label='Moving Avg')
        
        axes[idx].set_title(f'{name} ({unit})', fontweight='bold', fontsize=11)
        axes[idx].set_xlabel(f'Actual ({unit})', fontsize=9)
        axes[idx].set_ylabel(f'Predicted ({unit})', fontsize=9)
        axes[idx].grid(True, linestyle='--', alpha=0.4)
        axes[idx].legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "overall_model_performance_summary.png"), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  [SAVED] {os.path.join(models_dir, 'overall_model_performance_summary.png')}")

    print("\n" + "=" * 70)
    print("  MODEL RETRAINING COMPLETE — MODELS & EVALUATION PLOTS SAVED")
    print("=" * 70)

if __name__ == "__main__":
    main()
