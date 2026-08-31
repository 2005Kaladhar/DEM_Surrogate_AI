# ⚙️ DEM Surrogate AI — Real-Time Industrial Mill Simulation Accelerator

> *Replacing expensive physics simulations that take **hours** on HPC clusters with a sub-second AI prediction engine. Built during a live research internship at an industrial mill liner manufacturer.*

---

## 🎯 The Problem

**Discrete Element Method (DEM) simulations** are the gold standard for optimising industrial grinding mills — the machines that crush ore in mining operations worldwide. A single simulation:

- Requires expensive HPC/cloud compute clusters
- Takes **4–12 hours** to complete
- Costs thousands of rupees in cloud compute per run
- Cannot be run interactively by engineers on-site

This means engineering teams at large mining equipment manufacturers can only afford to test **3–5 design variations** per week. The entire product design cycle is bottlenecked by compute.

**What if an engineer could get a full simulation result in < 1 second for free, right in their browser?**

---

## 💡 The Solution — An AI Surrogate Model

This project trains a **machine learning surrogate model** that has learned the physics of grinding mill dynamics from 38 real EDEM simulations (≈3,800 data points) and can predict three critical engineering outputs in **milliseconds**:

| Output | Description | Unit |
|--------|-------------|------|
| 🔋 **Total Power Draw** | Energy consumed by the mill per revolution | kW |
| ⚡ **Max Particle Kinetic Energy** | Peak impact energy — drives wear & breakage | J |
| 💪 **Max Compressive Force** | Peak force on liner — drives structural fatigue | N |

These predictions are delivered across a **full 100-point rotation cycle** (0%–100% mill rotation), giving engineers a **dynamic simulation-equivalent curve** rather than a single-point estimate.

---

## 🧠 What Makes This Non-Trivial

### 1. Domain-Aware Feature Engineering (15+ Physics Features)
Raw simulation inputs (RPM, diameter, liner geometry) are insufficient. The model uses **derived physics features** that capture the actual governing equations of mill dynamics:

| Feature | Physics Rationale |
|---------|-------------------|
| `critical_speed_fraction` | `RPM / (42.3 / √D)` — the fundamental tumbling mill operating regime |
| `froude_number` | `ω²R / g` — governs cataract vs. cascade charge motion |
| `tip_speed` | `π·D·RPM / 60` — actual surface impact velocity |
| `charge_kinetic_head` | `0.5·ρ_mix·v²` — dynamic pressure on shell and lifters |
| `lifter_strike_freq` | `n_lifters · RPM / 60` — impact events per second |
| `power_flux_proxy` | `tip_speed² · media_mass · RPM` — total energy transfer rate |
| `froude_number` | Dimensionless governing number for charge cataracting behaviour |
| `shape_k0..k49` | **Fourier descriptors** of liner cross-section geometry (CAD → ML) |

### 2. Fourier Shape Encoding of CAD Geometry
The liner geometry (from industrial STEP/CAD files) is:
1. Sliced at the cross-sectional plane using **CadQuery**
2. Converted to a polar radial profile (r vs θ)
3. Decomposed into **50 Fourier harmonic coefficients** (shape_k0..k49)

This is not standard ML — it's a signal processing pipeline that encodes 3D CAD geometry into a 50-dimensional shape manifold that a tree ensemble can actually learn from.

### 3. Simulation-Level Grouped Stratified Split
With only 38 simulations (each producing 100 rows), naive random 80/20 splitting would **cause catastrophic data leakage** — the model would memorise time-step patterns from the same simulation it's being tested on.

The split was performed **at the simulation ID level** with **CF-stratified quartile bins** to ensure balanced target distribution, with 4 simulations force-assigned to training (sims 3, 4, 5, 24 have geometries far outside the normal distribution that must be learned, never predicted on).

### 4. The Resemblance Engine
When a user enters mill parameters, the app runs a **cosine similarity search** across all 38 historical simulations to find the closest physical match. This gives engineers a reference: *"Your design is most similar to Simulation 12 — a 5.2m SAG Mill at 72% critical speed."*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Streamlit Web Application                  │
│                                                             │
│  ┌─────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Data Creator│  │ Prediction Engine│  │  Data Report  │  │
│  │  (Tab 1)   │  │    (Tab 2)       │  │   (Tab 3)     │  │
│  └──────┬──────┘  └────────┬─────────┘  └───────────────┘  │
│         │                  │                                 │
│         ▼                  ▼                                 │
│  ┌─────────────┐  ┌─────────────────────────────────────┐   │
│  │Analysis     │  │  ML Inference Pipeline               │   │
│  │Engine       │  │                                     │   │
│  │(CadQuery    │  │  User Input → Feature Engineering → │   │
│  │+ Fourier)   │  │  ExtraTrees/RF → Plotly Curves      │   │
│  └─────────────┘  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Trained Model Artifacts     │
│  ├── model_power.pkl         │  ExtraTreesRegressor (300 trees)
│  ├── model_ke.pkl            │  ExtraTreesRegressor (300 trees)
│  ├── model_cf.pkl            │  RandomForestRegressor (500 trees)
│  └── scaler.pkl              │  StandardScaler (fit on train only)
└──────────────────────────────┘
```

---

## 📊 Model Performance

Training used 3 separate models to capture the distinct physical character of each output:

| Target | Model | Notes |
|--------|-------|-------|
| Power Draw | `ExtraTreesRegressor(n_estimators=300)` | Unscaled features (tree model) |
| Kinetic Energy | `ExtraTreesRegressor(n_estimators=300)` | Unscaled features |
| Compressive Force | `RandomForestRegressor(n_estimators=500)` | Scaled features (different feature sensitivity) |

Evaluation: **CF-stratified holdout** (70% train / 15% val / 15% test split at simulation level), with parity plots and moving-average trend overlays. **No cross-validation** — intentional, because the simulation grouping structure makes K-Fold CV invalid without careful group assignment.

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

> ⚠️ `cadquery` is required **only** for STEP file geometry analysis. All prediction and report features work without it.

### Run the App
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Load Pre-Trained Models
The `model_evaluation/` folder contains trained `.pkl` files ready to use.  
The `PRE PROCESSED/preprocessing_report/` folder contains `scaler.pkl` and training CSVs.

No retraining needed — just clone and run.

---

## 📁 Repository Structure

```
├── app.py                        # Main Streamlit application entry point
├── analysis_engine.py            # CadQuery STEP → Polar profile → Fourier pipeline
├── predictive_dashboard_page.py  # Prediction UI, resemblance engine, Plotly charts
├── data_report_page.py           # Interactive data gallery with Swiper.js carousel
├── preprocess_pipeline.py        # Full data preprocessing & feature engineering
├── train_final_models.py         # Model training & evaluation plot generation
├── generate_pdf_report.py        # ReportLab PDF export engine
├── requirements.txt
├── model_evaluation/
│   ├── final_model_power_total_geometry_kw.pkl
│   ├── final_model_ke_max_particle.pkl
│   ├── final_model_cf_max_particle.pkl
│   └── *.png                     # Evaluation parity plots
└── PRE PROCESSED/
    └── preprocessing_report/
        ├── scaler.pkl
        ├── train_unscaled.csv
        ├── val_unscaled.csv
        └── test_unscaled.csv
```

---

## 🔩 What Broke at 2AM (And How I Fixed It)

### The Bug: Silent Data Leakage That Looked Like Perfect Accuracy

At 2AM, with the first model checkpoint trained and predictions looking suspiciously good on the validation set, I ran a deeper inspection.

**The parity plot was nearly perfect. Too perfect.**

I traced it back:

```python
# WRONG — what was running initially:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

The dataset has 3,800 rows — **100 rows per simulation**. A naive `train_test_split` shuffles all rows randomly, so rows 47–52 from Simulation 7 end up in training while rows 53–58 from the same simulation end up in validation.

The model wasn't learning physics. It was **memorising simulation-specific patterns** and recognising its own training rows in validation.

**The Fix:** Rebuild the entire split at the simulation ID level:

```python
# RIGHT — grouped stratified split by simulation
sim_cf_mean = df.groupby("simulation_id")["cf_max_particle"].mean().sort_values()
sim_cf_mean_df["cf_stratum"] = pd.qcut(sim_cf_mean_df["mean_cf"], q=4, labels=False)

train_sims, val_sims, test_sims = [], [], []
for stratum, grp in sim_cf_mean_df.groupby("cf_stratum"):
    sids = grp["simulation_id"].tolist()
    np.random.shuffle(sids)
    n_tr = max(1, int(np.floor(0.70 * len(sids))))
    n_va = max(1, int(np.floor(0.15 * len(sids))))
    train_sims.extend(sids[:n_tr])
    val_sims.extend(sids[n_tr:n_tr + n_va])
    test_sims.extend(sids[n_tr + n_va:])
```

Validation R² dropped from a suspicious **0.998 → 0.87**, which is the honest performance on genuinely unseen simulation configurations. That's when I knew the model was actually generalising.

**The lesson:** When your dataset has group structure (patients, sessions, simulations), `train_test_split` is silently wrong. The real benchmark is whether the model works on a simulation it has **never seen a single row from**.

---

## 🌍 Real-World Impact

- A single EDEM simulation at an industrial scale costs ~₹8,000–₹25,000 in HPC compute time
- This surrogate replaces the inner loop of the design exploration phase
- An engineer can now evaluate **500 design variations in the time one simulation used to take**
- Applied to optimising liner geometry for reduced wear = directly extends liner life and reduces maintenance shutdowns in mining operations

---

## 👨‍💻 Built By

**Kaladhar** — Engineering Intern, Industrial Mill Liner R&D  
Built during a research internship focused on AI-accelerated DEM simulation for industrial mill optimisation.

---

## 📄 License

MIT License — see `LICENSE` for details.
