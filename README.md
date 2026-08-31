# Tree-Based Ensemble Surrogate Model for Industrial Grinding Mills

[![Live Web Application](https://img.shields.io/badge/Streamlit_Cloud-Live_Platform-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://dem-surrogate-model-ai.streamlit.app/)
[![GitHub Repository](https://img.shields.io/badge/GitHub-2005Kaladhar%2FDEM__Surrogate__AI-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/2005Kaladhar/DEM_Surrogate_AI)
[![Python Version](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

> **Live Application Access**  
> Access the interactive surrogate simulation platform in your browser:  
> **[https://dem-surrogate-model-ai.streamlit.app/](https://dem-surrogate-model-ai.streamlit.app/)**  
> *(No local installation or GPU/HPC hardware required — sub-second mill predictions directly on the web.)*

---

## Executive Summary

Industrial comminution (grinding mineral ores in SAG, AG, Ball, and Pebble mills) accounts for approximately **1.8% of global electrical energy consumption** and over **50% of the total energy budget of an industrial mining facility**.

Optimizing internal **mill liner profiles** (lifter heights, face angles, spacing, and wear profiles) is essential to maximize grinding throughput, reduce catastrophic liner cracking, and prevent mill shell structural fatigue. However:
* High-fidelity **Discrete Element Method (DEM)** simulations (e.g., Rocky DEM, EDEM) require **4 to 12 hours** per run on multi-node HPC clusters.
* Mining and equipment design engineers are constrained to evaluating only **3 to 5 design iterations per week**, creating a significant design bottleneck.

**DEM Surrogate AI** replaces expensive numerical particle simulations with a sub-second, tree-based ensemble surrogate model. Trained on high-fidelity DEM simulation rotation cycles across multi-scale industrial mills (diameters from 5.0 m to 11.2 m), it reconstructs complete 100-point rotational dynamics in **under 50 milliseconds**.

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                TRADITIONAL WORKFLOW vs. AI SURROGATE                            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│  TRADITIONAL DEM : CAD Ingestion ──► Mesh Setup ──► HPC Cluster (4-12 Hours) ──► Post-Processing │
│  AI SURROGATE    : CAD Ingestion ──► Fourier SVD ──► Tree Ensembles (<50 ms) ──► Live Dashboard │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Live Interactive Platform & Modules

**Hosted Application URL:** [https://dem-surrogate-model-ai.streamlit.app/](https://dem-surrogate-model-ai.streamlit.app/)

The platform integrates four specialized engineering modules:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     SURROGATE WEB SUITE                                         │
├───────────────────────┬───────────────────────┬────────────────────────┬────────────────────────┤
│    1. Data Creator    │  2. Record Viewer     │  3. Data Report & EDA  │  4. Predictive Engine  │
├───────────────────────┼───────────────────────┼────────────────────────┼────────────────────────┤
│ • Automated 3D STEP   │ • Interactive record  │ • Automated Data Audit │ • Sub-second inference │
│   B-Rep Slicing       │   browser & editor    │   & Outlier Detection  │   across 0-100% cycle  │
│ • Taubin Circle Fit   │ • 50-Harmonic Fourier │ • Isolation Forest     │ • 5-Point Moving Avg   │
│ • SVD Face Angles     │   spectrum inspector  │ • Dynamic Parity & R²  │ • QRF Uncertainty Band │
│ • Suggested Diameter  │ • In-place updates    │ • 30+ Sim Plot Gallery │ • Resemblance Engine   │
│   & Effective Area    │   & metadata logging  │ • Master PDF Export    │ • Excel / CSV Export   │
└───────────────────────┴───────────────────────┴────────────────────────┴────────────────────────┘
```

---

## Target Physical Quantities

The surrogate model predicts three foundational hydro-mechanical responses across a complete **100-point rotation cycle** ($0\% \le \theta \le 100\%$):

| Target Parameter | Symbol | Unit | Physical & Engineering Significance |
| :--- | :---: | :---: | :--- |
| **Total Power Draw** | $P_{\mathrm{total}}$ | **kW** | Governs motor power consumption, electrical sizing, and specific energy efficiency (kWh/t). |
| **Maximum Particle Kinetic Energy** | $E_{k,\max}$ | **J** | Measures peak impact energy delivered to ore chunks for comminution and liner impact gouging. |
| **Maximum Particle Compressive Force** | $F_{c,\max}$ | **N** | Measures localized peak compressive stress on liners, driving fatigue life and structural integrity. |

---

## Mathematical Formulations & Architecture

```
                                  CAD GEOMETRY PIPELINE
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  3D CAD STEP │────►│   CadQuery   │────►│    Taubin    │────►│  16,384-Bin  │────►│  50-Harmonic │
 │  Liner B-Rep │     │ Midplane Cut │     │  Circle Fit  │     │ Polar Map    │     │  DFT Amplit. │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                                            │
                                                                                            ▼
                               ML INFERENCE & PREDICTION                             ┌──────────────┐
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     │ 18 Physics   │
 │ Full 100-pt  │◄────│ Stochastic   │◄────│ ExtraTrees / │◄────│ 117-Dim Feat │◄────│ Derived      │
 │ Rotary Wave  │     │ Superposit.  │     │ Quantile RF  │     │ Vector       │     │ Features     │
 └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

### 1. CAD Ingestion & Geometry Discretization
1. **Midplane Slicing**: The liner solid geometry is ingested as a raw 3D STEP B-Rep file. The longitudinal mill axis (shortest bounding box span) is detected automatically, sectioning solids along the transverse midplane.
2. **Taubin Algebraic Circle Fitting**: The outer mill shell radius $R_{\mathrm{shell}}$ and center $(x_c, y_c)$ are extracted by minimizing algebraic distance with hyperaccuracy constraint:
   $$\mathcal{F}(a, b, R) = \sum_{i=1}^{M} \left[ (x_i - a)^2 + (y_i - b)^2 - R^2 \right]^2 \quad \text{subject to} \quad 4R^2 = 1$$
3. **16,384-Bin Polar Resampling**: Section boundary points are mapped to polar coordinates $(r, \theta)$ relative to $(x_c, y_c)$ and discretized into $N = 16,384$ uniform angular bins to form the unrolled protrusion waveform $h(\theta) = R_{\mathrm{shell}} - r(\theta)$.
4. **SVD Face Angle Sweep**: Singular Value Decomposition fits local tangent planes along leading and trailing lifter boundaries to extract leading face angle $\alpha_{\mathrm{lead}}$, trailing face angle $\beta_{\mathrm{trail}}$, and short-lifter dual-height parameters.

### 2. Rotational-Invariant Fourier Harmonic Descriptors
To create a compact representation of complex liner profiles invariant to starting angular orientation $\theta_0$, a 50-harmonic Discrete Fourier Transform (DFT) is computed over $h(\theta)$:
$$C_k = \frac{1}{N} \left| \sum_{n=0}^{N-1} h(s_n) e^{-j \frac{2\pi k n}{N}} \right|, \quad k \in [0, 49]$$
* **Rotation Invariance**: An angular phase shift $\theta \to \theta + \Delta\theta$ introduces a complex multiplier $e^{-j k \Delta\theta}$. Taking magnitude $|C_k|$ strips the phase, ensuring identical shape descriptors regardless of mill mesh orientation.

---

### 3. Derived Engineering Features Directory

18 domain-specific physical features are constructed to provide strong physical grounding to the ensemble models:

| Feature Name | Formula / Definition | Physical Impact |
| :--- | :---: | :--- |
| `critical_speed_fraction` | $\phi = \frac{N_{\mathrm{rpm}}}{42.3 / \sqrt{D_{\mathrm{eff}}}}$ | Governs the cascading vs. cataracting regime of the charge. |
| `froude_number` | $\mathrm{Fr} = \frac{\omega^2 R}{g}$ | Dimensionless ratio of centrifugal to gravitational forces. |
| `tip_speed` | $v_{\mathrm{tip}} = \frac{\pi D_{\mathrm{eff}} N_{\mathrm{rpm}}}{60}$ | Maximum tangential velocity of lifter tips impacting the charge toe (m/s). |
| `charge_kinetic_head` | $q_k = \frac{1}{2} \rho_{\mathrm{mix}} v_{\mathrm{tip}}^2$ | Dynamic kinetic pressure exerted on the mill shell and lifters (Pa). |
| `lifter_strike_freq` | $f_{\mathrm{strike}} = \frac{N_{\mathrm{rpm}} \cdot N_{\mathrm{lifters}}}{60}$ | Frequency of mechanical impact pulses per second (Hz). |
| `power_flux_proxy` | $\Pi_P = v_{\mathrm{tip}}^2 \cdot M_{\mathrm{media}} \cdot N_{\mathrm{rpm}}$ | Scaling proxy for mill motor power draw. |
| `specific_impact_energy` | $e_{\mathrm{impact}} = \frac{v_{\mathrm{tip}}^2}{D_{\mathrm{eff}}}$ | Per-meter collision energy capacity of the tumbling charge. |
| `face_angle_asymmetry` | $\Delta\alpha = \alpha_{\mathrm{lead}} - \beta_{\mathrm{trail}}$ | Angular differential governing charge trajectory and lift height. |
| `shape_energy` | $E_{\mathrm{shape}} = \sqrt{\sum_{k=0}^{49} C_k^2}$ | $L_2$ norm of harmonic amplitudes encoding overall lifter volume. |
| `shape_hf_sharpness` | $S_{\mathrm{hf}} = \frac{1}{20} \sum_{k=30}^{49} C_k$ | High-frequency Fourier mean capturing lifter edge sharpness vs. wear. |
| `rot_sin` / `rot_cos` | $\sin(2\pi \cdot \theta), \, \cos(2\pi \cdot \theta)$ | Enforces $360^\circ$ continuous cyclic boundary conditions across rotation. |
| `media_load_fraction` | $J = \frac{M_{\mathrm{media}}}{V_{\mathrm{mill}} \cdot \rho_{\mathrm{media}}}$ | Volumetric ball filling degree of the grinding chamber. |
| `has_short_lifter` | $\mathbb{I}(\alpha_{\mathrm{short}} > 0)$ | Indicator flag for hi-lo alternating lifter arrangements. |
| `has_ore` | $\mathbb{I}(M_{\mathrm{ore}} > 0)$ | Differentiates AG/SAG multi-component charges from ball milling. |
| `media_aspect_ratio` | $\mathrm{AR}_{\mathrm{media}} = R_{\mathrm{ball}} / D_{\mathrm{eff}}$ | Relative ball size ratio governing contact mechanics. |
| `total_charge_mass` | $M_{\mathrm{total}} = M_{\mathrm{media}} + M_{\mathrm{ore}}$ | Total charge load inside the rotating shell (kg). |
| `lifter_density` | $\rho_{\mathrm{lifter}} = \frac{N_{\mathrm{lifters}}}{\pi D_{\mathrm{eff}}}$ | Number of lifters per linear meter of shell circumference. |
| `media_count_proxy` | $N_{\mathrm{balls}} \approx M_{\mathrm{media}} / m_{\mathrm{ball}}$ | Estimated discrete particle count in the active grinding charge. |

---

## Model Performance & Validation Benchmarks

Models were evaluated on **strictly held-out test simulations** (grouped at the simulation level to eliminate data leakage):

| Target Variable | Model Architecture | Train $R^2$ | Val $R^2$ | Test $R^2$ | MAE | Engineering Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Power Draw** | 300-Tree `ExtraTreesRegressor` | **1.0000** | **0.9552** | **0.8773** | 27.82 kW | **98.2% Trend Accuracy** |
| **Max Kinetic Energy** | 300-Tree `ExtraTreesRegressor` | **1.0000** | **0.9692** | **0.8450** | 37.96 J | **97.4% Trend Accuracy** |
| **Max Compressive Force** | 500-Tree `QuantileRF` + Superposition | **0.9362** | **0.5000** | **0.4729** | 2489.4 N | **96.7% – 100.0% Peak Capture** |

```
                                    MODEL EVALUATION PARITY
         Total Power Draw (kW)              Max Kinetic Energy (J)            Max Compressive Force (N)
   ┌───────────────────────────────┐   ┌───────────────────────────────┐   ┌───────────────────────────────┐
   │  Test R² = 0.8773             │   │  Test R² = 0.8450             │   │  Peak Impact Acc = 98.4%      │
   │  MAE = 27.82 kW               │   │  MAE = 37.96 J                │   │  QRF 98th Pct Envelope        │
   │                               │   │                               │   │                               │
   │         /  Pred               │   │         /  Pred               │   │         /  Pred               │
   │       / ── Ground Truth       │   │       / ── Ground Truth       │   │       / ── Ground Truth       │
   │     /                         │   │     /                         │   │     /                         │
   └───────────────────────────────┘   └───────────────────────────────┘   └───────────────────────────────┘
```

> **Quantile Superposition for Compressive Force**  
> DEM compressive force signals contain extreme stochastic collision spikes as discrete particles strike lifters. Standard regressors suffer from regression-to-the-mean, underpredicting structural peaks by up to 60%. The Quantile Random Forest architecture predicts the median trajectory alongside a 98th percentile envelope, superimposing a localized Gumbel-Beta stochastic pulse to preserve full structural impact amplitudes.

---

## Local Installation & Setup

### Prerequisites
* Python 3.10+
* Git & Git LFS

### 1. Clone the Repository
```bash
# Clone with Git LFS to pull trained model weights (~277 MB)
git clone https://github.com/2005Kaladhar/DEM_Surrogate_AI.git
cd DEM_Surrogate_AI
git lfs pull
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Note for Linux users: ensure standard OpenGL runtime is present: `sudo apt-get install -y libgl1 libxrender1 libsm6 libice6 libxext6`)*

### 4. Launch the Application
```bash
streamlit run app.py
```
The application will launch locally at `http://localhost:8501`.

---

## Repository Structure

```
DEM_Surrogate_AI/
│
├── app.py                             # Main Streamlit web application & multi-page router
├── analysis_engine.py                 # CAD parsing, midplane slicing, Taubin circle fit & Fourier engine
├── predictive_dashboard_page.py       # Prediction engine, interactive waveforms & resemblance engine
├── data_report_page.py                # Preprocessing audit, parity gallery & PDF report launcher
├── preprocess_pipeline.py             # Automated data cleaning, outlier audit & physics feature generation
├── train_final_models.py              # ExtraTrees & Quantile Random Forest retraining engine
├── replot_all_evaluations_v2.py       # Ground-truth vs prediction evaluation plotting pipeline
├── generate_pdf_report.py             # Dynamic multi-page technical PDF report builder (ReportLab)
├── requirements.txt                   # Python library dependencies
├── packages.txt                       # Linux system package requirements for Streamlit Cloud
├── settings.json                      # Default application configurations & dataset binding
├── temp1_DataSet.xlsx                 # Master simulation dataset across 38 industrial configurations
│
└── DEM_Surrogate_temp1_DataSet/       # Git LFS Artifacts & Output Directory
    ├── DEM_Surrogate_Master_Report.pdf# Dynamic technical engineering report
    ├── model_evaluation/              # Trained ExtraTrees & QRF model weight pickles (.pkl)
    ├── evaluation_plots/              # Simulation ground-truth vs AI evaluation curves (.png)
    └── PRE PROCESSED/                 # Cleaned dataset splits & StandardScaler artifact (.pkl)
```
