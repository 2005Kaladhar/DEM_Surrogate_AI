# 📦 GitHub Upload Guide — DEM Surrogate AI

## What Goes on GitHub (and What Doesn't)

### ✅ Upload These
| File | Why |
|------|-----|
| `app.py` | Main Streamlit application |
| `analysis_engine.py` | CadQuery + Fourier geometry pipeline |
| `predictive_dashboard_page.py` | Prediction UI + Resemblance Engine |
| `data_report_page.py` | Data gallery + Swiper.js carousel |
| `preprocess_pipeline.py` | Full preprocessing pipeline |
| `train_final_models.py` | Model training script |
| `generate_pdf_report.py` | PDF report engine |
| `requirements.txt` | Dependencies |
| `README.md` | (The hackathon README from this folder) |
| `.gitignore` | To keep data out |
| `lottie_loading.json` | Loading animation asset |
| `splashscreen.json` | Splash screen animation |
| `model_evaluation/*.pkl` | **3 trained model files + scaler.pkl** |
| `model_evaluation/*.png` | Evaluation parity plots (good visual proof) |
| `config.json`, `settings.json` | App config |

### ❌ Do NOT Upload These
| File | Why |
|------|-----|
| `*.xlsx`, `*.csv` | Raw simulation data — keep private |
| `*.step`, `*.stp` | Industrial CAD geometry — large binary, not needed for the app to run |
| `masterDataSet_v1.csv` | Full training dataset — large file, not required on GitHub |
| `*.docx`, `*.pptx`, `*.pdf` | Documentation reports — not needed for running the app |
| `sim*_angular_profile.csv` | Per-simulation raw outputs — large files |
| `Anirban Sir/` | Entire folder — duplicate/backup, not needed |
| `*.mp4` | Demo video — upload separately to YouTube |
| `__pycache__/` | Python cache — auto-generated |
| `.vscode/`, `.cursor/`, `.gemini/` | IDE configs |

---

## Step-by-Step GitHub Setup

```bash
# 1. Create a new repo on github.com (name it: dem-surrogate-ai)
#    Set it to PUBLIC for hackathon visibility

# 2. From your Project folder, initialise git:
cd /path/to/your/project    # navigate to your Project folder
git init
git remote add origin https://github.com/YOUR_USERNAME/dem-surrogate-ai.git

# 3. Copy the hackathon README in:
copy Razorpay\README.md README.md   # overwrites the old one

# 4. Copy the .gitignore in:
copy Razorpay\.gitignore .gitignore

# 5. Stage only the safe files:
git add app.py analysis_engine.py predictive_dashboard_page.py
git add data_report_page.py preprocess_pipeline.py train_final_models.py
git add generate_pdf_report.py requirements.txt README.md .gitignore
git add lottie_loading.json splashscreen.json config.json settings.json
git add model_evaluation/*.pkl model_evaluation/*.png
git add PRE\ PROCESSED/preprocessing_report/scaler.pkl
git add PRE\ PROCESSED/preprocessing_report/train_unscaled.csv
git add PRE\ PROCESSED/preprocessing_report/val_unscaled.csv
git add PRE\ PROCESSED/preprocessing_report/test_unscaled.csv

# 6. Verify what you're committing:
git status

# 7. Commit and push:
git commit -m "feat: DEM Surrogate AI — Razorpay Open Innovation Hackathon"
git push -u origin main
```

---

## Pickle File Sizes — Check Before Upload

GitHub has a **50MB file size limit** (100MB hard limit).  
Check your pkl sizes first:

```powershell
# Run this in PowerShell from the Project folder:
Get-ChildItem model_evaluation\*.pkl | Select-Object Name, @{N='Size_MB';E={[math]::Round($_.Length/1MB, 2)}}
```

If any `.pkl` is over 50MB, use **Git LFS**:
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

---

## 5-Minute Demo Video Script

### What to Show (in order):
1. **(0:00–0:30)** Open the app. Show the clean Streamlit UI. Say: *"This is a real-time DEM simulation surrogate trained on 38 industrial mill configurations."*
2. **(0:30–1:30)** Go to **Prediction tab**. Set mill parameters (Ball Mill, 5m diameter, 12 RPM, etc.). Click predict. Show the three Plotly curves (Power, KE, CF) animate across the 100-rotation-% cycle.
3. **(1:30–2:30)** Point out the **Resemblance Engine** result — *"closest historical match is Simulation X — gives engineers a physical reference."*
4. **(2:30–3:30)** Switch to **Data Report tab**. Show the Swiper.js image gallery of simulation parity plots and profile analysis.
5. **(3:30–4:30)** Mention the engineering numbers: *"A normal EDEM simulation takes 4–12 hours and costs thousands in HPC compute. This gives the same curve in < 1 second."*
6. **(4:30–5:00)** Say: *"The real challenge wasn't the ML — it was the 2AM data leakage bug."* Briefly explain: naive `train_test_split` looked perfect (R² = 0.998) because it was memorising its own rows from the same simulations. Fixed with grouped stratified split — honest R² = 0.87 on genuinely unseen configurations.

---

## The "2AM" Story — For the Video

**Timestamp:** ~1AM during model validation sprint  
**Symptom:** Validation parity plot looked TOO perfect — near-zero error on all three targets  
**Root Cause:** `sklearn.train_test_split` shuffled 3,800 rows randomly. Since each simulation has 100 consecutive rows (timesteps), rows from the same simulation appeared in both train and test. The model memorised timestep patterns from simulations it had seen, not actual physics.  
**Fix:** Rebuilt the entire split at simulation ID level with CF-stratified quartile binning. Also force-assigned 4 geometrically-extreme simulations (IDs 3, 4, 5, 24) to training since they represent configurations far outside the normal population.  
**Result:** Validation R² honestly dropped from 0.998 → 0.87. That 0.87 is trustworthy. The 0.998 was a lie.  
**Takeaway you say on camera:** *"Lesson learned — when your data has group structure, `train_test_split` is silently wrong. Always validate on groups the model has never seen, not just rows."*

---

## Razorpay Open Category Positioning

**Frame this as:**
> *"An AI system that makes industrial simulation 10,000x cheaper and faster — democratising access to physics-accurate engineering design tools."*

**Keywords to use:**
- Physics-informed ML
- Industrial AI / Manufacturing AI  
- Real-time simulation surrogate
- CAD-to-ML pipeline (Fourier shape encoding)
- Open-source democratisation of HPC compute

**Why this beats typical hackathon projects:**
- It's **real data from a real company** (not a Kaggle dataset)
- It solves a **real cost problem** (HPC compute → free prediction)
- The **Fourier shape encoding** of CAD geometry is non-trivial and not commonly seen
- The **2AM bug story is genuine** — you can point to the exact code change that killed the inflated metrics
- It's **deployed and working** — not a Jupyter notebook, but a full multi-tab Streamlit app
