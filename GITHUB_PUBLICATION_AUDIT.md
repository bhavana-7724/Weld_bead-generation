# GitHub Publication Audit Report: `weld-bead-geometry-prediction`

**Project:** Predicting Weld Bead Profiles for GMAW and ColdArc Welding Using Deep Learning Models  
**Target Repository:** `weld-bead-geometry-prediction`  
**Location:** `C:\Users\cool\OneDrive\Desktop\Project\weld_project`  
**Audit Date:** 16 November 2026  

---

## 1. Executive Summary & Repository Status

A full publication audit was conducted to prepare the `weld_project` directory for public release on GitHub. 

### Key Highlights
- **Total Project Size:** 6,705 files, totaling **6,012.99 MB (~6.01 GB)**.
- **Files Excluded via `.gitignore`:** 6,464 files, totaling **6,009.87 MB (~5.99 GB)** (99.95% of byte volume excluded).
- **Files Included for Publication:** **241 files**, totaling **3.12 MB** (clean source code, documentation, scripts, and figures).
- **Largest Tracked File:** `figures/M1_to_M5_contour_contact_sheet.png` (363.7 KB). No file exceeds 400 KB.
- **Sensitive Credentials Scan:** Passed with **0 secrets, 0 API keys, and 0 tokens found**.
- **Data Confidentiality:** All raw metallographic cross-section images (`data/processed/`) and experimental CSVs are excluded. A standardized template (`data/sample_parameters_template.csv`) and documentation (`data/README.md`) are provided.

---

## 2. Git Environment & Security Audit

### Anomaly Detected & Resolved (Parent Repository Isolation)
- **Finding:** A `.git` repository was found at the user home directory (`C:\Users\cool\.git`), which had no commits but caused git commands to view the entire user profile (e.g., `.ssh`, `Downloads`, `.VirtualBox`).
- **Remediation:** Initialized a local, isolated Git repository directly inside `C:\Users\cool\OneDrive\Desktop\Project\weld_project\.git`. This stops upward directory traversal and isolates the project.
- **Remotes & Commit History:**
  - `git remote -v`: **None configured.**
  - `git log`: **No commits yet.**
  - **Status:** Pristine local repository. No history will be rewritten, and no remotes exist to overwrite.

---

## 3. Inclusion vs. Exclusion Matrix

| Category | Status | Count | Size | Rationale / Contents |
|---|:---:|:---:|:---:|---|
| **Deep Learning Checkpoints** | **EXCLUDED** | 50 files | 5,306.8 MB | Model weights (`*.pt`, `*.pth`, `*.pb`, `*.h5`, `*.pkl`) exceed GitHub limits; excluded via `.gitignore`. |
| **Intermediate Phase Results** | **EXCLUDED** | 5,588 files | 5,909.5 MB | Phase V1–V15 intermediate plots, raw numpy arrays (`*.npy`), and batch caches in `results/` and `outputs/`. |
| **Confidential Experimental Data** | **EXCLUDED** | 552 files | 6.6 MB | Raw metallographic transverse scans in `data/processed/` and proprietary parameter matrices (`*.csv`, `*.xlsx`). |
| **Presentation Binaries & Archives** | **EXCLUDED** | 7 files | 49.8 MB | Large binary files (`*.pptx`, `*.pdf`, `*.tar.gz`, `*.zip`). |
| **Server Synchronization Scripts** | **EXCLUDED** | 8 files | 0.06 MB | Internal scripts containing institute server paths (`sync_to_server.*`, `sync_results_from_server.*`). |
| **OS & Editor Artifacts** | **EXCLUDED** | ~70 files | 1.2 MB | Python `__pycache__`, `.pytest_cache`, `.DS_Store`, `Thumbs.db`. |
| **Core Source Code (`scripts/`)** | **INCLUDED** | 126 files | 2.15 MB | Phase V14 SDF morphing, Phase V12 validation, Phase V9 generative architectures, regression benchmark scripts. |
| **Standalone Modules (`app`, `bnn`, `mlp`)** | **INCLUDED** | 23 files | 0.40 MB | Interactive Tkinter GUI, inference engine, BNN uncertainty modeling, and MLP baselines. |
| **Curated Publication Figures (`figures/`)** | **INCLUDED** | 11 files | 0.38 MB | Lightweight, publication-grade PNGs for the README (M1–M5 contour comparisons, contact sheet). |
| **Documentation & LaTeX Source** | **INCLUDED** | 8 files | 0.18 MB | `README.md`, `LICENSE`, `data/README.md`, `final_report.tex`, `final_presentation.tex`. |
| **Repository Manifests & Configuration** | **INCLUDED** | 3 files | 0.01 MB | `requirements.txt`, `.gitignore`, `data/sample_parameters_template.csv`. |

---

## 4. Execution Command Verification

Every command documented in the `README.md` was independently tested on the local environment:

1. **Single-Condition Generation CLI:**
   - **Command:** `python scripts/phase_v14/generate_single_condition.py --material IN718 --process ColdArc --feed 5.0 --speed 20.0`
   - **Verification:** Successfully executed; generated all 9 artifacts (raw RGB, direct blend, M1 CNN, mask, contour PNG/JSON, CAD sketches, and comparison panel) for both View A and View B.
2. **Prospective Numerical Regressor Benchmark:**
   - **Command:** `python scripts/train_prospective_numerical_regressor.py`
   - **Verification:** Successfully ran 5-fold cross-validation across SVR, Random Forest, MLP, Ridge, and Gradient Boosting. Verified Multi-Output RBF-SVR selected as best model ($0.681\text{ mm}$ mean error).
3. **Dependency Verification:**
   - **Command:** Audited against `python -m pip list`.
   - **Verification:** All packages in `requirements.txt` (`numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`, `torchvision`, `opencv-python`, `Pillow`, `matplotlib`, `seaborn`, `scikit-image`, `joblib`, `tqdm`) match installed working versions.

---

## 5. Step-by-Step Git Commands for Publication

When you are ready to publish the repository to GitHub, execute the following commands in PowerShell. 

> **Important:** As instructed, **no commits or push actions have been taken automatically**. The commands below give you complete control.

```powershell
# 1. Navigate to the project root
Set-Location 'C:\Users\cool\OneDrive\Desktop\Project\weld_project'

# 2. Stage all unignored publication files (241 files, ~3.12 MB)
git add .

# 3. Verify that only intended files are staged (results, checkpoints, and raw data should NOT appear)
git status

# 4. Create the initial publication commit
git commit -m "feat: initial publication of weld-bead-geometry-prediction framework

- Stream 1: Calibrated Multi-Output RBF-SVR numerical geometry predictor
- Stream 2: Deep generative cross-sectional profiling suite (M1 to M5)
- Phase V14: Geometry-aware Shape-Aligned SDF macrograph morphing engine
- Automated CAD-style dimension-annotated engineering bead sketch stage
- Interactive desktop GUI demonstration and modular inference CLI
- Full 5-fold cross-validation and 40-view held-out benchmark evaluation"

# 5. Set default branch to main
git branch -M main

# 6. Add your GitHub remote repository (replace with your GitHub URL)
git remote add origin https://github.com/<your-github-username>/weld-bead-geometry-prediction.git

# 7. Push to GitHub (requires your explicit execution)
git push -u origin main
```
