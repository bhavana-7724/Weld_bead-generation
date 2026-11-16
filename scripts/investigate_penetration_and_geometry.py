"""
investigate_penetration_and_geometry.py

Comprehensive investigation addressing:
Problem 1: Penetration error, Sample 32 investigation, regression generalization,
           physically-motivated baselines, prospective sensitivity, robustness analysis.
Problem 2: Experimental macrograph orientation audit, alignment, aspect ratio analysis,
           fusion line contrast, and geometry-first rendering using out-of-fold predicted dimensions.
"""

import os
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v3")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36_v2.csv")

df = pd.read_csv(DATA_PATH)
print("Loaded dataset:", df.shape)

# Publication style
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12

# =============================================================================
# PART 1: PENETRATION & SAMPLE 32 INVESTIGATION
# =============================================================================
print("\n--- 1. Penetration Depth Analysis Across Quadrants ---")
quad_stats = df.groupby(['material', 'process'])['depth'].agg(['count', 'min', 'max', 'mean', 'std']).reset_index()
print(quad_stats.to_string(index=False))

# Look specifically at SS304L GMAW (Samples 28-36)
ss_gmaw = df[(df['material'] == 'SS304L') & (df['process'] == 'GMAW')].copy()
print("\nSS304L GMAW experiments:")
print(ss_gmaw[['sample', 'feed_rate', 'speed', 'HIP', 'efficiency', 'width', 'depth', 'reinforcement']].to_string(index=False))

# Baseline Models to compare on 5-fold CV and Holdouts [5, 14, 22, 32]:
# 1. Quadrant Mean Baseline (predict mean of training quadrant)
# 2. Linear Heat Input Proxy (depth ~ feed_rate / speed)
# 3. Prospective Ridge
# 4. Prospective SVR
# 5. Prospective Proposed DNN
# 6. Prospective BNN (MC Dropout)

material_le = LabelEncoder()
process_le = LabelEncoder()
df["mat_enc"] = material_le.fit_transform(df["material"])
df["proc_enc"] = process_le.fit_transform(df["process"])

PROSP_COLS = ["feed_rate", "speed", "mat_enc", "proc_enc"]
TARGET_COLS = ["width", "depth", "reinforcement"]
HOLDOUTS = [5, 14, 22, 32]

class ProposedDNN(nn.Module):
    def __init__(self, in_features=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

class BNNMCDropout(nn.Module):
    def __init__(self, in_features=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)

def train_torch(model_cls, X_tr, y_tr, X_val, y_val, epochs=400, lr=0.01):
    model = model_cls(X_tr.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss()
    
    xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32)
    xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    
    best_loss = float('inf')
    best_w = None
    no_imp = 0
    
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(xt), yt)
        loss.backward()
        opt.step()
        
        model.eval()
        with torch.no_grad():
            vl = crit(model(xv), yv).item()
        if vl < best_loss:
            best_loss = vl
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 30:
                break
    if best_w:
        model.load_state_dict(best_w)
    return model

# Run Benchmark comparing All Models with and without Sample 32
def evaluate_framework(exclude_sample_32=False):
    curr_df = df[df['sample'] != 32].copy().reset_index(drop=True) if exclude_sample_32 else df.copy()
    curr_holdouts = [h for h in HOLDOUTS if h in curr_df['sample'].values]
    
    train_pool = curr_df[~curr_df['sample'].isin(curr_holdouts)].copy().reset_index(drop=True)
    test_pool = curr_df[curr_df['sample'].isin(curr_holdouts)].copy().reset_index(drop=True)
    
    X_tr_full = train_pool[PROSP_COLS].values.astype(np.float32)
    y_tr_full = train_pool[TARGET_COLS].values.astype(np.float32)
    X_te = test_pool[PROSP_COLS].values.astype(np.float32)
    y_te = test_pool[TARGET_COLS].values.astype(np.float32)
    
    # 5-fold CV on train_pool
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    models = {
        "QuadrantMean": "baseline_quadrant",
        "HeatInputRatio": "baseline_ratio",
        "Ridge": MultiOutputRegressor(Ridge(alpha=1.0)),
        "SVR": MultiOutputRegressor(SVR(C=10.0, epsilon=0.1)),
        "Proposed_DNN": "torch_dnn",
        "BNN_MCDO": "torch_bnn"
    }
    
    cv_records = []
    
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_tr_full), 1):
        X_tr, X_va = X_tr_full[tr_idx], X_tr_full[va_idx]
        y_tr, y_va = y_tr_full[tr_idx], y_tr_full[va_idx]
        
        # Scaling continuous (cols 0, 1)
        sx = StandardScaler()
        X_tr_s = X_tr.copy()
        X_tr_s[:, :2] = sx.fit_transform(X_tr[:, :2])
        X_va_s = X_va.copy()
        X_va_s[:, :2] = sx.transform(X_va[:, :2])
        
        sy = StandardScaler()
        y_tr_s = sy.fit_transform(y_tr)
        y_va_s = sy.transform(y_va)
        
        for name, m_obj in models.items():
            if m_obj == "baseline_quadrant":
                # Predict mean of (mat, proc) quadrant from training fold
                tr_sub = train_pool.iloc[tr_idx]
                means = tr_sub.groupby(['mat_enc', 'proc_enc'])[TARGET_COLS].mean()
                y_pred = []
                for row in X_va:
                    m_enc, p_enc = row[2], row[3]
                    if (m_enc, p_enc) in means.index:
                        y_pred.append(means.loc[(m_enc, p_enc)].values)
                    else:
                        y_pred.append(tr_sub[TARGET_COLS].mean().values)
                y_pred = np.array(y_pred)
            elif m_obj == "baseline_ratio":
                # Linear model on (F / S) ratio
                ratio_tr = (X_tr[:, 0] / X_tr[:, 1]).reshape(-1, 1)
                ratio_va = (X_va[:, 0] / X_va[:, 1]).reshape(-1, 1)
                lr_ratio = MultiOutputRegressor(LinearRegression())
                lr_ratio.fit(ratio_tr, y_tr)
                y_pred = lr_ratio.predict(ratio_va)
            elif m_obj == "torch_dnn":
                net = train_torch(ProposedDNN, X_tr_s, y_tr_s, X_va_s, y_va_s, epochs=400, lr=0.01)
                net.eval()
                with torch.no_grad():
                    pred_s = net(torch.tensor(X_va_s, dtype=torch.float32)).numpy()
                y_pred = sy.inverse_transform(pred_s)
            elif m_obj == "torch_bnn":
                net = train_torch(BNNMCDropout, X_tr_s, y_tr_s, X_va_s, y_va_s, epochs=400, lr=0.008)
                net.train()
                mc_p = []
                tin = torch.tensor(X_va_s, dtype=torch.float32)
                for _ in range(50):
                    with torch.no_grad():
                        mc_p.append(net(tin).numpy())
                y_pred = sy.inverse_transform(np.mean(mc_p, axis=0))
            else:
                m = m_obj
                m.fit(X_tr_s, y_tr)
                y_pred = m.predict(X_va_s)
                
            w_mae = mean_absolute_error(y_va[:, 0], y_pred[:, 0])
            d_mae = mean_absolute_error(y_va[:, 1], y_pred[:, 1])
            r_mae = mean_absolute_error(y_va[:, 2], y_pred[:, 2])
            avg_mae = (w_mae + d_mae + r_mae) / 3.0
            
            cv_records.append({
                "Excluded_Sample_32": exclude_sample_32,
                "Fold": fold,
                "Model": name,
                "Width_MAE": w_mae,
                "Depth_MAE": d_mae,
                "Reinf_MAE": r_mae,
                "Avg_MAE": avg_mae
            })
            
    df_cv = pd.DataFrame(cv_records)
    
    # Train on Full Train Pool -> Test on Holdouts
    sx_f = StandardScaler()
    X_tr_fs = X_tr_full.copy()
    X_tr_fs[:, :2] = sx_f.fit_transform(X_tr_full[:, :2])
    X_te_s = X_te.copy()
    X_te_s[:, :2] = sx_f.transform(X_te[:, :2])
    
    sy_f = StandardScaler()
    y_tr_fs = sy_f.fit_transform(y_tr_full)
    
    val_cut = int(0.15 * len(X_tr_fs))
    X_t_sub, X_v_sub = X_tr_fs[val_cut:], X_tr_fs[:val_cut]
    y_t_sub_s, y_v_sub_s = y_tr_fs[val_cut:], y_tr_fs[:val_cut]
    
    te_records = []
    oof_predictions = {}
    
    for name, m_obj in models.items():
        if m_obj == "baseline_quadrant":
            means = train_pool.groupby(['mat_enc', 'proc_enc'])[TARGET_COLS].mean()
            y_pred = []
            for row in X_te:
                m_enc, p_enc = row[2], row[3]
                if (m_enc, p_enc) in means.index:
                    y_pred.append(means.loc[(m_enc, p_enc)].values)
                else:
                    y_pred.append(train_pool[TARGET_COLS].mean().values)
            y_pred = np.array(y_pred)
        elif m_obj == "baseline_ratio":
            ratio_tr = (X_tr_full[:, 0] / X_tr_full[:, 1]).reshape(-1, 1)
            ratio_te = (X_te[:, 0] / X_te[:, 1]).reshape(-1, 1)
            lr_ratio = MultiOutputRegressor(LinearRegression())
            lr_ratio.fit(ratio_tr, y_tr_full)
            y_pred = lr_ratio.predict(ratio_te)
        elif m_obj == "torch_dnn":
            net = train_torch(ProposedDNN, X_t_sub, y_t_sub_s, X_v_sub, y_v_sub_s, epochs=400, lr=0.01)
            net.eval()
            with torch.no_grad():
                pred_s = net(torch.tensor(X_te_s, dtype=torch.float32)).numpy()
            y_pred = sy_f.inverse_transform(pred_s)
        elif m_obj == "torch_bnn":
            net = train_torch(BNNMCDropout, X_t_sub, y_t_sub_s, X_v_sub, y_v_sub_s, epochs=400, lr=0.008)
            net.train()
            mc_p = []
            tin = torch.tensor(X_te_s, dtype=torch.float32)
            for _ in range(50):
                with torch.no_grad():
                    mc_p.append(net(tin).numpy())
            y_pred = sy_f.inverse_transform(np.mean(mc_p, axis=0))
        else:
            m = m_obj
            m.fit(X_tr_fs, y_tr_full)
            y_pred = m.predict(X_te_s)
            
        oof_predictions[name] = y_pred
        
        for i, h in enumerate(curr_holdouts):
            w_err = abs(y_te[i, 0] - y_pred[i, 0])
            d_err = abs(y_te[i, 1] - y_pred[i, 1])
            r_err = abs(y_te[i, 2] - y_pred[i, 2])
            avg_e = (w_err + d_err + r_err) / 3.0
            
            te_records.append({
                "Excluded_Sample_32": exclude_sample_32,
                "Model": name,
                "Sample": h,
                "Alloy": test_pool.iloc[i]['material'],
                "Process": test_pool.iloc[i]['process'],
                "Width_Actual": y_te[i, 0], "Width_Pred": y_pred[i, 0], "Width_Error": w_err,
                "Depth_Actual": y_te[i, 1], "Depth_Pred": y_pred[i, 1], "Depth_Error": d_err,
                "Reinf_Actual": y_te[i, 2], "Reinf_Pred": y_pred[i, 2], "Reinf_Error": r_err,
                "Avg_Error": avg_e
            })
            
    df_te = pd.DataFrame(te_records)
    return df_cv, df_te, oof_predictions

print("\n--- Running Evaluation WITH Sample 32 (Standard) ---")
cv_with_32, te_with_32, preds_with_32 = evaluate_framework(exclude_sample_32=False)

print("\n--- Running Diagnostic Evaluation WITHOUT Sample 32 ---")
cv_no_32, te_no_32, _ = evaluate_framework(exclude_sample_32=True)

# Combine and save
cv_comp = pd.concat([cv_with_32, cv_no_32], ignore_index=True)
te_comp = pd.concat([te_with_32, te_no_32], ignore_index=True)

cv_comp.to_csv(os.path.join(OUT_DIR, "penetration_cv_comparison.csv"), index=False)
te_comp.to_csv(os.path.join(OUT_DIR, "penetration_holdout_comparison.csv"), index=False)

# Aggregate CV summary
cv_summary = cv_comp.groupby(['Excluded_Sample_32', 'Model'])[['Width_MAE', 'Depth_MAE', 'Reinf_MAE', 'Avg_MAE']].mean().reset_index()
print("\n=== CV SUMMARY (Comparison With vs Without Sample 32) ===")
print(cv_summary.to_string(index=False))

# Holdout summary
te_summary = te_comp.groupby(['Excluded_Sample_32', 'Model'])[['Width_Error', 'Depth_Error', 'Reinf_Error', 'Avg_Error']].mean().reset_index()
print("\n=== HOLDOUT SUMMARY (Comparison With vs Without Sample 32) ===")
print(te_summary.to_string(index=False))

# =============================================================================
# FIGURE 10: Sample 32 Penetration Investigation & Thermodynamic Mode Shift
# =============================================================================
print("\nGenerating Figure 10: Sample 32 Penetration Investigation...")
fig10, axs = plt.subplots(2, 2, figsize=(15, 12))

# Subplot (a): Depth vs Wire Feed Rate across Travel Speeds in SS304L GMAW
sns.lineplot(
    data=ss_gmaw, x="feed_rate", y="depth", hue="speed", marker="o", markersize=10,
    palette="tab10", ax=axs[0, 0], lw=2
)
# Highlight Sample 32
s32_pt = ss_gmaw[ss_gmaw['sample'] == 32].iloc[0]
axs[0, 0].scatter([s32_pt['feed_rate']], [s32_pt['depth']], color="red", s=250, zorder=5, edgecolors="black", lw=2)
axs[0, 0].annotate(
    "Sample 32 (D=5.25 mm)\nExtreme Non-Monotonic Spike",
    xy=(s32_pt['feed_rate'], s32_pt['depth']),
    xytext=(s32_pt['feed_rate'] - 0.35, s32_pt['depth'] + 0.4),
    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
    fontweight='bold', color='darkred'
)
axs[0, 0].set_title("(a) Penetration Depth vs. Wire Feed Rate in SS304L GMAW", fontweight='bold')
axs[0, 0].set_xlabel("Wire Feed Rate (m/min)")
axs[0, 0].set_ylabel("Penetration Depth (mm)")
axs[0, 0].grid(True, linestyle="--", alpha=0.5)

# Subplot (b): Penetration vs Heat Input Parameter (HIP) across All 36 Experiments
sns.scatterplot(
    data=df, x="HIP", y="depth", hue="process", style="material", s=130,
    palette={"ColdArc": "#2ca02c", "GMAW": "#d62728"}, ax=axs[0, 1]
)
axs[0, 1].scatter([s32_pt['HIP']], [s32_pt['depth']], color="red", s=250, zorder=5, edgecolors="black", lw=2)
axs[0, 1].annotate(
    "Sample 32 (Global Outlier)\nModerate HIP (115.7 J/mm)\nMaximum Depth (5.25 mm)",
    xy=(s32_pt['HIP'], s32_pt['depth']),
    xytext=(s32_pt['HIP'] - 35, s32_pt['depth'] - 0.8),
    arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8),
    fontweight='bold', color='darkred'
)
axs[0, 1].set_title("(b) Penetration Depth vs. Heat Input across Dataset", fontweight='bold')
axs[0, 1].set_xlabel("Heat Input Parameter HIP (J/mm)")
axs[0, 1].set_ylabel("Penetration Depth (mm)")
axs[0, 1].grid(True, linestyle="--", alpha=0.5)

# Subplot (c): Depth Prediction Error on Holdouts (BNN, DNN, SVR, Baselines)
te_plot_df = te_comp[te_comp['Excluded_Sample_32'] == False].copy()
sns.barplot(
    data=te_plot_df, x="Sample", y="Depth_Error", hue="Model",
    palette="Set2", ax=axs[1, 0]
)
axs[1, 0].set_title("(c) Penetration Depth Error across Development Holdouts", fontweight='bold')
axs[1, 0].set_xlabel("Holdout Sample ID")
axs[1, 0].set_ylabel("Absolute Depth Error (mm)")
axs[1, 0].grid(True, linestyle="--", alpha=0.5)

# Subplot (d): Robustness Analysis (Model CV MAE With vs. Without Sample 32)
cv_pvt = cv_summary.pivot(index="Model", columns="Excluded_Sample_32", values="Depth_MAE").reset_index()
cv_pvt.columns = ["Model", "With_Sample_32", "Without_Sample_32"]
x_idx = np.arange(len(cv_pvt))
w_bar = 0.35
axs[1, 1].bar(x_idx - w_bar/2, cv_pvt["With_Sample_32"], width=w_bar, label="With Sample 32", color="#e76f51")
axs[1, 1].bar(x_idx + w_bar/2, cv_pvt["Without_Sample_32"], width=w_bar, label="Without Sample 32", color="#2a9d8f")
axs[1, 1].set_xticks(x_idx)
axs[1, 1].set_xticklabels(cv_pvt["Model"], rotation=30, ha="right")
axs[1, 1].set_ylabel("5-Fold CV Depth MAE (mm)")
axs[1, 1].set_title("(d) Depth Sensitivity to Sample 32 Inclusion in Training Pool", fontweight='bold')
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
fig10_path = os.path.join(OUT_DIR, "fig10_sample32_penetration_investigation.png")
fig10.savefig(fig10_path, dpi=300)
plt.close(fig10)
print(f"Saved: {fig10_path}")

# =============================================================================
# PART 2: WELD IMAGE ORIENTATION, ALIGNMENT & GEOMETRY-FIRST AUDIT
# =============================================================================
print("\n--- 2. Auditing Image Orientation and Physical Alignment ---")

# Let's inspect raw image zip files for actual orientation
raw_zip_in718 = os.path.join(BASE_DIR, "..", "Dataset 3(1).zip")
raw_zip_ss304 = os.path.join(BASE_DIR, "..", "Dataset 3(2).zip")

# Load Sample 5, 14, 22, 32 raw images and compare against current processed crops
holdout_raw_crops = {}

def get_raw_image(sample_id):
    zip_p = raw_zip_in718 if sample_id <= 18 else raw_zip_ss304
    prefix = "Dataset 3(1)" if sample_id <= 18 else "Dataset 3(2)"
    with zipfile.ZipFile(zip_p, 'r') as z:
        # Check candidate names
        cands = [n for n in z.namelist() if f"{sample_id}-A." in n or f"{sample_id}- A." in n or f"{sample_id}-B." in n or f"{sample_id}- B." in n]
        if not cands:
            raise FileNotFoundError(f"Sample {sample_id} not in {zip_p}")
        with z.open(cands[0]) as f:
            b = f.read()
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            return im, cands[0]

# Render Geometry-First in BOTH Orientations:
# 1. Legacy Horizontal Template (plate at horizontal line, crown upwards)
# 2. Correct Physical Orientation (plate at vertical line, base metal on left, crown to right, depth into left)
def render_parametric_profile(width_mm, depth_mm, reinf_mm, orientation="physical_vertical", canvas_size=(256, 256), scale_px_per_mm=12.0):
    h, w = canvas_size
    img = np.ones(canvas_size, dtype=np.uint8) * 170 # base metal gray
    
    w_px = width_mm * scale_px_per_mm
    d_px = depth_mm * scale_px_per_mm
    r_px = reinf_mm * scale_px_per_mm
    
    if orientation == "legacy_horizontal":
        y_surf = 140
        cx = w // 2
        half_w = int(w_px / 2.0)
        x_left = max(0, cx - half_w)
        x_right = min(w, cx + half_w)
        
        for x in range(w):
            if x < x_left or x > x_right:
                img[:y_surf, x] = 40
            else:
                rel = (x - cx) / float(half_w + 1e-5)
                y_crown = int(y_surf - r_px * (1.0 - rel**2))
                img[:y_crown, x] = 40
                img[y_crown:y_surf, x] = 215
        for x in range(x_left, x_right):
            rel = (x - cx) / float(half_w + 1e-5)
            y_pen = int(y_surf + d_px * (1.0 - rel**2))
            img[y_surf:y_pen, x] = 205
            if y_pen < h:
                img[y_pen-2:y_pen+1, x] = 90
        img[y_surf-1:y_surf+1, :x_left] = 80
        img[y_surf-1:y_surf+1, x_right:] = 80
        
    elif orientation == "physical_vertical":
        # Physical Orientation matching Raw Micrographs:
        # Plate boundary is at x = 128 (vertical line)
        # Base metal is on LEFT (x < 128)
        # Ambient mounting resin / background is on RIGHT (x > 128)
        # Crown reinforcement projects TO THE RIGHT (x > 128)
        # Penetration extends INTO THE PLATE TO THE LEFT (x < 128)
        # Bead width is along VERTICAL (Y axis), centered at cy = 128
        x_surf = 128
        cy = h // 2
        half_w = int(w_px / 2.0)
        y_top = max(0, cy - half_w)
        y_bot = min(h, cy + half_w)
        
        # Right of plate is ambient resin (dark ~40)
        img[:, x_surf:] = 40
        
        # Crown projection to the right (x > x_surf)
        for y in range(y_top, y_bot):
            rel = (y - cy) / float(half_w + 1e-5)
            x_crown = int(x_surf + r_px * (1.0 - rel**2))
            img[y, x_surf:x_crown] = 215 # weld metal crown
            if x_crown < w:
                img[y, x_crown-1:x_crown+1] = 80 # outer crown contour
                
        # Penetration to the left (x < x_surf)
        for y in range(y_top, y_bot):
            rel = (y - cy) / float(half_w + 1e-5)
            x_pen = int(x_surf - d_px * (1.0 - rel**2))
            x_pen = max(0, x_pen)
            img[y, x_pen:x_surf] = 205 # fusion zone
            if x_pen > 0:
                img[y, x_pen-1:x_pen+1] = 90 # fusion boundary line
                
        # Plate vertical surface lines above and below bead
        img[:y_top, x_surf-1:x_surf+1] = 80
        img[y_bot:, x_surf-1:x_surf+1] = 80
        
    return cv2.GaussianBlur(img, (3, 3), 0.5)

# =============================================================================
# FIGURE 11: Experimental Image Orientation & Geometry-First Audit
# =============================================================================
print("\nGenerating Figure 11: Experimental Image Orientation & Alignment...")
fig11, axes = plt.subplots(4, 5, figsize=(20, 15), dpi=300)

col_labels = [
    "Raw Experimental Photo\n(Full Frame Aspect)",
    "Current Processed Crop\n(Vertical Plate Baseline)",
    "Legacy Model E\n(Assumed Horizontal)",
    "Physical Model E\n(Out-of-Fold Predicted)",
    "Physical Model E\n(Ground Truth Dimensions)"
]

for col_idx, cl in enumerate(col_labels):
    axes[0, col_idx].text(0.5, 1.15, cl, ha='center', va='bottom', fontsize=11, fontweight='bold', transform=axes[0, col_idx].transAxes)

# Load Prospective DNN predictions from the 5-fold CV run to test out-of-fold geometry rendering!
dnn_oof_preds = preds_with_32["Proposed_DNN"]

for row_idx, h in enumerate(HOLDOUTS):
    r_info = df[df['sample'] == h].iloc[0]
    raw_im, raw_name = get_raw_image(h)
    
    # Raw thumbnail
    raw_thumb = cv2.resize(raw_im, (256, 384))
    
    # Processed crop in images_36_v3
    proc_path = os.path.join(BASE_DIR, "data", "processed", "images_36_v3", f"{h}-A.png")
    proc_im = cv2.imread(proc_path, cv2.IMREAD_GRAYSCALE)
    
    # Predicted dimensions from Proposed DNN
    w_pred, d_pred, r_pred = dnn_oof_preds[row_idx]
    
    # Actual dimensions
    w_act, d_act, r_act = r_info['width'], r_info['depth'], r_info['reinforcement']
    
    # Legacy horizontal rendering
    legacy_im = render_parametric_profile(w_act, d_act, r_act, orientation="legacy_horizontal")
    
    # Physical vertical rendering with OOF PREDICTED dimensions
    oof_phys_im = render_parametric_profile(w_pred, d_pred, r_pred, orientation="physical_vertical")
    
    # Physical vertical rendering with GROUND TRUTH dimensions
    gt_phys_im = render_parametric_profile(w_act, d_act, r_act, orientation="physical_vertical")
    
    # Plotting
    axes[row_idx, 0].imshow(cv2.cvtColor(raw_thumb, cv2.COLOR_BGR2RGB))
    axes[row_idx, 0].axis('off')
    
    axes[row_idx, 1].imshow(proc_im, cmap='gray')
    axes[row_idx, 1].axis('off')
    
    axes[row_idx, 2].imshow(legacy_im, cmap='gray')
    axes[row_idx, 2].axis('off')
    
    axes[row_idx, 3].imshow(oof_phys_im, cmap='gray')
    axes[row_idx, 3].axis('off')
    
    axes[row_idx, 4].imshow(gt_phys_im, cmap='gray')
    axes[row_idx, 4].axis('off')
    
    row_text = (
        f"Holdout {h} ({r_info['material']}-{r_info['process']})\n"
        f"Actual: W={w_act:.1f}, D={d_act:.2f}, R={r_act:.2f}\n"
        f"DNN Pred: W={w_pred:.1f}, D={d_pred:.2f}, R={r_pred:.2f}"
    )
    axes[row_idx, 0].set_ylabel(row_text, fontsize=9.5, fontweight='bold', labelpad=10)

plt.subplots_adjust(top=0.92, bottom=0.03, left=0.10, right=0.98, hspace=0.15, wspace=0.10)
fig11_path = os.path.join(OUT_DIR, "fig11_image_orientation_and_alignment.png")
fig11.savefig(fig11_path, dpi=300)
plt.close(fig11)
print(f"Saved: {fig11_path}")

print("\n--- Investigation and Figure Generation Complete! ---")
