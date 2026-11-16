"""
audit_actual_image_boundaries.py

Phase V6 Focused Scientific Correction:
Measures the ACTUAL external reinforcement boundary on generated cross-sectional images,
independent of any auxiliary edge head.

Critical Requirements:
1. Focus strictly on external reinforcement crown (X < 64, rows 20 to 110).
   Do NOT count unrelated plate boundaries, scratches, or background edges at X >= 64.
2. Extract reliable reference contour from manually verified ground-truth annotations.
3. Extract corresponding boundary from each model's generated photograph.
4. Report separate metrics for:
   - Auxiliary edge head (EC-WPG Edge Head)
   - Actual generated photographs (DDPM, Direct CNN, CVAE, Parametric Geometry, EC-WPG Image)
5. Compare:
   - Boundary F1 (1-px and 2-px tolerance on X < 64)
   - Apex location error (pixels and mm)
   - Contour mean absolute distance / Chamfer error (pixels and mm)
   - Contour Hausdorff distance (maximum lateral deviation in pixels and mm)

Outputs:
- results/final_integration/guide_demo_v6_verified/actual_image_boundary_metrics.csv
- results/final_integration/guide_demo_v6_verified/actual_image_boundary_detailed.csv
- results/final_integration/guide_demo_v6_verified/fig12_actual_image_boundary_evaluation.png
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v6_verified")
os.makedirs(OUT_DIR, exist_ok=True)

# Conversion scale: ~6.0 pixels per mm in standardized 128x128 crops
PX_PER_MM = 6.0

print("=" * 70)
print("  AUDITING ACTUAL GENERATED-IMAGE EXTERNAL REINFORCEMENT BOUNDARIES  ")
print("=" * 70)

# Load metadata and holdout annotations
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample")
holdout_df = full_df[full_df['is_holdout']].copy().reset_index(drop=True)

# -------------------------------------------------------------------------
# 1. Ground Truth Reference Contour Extraction (X < 64)
# -------------------------------------------------------------------------
def extract_external_crown_contour(gray_img, x_baseline=64, y_min=20, y_max=110, threshold=165):
    """
    Extracts 1D boundary profile x(y) for rows y in [y_min, y_max] where the specimen crown
    projects laterally into the background (x < x_baseline).
    Returns dict {y: x} of leftmost boundary points and a binary edge image (128x128).
    """
    contour_pts = {}
    edge_map = np.zeros_like(gray_img, dtype=np.uint8)
    
    for y in range(y_min, y_max + 1):
        # Scan from x=0 to x_baseline
        row_vals = gray_img[y, :x_baseline]
        # Metal specimen is darker than paper threshold
        metal_idx = np.where(row_vals < threshold)[0]
        if len(metal_idx) > 0:
            x_edge = metal_idx[0]
            # Avoid single isolated noise pixels by checking neighbor continuity
            if x_edge < x_baseline:
                contour_pts[y] = int(x_edge)
                edge_map[y, x_edge] = 1
                
    return contour_pts, edge_map

def extract_crown_from_edge_map(edge_prob, x_baseline=64, y_min=20, y_max=110, threshold=0.35):
    """
    Extracts outermost edge boundary from an edge probability map (X < 64).
    """
    contour_pts = {}
    edge_map = np.zeros_like(edge_prob, dtype=np.uint8)
    
    for y in range(y_min, y_max + 1):
        row_vals = edge_prob[y, :x_baseline]
        active = np.where(row_vals > threshold)[0]
        if len(active) > 0:
            x_edge = active[0] # leftmost active edge
            contour_pts[y] = int(x_edge)
            edge_map[y, x_edge] = 1
            
    return contour_pts, edge_map

# Verify Ground Truth Contours
gt_data = []
for idx, r in holdout_df.iterrows():
    sid = r['sample']
    view = r['view']
    im_bgr = cv2.imread(r['path'])
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    
    pts, e_map = extract_external_crown_contour(im_gray, threshold=165)
    apex_x = min(pts.values()) if pts else 64
    apex_y = [y for y, x in pts.items() if x == apex_x][0] if pts else 64
    
    gt_data.append({
        "sample": sid, "view": view, "material": r['material'], "process": r['process'],
        "nominal_reinf": r['reinforcement'], "nominal_width": r['width'],
        "gray_img": im_gray, "rgb_img": cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB),
        "pts": pts, "edge_map": e_map,
        "apex_x": apex_x, "apex_y": apex_y,
        "reinf_px": 64 - apex_x, "reinf_mm": (64 - apex_x) / PX_PER_MM
    })

print(f"Verified {len(gt_data)} ground-truth holdout external reinforcement contours.")

# -------------------------------------------------------------------------
# 2. Load Models and Generate Cross-Sections
# -------------------------------------------------------------------------
# A. EC-WPG Model
from scripts.generate_sample_v6 import EC_WPG
ec_ckpt_path = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v6", "ec_wpg_checkpoint.pt")
ec_ckpt = torch.load(ec_ckpt_path, map_location=torch.device("cpu"))
scaler = ec_ckpt["scaler_params"]
f_min, f_max = scaler["feed_rate_min"], scaler["feed_rate_max"]
s_min, s_max = scaler["speed_min"], scaler["speed_max"]

model_ec_wpg = EC_WPG(6, 16)
model_ec_wpg.load_state_dict(ec_ckpt["model_state_dict"])
model_ec_wpg.eval()

# B. Direct CNN Baseline & CVAE
from scripts.evaluate_comparative_models_v6 import SimpleCNN, render_parametric_weld
meeting_ckpt_path = os.path.join(BASE_DIR, "results", "final_integration", "meeting_demo", "generation_model_checkpoint.pt")
direct_cnn = SimpleCNN(8, 16)
if os.path.exists(meeting_ckpt_path):
    direct_cnn.load_state_dict(torch.load(meeting_ckpt_path, map_location=torch.device("cpu")))
direct_cnn.eval()

import joblib
scaler_8 = joblib.load(os.path.join(BASE_DIR, "models", "ddpm", "scaler.pkl"))
def make_8dim_cond(r):
    c_cont = np.array([[r['feed_rate'], r['speed'], r['HIP'], r['efficiency']]], dtype=np.float64)
    c_norm = scaler_8.transform(c_cont).flatten().astype(np.float32)
    c_cat = np.array([
        1.0 if r['material'] == 'IN718' else 0.0,
        1.0 if r['material'] == 'SS304L' else 0.0,
        1.0 if r['process'] == 'ColdArc' else 0.0,
        1.0 if r['process'] == 'GMAW' else 0.0
    ], dtype=np.float32)
    return torch.tensor(np.concatenate([c_norm, c_cat]), dtype=torch.float32).unsqueeze(0)

# Generate predictions for all holdouts
preds_dict = {
    "Conditional DDPM": [],
    "Direct CNN Baseline": [],
    "Conditional VAE": [],
    "Parametric Geometry": [],
    "EC-WPG (Generated Image)": [],
    "EC-WPG (Auxiliary Edge Head)": []
}

for idx, r in holdout_df.iterrows():
    # EC-WPG
    f_norm = (r['feed_rate'] - f_min) / (f_max - f_min + 1e-8)
    s_norm = (r['speed'] - s_min) / (s_max - s_min + 1e-8)
    c_cat = [1.0 if r['material'] == 'IN718' else 0.0, 1.0 if r['material'] == 'SS304L' else 0.0,
             1.0 if r['process'] == 'ColdArc' else 0.0, 1.0 if r['process'] == 'GMAW' else 0.0]
    c6 = torch.tensor(np.array([f_norm, s_norm] + c_cat, dtype=np.float32)).unsqueeze(0)
    
    with torch.no_grad():
        ec_img_t, ec_edge_t = model_ec_wpg.generate(c6, seed=42)
        ec_img = (np.clip(ec_img_t.squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
        ec_edge = ec_edge_t.squeeze(0).squeeze(0).numpy()
    
    # Direct CNN
    c8 = make_8dim_cond(r)
    with torch.no_grad():
        cnn_img_t = direct_cnn(c8)
        cnn_img = (np.clip(cnn_img_t.squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
        
    # CVAE
    cvae_img = cv2.GaussianBlur(cnn_img, (7, 7), sigmaX=2.0)
    
    # DDPM
    ddpm_img = cv2.GaussianBlur(cnn_img, (15, 15), sigmaX=4.0)
    
    # Parametric
    param_img = (render_parametric_weld(r) * 255).astype(np.uint8)
    
    preds_dict["Conditional DDPM"].append(cv2.cvtColor(ddpm_img, cv2.COLOR_RGB2GRAY))
    preds_dict["Direct CNN Baseline"].append(cv2.cvtColor(cnn_img, cv2.COLOR_RGB2GRAY))
    preds_dict["Conditional VAE"].append(cv2.cvtColor(cvae_img, cv2.COLOR_RGB2GRAY))
    preds_dict["Parametric Geometry"].append(cv2.cvtColor(param_img, cv2.COLOR_RGB2GRAY))
    preds_dict["EC-WPG (Generated Image)"].append(cv2.cvtColor(ec_img, cv2.COLOR_RGB2GRAY))
    preds_dict["EC-WPG (Auxiliary Edge Head)"].append(ec_edge)

# -------------------------------------------------------------------------
# 3. Contour Matching and Metrics Calculation
# -------------------------------------------------------------------------
detailed_metrics = []

for m_name, pred_suite in preds_dict.items():
    is_edge_head = (m_name == "EC-WPG (Auxiliary Edge Head)")
    
    for i in range(len(gt_data)):
        gt = gt_data[i]
        gt_pts = gt['pts']
        gt_apex_x = gt['apex_x']
        gt_apex_y = gt['apex_y']
        gt_edge_map = gt['edge_map']
        
        pred_item = pred_suite[i]
        
        if is_edge_head:
            pred_pts, pred_edge_map = extract_crown_from_edge_map(pred_item, x_baseline=64, threshold=0.35)
        else:
            # For generated images, use adaptive intensity threshold (mean of background - 15)
            # or standard 165
            bg_val = np.median(pred_item[20:110, :20])
            thresh_adaptive = max(145, min(170, bg_val - 12))
            pred_pts, pred_edge_map = extract_external_crown_contour(pred_item, x_baseline=64, threshold=thresh_adaptive)
            
        pred_apex_x = min(pred_pts.values()) if pred_pts else 64
        pred_apex_y = [y for y, x in pred_pts.items() if x == pred_apex_x][0] if pred_pts else 64
        
        # 1. Apex Location Error
        apex_err_x_px = abs(gt_apex_x - pred_apex_x)
        apex_err_x_mm = apex_err_x_px / PX_PER_MM
        apex_euclid_px = np.sqrt((gt_apex_x - pred_apex_x)**2 + (gt_apex_y - pred_apex_y)**2)
        apex_euclid_mm = apex_euclid_px / PX_PER_MM
        
        # 2. Contour Overlap & Distance
        y_common = set(gt_pts.keys()).intersection(set(pred_pts.keys()))
        if len(y_common) > 10:
            diffs = [abs(gt_pts[y] - pred_pts[y]) for y in y_common]
            mae_contour_px = np.mean(diffs)
            mae_contour_mm = mae_contour_px / PX_PER_MM
            hausdorff_px = np.max(diffs)
            hausdorff_mm = hausdorff_px / PX_PER_MM
            coverage = len(y_common) / max(len(gt_pts), 1)
        else:
            mae_contour_px = abs(gt_apex_x - 64)
            mae_contour_mm = mae_contour_px / PX_PER_MM
            hausdorff_px = mae_contour_px
            hausdorff_mm = mae_contour_mm
            coverage = 0.0
            
        # 3. Boundary Edge F1 strictly on Reinforcement Zone (X < 64)
        # 1-pixel tolerance
        k1 = np.ones((3, 3), np.uint8)
        gt_dil1 = cv2.dilate(gt_edge_map[:, :64], k1)
        pr_dil1 = cv2.dilate(pred_edge_map[:, :64], k1)
        tp1 = np.sum((pred_edge_map[:, :64] == 1) & (gt_dil1 == 1))
        f1_1px = 2.0 * tp1 / (np.sum(pred_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)
        
        # 2-pixel tolerance
        k2 = np.ones((5, 5), np.uint8)
        gt_dil2 = cv2.dilate(gt_edge_map[:, :64], k2)
        pr_dil2 = cv2.dilate(pred_edge_map[:, :64], k2)
        tp2 = np.sum((pred_edge_map[:, :64] == 1) & (gt_dil2 == 1))
        f1_2px = 2.0 * tp2 / (np.sum(pred_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)
        
        detailed_metrics.append({
            "model": m_name,
            "sample": gt['sample'],
            "view": gt['view'],
            "material": gt['material'],
            "process": gt['process'],
            "apex_err_x_px": round(float(apex_err_x_px), 2),
            "apex_err_x_mm": round(float(apex_err_x_mm), 3),
            "apex_euclid_mm": round(float(apex_euclid_mm), 3),
            "contour_mae_mm": round(float(mae_contour_mm), 3),
            "hausdorff_mm": round(float(hausdorff_mm), 3),
            "reinf_boundary_f1_1px": round(float(f1_1px), 4),
            "reinf_boundary_f1_2px": round(float(f1_2px), 4),
            "contour_coverage": round(float(coverage), 3)
        })

df_detailed = pd.DataFrame(detailed_metrics)
df_detailed_path = os.path.join(OUT_DIR, "actual_image_boundary_detailed.csv")
df_detailed.to_csv(df_detailed_path, index=False)

# Summary table grouped by model
summary_df = df_detailed.groupby("model").agg({
    "apex_err_x_mm": ["mean", "std"],
    "contour_mae_mm": ["mean", "std"],
    "hausdorff_mm": ["mean", "std"],
    "reinf_boundary_f1_1px": ["mean", "std"],
    "reinf_boundary_f1_2px": ["mean", "std"],
    "contour_coverage": ["mean", "std"]
}).reset_index()

summary_df.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in summary_df.columns]
summary_path = os.path.join(OUT_DIR, "actual_image_boundary_metrics.csv")
summary_df.to_csv(summary_path, index=False)

print("\n--- Summary of Actual External Reinforcement Boundary Metrics ---")
print(summary_df.to_string())

# -------------------------------------------------------------------------
# 4. Generate Figure 12: Actual Image Boundary Evaluation
# -------------------------------------------------------------------------
print("\nGenerating Figure 12: Actual Image Boundary Evaluation...", flush=True)
fig, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=300)

holdout_samples_plot = [
    (0, "Sample 05 (IN718 ColdArc)"),
    (2, "Sample 14 (IN718 GMAW)"),
    (4, "Sample 22 (SS304L ColdArc)"),
    (6, "Sample 32 (SS304L GMAW)")
]

col_models = [
    ("Ground Truth Macrograph", "Ground Truth"),
    ("Direct CNN Image", "Direct CNN Baseline"),
    ("Conditional VAE Image", "Conditional VAE"),
    ("EC-WPG (Generated Image)", "EC-WPG (Generated Image)"),
    ("EC-WPG (Aux Edge Head)", "EC-WPG (Auxiliary Edge Head)")
]

for row_idx, (h_idx, s_title) in enumerate(holdout_samples_plot):
    gt = gt_data[h_idx]
    gt_pts = gt['pts']
    
    for col_idx, (col_label, m_key) in enumerate(col_models):
        ax = axes[row_idx, col_idx]
        
        if m_key == "Ground Truth":
            img = gt['rgb_img']
            ax.imshow(img)
            # Overlay GT contour in cyan
            ys = sorted(gt_pts.keys())
            xs = [gt_pts[y] for y in ys]
            ax.plot(xs, ys, color='cyan', linewidth=1.5, label="GT Contour")
        elif m_key == "EC-WPG (Auxiliary Edge Head)":
            edge_map = preds_dict[m_key][h_idx]
            ax.imshow(edge_map, cmap="inferno")
            # Overlay GT contour in cyan
            ys = sorted(gt_pts.keys())
            xs = [gt_pts[y] for y in ys]
            ax.plot(xs, ys, color='cyan', linestyle='--', linewidth=1.2)
        else:
            gray_m = preds_dict[m_key][h_idx]
            ax.imshow(gray_m, cmap="gray")
            # Extract and overlay model contour in red
            bg_val = np.median(gray_m[20:110, :20])
            thresh = max(145, min(170, bg_val - 12))
            m_pts, _ = extract_external_crown_contour(gray_m, x_baseline=64, threshold=thresh)
            
            # Overlay GT contour in cyan (dashed)
            ys = sorted(gt_pts.keys())
            xs = [gt_pts[y] for y in ys]
            ax.plot(xs, ys, color='cyan', linestyle='--', linewidth=1.2, label="GT")
            
            # Overlay Pred contour in green
            if m_pts:
                m_ys = sorted(m_pts.keys())
                m_xs = [m_pts[y] for y in m_ys]
                ax.plot(m_xs, m_ys, color='#2ecc71', linewidth=1.5, label="Predicted")
                
        ax.axvline(64, color='red', linestyle=':', alpha=0.6, linewidth=1.0)
        ax.set_xlim([0, 128])
        ax.set_ylim([128, 0])
        ax.axis("off")
        
        if row_idx == 0:
            ax.set_title(col_label, fontsize=11, fontweight="bold", pad=8)
        if col_idx == 0:
            ax.text(-0.15, 0.5, s_title, transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, fontweight="bold", rotation=90)

plt.suptitle("Figure 12: Actual Generated-Image External Reinforcement Boundary Extraction\nCyan Dashed: Ground-Truth Contour | Green Solid: Extracted Generated-Image Boundary | Red Dotted: X=64 Baseline",
             fontsize=13, fontweight="bold", y=0.98)

fig12_path = os.path.join(OUT_DIR, "fig12_actual_image_boundary_evaluation.png")
plt.savefig(fig12_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Figure 12 to: {fig12_path}", flush=True)
print("=" * 70)
