"""
generate_v7_diagnostic_figures.py

Generates:
1. fig3_failure_cases_and_mismatches_v7.png (Visualizing top contour mismatches and failure modes)
2. fig4_cv_generalization_boxplot_v7.png (5-Fold CV distribution boxplots for all metrics)
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7")

detailed_csv = os.path.join(OUT_DIR, "EXTERNAL_CONTOUR_METRICS.csv")
df = pd.read_csv(detailed_csv)
cv_df = df[df['evaluation_suite'].str.contains("5-Fold")].copy()

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

# -------------------------------------------------------------------------
# 1. Figure 4: 5-Fold Cross-Validation Boxplots
# -------------------------------------------------------------------------
print("Generating Figure 4: 5-Fold CV Generalization Boxplots...", flush=True)
fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

models = ["Direct Conditional CNN", "Conditional VAE", "GC-WPG Prototype", "Prospective Parametric"]
colors = ['#3498db', '#9b59b6', '#2ecc71', '#e67e22']

# Subplot (0, 0): Whole SSIM
data_ssim = [cv_df[cv_df['model'] == m]['ssim'].values for m in models]
bplot1 = axes[0, 0].boxplot(data_ssim, tick_labels=["Direct CNN", "CVAE", "GC-WPG", "Parametric"],
                            patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bplot1['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0, 0].set_ylabel("SSIM (Higher is Better)", fontsize=10, fontweight='bold')
axes[0, 0].set_title("(a) Whole-Image Structural Similarity (SSIM)", fontsize=11, fontweight='bold')
axes[0, 0].grid(axis='y', linestyle=':', alpha=0.6)

# Subplot (0, 1): Weld ROI SSIM
data_roi_ssim = [cv_df[cv_df['model'] == m]['roi_ssim'].values for m in models]
bplot2 = axes[0, 1].boxplot(data_roi_ssim, tick_labels=["Direct CNN", "CVAE", "GC-WPG", "Parametric"],
                            patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[0, 1].set_ylabel("Weld-ROI SSIM (Higher is Better)", fontsize=10, fontweight='bold')
axes[0, 1].set_title("(b) Active Weld Zone Structural Similarity (ROI SSIM)", fontsize=11, fontweight='bold')
axes[0, 1].grid(axis='y', linestyle=':', alpha=0.6)

# Subplot (1, 0): Contour MAE (Pixels)
data_mae = [cv_df[cv_df['model'] == m]['contour_mae_px'].values for m in models]
bplot3 = axes[1, 0].boxplot(data_mae, tick_labels=["Direct CNN", "CVAE", "GC-WPG", "Parametric"],
                            patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bplot3['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1, 0].set_ylabel("Contour Distance Error (Pixels)", fontsize=10, fontweight='bold')
axes[1, 0].set_title("(c) External Reinforcement Contour MAE (Lower is Better)", fontsize=11, fontweight='bold')
axes[1, 0].set_ylim([0, 15])
axes[1, 0].grid(axis='y', linestyle=':', alpha=0.6)

# Subplot (1, 1): Boundary F1 (1-pixel tolerance)
data_f1 = [cv_df[cv_df['model'] == m]['reinf_boundary_f1_1px'].values for m in models]
bplot4 = axes[1, 1].boxplot(data_f1, tick_labels=["Direct CNN", "CVAE", "GC-WPG", "Parametric"],
                            patch_artist=True, medianprops=dict(color='black', linewidth=1.5))
for patch, color in zip(bplot4['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[1, 1].set_ylabel("Boundary F1 (1-px tolerance)", fontsize=10, fontweight='bold')
axes[1, 1].set_title("(d) Crown Boundary Precision (1-px Tolerance F1)", fontsize=11, fontweight='bold')
axes[1, 1].grid(axis='y', linestyle=':', alpha=0.6)

plt.suptitle("Figure 4 (Phase V7): 5-Fold Grouped Cross-Validation Metric Distributions\n"
             "All 36 Independent Physical Experiments (72 Out-of-Fold Macrograph Views)",
             fontsize=13, fontweight='bold', y=0.98)
fig4_path = os.path.join(OUT_DIR, "fig4_cv_generalization_boxplot_v7.png")
plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Figure 4 to {fig4_path}")

# -------------------------------------------------------------------------
# 2. Figure 3: Failure Cases & Mismatches
# -------------------------------------------------------------------------
print("Generating Figure 3: Mismatch & Failure Case Analysis...", flush=True)
# Find specimens with highest contour error in 5-fold CV
worst_records = cv_df.sort_values(by="contour_mae_px", ascending=False).drop_duplicates(subset=["sample", "model"]).head(4)

fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), dpi=300)
for idx, (_, r) in enumerate(worst_records.iterrows()):
    sid = int(r['sample'])
    view = r['view']
    m_name = r['model']
    err_px = r['contour_mae_px']
    apex_px = r['apex_err_x_px']
    
    # Load raw image
    im_path = os.path.join(BASE_DIR, "data", "processed", "aligned_128", f"{sid}-{view}.png")
    im = cv2.imread(im_path)
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    ax = axes[idx]
    ax.imshow(im_rgb)
    ax.axvline(64, color='white', linestyle=':', alpha=0.7)
    
    # Plot GT contour
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    pts = {}
    for y in range(20, 111):
        m = np.where(im_gray[y, :64] < 165)[0]
        if len(m) > 0 and m[0] < 64: pts[y] = int(m[0])
    if pts:
        ax.plot([pts[y] for y in sorted(pts.keys())], sorted(pts.keys()), color='cyan', linewidth=2.0, linestyle='--', label='GT Contour')
        
    ax.set_title(f"Sample {sid}-{view} ({r['material']}-{r['process']})\nModel: {m_name}\nContour MAE: {err_px:.1f} px | Apex Err: {apex_px:.1f} px",
                 fontsize=9.5, fontweight='bold')
    ax.axis('off')
    if idx == 0:
        ax.legend(loc="lower left", fontsize=8)

plt.suptitle("Figure 3 (Phase V7): Failure Mode Analysis on Out-of-Fold Test Specimens\n"
             "Characterized by Extreme Aspect Ratios, Faint Etching, and Low Toe Contrast",
             fontsize=13, fontweight='bold', y=0.98)
fig3_path = os.path.join(OUT_DIR, "fig3_failure_cases_and_mismatches_v7.png")
plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Figure 3 to {fig3_path}")
print("Diagnostic figures generated successfully.")
