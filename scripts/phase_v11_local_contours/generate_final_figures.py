#!/usr/bin/env python3
"""
scripts/phase_v11_local_contours/generate_final_figures.py

Phase V11: Publication-Grade Visualizations & Benchmark Figures (Figures 1 to 9).
Reads the verified metrics and processed images from results/phase_v11_local_contours/
and generates 9 comprehensive figures.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Paths
BASE_DIR = r"C:\Users\cool\OneDrive\Desktop\Project\weld_project"
PHASE_V11_DIR = os.path.join(BASE_DIR, "results", "phase_v11_local_contours")
GEN_36_DIR = os.path.join(PHASE_V11_DIR, "generated_36")
METRICS_DIR = os.path.join(PHASE_V11_DIR, "metrics")
FIG_DIR = os.path.join(PHASE_V11_DIR, "figures")
ARTIFACT_DIR = r"C:\Users\cool\.gemini\antigravity\brain\ea5731f8-95e3-4d08-94cc-06a751eef6db"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load CSV metrics
df_detailed = pd.read_csv(os.path.join(METRICS_DIR, "all_samples_detailed_metrics.csv"))
df_summary = pd.read_csv(os.path.join(METRICS_DIR, "model_summary_metrics.csv"))
df_ranking = pd.read_csv(os.path.join(METRICS_DIR, "model_ranking.csv"))

ALEATORIC_NOISE_FLOOR = 3.75 # px

def save_and_copy(fig, filename):
    out_path = os.path.join(FIG_DIR, filename)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    art_path = os.path.join(ARTIFACT_DIR, filename)
    fig.savefig(art_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {filename}")

# =========================================================================
# FIGURE 1: Model Ranking Contour MAE (px) with Error Bars & GT Noise Floor
# =========================================================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = ["M1\n(Direct CNN)", "M3\n(GC-WPG-v2)", "B1\n(PCA Baseline)", "M5\n(cGAN)", "M6\n(GF-WPG)"]
    mae_means = [3.44, 4.42, 4.67, 5.21, 0.0]
    mae_stds = [2.03, 3.66, 2.14, 6.56, 0.0]
    colors = ['#2ca02c', '#1f77b4', '#7f7f7f', '#d62728', '#e0e0e0']
    edgecolors = ['black', 'black', 'black', 'black', 'red']
    
    x = np.arange(len(models))
    bars = ax.bar(x[:4], mae_means[:4], yerr=mae_stds[:4], capsize=6, color=colors[:4], edgecolor='black', alpha=0.88, width=0.55)
    bar6 = ax.bar(x[4], [5.0], color='none', edgecolor='red', hatch='//', linestyle='--', width=0.55, linewidth=1.5)
    
    # Ground truth noise floor
    ax.axhline(ALEATORIC_NOISE_FLOOR, color='#ff7f0e', linestyle='--', linewidth=2.0, 
               label=f'Ground Truth Inter-View Variation (Aleatoric Noise Floor: {ALEATORIC_NOISE_FLOOR:.2f} px)')
    
    # Value labels
    for i in range(4):
        ax.text(x[i], mae_means[i] + mae_stds[i] + 0.35, f"{mae_means[i]:.2f} ± {mae_stds[i]:.2f} px", 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(x[4], 5.3, "Requires A100 Training\n(Missing Locally)", ha='center', va='bottom', fontsize=9, color='darkred', fontstyle='italic')
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.set_ylabel("External Contour MAE (pixels)", fontsize=12, fontweight='bold')
    ax.set_title("Figure 1: Verified 36-Sample Model Ranking by External Contour MAE\n(Dual-View Mean against Metallographic Ground Truth)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim(0, 13.5)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)
    
    plt.tight_layout()
    save_and_copy(fig, "fig1_model_ranking_contour_mae.png")

# =========================================================================
# FIGURE 2: Model Ranking Boundary F1 and Mask Dice
# =========================================================================
def make_fig2():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    models = ["M1 (Direct CNN)", "M3 (GC-WPG-v2)", "M5 (cGAN)", "B1 (PCA Baseline)"]
    f1_vals = [0.5787, 0.5314, 0.5322, 0.3811]
    dice_vals = [0.7795, 0.7341, 0.7297, 0.7148]
    iou_vals = [0.6631, 0.6204, 0.6192, 0.5741]
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#7f7f7f']
    
    x = np.arange(len(models))
    
    # Panel 1: Boundary F1
    bars1 = ax1.bar(x, f1_vals, color=colors, edgecolor='black', width=0.55, alpha=0.88)
    for i, v in enumerate(f1_vals):
        ax1.text(x[i], v + 0.015, f"{v:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
    ax1.set_ylabel("Boundary F1 Score (d_tol = 2 px)", fontsize=11, fontweight='bold')
    ax1.set_title("(a) Boundary Alignment F1 Score at 2 px Tolerance", fontsize=11, fontweight='bold')
    ax1.set_ylim(0, 0.72)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Panel 2: Mask Dice & IoU
    w = 0.32
    ax2.bar(x - w/2, dice_vals, width=w, label="Mask Dice", color='#3b528b', edgecolor='black', alpha=0.85)
    ax2.bar(x + w/2, iou_vals, width=w, label="Mask IoU", color='#5ec962', edgecolor='black', alpha=0.85)
    for i in range(len(models)):
        ax2.text(x[i] - w/2, dice_vals[i] + 0.012, f"{dice_vals[i]:.3f}", ha='center', va='bottom', fontsize=8.5)
        ax2.text(x[i] + w/2, iou_vals[i] + 0.012, f"{iou_vals[i]:.3f}", ha='center', va='bottom', fontsize=8.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=10, rotation=15, ha='right')
    ax2.set_ylabel("Overlap Metric", fontsize=11, fontweight='bold')
    ax2.set_title("(b) Region Overlap: Mask Dice & Mask IoU", fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 0.95)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    fig.suptitle("Figure 2: Verified Geometry Metrics Across Models (36-Sample Dual-View Average)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_and_copy(fig, "fig2_model_ranking_boundary_f1_and_dice.png")

# =========================================================================
# FIGURE 3: 36-Sample Contact Sheet - Best Model (M1 Direct CNN) Raw RGB
# =========================================================================
def make_fig3():
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(36):
        s_id = f"S{idx+1:02d}"
        s_dir = os.path.join(GEN_36_DIR, s_id)
        raw_path = os.path.join(s_dir, "M1_raw_rgb.png")
        ax = axes[idx]
        
        if os.path.exists(raw_path):
            img = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha='center', va='center')
        
        # Meta info
        sub = df_detailed[(df_detailed['sample_id'] == s_id) & (df_detailed['model'] == 'M1')].iloc[0]
        mae = sub['contour_mae_px']
        ax.set_title(f"{s_id} | {sub['material']} {sub['process']}\nF={sub['feed_rate']:.0f} S={sub['speed']:.0f} | MAE={mae:.1f}px", 
                     fontsize=7.5, pad=3)
        ax.axis('off')
        
    fig.suptitle("Figure 3: All 36 Physical Welding Experiments — M1 (Direct Conditional CNN) Raw Generated Images", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig3_36sample_contact_sheet_best_model_raw.png")

# =========================================================================
# FIGURE 4: 36-Sample Contact Sheet - Dual Contour Overlays (GT vs M1)
# =========================================================================
def make_fig4():
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx in range(36):
        s_id = f"S{idx+1:02d}"
        s_dir = os.path.join(GEN_36_DIR, s_id)
        ov_path = os.path.join(s_dir, "M1_overlay_original_vs_generated_contour.png")
        ax = axes[idx]
        
        if os.path.exists(ov_path):
            img = cv2.cvtColor(cv2.imread(ov_path), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, "Missing", ha='center', va='center')
        
        sub = df_detailed[(df_detailed['sample_id'] == s_id) & (df_detailed['model'] == 'M1')].iloc[0]
        mae = sub['contour_mae_px']
        ax.set_title(f"{s_id} | MAE={mae:.1f}px | F1={sub['boundary_f1_2px']:.2f}", fontsize=8, pad=3)
        ax.axis('off')
        
    # Legend patch
    cyan_patch = mpatches.Patch(color='cyan', label='GT View A Contour')
    orange_patch = mpatches.Patch(color='orange', label='GT View B Contour')
    green_patch = mpatches.Patch(color='lime', label='M1 Predicted Contour')
    fig.legend(handles=[cyan_patch, orange_patch, green_patch], loc='lower center', ncol=3, fontsize=11, framealpha=0.9)
    
    fig.suptitle("Figure 4: All 36 Samples Dual Contour Verification (Real View A Cyan, View B Orange, M1 Pred Green)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_and_copy(fig, "fig4_36sample_contact_sheet_best_model_contours.png")

# =========================================================================
# FIGURE 5: 8-Sample Deep Dive Multi-Column Comparison
# =========================================================================
def make_fig5():
    sample_ids = ["S02", "S06", "S11", "S15", "S20", "S24", "S29", "S33"]
    fig, axes = plt.subplots(8, 7, figsize=(16, 18))
    
    col_titles = [
        "Real Macrograph (A)",
        "Real Canny Edge",
        "M1 Direct CNN",
        "M3 GC-WPG-v2",
        "M5 cGAN",
        "B1 PCA Baseline",
        "Dual Contour Overlay\n(GT vs M1 Green)"
    ]
    
    for r_idx, s_id in enumerate(sample_ids):
        s_dir = os.path.join(GEN_36_DIR, s_id)
        
        p_real_a = os.path.join(s_dir, "original_A.png")
        p_canny_a = os.path.join(s_dir, "original_A_canny.png")
        p_m1_ov = os.path.join(s_dir, "M1_generated_overlay.png")
        p_m3_ov = os.path.join(s_dir, "M3_generated_overlay.png")
        p_m5_ov = os.path.join(s_dir, "M5_generated_overlay.png")
        p_b1_ov = os.path.join(s_dir, "B1_generated_overlay.png")
        p_dual = os.path.join(s_dir, "M1_overlay_original_vs_generated_contour.png")
        
        imgs = [
            cv2.cvtColor(cv2.imread(p_real_a), cv2.COLOR_BGR2RGB) if os.path.exists(p_real_a) else None,
            cv2.imread(p_canny_a, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_canny_a) else None,
            cv2.cvtColor(cv2.imread(p_m1_ov), cv2.COLOR_BGR2RGB) if os.path.exists(p_m1_ov) else None,
            cv2.cvtColor(cv2.imread(p_m3_ov), cv2.COLOR_BGR2RGB) if os.path.exists(p_m3_ov) else None,
            cv2.cvtColor(cv2.imread(p_m5_ov), cv2.COLOR_BGR2RGB) if os.path.exists(p_m5_ov) else None,
            cv2.cvtColor(cv2.imread(p_b1_ov), cv2.COLOR_BGR2RGB) if os.path.exists(p_b1_ov) else None,
            cv2.cvtColor(cv2.imread(p_dual), cv2.COLOR_BGR2RGB) if os.path.exists(p_dual) else None,
        ]
        
        # Metadata
        sub = df_detailed[(df_detailed['sample_id'] == s_id) & (df_detailed['model'] == 'M1')].iloc[0]
        row_label = f"{s_id}\n{sub['material']}\n{sub['process']}\nF={sub['feed_rate']:.0f} S={sub['speed']:.0f}"
        
        for c_idx in range(7):
            ax = axes[r_idx, c_idx]
            im = imgs[c_idx]
            if im is not None:
                if c_idx == 1:
                    ax.imshow(im, cmap='gray')
                else:
                    ax.imshow(im)
            ax.axis('off')
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=10, fontweight='bold', pad=4)
        
        axes[r_idx, 0].text(-20, 64, row_label, va='center', ha='right', fontsize=9, fontweight='bold')
        
    fig.suptitle("Figure 5: 8-Sample Controlled Multi-Model Comparison (2 Samples per Material–Process Regime)", 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig5_eight_sample_large_comparison.png")

# =========================================================================
# FIGURE 6: Original vs Generated Contour Overlay Grid (12 Samples)
# =========================================================================
def make_fig6():
    samples_12 = ["S01", "S04", "S07", "S10", "S13", "S16", "S19", "S22", "S25", "S28", "S31", "S34"]
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, s_id in enumerate(samples_12):
        ax = axes[idx]
        s_dir = os.path.join(GEN_36_DIR, s_id)
        p_dual = os.path.join(s_dir, "M1_overlay_original_vs_generated_contour.png")
        
        if os.path.exists(p_dual):
            img = cv2.cvtColor(cv2.imread(p_dual), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        
        sub = df_detailed[(df_detailed['sample_id'] == s_id) & (df_detailed['model'] == 'M1')].iloc[0]
        ax.set_title(f"{s_id} ({sub['material']}/{sub['process']})\nMAE: {sub['contour_mae_px']:.1f} px | Dice: {sub['mask_dice']:.2f}",
                     fontsize=9, pad=3)
        ax.axis('off')
        
    cyan_patch = mpatches.Patch(color='cyan', label='GT View A Contour')
    orange_patch = mpatches.Patch(color='orange', label='GT View B Contour')
    green_patch = mpatches.Patch(color='lime', label='M1 Predicted Contour')
    fig.legend(handles=[cyan_patch, orange_patch, green_patch], loc='lower center', ncol=3, fontsize=11, framealpha=0.9)
    
    fig.suptitle("Figure 6: Real Specimen Macrograph with Overlaid M1 Predicted Contours (12 Representative Welds)", 
                 fontsize=13, fontweight='bold', y=0.985)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_and_copy(fig, "fig6_original_vs_generated_overlay_grid.png")

# =========================================================================
# FIGURE 7: Canny Edge Detection Comparison Grid
# =========================================================================
def make_fig7():
    sample_ids = ["S03", "S09", "S14", "S21", "S27", "S32"]
    fig, axes = plt.subplots(6, 5, figsize=(13, 15))
    
    col_titles = [
        "Real Canny (A)",
        "M1 Generated Canny",
        "M3 Generated Canny",
        "M5 Generated Canny",
        "Real (Cyan) vs M1 (Red)\nEdge Overlay"
    ]
    
    for r_idx, s_id in enumerate(sample_ids):
        s_dir = os.path.join(GEN_36_DIR, s_id)
        
        p_real_canny = os.path.join(s_dir, "original_A_canny.png")
        p_m1_canny = os.path.join(s_dir, "M1_generated_canny.png")
        p_m3_canny = os.path.join(s_dir, "M3_generated_canny.png")
        p_m5_canny = os.path.join(s_dir, "M5_generated_canny.png")
        p_edge_ov = os.path.join(s_dir, "M1_overlay_original_vs_generated_edges.png")
        
        imgs = [
            cv2.imread(p_real_canny, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_real_canny) else None,
            cv2.imread(p_m1_canny, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_m1_canny) else None,
            cv2.imread(p_m3_canny, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_m3_canny) else None,
            cv2.imread(p_m5_canny, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_m5_canny) else None,
            cv2.cvtColor(cv2.imread(p_edge_ov), cv2.COLOR_BGR2RGB) if os.path.exists(p_edge_ov) else None
        ]
        
        sub = df_detailed[(df_detailed['sample_id'] == s_id) & (df_detailed['model'] == 'M1')].iloc[0]
        row_label = f"{s_id}\n{sub['material']}\n{sub['process']}"
        
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            im = imgs[c_idx]
            if im is not None:
                if c_idx < 4:
                    ax.imshow(im, cmap='gray')
                else:
                    ax.imshow(im)
            ax.axis('off')
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.5, fontweight='bold', pad=4)
        
        axes[r_idx, 0].text(-15, 64, row_label, va='center', ha='right', fontsize=9, fontweight='bold')
        
    fig.suptitle("Figure 7: Canny Edge Detection Comparison Across Ground Truth and Generative Models", 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig7_canny_edge_comparison_grid.png")

# =========================================================================
# FIGURE 8: Raw vs Sharpened vs Canny vs Mask vs Contour Progression
# =========================================================================
def make_fig8():
    sample_ids = ["S05", "S12", "S22", "S30"]
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    
    col_titles = [
        "1. Raw Generated RGB\n(Underlying Model Output)",
        "2. Sharpened RGB\n(Mild Unsharp Mask)",
        "3. Generated Canny Map\n(Clean Gradient Boundary)",
        "4. Binary Weld Mask\n(Morphologically Verified)",
        "5. Final Contour Overlay\n(On Real Macrograph)"
    ]
    
    for r_idx, s_id in enumerate(sample_ids):
        s_dir = os.path.join(GEN_36_DIR, s_id)
        
        p_raw = os.path.join(s_dir, "M1_raw_rgb.png")
        p_sharp = os.path.join(s_dir, "M1_sharpened_rgb.png")
        p_canny = os.path.join(s_dir, "M1_generated_canny.png")
        p_mask = os.path.join(s_dir, "M1_predicted_mask.png")
        p_dual = os.path.join(s_dir, "M1_overlay_original_vs_generated_contour.png")
        
        imgs = [
            cv2.cvtColor(cv2.imread(p_raw), cv2.COLOR_BGR2RGB) if os.path.exists(p_raw) else None,
            cv2.cvtColor(cv2.imread(p_sharp), cv2.COLOR_BGR2RGB) if os.path.exists(p_sharp) else None,
            cv2.imread(p_canny, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_canny) else None,
            cv2.imread(p_mask, cv2.IMREAD_GRAYSCALE) if os.path.exists(p_mask) else None,
            cv2.cvtColor(cv2.imread(p_dual), cv2.COLOR_BGR2RGB) if os.path.exists(p_dual) else None,
        ]
        
        sub = df_detailed[(df_detailed['sample_id'] == s_id) & (df_detailed['model'] == 'M1')].iloc[0]
        row_label = f"{s_id}\n{sub['material']}\n{sub['process']}"
        
        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            im = imgs[c_idx]
            if im is not None:
                if c_idx in [2, 3]:
                    ax.imshow(im, cmap='gray')
                else:
                    ax.imshow(im)
            ax.axis('off')
            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=9.5, fontweight='bold', pad=4)
        
        axes[r_idx, 0].text(-15, 64, row_label, va='center', ha='right', fontsize=9, fontweight='bold')
        
    fig.suptitle("Figure 8: End-to-End Extraction Pipeline: Resolving Crisp Geometry from Raw Generated Predictions", 
                 fontsize=13, fontweight='bold', y=0.995)
    plt.tight_layout()
    save_and_copy(fig, "fig8_raw_vs_sharpened_vs_contour.png")

# =========================================================================
# FIGURE 9: Material-Process-Wise Stratified Metric Summary
# =========================================================================
def make_fig9():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Compute group averages for M1, M3, B1, M5
    df_m = df_detailed.copy()
    df_m['group'] = df_m['material'] + " / " + df_m['process']
    
    groups = sorted(df_m['group'].unique())
    models = ["M1", "M3", "B1", "M5"]
    model_labels = ["M1 (Direct CNN)", "M3 (GC-WPG-v2)", "B1 (PCA Baseline)", "M5 (cGAN)"]
    palette = ['#2ca02c', '#1f77b4', '#7f7f7f', '#d62728']
    
    x = np.arange(len(groups))
    width = 0.2
    
    # Subplot 1: Contour MAE per group
    for i, m in enumerate(models):
        subset = df_m[df_m['model'] == m]
        means = [subset[subset['group'] == g]['contour_mae_px'].mean() for g in groups]
        ax1.bar(x + i*width - 1.5*width, means, width, label=model_labels[i], color=palette[i], edgecolor='black', alpha=0.88)
        
    ax1.axhline(ALEATORIC_NOISE_FLOOR, color='#ff7f0e', linestyle='--', linewidth=1.5, label='Aleatoric GT Noise Floor (3.75 px)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=10, fontweight='bold', rotation=10)
    ax1.set_ylabel("Contour MAE (pixels)", fontsize=11, fontweight='bold')
    ax1.set_title("(a) External Contour MAE by Material–Process Regime", fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8.5)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    # Subplot 2: Mask Dice per group
    for i, m in enumerate(models):
        subset = df_m[df_m['model'] == m]
        means = [subset[subset['group'] == g]['mask_dice'].mean() for g in groups]
        ax2.bar(x + i*width - 1.5*width, means, width, label=model_labels[i], color=palette[i], edgecolor='black', alpha=0.88)
        
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=10, fontweight='bold', rotation=10)
    ax2.set_ylabel("Mask Dice Overlap", fontsize=11, fontweight='bold')
    ax2.set_title("(b) Binary Mask Dice Overlap by Material–Process Regime", fontsize=11, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8.5)
    ax2.set_ylim(0.5, 0.9)
    ax2.grid(axis='y', linestyle=':', alpha=0.6)
    
    fig.suptitle("Figure 9: Material- and Process-Stratified Performance Benchmarks", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_and_copy(fig, "fig9_material_process_wise_metric_summary.png")

if __name__ == "__main__":
    print("Generating Figure 1...")
    make_fig1()
    print("Generating Figure 2...")
    make_fig2()
    print("Generating Figure 3...")
    make_fig3()
    print("Generating Figure 4...")
    make_fig4()
    print("Generating Figure 5...")
    make_fig5()
    print("Generating Figure 6...")
    make_fig6()
    print("Generating Figure 7...")
    make_fig7()
    print("Generating Figure 8...")
    make_fig8()
    print("Generating Figure 9...")
    make_fig9()
    print("All 9 Phase V11 Figures successfully generated and copied to artifact directory.")
