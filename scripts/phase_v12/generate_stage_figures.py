"""
generate_stage_figures.py

Phase V12 Finalized Project Stage: Summary & Presentation Figures
Generates publication-quality summary figures:
  - fig1_model_ranking_contour_mae.png: Bar chart of contour MAE across M1-M5 with std dev & 3.75 px inter-view reference
  - fig2_model_ranking_boundary_f1_dice.png: Grouped bar chart of Boundary F1 (2px) and Mask Dice
  - fig3_best_model_36_sample_overview.png: 36-sample overview plate (6x6) of top model (M1 Direct CNN) dual-overlays
  - fig4_representative_8_sample_comparison.png: 8-sample comparison plate across all 5 architectures & ground truth
  - fig5_raw_vs_contour_vs_overlay.png: Methodological progression plate (Raw -> Sharpened -> Canny -> Contour -> Overlay)

Outputs saved to:
  results/phase_v12_finalized_project_stage/figures/
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v12.run_model_36_panels import (
    extract_robust_crown_contour, mild_unsharp_mask, compute_clean_canny,
    create_contour_image, create_contour_overlay, create_dual_contour_overlay
)

STAGE_DIR = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage")
METRICS_DIR = os.path.join(STAGE_DIR, "metrics")
FIGURES_DIR = os.path.join(STAGE_DIR, "figures")
V9_GEN_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")

os.makedirs(FIGURES_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Figure 1: Model Ranking by Contour MAE
# -------------------------------------------------------------------------
def generate_fig1():
    ranking_csv = os.path.join(METRICS_DIR, "model_ranking.csv")
    if not os.path.exists(ranking_csv):
        print(f"Fig 1: Ranking CSV not yet available at {ranking_csv}")
        return
    df = pd.read_csv(ranking_csv).sort_values("contour_mae_mean_px")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200, facecolor='#0D1117')
    ax.set_facecolor('#161B22')

    models = [r['model_key'] for _, r in df.iterrows()]
    names = [r['architecture'].split(':')[1].strip() if ':' in r['architecture'] else r['architecture'] for _, r in df.iterrows()]
    maes = df['contour_mae_mean_px'].values
    stds = df['contour_mae_std_px'].values

    colors = ['#238636', '#1F6FEB', '#8957E5', '#D29922', '#DA3633']
    bars = ax.bar(models, maes, yerr=stds, capsize=6, color=colors[:len(models)], edgecolor='#30363D', linewidth=1.2, alpha=0.9)

    # Inter-view reference line (3.75 px)
    ax.axhline(3.75, color='#F78166', linestyle='--', linewidth=2.0, label='Empirical Transverse Variation (View A vs View B: 3.75 px)')

    for bar, mae, std, name in zip(bars, maes, stds, names):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + std + 0.3,
                f"{mae:.2f} px\n(±{std:.2f})", ha='center', va='bottom', color='#F0F6FC', fontsize=9.5, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width() / 2.0, 0.4,
                name, ha='center', va='bottom', color='#FFFFFF', fontsize=9, rotation=90, fontweight='semibold')

    ax.set_ylabel("Crown Contour MAE (pixels)", color='#F0F6FC', fontsize=12, fontweight='bold')
    ax.set_title("Phase V12: Generative Model Benchmark — Crown Contour MAE (N=36 Experiments)\n[Full Dataset Reconstruction Benchmark]",
                 color='#58A6FF', fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors='#C9D1D9', labelsize=11)
    ax.grid(axis='y', color='#30363D', linestyle=':', alpha=0.7)
    ax.set_ylim(0, max(maes + stds) + 2.5)

    legend = ax.legend(loc='upper right', facecolor='#21262D', edgecolor='#30363D', fontsize=10)
    for text in legend.get_texts():
        text.set_color('#F0F6FC')

    out_path = os.path.join(FIGURES_DIR, "fig1_model_ranking_contour_mae.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[OK] Fig 1 saved to: {out_path}")

# -------------------------------------------------------------------------
# Figure 2: Boundary F1 & Mask Dice Comparison
# -------------------------------------------------------------------------
def generate_fig2():
    ranking_csv = os.path.join(METRICS_DIR, "model_ranking.csv")
    if not os.path.exists(ranking_csv):
        return
    df = pd.read_csv(ranking_csv).sort_values("model_key")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200, facecolor='#0D1117')
    ax.set_facecolor('#161B22')

    models = [r['model_key'] for _, r in df.iterrows()]
    dice = df['mask_dice_mean'].values
    f1 = df['boundary_f1_mean'].values

    x = np.arange(len(models))
    width = 0.35

    rects1 = ax.bar(x - width/2, dice, width, label='Mask Dice Score', color='#3FB950', edgecolor='#30363D')
    rects2 = ax.bar(x + width/2, f1, width, label='Boundary F1 (tol=2px)', color='#58A6FF', edgecolor='#30363D')

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.015, f"{h:.3f}", ha='center', va='bottom', color='#3FB950', fontsize=9, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., h + 0.015, f"{h:.3f}", ha='center', va='bottom', color='#58A6FF', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, color='#F0F6FC', fontsize=11, fontweight='bold')
    ax.set_ylabel("Score (0.0 to 1.0)", color='#F0F6FC', fontsize=12, fontweight='bold')
    ax.set_title("Phase V12: Mask Segmentation Dice & Boundary F1 (tol = 2px) Across 5 Models\n[Geometric Overlap & Boundary Alignment Metrics]",
                 color='#58A6FF', fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(colors='#C9D1D9', labelsize=11)
    ax.grid(axis='y', color='#30363D', linestyle=':', alpha=0.7)
    ax.set_ylim(0, 1.05)

    legend = ax.legend(loc='upper right', facecolor='#21262D', edgecolor='#30363D', fontsize=10)
    for text in legend.get_texts():
        text.set_color('#F0F6FC')

    out_path = os.path.join(FIGURES_DIR, "fig2_model_ranking_boundary_f1_dice.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[OK] Fig 2 saved to: {out_path}")

# -------------------------------------------------------------------------
# Figure 3: Best Model (M1 Direct CNN) 36-Sample Overview Plate (6x6 Grid)
# -------------------------------------------------------------------------
def generate_fig3():
    df_clean = pd.read_csv(CLEAN_CSV)
    m1_metrics_csv = os.path.join(METRICS_DIR, "M1_DirectCNN_metrics.csv")
    df_m = pd.read_csv(m1_metrics_csv) if os.path.exists(m1_metrics_csv) else None

    fig, axes = plt.subplots(6, 6, figsize=(18, 18), dpi=150, facecolor='#0D1117')
    fig.subplots_adjust(top=0.94, bottom=0.03, left=0.03, right=0.97, hspace=0.25, wspace=0.15)

    fig.suptitle("Phase V12: Model M1 (Direct Conditional CNN) — Complete 36 Physical Welding Experiments Overview\n[Dual Contour Overlays: Cyan = Real A, Magenta = Real B, Lime Green = Generated M1]",
                 color='#58A6FF', fontsize=16, fontweight='bold', y=0.97)

    for i in range(36):
        ax = axes[i // 6, i % 6]
        s_id = i + 1
        s_code = f"S{s_id:02d}"
        s_dir = os.path.join(V9_GEN_DIR, s_code)

        real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_A.png")), cv2.COLOR_BGR2RGB)
        real_b_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_B.png")), cv2.COLOR_BGR2RGB)
        gen_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "M1_DirectCNN_raw.png")), cv2.COLOR_BGR2RGB)

        cnt_a = extract_robust_crown_contour(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY))
        cnt_b = extract_robust_crown_contour(cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY))
        cnt_gen = extract_robust_crown_contour(cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY))

        dual = create_dual_contour_overlay(real_a_rgb, cnt_a['pts'], cnt_b['pts'], cnt_gen['pts'])
        ax.imshow(dual)
        ax.axis('off')

        mae_str = ""
        if df_m is not None:
            r = df_m[df_m['sample'] == s_id].iloc[0]
            mae_str = f" | {r['contour_mae_px_avg']:.1f}px"

        row_c = df_clean[df_clean['sample'] == s_id].iloc[0]
        ax.set_title(f"{s_code}: {row_c['material']} {row_c['process']}{mae_str}", color='#F0F6FC', fontsize=8.5, pad=3)

    out_path = os.path.join(FIGURES_DIR, "fig3_best_model_36_sample_overview.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[OK] Fig 3 saved to: {out_path}")

# -------------------------------------------------------------------------
# Figure 4: Representative 8-Sample Comparison Across All 5 Models
# -------------------------------------------------------------------------
def generate_fig4():
    rep_samples = [1, 5, 9, 14, 20, 26, 30, 36]
    models = ["M1_DirectCNN", "M2_CVAE", "M3_GCWPGv2", "M4_GeoSDFCVAE", "M5_cGAN"]
    col_titles = ["Real Macrograph\n(View A)", "M1: Direct CNN\n(Direct Conv)", "M2: CVAE\n(Latent VAE)",
                  "M3: GC-WPG-v2\n(Spatial Prior)", "M4: GeoSDF-CVAE\n(Signed Distance)", "M5: cGAN\n(PatchGAN)"]

    df_clean = pd.read_csv(CLEAN_CSV)
    fig, axes = plt.subplots(8, 6, figsize=(18, 22), dpi=150, facecolor='#0D1117')
    fig.subplots_adjust(top=0.95, bottom=0.03, left=0.06, right=0.96, hspace=0.25, wspace=0.15)

    fig.suptitle("Phase V12: Multi-Architecture Comparison Across 8 Representative Physical Welding Experiments\n[Covering Both IN718 & SS304L across ColdArc and GMAW Regimes]",
                 color='#58A6FF', fontsize=16, fontweight='bold', y=0.98)

    for row_idx, s_id in enumerate(rep_samples):
        s_code = f"S{s_id:02d}"
        s_dir = os.path.join(V9_GEN_DIR, s_code)
        r_info = df_clean[df_clean['sample'] == s_id].iloc[0]

        # Col 0: Real A with detected contour
        real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_A.png")), cv2.COLOR_BGR2RGB)
        cnt_a = extract_robust_crown_contour(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY))
        overlay_a = create_contour_overlay(real_a_rgb, cnt_a['pts'], (cnt_a['apex_x'], cnt_a['apex_y']), line_color=(0, 220, 255))

        ax0 = axes[row_idx, 0]
        ax0.imshow(overlay_a)
        ax0.axis('off')
        if row_idx == 0:
            ax0.set_title(col_titles[0], color='#58A6FF', fontsize=10.5, fontweight='bold', pad=8)
        ax0.text(-0.25, 0.5, f"{s_code}\n{r_info['material']}\n{r_info['process']}\nF={r_info['feed_rate']:.1f}\nS={r_info['speed']:.0f}",
                 transform=ax0.transAxes, color='#F0F6FC', fontsize=9, fontweight='bold', va='center', ha='center')

        # Cols 1-5: Generative Models
        for col_idx, m_name in enumerate(models, start=1):
            ax = axes[row_idx, col_idx]
            gen_path = os.path.join(s_dir, f"{m_name}_raw.png")
            gen_rgb = cv2.cvtColor(cv2.imread(gen_path), cv2.COLOR_BGR2RGB)
            cnt_gen = extract_robust_crown_contour(cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY))
            overlay_gen = create_contour_overlay(gen_rgb, cnt_gen['pts'], (cnt_gen['apex_x'], cnt_gen['apex_y']), line_color=(0, 255, 0))

            ax.imshow(overlay_gen)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], color='#58A6FF', fontsize=10.5, fontweight='bold', pad=8)

    out_path = os.path.join(FIGURES_DIR, "fig4_representative_8_sample_comparison.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[OK] Fig 4 saved to: {out_path}")

# -------------------------------------------------------------------------
# Figure 5: Methodological Progression (Raw -> Sharpened -> Canny -> Contour -> Overlay)
# -------------------------------------------------------------------------
def generate_fig5():
    sample_ids = [1, 9, 20, 36]
    df_clean = pd.read_csv(CLEAN_CSV)

    fig, axes = plt.subplots(4, 6, figsize=(18, 12), dpi=150, facecolor='#0D1117')
    fig.subplots_adjust(top=0.92, bottom=0.04, left=0.06, right=0.96, hspace=0.30, wspace=0.15)

    fig.suptitle("Phase V12: Methodological Processing Pipeline — Raw RGB to Validated Dual-Contour Overlay\n[Note: Canny & unsharp masking provide edge visualization; quantitative comparison is performed against physical macrograph ground-truth]",
                 color='#58A6FF', fontsize=15, fontweight='bold', y=0.97)

    col_headers = [
        "1. Real Macrograph\n(Ground Truth)",
        "2. Raw Generated RGB\n(M1 Direct CNN)",
        "3. Sharpened RGB\n(Mild Unsharp Mask)",
        "4. Canny Edge Map\n(Gradient Boundaries)",
        "5. Extracted Contour\n(Baseline X=64)",
        "6. Dual-Contour Overlay\n(Cyan: Real | Green: Gen)"
    ]

    for row_idx, s_id in enumerate(sample_ids):
        s_code = f"S{s_id:02d}"
        s_dir = os.path.join(V9_GEN_DIR, s_code)
        r_info = df_clean[df_clean['sample'] == s_id].iloc[0]

        real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_A.png")), cv2.COLOR_BGR2RGB)
        real_b_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_B.png")), cv2.COLOR_BGR2RGB)
        gen_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "M1_DirectCNN_raw.png")), cv2.COLOR_BGR2RGB)

        gray_a = cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY)
        gray_gen = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)

        cnt_a = extract_robust_crown_contour(gray_a)
        cnt_b = extract_robust_crown_contour(gray_b)
        cnt_gen = extract_robust_crown_contour(gray_gen)

        sharp_gen = mild_unsharp_mask(gen_rgb)
        canny_gen = compute_clean_canny(gray_gen)
        cnt_img = create_contour_image(cnt_gen['pts'])
        dual = create_dual_contour_overlay(real_a_rgb, cnt_a['pts'], cnt_b['pts'], cnt_gen['pts'])

        imgs = [real_a_rgb, gen_rgb, sharp_gen, canny_gen, cnt_img, dual]
        cmaps = [None, None, None, 'inferno', 'gray', None]

        for col_idx, (im, cm) in enumerate(zip(imgs, cmaps)):
            ax = axes[row_idx, col_idx]
            if cm:
                ax.imshow(im, cmap=cm)
            else:
                ax.imshow(im)
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_headers[col_idx], color='#58A6FF', fontsize=10, fontweight='bold', pad=6)

        axes[row_idx, 0].text(-0.25, 0.5, f"{s_code}\n{r_info['material']}\n{r_info['process']}\nF={r_info['feed_rate']:.1f}\nS={r_info['speed']:.0f}",
                               transform=axes[row_idx, 0].transAxes, color='#F0F6FC', fontsize=9.5, fontweight='bold', va='center', ha='center')

    out_path = os.path.join(FIGURES_DIR, "fig5_raw_vs_contour_vs_overlay.png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[OK] Fig 5 saved to: {out_path}")

def main():
    print("=" * 80)
    print("PHASE V12: GENERATING ALL SUMMARY & PRESENTATION FIGURES")
    print("=" * 80)
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig4()
    generate_fig5()
    print("=" * 80)
    print("ALL 5 PRESENTATION FIGURES GENERATED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
