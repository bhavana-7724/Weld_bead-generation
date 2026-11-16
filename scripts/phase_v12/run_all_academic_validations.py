"""
run_all_academic_validations.py

Executes complete academic validation across all 5 models (M1–M5) and all 36 samples (S01–S36),
generating 180 individual publication-grade panels, CSV metrics, galleries, and summary figures.

Directory Structure:
results/phase_v12_finalized_project_stage/ACADEMIC_FINAL_VALIDATION/
├── S02_reference_panel.png
├── ALL_MODELS_METRICS.csv
├── MODEL_COMPARISON_SUMMARY.csv
├── GALLERY.html
├── FINAL_ACADEMIC_VALIDATION_REPORT.md
├── summary_figures/
│   ├── fig1_model_mae_comparison.png
│   └── fig2_model_dice_f1_comparison.png
├── M1_DirectCNN/
│   ├── S01_panel.png ... S36_panel.png (36 panels)
│   ├── metrics.csv
│   └── INDEX.html
├── M2_CVAE/
│   ├── S01_panel.png ... S36_panel.png (36 panels)
│   ├── metrics.csv
│   └── INDEX.html
├── M3_GCWPGv2/
│   ├── S01_panel.png ... S36_panel.png (36 panels)
│   ├── metrics.csv
│   └── INDEX.html
├── M4_GeoSDFCVAE/
│   ├── S01_panel.png ... S36_panel.png (36 panels)
│   ├── metrics.csv
│   └── INDEX.html
└── M5_cGAN/
    ├── S01_panel.png ... S36_panel.png (36 panels)
    ├── metrics.csv
    └── INDEX.html
"""

import os
import sys
import json
import time
import pickle
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

from scripts.phase_v12.generate_academic_panels import (
    render_academic_panel,
    MODEL_CONFIGS,
    OUTPUT_BASE,
    CLEAN_CSV
)

SUMMARY_FIG_DIR = os.path.join(OUTPUT_BASE, "summary_figures")
os.makedirs(SUMMARY_FIG_DIR, exist_ok=True)


def build_model_index_html(model_key, records, out_html_path):
    m_cfg = MODEL_CONFIGS[model_key]
    rows_html = ""
    for r in records:
        s_code = r["sample"]
        rows_html += f"""
        <tr>
            <td style="font-weight:bold;">{s_code}</td>
            <td>{r['material']}</td>
            <td>{r['process']}</td>
            <td>{r['feed_rate']:.1f} m/min</td>
            <td>{r['speed']:.1f} cm/min</td>
            <td style="color:#0969da; font-weight:bold;">{r['dual_view_avg_mae_px']:.2f} px</td>
            <td>{r['dual_view_avg_f1_2px']:.3f}</td>
            <td>{r['dual_view_avg_dice']:.3f}</td>
            <td>{r['dual_view_avg_apex_err_px']:.1f} px</td>
            <td>{r['abs_error_reinf_mm']:.2f} mm</td>
            <td><a href="{s_code}_panel.png" target="_blank" style="color:#0969da; text-decoration:none; font-weight:bold;">View Panel &rarr;</a></td>
        </tr>
        """

    cards_html = ""
    for r in records:
        s_code = r["sample"]
        cards_html += f"""
        <div class="card">
            <h4>{s_code}: {r['material']} | {r['process']} (F={r['feed_rate']:.1f}, S={r['speed']:.1f})</h4>
            <p>Dual MAE: <b>{r['dual_view_avg_mae_px']:.2f} px</b> | Dice: <b>{r['dual_view_avg_dice']:.3f}</b> | Apex Err: <b>{r['dual_view_avg_apex_err_px']:.1f} px</b></p>
            <a href="{s_code}_panel.png" target="_blank">
                <img src="{s_code}_panel.png" alt="{s_code} Panel" loading="lazy">
            </a>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{m_cfg['display_name']} — Academic Validation Collection</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; background: #fafbfc; color: #1f2328; }}
    h1 {{ color: #0969da; margin-bottom: 4px; }}
    .subtitle {{ color: #57606a; font-size: 14px; margin-bottom: 20px; }}
    .nav-bar {{ margin-bottom: 24px; padding: 12px; background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px; }}
    .nav-bar a {{ margin-right: 16px; color: #0969da; text-decoration: none; font-weight: 600; font-size: 13.5px; }}
    .nav-bar a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; background: #ffffff; border: 1px solid #d0d7de; font-size: 13px; }}
    th, td {{ padding: 8px 12px; border: 1px solid #d0d7de; text-align: left; }}
    th {{ background: #f6f8fa; font-weight: 600; }}
    tr:nth-child(even) {{ background: #fcfcfc; }}
    .gallery-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 18px; }}
    .card {{ background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }}
    .card h4 {{ margin: 0 0 6px 0; font-size: 13.5px; color: #0969da; }}
    .card p {{ margin: 0 0 10px 0; font-size: 12px; color: #57606a; }}
    .card img {{ width: 100%; border-radius: 4px; border: 1px solid #e1e4e8; display: block; }}
</style>
</head>
<body>
    <div class="nav-bar">
        <a href="../GALLERY.html">&larr; Back to Unified Model Gallery</a>
        <a href="../M1_DirectCNN/INDEX.html">M1 DirectCNN</a>
        <a href="../M2_CVAE/INDEX.html">M2 CVAE</a>
        <a href="../M3_GCWPGv2/INDEX.html">M3 GCWPGv2</a>
        <a href="../M4_GeoSDFCVAE/INDEX.html">M4 GeoSDFCVAE</a>
        <a href="../M5_cGAN/INDEX.html">M5 cGAN</a>
    </div>

    <h1>{m_cfg['display_name']}</h1>
    <div class="subtitle">{m_cfg['short_desc']} &bull; Complete 36-Sample Ground-Truth Reconstruction Benchmark</div>

    <h2>Quantitative Performance Summary Table</h2>
    <table>
        <thead>
            <tr>
                <th>Sample</th><th>Material</th><th>Process</th><th>Feed Rate</th><th>Speed</th>
                <th>Dual-View MAE</th><th>Boundary F1</th><th>Mask Dice</th><th>Apex Error</th><th>Reinf Err (mm)</th><th>Link</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>

    <h2>Visual Gallery (36 Panels)</h2>
    <div class="gallery-grid">
        {cards_html}
    </div>
</body>
</html>
"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_unified_gallery_html(all_records, summary_df, out_html_path):
    summary_rows = ""
    for _, r in summary_df.iterrows():
        summary_rows += f"""
        <tr>
            <td style="font-weight:bold;"><a href="{r['model']}/INDEX.html" style="color:#0969da; text-decoration:none;">{r['display_name']}</a></td>
            <td style="font-weight:bold; color:#0969da;">{r['mean_mae_px']:.2f} px</td>
            <td>{r['mean_f1']:.3f}</td>
            <td>{r['mean_dice']:.3f}</td>
            <td>{r['mean_iou']:.3f}</td>
            <td>{r['mean_apex_err_px']:.2f} px</td>
            <td>{r['best_sample']} ({r['best_mae_px']:.2f} px)</td>
            <td>{r['worst_sample']} ({r['worst_mae_px']:.2f} px)</td>
            <td><a href="{r['model']}/INDEX.html" style="color:#0969da; font-weight:600;">Open Collection (36) &rarr;</a></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase V12 Academic Validation — Unified Model Gallery</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 28px; background: #fafbfc; color: #1f2328; }}
    h1 {{ color: #0969da; margin-bottom: 6px; font-size: 26px; }}
    .subtitle {{ color: #57606a; font-size: 15px; margin-bottom: 24px; }}
    .highlight-box {{ background: #ffffff; border: 1px solid #d0d7de; border-left: 4px solid #0969da; border-radius: 6px; padding: 16px; margin-bottom: 28px; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 32px; background: #ffffff; border: 1px solid #d0d7de; font-size: 13.5px; }}
    th, td {{ padding: 10px 14px; border: 1px solid #d0d7de; text-align: left; }}
    th {{ background: #f6f8fa; font-weight: 600; }}
    tr:nth-child(even) {{ background: #fcfcfc; }}
    .btn-group {{ margin-bottom: 28px; }}
    .btn {{ display: inline-block; padding: 9px 16px; margin-right: 10px; background: #0969da; color: #ffffff; border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 13px; }}
    .btn:hover {{ background: #0858b9; }}
    .btn-sec {{ background: #f6f8fa; color: #24292F; border: 1px solid #d0d7de; }}
    .btn-sec:hover {{ background: #eaeef2; }}
    .fig-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 32px; }}
    .fig-card {{ background: #ffffff; border: 1px solid #d0d7de; border-radius: 6px; padding: 14px; }}
    .fig-card h3 {{ margin: 0 0 10px 0; font-size: 14px; color: #0969da; }}
    .fig-card img {{ width: 100%; border-radius: 4px; border: 1px solid #e1e4e8; display: block; }}
</style>
</head>
<body>
    <h1>Phase V12 Academic Validation Master Gallery</h1>
    <div class="subtitle">Complete 180-Panel Scientific Ground-Truth Reconstruction Suite &bull; 5 Generative Models &times; 36 Experimental Conditions</div>

    <div class="highlight-box">
        <h3 style="margin-top:0; color:#0969da;">Academic Validation Overview</h3>
        <p style="margin:4px 0 8px 0; font-size:13.5px; line-height:1.5;">
            Every experimental condition (S01–S36) has been rendered in a clean white-background academic plate comparing 
            <b>Experimental View A</b>, <b>Experimental View B</b>, <b>Raw Generated RGB</b>, <b>Predicted Crown Contour & Mask</b>, 
            <b>Edge Diagnostic</b>, and <b>Multi-View Overlay</b>. Physical dimensions (mm) and image pixel metrics (px) are strictly separated.
        </p>
        <p style="margin:0; font-size:13px; color:#57606a;">
            Reference Specimen: <a href="S02_reference_panel.png" target="_blank" style="color:#0969da; font-weight:bold;">S02 Academic Reference Panel &rarr;</a>
        </p>
    </div>

    <div class="btn-group">
        <a class="btn" href="M1_DirectCNN/INDEX.html">M1 DirectCNN (36)</a>
        <a class="btn" href="M2_CVAE/INDEX.html">M2 CVAE (36)</a>
        <a class="btn" href="M3_GCWPGv2/INDEX.html">M3 GCWPGv2 (36)</a>
        <a class="btn" href="M4_GeoSDFCVAE/INDEX.html">M4 GeoSDFCVAE (36)</a>
        <a class="btn" href="M5_cGAN/INDEX.html">M5 cGAN (36)</a>
        <a class="btn btn-sec" href="ALL_MODELS_METRICS.csv">Download Master Metrics CSV</a>
    </div>

    <h2>Comparative 5-Model Benchmark Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Generative Architecture</th>
                <th>Mean Dual MAE</th>
                <th>Mean F1 (2px)</th>
                <th>Mean Dice</th>
                <th>Mean IoU</th>
                <th>Mean Apex Err</th>
                <th>Best Case</th>
                <th>Worst Case</th>
                <th>Panel Gallery</th>
            </tr>
        </thead>
        <tbody>
            {summary_rows}
        </tbody>
    </table>

    <h2>Summary Validation Visualizations</h2>
    <div class="fig-grid">
        <div class="fig-card">
            <h3>Figure 1: Mean Dual-View Contour MAE Across 5 Models</h3>
            <img src="summary_figures/fig1_model_mae_comparison.png" alt="Model MAE Comparison">
        </div>
        <div class="fig-card">
            <h3>Figure 2: Boundary F1 (2px) and Mask Dice Comparison</h3>
            <img src="summary_figures/fig2_model_dice_f1_comparison.png" alt="Model Dice F1 Comparison">
        </div>
    </div>
</body>
</html>
"""
    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def generate_summary_figures(summary_df, all_df):
    # Figure 1: Mean Contour MAE Comparison
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, facecolor='#FFFFFF')
    ax.set_facecolor('#FAFAFA')
    models = summary_df["model"].tolist()
    names = [m.replace("_", " ") for m in models]
    maes = summary_df["mean_mae_px"].tolist()
    colors = ['#0969DA', '#2DA44E', '#8250DF', '#BF8700', '#CF222E']

    bars = ax.bar(names, maes, color=colors, width=0.55, edgecolor='#D0D7DE', linewidth=1.2)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{val:.2f} px",
                ha='center', va='bottom', fontsize=10.5, fontweight='bold', color='#24292F')

    ax.set_ylabel("Mean Dual-View Contour MAE (pixels)", fontsize=11, fontweight='semibold')
    ax.set_title("Weld Crown Geometric Accuracy: Mean Contour MAE Across 5 Models (36 Samples)", fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0, max(maes) * 1.25)
    plt.tight_layout()
    p1 = os.path.join(SUMMARY_FIG_DIR, "fig1_model_mae_comparison.png")
    plt.savefig(p1, dpi=150, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: Boundary F1 & Mask Dice
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150, facecolor='#FFFFFF')
    ax.set_facecolor('#FAFAFA')
    x = np.arange(len(models))
    w = 0.35
    f1s = summary_df["mean_f1"].tolist()
    dices = summary_df["mean_dice"].tolist()

    b1 = ax.bar(x - w/2, f1s, width=w, label="Boundary F1 (2px)", color='#0969DA', edgecolor='#D0D7DE')
    b2 = ax.bar(x + w/2, dices, width=w, label="Mask Dice Score", color='#2DA44E', edgecolor='#D0D7DE')

    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{bar.get_height():.3f}",
                ha='center', va='bottom', fontsize=9.5, fontweight='semibold')
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{bar.get_height():.3f}",
                ha='center', va='bottom', fontsize=9.5, fontweight='semibold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("Metric Score [0.0 - 1.0]", fontsize=11, fontweight='semibold')
    ax.set_title("Weld Crown Overlap and Boundary Precision: F1 & Dice (36 Samples)", fontsize=12, fontweight='bold', pad=12)
    ax.legend(frameon=True, facecolor='#FFFFFF', edgecolor='#D0D7DE')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    p2 = os.path.join(SUMMARY_FIG_DIR, "fig2_model_dice_f1_comparison.png")
    plt.savefig(p2, dpi=150, facecolor='#FFFFFF', bbox_inches='tight')
    plt.close(fig)


def run_full_validation_suite():
    t_start = time.time()
    all_records = []
    summary_records = []

    print("=" * 80)
    print("  PHASE V12 FINAL ACADEMIC VALIDATION SUITE: 5 MODELS x 36 SAMPLES")
    print(f"  Target Base: {OUTPUT_BASE}")
    print("=" * 80)

    for m_key in MODEL_CONFIGS.keys():
        m_cfg = MODEL_CONFIGS[m_key]
        m_dir = os.path.join(OUTPUT_BASE, m_key)
        os.makedirs(m_dir, exist_ok=True)
        m_records = []

        print(f"\n>>> Processing Model: {m_cfg['display_name']} ...")
        t_m = time.time()

        for s_idx in range(1, 37):
            s_code = f"S{s_idx:02d}"
            panel_path = os.path.join(m_dir, f"{s_code}_panel.png")
            rec = render_academic_panel(s_code, s_idx, m_key, panel_path)
            m_records.append(rec)
            all_records.append(rec)

        # Save per-model CSV
        df_m = pd.DataFrame(m_records)
        csv_m = os.path.join(m_dir, "metrics.csv")
        df_m.to_csv(csv_m, index=False)

        # Build per-model INDEX.html
        html_m = os.path.join(m_dir, "INDEX.html")
        build_model_index_html(m_key, m_records, html_m)

        # Compute model summary
        mean_mae = df_m["dual_view_avg_mae_px"].mean()
        mean_f1 = df_m["dual_view_avg_f1_2px"].mean()
        mean_dice = df_m["dual_view_avg_dice"].mean()
        mean_iou = df_m["dual_view_avg_iou"].mean()
        mean_apex = df_m["dual_view_avg_apex_err_px"].mean()

        best_row = df_m.loc[df_m["dual_view_avg_mae_px"].idxmin()]
        worst_row = df_m.loc[df_m["dual_view_avg_mae_px"].idxmax()]

        summary_records.append({
            "model": m_key,
            "display_name": m_cfg["display_name"],
            "mean_mae_px": round(mean_mae, 2),
            "mean_f1": round(mean_f1, 3),
            "mean_dice": round(mean_dice, 3),
            "mean_iou": round(mean_iou, 3),
            "mean_apex_err_px": round(mean_apex, 2),
            "best_sample": best_row["sample"],
            "best_mae_px": round(best_row["dual_view_avg_mae_px"], 2),
            "worst_sample": worst_row["sample"],
            "worst_mae_px": round(worst_row["dual_view_avg_mae_px"], 2),
            "panels_rendered": len(m_records)
        })

        print(f"    Completed {len(m_records)} panels in {time.time() - t_m:.1f}s | Mean MAE: {mean_mae:.2f} px | Dice: {mean_dice:.3f}")

    # Save Master CSV
    df_all = pd.DataFrame(all_records)
    all_csv_path = os.path.join(OUTPUT_BASE, "ALL_MODELS_METRICS.csv")
    df_all.to_csv(all_csv_path, index=False)

    df_summary = pd.DataFrame(summary_records).sort_values("mean_mae_px")
    summary_csv_path = os.path.join(OUTPUT_BASE, "MODEL_COMPARISON_SUMMARY.csv")
    df_summary.to_csv(summary_csv_path, index=False)

    # Generate figures
    generate_summary_figures(df_summary, df_all)

    # Generate master gallery HTML
    master_html_path = os.path.join(OUTPUT_BASE, "GALLERY.html")
    build_unified_gallery_html(all_records, df_summary, master_html_path)

    total_time = time.time() - t_start
    print("\n" + "=" * 80)
    print(f"  ALL 180 ACADEMIC VALIDATION PANELS SUCCESSFULLY GENERATED! ({total_time:.1f} s)")
    print(f"  Master Metrics CSV:     {all_csv_path}")
    print(f"  Model Summary CSV:      {summary_csv_path}")
    print(f"  Unified Master Gallery: {master_html_path}")
    print("=" * 80)
    print("\nModel Benchmark Summary:")
    print(df_summary[["model", "mean_mae_px", "mean_f1", "mean_dice", "mean_apex_err_px", "best_sample", "worst_sample"]].to_string(index=False))


if __name__ == "__main__":
    run_full_validation_suite()
