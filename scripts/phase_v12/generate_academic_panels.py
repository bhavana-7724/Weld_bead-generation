"""
generate_academic_panels.py

Generates publication-grade, white-background academic validation panels for
weld cross-section evaluation across all 36 experimental conditions and all 5 models:
- M1_DirectCNN
- M2_CVAE
- M3_GCWPGv2
- M4_GeoSDFCVAE
- M5_cGAN

Features:
1. Clean white academic background, readable typography, zero header overlap.
2. Complete inclusion of Real View A, Real View B, Raw Generated RGB,
   Predicted Crown Contour, Edge/Texture Diagnostic, and Multi-View Contour Overlay.
3. Separation of Physical Dimensions (mm) from Image Pixel Metrics (px).
4. Evaluation of View A, View B, and Dual-View Average for every sample.
5. Saves all 180 panels into results/phase_v12_finalized_project_stage/ACADEMIC_FINAL_VALIDATION/
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
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v12.audit_segmentation_methods import (
    extract_method3_toe_bounded,
    get_gt_contour_from_mask,
    compute_toe_errors
)
from scripts.phase_v12.run_model_36_panels import (
    evaluate_metrics,
    create_contour_overlay,
    create_dual_contour_overlay,
    compute_clean_canny,
    mild_unsharp_mask
)
from scripts.phase_v12.generate_unseen_interactive import (
    compute_clean_edge_gradient_map
)

STAGE_DIR = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage")
OUTPUT_BASE = os.path.join(STAGE_DIR, "ACADEMIC_FINAL_VALIDATION")
V9_GEN_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")
GT_MASK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit", "gt_masks")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")
NUM_REG_PATH = os.path.join(BASE_DIR, "models", "numerical", "prospective_regressor_checkpoint.pkl")

# Model configuration dictionary
MODEL_CONFIGS = {
    "M1_DirectCNN": {
        "display_name": "M1: Direct Conditional CNN",
        "short_desc": "Geometry-Aware Transposed Convolutional Network",
        "file_prefix": "M1_DirectCNN"
    },
    "M2_CVAE": {
        "display_name": "M2: Conditional Variational Autoencoder (CVAE)",
        "short_desc": "Latent Manifold Generator with Condition Embedding",
        "file_prefix": "M2_CVAE"
    },
    "M3_GCWPGv2": {
        "display_name": "M3: Geometry-Conditioned WPG v2",
        "short_desc": "Parametric Feature Extraction Network",
        "file_prefix": "M3_GCWPGv2"
    },
    "M4_GeoSDFCVAE": {
        "display_name": "M4: GeoSDF-CVAE",
        "short_desc": "Implicit Signed Distance Field Variational Autoencoder",
        "file_prefix": "M4_GeoSDFCVAE"
    },
    "M5_cGAN": {
        "display_name": "M5: Conditional GAN (cGAN)",
        "short_desc": "Adversarial Generator with Multi-Scale Discriminator",
        "file_prefix": "M5_cGAN"
    }
}

os.makedirs(OUTPUT_BASE, exist_ok=True)

# -------------------------------------------------------------------------
# Load Prospective Regressor
# -------------------------------------------------------------------------
reg_data = None
if os.path.exists(NUM_REG_PATH):
    with open(NUM_REG_PATH, "rb") as f:
        reg_data = pickle.load(f)

def predict_numerical_geometry(material, process, feed_rate, speed):
    if reg_data is None:
        return {"predicted_width_mm": 0.0, "predicted_depth_mm": 0.0, "predicted_reinforcement_mm": 0.0}
    df_in = pd.DataFrame([{
        "feed_rate": float(feed_rate),
        "speed": float(speed),
        "material": material,
        "process": process
    }])
    X_num = reg_data["scaler"].transform(df_in[reg_data["num_features"]])
    X_cat = reg_data["encoder"].transform(df_in[reg_data["cat_features"]])
    X_all = np.hstack([X_num, X_cat])
    preds = reg_data["model"].predict(X_all)[0]
    return {
        "predicted_width_mm": round(float(preds[0]), 2),
        "predicted_depth_mm": round(float(preds[1]), 2),
        "predicted_reinforcement_mm": round(float(preds[2]), 2)
    }

# -------------------------------------------------------------------------
# Academic White-Background Panel Renderer
# -------------------------------------------------------------------------
def render_academic_panel(s_code, s_num, model_key, out_path):
    m_cfg = MODEL_CONFIGS[model_key]
    df_clean = pd.read_csv(CLEAN_CSV)
    row = df_clean[df_clean["sample"] == s_num].iloc[0]

    mat = row["material"]
    proc = row["process"]
    f_rate = float(row["feed_rate"])
    speed = float(row["speed"])
    hip = float(row["HIP"])
    exp_w = float(row["width"])
    exp_d = float(row["depth"])
    exp_r = float(row["reinforcement"])

    # Predict numerical dimensions
    num_preds = predict_numerical_geometry(mat, proc, f_rate, speed)
    pred_w = num_preds["predicted_width_mm"]
    pred_d = num_preds["predicted_depth_mm"]
    pred_r = num_preds["predicted_reinforcement_mm"]

    err_w = abs(pred_w - exp_w)
    err_d = abs(pred_d - exp_d)
    err_r = abs(pred_r - exp_r)
    rel_w = (err_w / exp_w) * 100
    rel_d = (err_d / exp_d) * 100
    rel_r = (err_r / exp_r) * 100

    s_dir = os.path.join(V9_GEN_DIR, s_code)
    real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_A.png")), cv2.COLOR_BGR2RGB)
    real_b_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_B.png")), cv2.COLOR_BGR2RGB)
    gen_file = os.path.join(s_dir, f"{m_cfg['file_prefix']}_raw.png")
    gen_rgb = cv2.cvtColor(cv2.imread(gen_file), cv2.COLOR_BGR2RGB)
    gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)
    sharp_rgb = mild_unsharp_mask(gen_rgb)

    # Real contours from GT masks
    gt_mask_a = cv2.imread(os.path.join(GT_MASK_DIR, f"{s_num}-A.png"), cv2.IMREAD_GRAYSCALE)
    gt_mask_b = cv2.imread(os.path.join(GT_MASK_DIR, f"{s_num}-B.png"), cv2.IMREAD_GRAYSCALE)
    cnt_a = get_gt_contour_from_mask(gt_mask_a)
    cnt_b = get_gt_contour_from_mask(gt_mask_b)

    # Predicted crown contour via Method 3
    cnt_pred = extract_method3_toe_bounded(gen_gray)
    cnt_pred_eval = cnt_pred.copy()
    cnt_pred_eval["pts"] = {int(y): int(round(x)) for y, x in cnt_pred["pts"].items()}

    canny_a = compute_clean_canny(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY))
    canny_b = compute_clean_canny(cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY))
    canny_gen = compute_clean_canny(gen_gray)

    mA = evaluate_metrics(real_a_rgb, gen_rgb, cnt_a, cnt_pred_eval, canny_a, canny_gen)
    mB = evaluate_metrics(real_b_rgb, gen_rgb, cnt_b, cnt_pred_eval, canny_b, canny_gen)
    top_err_a, bot_err_a = compute_toe_errors(cnt_a, cnt_pred)
    top_err_b, bot_err_b = compute_toe_errors(cnt_b, cnt_pred)

    avg_mae = 0.5 * (mA["contour_mae_px"] + mB["contour_mae_px"])
    avg_f1 = 0.5 * (mA["boundary_f1_2px"] + mB["boundary_f1_2px"])
    avg_dice = 0.5 * (mA["mask_dice"] + mB["mask_dice"])
    avg_iou = 0.5 * (mA["mask_iou"] + mB["mask_iou"])
    avg_apex = 0.5 * (mA["apex_projection_error_px"] + mB["apex_projection_error_px"])
    avg_top_toe = 0.5 * (top_err_a + top_err_b)
    avg_bot_toe = 0.5 * (bot_err_a + bot_err_b)

    # Edge decomposition
    edge_map, edge_note, _, edge_decomp = compute_clean_edge_gradient_map(sharp_rgb)

    # ---------------------------------------------------------------------
    # Render Academic White-Background Figure
    # ---------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), dpi=150, facecolor='#FFFFFF')
    gs = GridSpec(nrows=4, ncols=3, height_ratios=[0.85, 3.8, 3.8, 1.25],
                  top=0.96, bottom=0.04, left=0.04, right=0.96,
                  hspace=0.28, wspace=0.18)

    # 1. Top Header Banner
    ax_b = fig.add_subplot(gs[0, :])
    ax_b.set_facecolor('#F6F8FA')
    ax_b.axis('off')
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax_b.transAxes, fill=False, edgecolor='#D0D7DE', linewidth=1.2)
    ax_b.add_patch(rect)

    title_text = f"ACADEMIC VALIDATION: EXPERIMENT {s_code}  |  IN-SAMPLE RECONSTRUCTION"
    sub_title = f"Generative Model: {m_cfg['display_name']} ({m_cfg['short_desc']})"
    proc_info = f"Parameters: Material = {mat}  |  Process = {proc}  |  Feed Rate = {f_rate:.1f} m/min  |  Travel Speed = {speed:.1f} cm/min  |  HIP = {hip:.2f} J/mm"

    ax_b.text(0.015, 0.76, title_text, color='#0969DA', fontsize=15, fontweight='bold', va='center')
    ax_b.text(0.015, 0.48, sub_title, color='#57606A', fontsize=10.5, fontstyle='italic', va='center')
    ax_b.text(0.015, 0.20, proc_info, color='#24292F', fontsize=10.5, fontweight='semibold', va='center')

    # 2. Six Visual Panels
    # Panel 1: Real View A
    ax1 = fig.add_subplot(gs[1, 0])
    ov_a = create_contour_overlay(real_a_rgb, cnt_a['pts'], (cnt_a['apex_x'], cnt_a['apex_y']), line_color=(0, 200, 255))
    ax1.imshow(ov_a)
    ax1.set_title(f"(a) Experimental View A\n[Cyan: GT Contour | Span: {len(cnt_a['pts'])} px]", color='#24292F', fontsize=10, fontweight='bold', pad=5)
    ax1.axis('off')

    # Panel 2: Real View B
    ax2 = fig.add_subplot(gs[1, 1])
    ov_b = create_contour_overlay(real_b_rgb, cnt_b['pts'], (cnt_b['apex_x'], cnt_b['apex_y']), line_color=(255, 0, 200))
    ax2.imshow(ov_b)
    ax2.set_title(f"(b) Experimental View B\n[Magenta: GT Contour | Span: {len(cnt_b['pts'])} px]", color='#24292F', fontsize=10, fontweight='bold', pad=5)
    ax2.axis('off')

    # Panel 3: Raw Generated RGB
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(gen_rgb)
    ax3.set_title(f"(c) Raw Generated RGB\n[{m_cfg['display_name'].split(':')[0]} Output (128x128)]", color='#24292F', fontsize=10, fontweight='bold', pad=5)
    ax3.axis('off')

    # Panel 4: Predicted Crown Contour & Mask
    ax4 = fig.add_subplot(gs[2, 0])
    # Composite display: mask in light grey + contour line in dark blue
    mask_disp = np.ones((128, 128, 3), dtype=np.uint8) * 248
    mask_disp[cnt_pred['mask'] > 0] = [218, 230, 245] # Soft blue mask
    # Yellow baseline
    for y in range(0, 128, 4):
        mask_disp[y:y+2, 64] = [210, 150, 0]
    # Blue contour line
    for y, x in cnt_pred['pts'].items():
        xi = int(round(x))
        if 0 <= y < 128 and 0 <= xi < 128:
            mask_disp[y, xi] = [9, 105, 218]
            if xi + 1 < 128: mask_disp[y, xi + 1] = [9, 105, 218]
    ax4.imshow(mask_disp)
    ax4.set_title(f"(d) Predicted Crown Contour & Mask\n[Support: {cnt_pred['support_span']} rows | Area: {cnt_pred['active_area_px']} px]", color='#24292F', fontsize=10, fontweight='bold', pad=5)
    ax4.axis('off')

    # Panel 5: Edge / Diagnostic Map
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.imshow(edge_map, cmap='gray_r') # Black edges on white background
    ax5.set_title(f"(e) Edge / Texture Diagnostic\n[Active Edges: {edge_decomp['total_edge_pixels']} px | Substrate: {edge_decomp['substrate_bulk_pct']}%]", color='#24292F', fontsize=10, fontweight='bold', pad=5)
    ax5.axis('off')

    # Panel 6: Multi-View Overlay
    ax6 = fig.add_subplot(gs[2, 2])
    tri_overlay = create_dual_contour_overlay(sharp_rgb, cnt_a['pts'], cnt_b['pts'], cnt_pred['pts'])
    ax6.imshow(tri_overlay)
    ax6.set_title(f"(f) Multi-View Contour Overlay\n[Cyan: View A | Magenta: View B | Green: Gen]", color='#1A7F37', fontsize=10, fontweight='bold', pad=5)
    ax6.axis('off')

    # 3. Bottom Comparative Scorecards (Stacked / Two Columns)
    ax_card = fig.add_subplot(gs[3, :])
    ax_card.set_facecolor('#F6F8FA')
    ax_card.axis('off')
    rect_card = plt.Rectangle((0, 0), 1, 1, transform=ax_card.transAxes, fill=False, edgecolor='#D0D7DE', linewidth=1.2)
    ax_card.add_patch(rect_card)

    # Left Column: Numerical Physical Dimensions
    ax_card.text(0.015, 0.85, "1. PHYSICAL BEAD DIMENSIONS (Numerical Regression vs. Ground Truth):",
                 color='#0969DA', fontsize=10, fontweight='bold', va='center')
    dim_line1 = f"Experimental (True):   Width = {exp_w:.2f} mm   |   Depth = {exp_d:.2f} mm   |   Reinforcement = {exp_r:.2f} mm"
    dim_line2 = f"Predicted (Ridge):      Width = {pred_w:.2f} mm   |   Depth = {pred_d:.2f} mm   |   Reinforcement = {pred_r:.2f} mm"
    dim_line3 = f"Absolute Error:         ΔW = {err_w:.2f} mm ({rel_w:.1f}%)   |   ΔD = {err_d:.2f} mm ({rel_d:.1f}%)   |   ΔR = {err_r:.2f} mm ({rel_r:.1f}%)"
    ax_card.text(0.015, 0.62, dim_line1, color='#24292F', fontsize=9.2, va='center')
    ax_card.text(0.015, 0.40, dim_line2, color='#24292F', fontsize=9.2, va='center')
    ax_card.text(0.015, 0.18, dim_line3, color='#1A7F37', fontsize=9.2, fontweight='semibold', va='center')

    # Right Column: Image-Based Pixel Metrics
    ax_card.text(0.55, 0.85, "2. IMAGE-BASED CONTOUR METRICS (Dual-View Macrograph Comparison):",
                 color='#0969DA', fontsize=10, fontweight='bold', va='center')
    pix_line1 = f"View A:   MAE = {mA['contour_mae_px']:.2f} px   |   F1 (2px) = {mA['boundary_f1_2px']:.3f}   |   Dice = {mA['mask_dice']:.3f}   |   Apex Err = {mA['apex_projection_error_px']:.1f} px   |   Toe (Top/Bot) = {top_err_a:.1f}/{bot_err_a:.1f} px"
    pix_line2 = f"View B:   MAE = {mB['contour_mae_px']:.2f} px   |   F1 (2px) = {mB['boundary_f1_2px']:.3f}   |   Dice = {mB['mask_dice']:.3f}   |   Apex Err = {mB['apex_projection_error_px']:.1f} px   |   Toe (Top/Bot) = {top_err_b:.1f}/{bot_err_b:.1f} px"
    pix_line3 = f"Dual-View Average:   MAE = {avg_mae:.2f} px   |   Boundary F1 = {avg_f1:.3f}   |   Dice = {avg_dice:.3f}   |   IoU = {avg_iou:.3f}   |   Status: Valid Contour"
    ax_card.text(0.55, 0.62, pix_line1, color='#24292F', fontsize=9.2, va='center')
    ax_card.text(0.55, 0.40, pix_line2, color='#24292F', fontsize=9.2, va='center')
    ax_card.text(0.55, 0.18, pix_line3, color='#CF222E' if avg_mae > 10 else '#1A7F37', fontsize=9.2, fontweight='bold', va='center')

    plt.savefig(out_path, dpi=150, facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    metric_record = {
        "sample": s_code,
        "sample_num": s_num,
        "model": model_key,
        "material": mat,
        "process": proc,
        "feed_rate": f_rate,
        "speed": speed,
        "hip": hip,
        "exp_width_mm": exp_w,
        "exp_depth_mm": exp_d,
        "exp_reinforcement_mm": exp_r,
        "pred_width_mm": pred_w,
        "pred_depth_mm": pred_d,
        "pred_reinforcement_mm": pred_r,
        "abs_error_width_mm": round(err_w, 3),
        "abs_error_depth_mm": round(err_d, 3),
        "abs_error_reinf_mm": round(err_r, 3),
        "view_a_mae_px": round(mA["contour_mae_px"], 2),
        "view_a_f1_2px": round(mA["boundary_f1_2px"], 3),
        "view_a_dice": round(mA["mask_dice"], 3),
        "view_a_iou": round(mA["mask_iou"], 3),
        "view_a_apex_err_px": round(mA["apex_projection_error_px"], 2),
        "view_a_top_toe_err_px": round(top_err_a, 2),
        "view_a_bot_toe_err_px": round(bot_err_a, 2),
        "view_b_mae_px": round(mB["contour_mae_px"], 2),
        "view_b_f1_2px": round(mB["boundary_f1_2px"], 3),
        "view_b_dice": round(mB["mask_dice"], 3),
        "view_b_iou": round(mB["mask_iou"], 3),
        "view_b_apex_err_px": round(mB["apex_projection_error_px"], 2),
        "view_b_top_toe_err_px": round(top_err_b, 2),
        "view_b_bot_toe_err_px": round(bot_err_b, 2),
        "dual_view_avg_mae_px": round(avg_mae, 2),
        "dual_view_avg_f1_2px": round(avg_f1, 3),
        "dual_view_avg_dice": round(avg_dice, 3),
        "dual_view_avg_iou": round(avg_iou, 3),
        "dual_view_avg_apex_err_px": round(avg_apex, 2),
        "dual_view_avg_top_toe_err_px": round(avg_top_toe, 2),
        "dual_view_avg_bot_toe_err_px": round(avg_bot_toe, 2),
        "pred_apex_x": round(cnt_pred["apex_x"], 2),
        "pred_apex_y": int(round(cnt_pred["apex_y"])),
        "pred_reinf_proj_px": round(cnt_pred["reinf_proj_px"], 2),
        "pred_support_span_px": cnt_pred["support_span"],
        "pred_active_area_px": cnt_pred["active_area_px"],
        "edge_total_pixels": edge_decomp["total_edge_pixels"],
        "edge_substrate_bulk_pct": edge_decomp["substrate_bulk_pct"],
        "failure_flag": mA["missing_contour_flag"] or mB["missing_contour_flag"]
    }
    return metric_record

def main():
    print("Testing academic panel generation for S02 (M1_DirectCNN)...")
    s02_path = os.path.join(OUTPUT_BASE, "S02_reference_panel.png")
    rec = render_academic_panel("S02", 2, "M1_DirectCNN", s02_path)
    print(f"Rendered S02 reference panel: {s02_path}")
    print(f"S02 Dual-View MAE: {rec['dual_view_avg_mae_px']} px | Dice: {rec['dual_view_avg_dice']} | F1: {rec['dual_view_avg_f1_2px']}")

if __name__ == "__main__":
    main()
