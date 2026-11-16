"""
run_model_36_panels.py

Phase V12 Finalized Project Stage: Individual 36-Sample Panel Generator & Benchmarking
Generates 36 individual high-resolution presentation panels for a designated model.

Each individual sample panel contains:
1. Header: Sample ID, Material, Process, Feed rate, Travel speed, Heat Input Parameter (HIP).
2. Ground-Truth Experimental Specs: Width, Depth, Reinforcement (mm).
3. Visual Blocks (8 panels):
   - Experimental Macrograph View A with detected crown contour & apex
   - Experimental Macrograph View B (inter-view reference) with detected crown contour
   - Raw Generated RGB Image from model
   - Sharpened Generated RGB Image (mild unsharp-mask for visual inspection)
   - Extracted Generated Crown Contour (B&W binary line map)
   - Canny Edge Detection Map (intensity boundary gradients)
   - Dual-Contour Overlay (Real A cyan, Real B magenta, Generated lime green on macrograph)
   - Predicted Binary Weld Crown Mask (deposit region X < 64)
4. Comprehensive Quantitative Pixel Metrics Card:
   - Contour MAE (px), Contour RMSE (px), Boundary F1 (2px), Mask Dice, Mask IoU,
     Apex Projection Error (px), Edge F1, ROI SSIM, PSNR (dB), Contour Quality Status.

Outputs are saved to:
  results/phase_v12_finalized_project_stage/sample_panels/{MODEL_FOLDER}/S{xx}_panel.png
  results/phase_v12_finalized_project_stage/metrics/{MODEL_FOLDER}_metrics.csv
"""

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")
ALIGNED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")

V9_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware")
V9_GEN_DIR = os.path.join(V9_DIR, "generated_36")

STAGE_DIR = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage")
SAMPLE_PANELS_BASE = os.path.join(STAGE_DIR, "sample_panels")
METRICS_DIR = os.path.join(STAGE_DIR, "metrics")

MODEL_CONFIGS = {
    "M1": {
        "folder": "M1_DirectCNN",
        "raw_prefix": "M1_DirectCNN",
        "display_name": "M1: Direct Conditional CNN",
        "short_desc": "End-to-end conditional convolutional decoder directly predicting weld macrograph RGB"
    },
    "M2": {
        "folder": "M2_CVAE",
        "raw_prefix": "M2_CVAE",
        "display_name": "M2: Conditional VAE (CVAE)",
        "short_desc": "Probabilistic latent generative model conditioned on weld process parameters"
    },
    "M3": {
        "folder": "M3_GCWPGv2",
        "raw_prefix": "M3_GCWPGv2",
        "display_name": "M3: GC-WPG-v2 Multi-Scale Spatial Prior",
        "short_desc": "Multi-scale spatial prior CNN incorporating geometric constraint loss"
    },
    "M4": {
        "folder": "M4_GeoSDFCVAE",
        "raw_prefix": "M4_GeoSDFCVAE",
        "display_name": "M4: GeoSDF-CVAE Continuous SDF",
        "short_desc": "Continuous Signed Distance Field VAE explicitly predicting geometry distance maps"
    },
    "M5": {
        "folder": "M5_cGAN",
        "raw_prefix": "M5_cGAN",
        "display_name": "M5: Geometry-Conditioned cGAN",
        "short_desc": "Adversarial network combining PatchGAN discriminator with spatial geometry conditioning"
    }
}

# =========================================================================
# Robust Contour Extraction & Preprocessing
# =========================================================================
def extract_robust_crown_contour(gray_img, x_baseline=64, y_min=18, y_max=112):
    """
    Extracts high-confidence external weld crown boundary:
    - Adaptive background estimation from far-left margin
    - Local thresholding on crown deposit region (X < 64)
    - Morphological cleaning to prevent isolated spatter or noise
    - Enforces leftward crown boundary: X_crown(Y) <= X_baseline
    """
    bg_val = np.median(gray_img[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))
    
    metal_roi = (gray_img[:, :x_baseline] < adaptive_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    metal_cleaned = cv2.morphologyEx(metal_roi, cv2.MORPH_OPEN, kernel)
    
    pts = {}
    for y in range(y_min, y_max + 1):
        idx = np.where(metal_cleaned[y, :x_baseline] == 1)[0]
        if len(idx) > 0:
            x_min = int(idx[0])
            if x_min < x_baseline:
                pts[y] = x_min
                
    mask = np.zeros((128, 128), dtype=np.uint8)
    if len(pts) >= 8:
        for y, x_min in pts.items():
            mask[y, x_min:x_baseline] = 1

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + np.argmax(areas)
        mask = (labels == largest_label).astype(np.uint8)
        pts = {}
        for y in range(y_min, y_max + 1):
            idx = np.where(mask[y, :x_baseline] == 1)[0]
            if len(idx) > 0:
                pts[y] = int(idx[0])

    if len(pts) > 0:
        apex_y = min(pts, key=pts.get)
        apex_x = pts[apex_y]
        reinf_proj = 64.0 - apex_x
        active_area = int(np.sum(mask))
        sorted_ys = sorted(pts.keys())
        top_toe = (float(pts[sorted_ys[0]]), float(sorted_ys[0]))
        bot_toe = (float(pts[sorted_ys[-1]]), float(sorted_ys[-1]))
    else:
        apex_y, apex_x, reinf_proj, active_area = 64, 64, 0.0, 0
        top_toe, bot_toe = (64.0, 64.0), (64.0, 64.0)

    return {
        "pts": pts,
        "apex_x": float(apex_x),
        "apex_y": float(apex_y),
        "reinf_proj_px": float(reinf_proj),
        "top_toe": top_toe,
        "bot_toe": bot_toe,
        "active_area_px": active_area,
        "mask": mask
    }

def mild_unsharp_mask(rgb_img, alpha=0.8, sigma=1.0):
    blurred = cv2.GaussianBlur(rgb_img, (0, 0), sigma)
    sharp = rgb_img.astype(np.float32) + alpha * (rgb_img.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)

def compute_clean_canny(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return cv2.Canny(blurred, lower, upper)

def create_contour_image(pts, x_baseline=64):
    cnt_img = np.zeros((128, 128), dtype=np.uint8)
    for y, x in pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            cnt_img[y, x] = 255
    return cnt_img

def create_contour_overlay(base_rgb, pts, apex_xy=None, x_baseline=64, line_color=(0, 255, 0)):
    overlay = base_rgb.copy()
    for y in range(0, 128, 4):
        overlay[y:y+2, x_baseline] = [255, 255, 0] # Yellow dashed baseline

    for y, x in pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            overlay[y, x] = list(line_color)
            if x + 1 < 128: overlay[y, x + 1] = list(line_color)

    if apex_xy is not None:
        ax, ay = int(round(apex_xy[0])), int(round(apex_xy[1]))
        if 0 <= ay < 128 and 0 <= ax < 128:
            cv2.circle(overlay, (ax, ay), 2, (255, 0, 0), -1)
    return overlay

def create_dual_contour_overlay(base_rgb, pts_real_a, pts_real_b, pts_gen, x_baseline=64):
    overlay = cv2.cvtColor(cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    overlay = (overlay.astype(np.float32) * 0.75).astype(np.uint8)

    # Baseline
    for y in range(0, 128, 4):
        overlay[y:y+2, x_baseline] = [255, 255, 0]

    # Real A (Cyan)
    for y, x in pts_real_a.items():
        if 0 <= y < 128 and 0 <= x < 128:
            overlay[y, x] = [0, 220, 255]

    # Real B (Magenta)
    for y, x in pts_real_b.items():
        if 0 <= y < 128 and 0 <= x < 128:
            overlay[y, x] = [255, 0, 220]

    # Generated (Lime Green - 2px thick)
    for y, x in pts_gen.items():
        if 0 <= y < 128 and 0 <= x < 128:
            overlay[y, x] = [0, 255, 0]
            if x - 1 >= 0: overlay[y, x - 1] = [0, 255, 0]

    return overlay

# =========================================================================
# Quantitative Pixel Metric Evaluator
# =========================================================================
def evaluate_metrics(real_rgb, gen_rgb, real_cnt, gen_cnt, real_canny, gen_canny):
    # SSIM & PSNR
    full_ssim = float(ssim_func(real_rgb, gen_rgb, channel_axis=2, data_range=255))
    psnr = float(psnr_func(real_rgb, gen_rgb, data_range=255))

    # ROI SSIM (Deposit region X < 64)
    r_roi = real_rgb.copy()
    g_roi = gen_rgb.copy()
    r_roi[:, 64:] = 0
    g_roi[:, 64:] = 0
    roi_ssim = float(ssim_func(r_roi, g_roi, channel_axis=2, data_range=255))

    # Contour MAE and RMSE
    r_pts = real_cnt.get("pts", {})
    g_pts = gen_cnt.get("pts", {})
    common_ys = sorted(list(set(r_pts.keys()).intersection(set(g_pts.keys()))))

    if len(common_ys) >= 5:
        diffs = [abs(r_pts[y] - g_pts[y]) for y in common_ys]
        contour_mae_px = float(np.mean(diffs))
        contour_rmse_px = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
        missing_contour_flag = False
        quality_note = "Continuous valid contour"
    else:
        contour_mae_px = 35.0
        contour_rmse_px = 40.0
        missing_contour_flag = True
        quality_note = "Sparse/failed contour extraction"

    # Apex Projection Error
    apex_proj_err = float(abs(real_cnt.get("reinf_proj_px", 0) - gen_cnt.get("reinf_proj_px", 0)))

    # Mask Dice & IoU
    r_mask = real_cnt.get("mask", np.zeros((128, 128), dtype=np.uint8))
    g_mask = gen_cnt.get("mask", np.zeros((128, 128), dtype=np.uint8))
    inter = np.sum((r_mask > 0) & (g_mask > 0))
    union = np.sum((r_mask > 0) | (g_mask > 0))
    mask_iou = float(inter / (union + 1e-8))
    mask_dice = float(2.0 * inter / (np.sum(r_mask > 0) + np.sum(g_mask > 0) + 1e-8))

    # Boundary F1 Score at d_tol = 2 px
    r_edge = create_contour_image(r_pts)
    g_edge = create_contour_image(g_pts)
    kernel_2px = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dil_r = cv2.dilate(r_edge, kernel_2px)
    dil_g = cv2.dilate(g_edge, kernel_2px)
    tp_p = np.sum((g_edge == 255) & (dil_r == 255))
    prec = float(tp_p / (np.sum(g_edge == 255) + 1e-8))
    tp_r = np.sum((r_edge == 255) & (dil_g == 255))
    rec = float(tp_r / (np.sum(r_edge == 255) + 1e-8))
    b_f1 = float(2 * prec * rec / (prec + rec + 1e-8))

    # Edge F1 on Canny Maps
    dil_rc = cv2.dilate(real_canny, kernel_2px)
    dil_gc = cv2.dilate(gen_canny, kernel_2px)
    tp_pc = np.sum((gen_canny == 255) & (dil_rc == 255))
    p_c = float(tp_pc / (np.sum(gen_canny == 255) + 1e-8))
    tp_rc = np.sum((real_canny == 255) & (dil_gc == 255))
    r_c = float(tp_rc / (np.sum(real_canny == 255) + 1e-8))
    edge_f1 = float(2 * p_c * r_c / (p_c + r_c + 1e-8))

    return {
        "contour_mae_px": contour_mae_px,
        "contour_rmse_px": contour_rmse_px,
        "apex_projection_error_px": apex_proj_err,
        "boundary_f1_2px": b_f1,
        "mask_dice": mask_dice,
        "mask_iou": mask_iou,
        "edge_f1": edge_f1,
        "roi_ssim": roi_ssim,
        "full_image_ssim": full_ssim,
        "psnr": psnr,
        "missing_contour_flag": missing_contour_flag,
        "segmentation_quality_note": quality_note
    }

# =========================================================================
# High-Resolution Panel Assembly
# =========================================================================
def render_sample_panel(s_code, row, m_cfg, real_a_rgb, real_b_rgb, gen_rgb,
                        cnt_a, cnt_b, cnt_gen, canny_a, canny_gen, metrics_a, metrics_avg,
                        out_path):
    """
    Renders an 1800x1200 academic presentation plate with 8 visual subpanels,
    clear headers, experimental bead geometry, and quantitative scorecards.
    """
    fig = plt.figure(figsize=(18, 12), dpi=150, facecolor='#0D1117')
    
    # Grid: Top banner (0.0 to 0.14), Visuals (0.16 to 0.82), Scorecard (0.84 to 0.98)
    gs = GridSpec(nrows=3, ncols=4, height_ratios=[0.9, 3.8, 3.8],
                  top=0.96, bottom=0.04, left=0.04, right=0.96,
                  hspace=0.28, wspace=0.18)

    # ---------------- Top Banner ----------------
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_facecolor('#161B22')
    ax_banner.axis('off')

    # Border around banner
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax_banner.transAxes,
                         fill=False, edgecolor='#30363D', linewidth=1.5)
    ax_banner.add_patch(rect)

    # Text in banner
    mat = row['material']
    proc = row['process']
    f_rate = float(row['feed_rate'])
    speed = float(row['speed'])
    hip = float(row['HIP'])
    width = float(row['width'])
    depth = float(row['depth'])
    reinf = float(row['reinforcement'])

    title_str = f"WELD BEAD CROSS-SECTION ANALYSIS — EXPERIMENT {s_code}"
    sub_title = f"MODEL: {m_cfg['display_name']} ({m_cfg['short_desc']})"
    proc_info = f"Material: {mat}  |  Process: {proc}  |  Wire Feed Rate: {f_rate:.1f} m/min  |  Travel Speed: {speed:.1f} cm/min  |  HIP: {hip:.2f} J/mm"
    geom_info = f"Experimental Ground-Truth: Width = {width:.2f} mm  |  Depth = {depth:.2f} mm  |  Reinforcement = {reinf:.2f} mm"

    ax_banner.text(0.02, 0.78, title_str, color='#58A6FF', fontsize=16, fontweight='bold', va='center')
    ax_banner.text(0.02, 0.52, sub_title, color='#8B949E', fontsize=11, fontstyle='italic', va='center')
    ax_banner.text(0.02, 0.24, proc_info, color='#F0F6FC', fontsize=11, fontweight='semibold', va='center')
    ax_banner.text(0.98, 0.24, geom_info, color='#3FB950', fontsize=11, fontweight='bold', ha='right', va='center')

    # ---------------- Image Visuals (2 rows x 4 cols) ----------------
    # 1. Real View A
    ax1 = fig.add_subplot(gs[1, 0])
    overlay_a = create_contour_overlay(real_a_rgb, cnt_a['pts'], (cnt_a['apex_x'], cnt_a['apex_y']), line_color=(0, 220, 255))
    ax1.imshow(overlay_a)
    ax1.set_title("1. Experimental View A (Macrograph)\n[Cyan: Extracted Crown Contour]", color='#F0F6FC', fontsize=10, pad=6)
    ax1.axis('off')

    # 2. Real View B
    ax2 = fig.add_subplot(gs[1, 1])
    overlay_b = create_contour_overlay(real_b_rgb, cnt_b['pts'], (cnt_b['apex_x'], cnt_b['apex_y']), line_color=(255, 0, 220))
    ax2.imshow(overlay_b)
    ax2.set_title("2. Experimental View B (Reference)\n[Magenta: Crown Contour]", color='#F0F6FC', fontsize=10, pad=6)
    ax2.axis('off')

    # 3. Raw Generated RGB
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(gen_rgb)
    ax3.set_title("3. Raw Generated RGB\n[Direct Neural Output (128x128)]", color='#F0F6FC', fontsize=10, pad=6)
    ax3.axis('off')

    # 4. Sharpened Generated RGB
    ax4 = fig.add_subplot(gs[1, 3])
    gen_sharp = mild_unsharp_mask(gen_rgb)
    ax4.imshow(gen_sharp)
    ax4.set_title("4. Sharpened RGB (Visual Inspection)\n[Mild Unsharp Mask: alpha=0.8]", color='#F0F6FC', fontsize=10, pad=6)
    ax4.axis('off')

    # 5. Extracted Generated Contour
    ax5 = fig.add_subplot(gs[2, 0])
    cnt_img = create_contour_image(cnt_gen['pts'])
    ax5.imshow(cnt_img, cmap='gray')
    ax5.axvline(64, color='yellow', linestyle='--', linewidth=1.0)
    ax5.set_title("5. Generated Crown Contour (B&W)\n[Outer Boundary on Baseline X=64]", color='#F0F6FC', fontsize=10, pad=6)
    ax5.axis('off')

    # 6. Canny Edge Detection
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.imshow(canny_gen, cmap='inferno')
    ax6.set_title("6. Canny Edge Detection Map\n[Gradient Intensity Boundaries]", color='#F0F6FC', fontsize=10, pad=6)
    ax6.axis('off')

    # 7. Dual Contour Overlay (Real vs Generated)
    ax7 = fig.add_subplot(gs[2, 2])
    dual_overlay = create_dual_contour_overlay(real_a_rgb, cnt_a['pts'], cnt_b['pts'], cnt_gen['pts'])
    ax7.imshow(dual_overlay)
    ax7.set_title("7. Dual Contour Overlay\n[Cyan: Real A | Mag: Real B | Green: Gen]", color='#F0F6FC', fontsize=10, pad=6)
    ax7.axis('off')

    # 8. Predicted Binary Weld Mask
    ax8 = fig.add_subplot(gs[2, 3])
    gen_mask = cnt_gen['mask']
    ax8.imshow(gen_mask, cmap='viridis')
    ax8.axvline(64, color='yellow', linestyle='--', linewidth=1.0)
    ax8.set_title("8. Predicted Binary Crown Mask\n[Deposit Region X < 64]", color='#F0F6FC', fontsize=10, pad=6)
    ax8.axis('off')

    # ---------------- Bottom Scorecard Bar ----------------
    # Add a custom text summary across the bottom
    score_text = (
        f"QUANTITATIVE EVALUATION (vs View A / Dual-View Avg):  "
        f"Contour MAE = {metrics_a['contour_mae_px']:.2f} px ({metrics_avg['contour_mae_px']:.2f} px avg)  |  "
        f"Boundary F1 (2px) = {metrics_a['boundary_f1_2px']:.3f}  |  "
        f"Mask Dice = {metrics_a['mask_dice']:.3f}  |  "
        f"Mask IoU = {metrics_a['mask_iou']:.3f}  |  "
        f"Apex Error = {metrics_a['apex_projection_error_px']:.2f} px  |  "
        f"ROI SSIM = {metrics_a['roi_ssim']:.3f}  |  "
        f"PSNR = {metrics_a['psnr']:.1f} dB  |  "
        f"Status: {metrics_a['segmentation_quality_note']}"
    )
    fig.text(0.5, 0.012, score_text, color='#E6EDF3', fontsize=9.5,
             ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262D', edgecolor='#30363D', linewidth=1))

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

# =========================================================================
# Main Execution for a Specified Model
# =========================================================================
def process_model(model_key):
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{model_key}'. Allowed: {list(MODEL_CONFIGS.keys())}")

    m_cfg = MODEL_CONFIGS[model_key]
    folder_name = m_cfg["folder"]
    display_name = m_cfg["display_name"]
    raw_prefix = m_cfg["raw_prefix"]

    out_folder = os.path.join(SAMPLE_PANELS_BASE, folder_name)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    print("=" * 80)
    print(f"PROCESSING MODEL: {display_name}")
    print(f"Output Directory: {out_folder}")
    print("=" * 80)

    df_clean = pd.read_csv(CLEAN_CSV)
    metric_rows = []

    t0 = time.time()
    for idx, row in df_clean.iterrows():
        s_id = int(row['sample'])
        s_code = f"S{s_id:02d}"
        s_dir = os.path.join(V9_GEN_DIR, s_code)

        # 1. Load Real Images
        path_a = os.path.join(s_dir, "Real_A.png")
        path_b = os.path.join(s_dir, "Real_B.png")
        if not os.path.exists(path_a):
            path_a = os.path.join(ALIGNED_DIR, f"{s_id}-A.png")
        if not os.path.exists(path_b):
            path_b = os.path.join(ALIGNED_DIR, f"{s_id}-B.png")

        real_a_bgr = cv2.imread(path_a)
        real_b_bgr = cv2.imread(path_b)
        real_a_rgb = cv2.cvtColor(real_a_bgr, cv2.COLOR_BGR2RGB)
        real_b_rgb = cv2.cvtColor(real_b_bgr, cv2.COLOR_BGR2RGB)

        # 2. Load Model Raw Output
        gen_path = os.path.join(s_dir, f"{raw_prefix}_raw.png")
        if not os.path.exists(gen_path):
            raise FileNotFoundError(f"Missing raw generated image for {s_code}: {gen_path}")
        gen_bgr = cv2.imread(gen_path)
        gen_rgb = cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2RGB)

        # 3. Contour & Canny Extraction
        gray_a = cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY)
        gray_b = cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY)
        gray_gen = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)

        cnt_a = extract_robust_crown_contour(gray_a)
        cnt_b = extract_robust_crown_contour(gray_b)
        cnt_gen = extract_robust_crown_contour(gray_gen)

        canny_a = compute_clean_canny(gray_a)
        canny_b = compute_clean_canny(gray_b)
        canny_gen = compute_clean_canny(gray_gen)

        # 4. Metrics against View A, View B, and Dual-View Average
        m_a = evaluate_metrics(real_a_rgb, gen_rgb, cnt_a, cnt_gen, canny_a, canny_gen)
        m_b = evaluate_metrics(real_b_rgb, gen_rgb, cnt_b, cnt_gen, canny_b, canny_gen)

        m_avg = {
            "contour_mae_px": (m_a["contour_mae_px"] + m_b["contour_mae_px"]) / 2.0,
            "contour_rmse_px": (m_a["contour_rmse_px"] + m_b["contour_rmse_px"]) / 2.0,
            "apex_projection_error_px": (m_a["apex_projection_error_px"] + m_b["apex_projection_error_px"]) / 2.0,
            "boundary_f1_2px": (m_a["boundary_f1_2px"] + m_b["boundary_f1_2px"]) / 2.0,
            "mask_dice": (m_a["mask_dice"] + m_b["mask_dice"]) / 2.0,
            "mask_iou": (m_a["mask_iou"] + m_b["mask_iou"]) / 2.0,
            "edge_f1": (m_a["edge_f1"] + m_b["edge_f1"]) / 2.0,
            "roi_ssim": (m_a["roi_ssim"] + m_b["roi_ssim"]) / 2.0,
            "full_image_ssim": (m_a["full_image_ssim"] + m_b["full_image_ssim"]) / 2.0,
            "psnr": (m_a["psnr"] + m_b["psnr"]) / 2.0,
            "missing_contour_flag": m_a["missing_contour_flag"] or m_b["missing_contour_flag"],
            "segmentation_quality_note": m_a["segmentation_quality_note"]
        }

        # 5. Render & Save Panel
        panel_path = os.path.join(out_folder, f"{s_code}_panel.png")
        render_sample_panel(s_code, row, m_cfg, real_a_rgb, real_b_rgb, gen_rgb,
                            cnt_a, cnt_b, cnt_gen, canny_a, canny_gen, m_a, m_avg,
                            panel_path)

        # 6. Record Metrics
        metric_rows.append({
            "sample": s_id,
            "sample_code": s_code,
            "model": folder_name,
            "material": row['material'],
            "process": row['process'],
            "feed_rate": float(row['feed_rate']),
            "speed": float(row['speed']),
            "width_mm": float(row['width']),
            "depth_mm": float(row['depth']),
            "reinforcement_mm": float(row['reinforcement']),
            # View A Metrics
            "contour_mae_px_view_a": m_a["contour_mae_px"],
            "boundary_f1_2px_view_a": m_a["boundary_f1_2px"],
            "mask_dice_view_a": m_a["mask_dice"],
            "mask_iou_view_a": m_a["mask_iou"],
            "apex_err_px_view_a": m_a["apex_projection_error_px"],
            "edge_f1_view_a": m_a["edge_f1"],
            "roi_ssim_view_a": m_a["roi_ssim"],
            "psnr_view_a": m_a["psnr"],
            # Dual-View Average Metrics
            "contour_mae_px_avg": m_avg["contour_mae_px"],
            "boundary_f1_2px_avg": m_avg["boundary_f1_2px"],
            "mask_dice_avg": m_avg["mask_dice"],
            "mask_iou_avg": m_avg["mask_iou"],
            "apex_err_px_avg": m_avg["apex_projection_error_px"],
            "edge_f1_avg": m_avg["edge_f1"],
            "roi_ssim_avg": m_avg["roi_ssim"],
            "psnr_avg": m_avg["psnr"],
            "missing_contour": m_avg["missing_contour_flag"],
            "quality_status": m_avg["segmentation_quality_note"],
            "panel_path": panel_path
        })

        if (idx + 1) % 6 == 0 or (idx + 1) == len(df_clean):
            print(f"  Processed {idx + 1:02d}/36 samples -> {s_code} MAE View A: {m_a['contour_mae_px']:.2f} px, Dice: {m_a['mask_dice']:.3f}")

    df_metrics = pd.DataFrame(metric_rows)
    metrics_csv = os.path.join(METRICS_DIR, f"{folder_name}_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    elapsed = time.time() - t0

    print(f"\n[DONE] Completed {len(df_metrics)} sample panels in {elapsed:.2f}s ({elapsed/36:.3f}s/sample)")
    print(f"Summary for {folder_name}:")
    print(f"  Mean Contour MAE (View A): {df_metrics['contour_mae_px_view_a'].mean():.2f} +/- {df_metrics['contour_mae_px_view_a'].std():.2f} px")
    print(f"  Mean Contour MAE (Dual Avg): {df_metrics['contour_mae_px_avg'].mean():.2f} +/- {df_metrics['contour_mae_px_avg'].std():.2f} px")
    print(f"  Mean Mask Dice (Dual Avg): {df_metrics['mask_dice_avg'].mean():.3f} +/- {df_metrics['mask_dice_avg'].std():.3f}")
    print(f"  Mean Boundary F1 (Dual Avg): {df_metrics['boundary_f1_2px_avg'].mean():.3f} +/- {df_metrics['boundary_f1_2px_avg'].std():.3f}")
    print(f"  Panels Saved To: {out_folder}")
    print(f"  Metrics Saved To: {metrics_csv}")

    return df_metrics

def main():
    parser = argparse.ArgumentParser(description="Phase V12: Generate 36 Sample Panels for a Model")
    parser.add_argument("--model", type=str, default="M1",
                        choices=["M1", "M2", "M3", "M4", "M5", "ALL"],
                        help="Model key to run: M1, M2, M3, M4, M5, or ALL")
    args = parser.parse_args()

    if args.model == "ALL":
        for k in ["M1", "M2", "M3", "M4", "M5"]:
            process_model(k)
    else:
        process_model(args.model)

if __name__ == "__main__":
    main()
