"""
process_and_evaluate_36_samples.py

Stages B, C, D, and E for Phase V11 Local Contour Evaluation Pipeline

Processes all 36 physical welding experiments:
Stage B — Original Image Processing:
  - Loads aligned original macrographs (Views A and B)
  - Saves copies, grayscale, Canny edge detection, binary masks, clean outer weld contours, overlays.

Stage C — Generated Image Processing:
  - For available models: M1 (DirectCNN), M3 (GC-WPG-v2), M5 (cGAN), and B1 (PCA-Contour Baseline)
  - M6 (GF-WPG): Honestly reported as missing locally (awaiting remote A100 GPU training)
  - Saves: raw RGB, sharpened RGB (mild unsharp mask), Canny edge, binary mask, contour image,
    generated overlay, original vs generated contour overlay, original vs generated edge overlay.

Stage D — Pixel-Based Comparison Metrics:
  - Computes all 17 pixel metrics (Contour MAE, RMSE, Apex Projection Error, Top/Bottom Toe Error,
    Area Error, Symmetric Distance, Hausdorff-95, Boundary F1 2px, Mask Dice, Mask IoU, Edge F1,
    ROI SSIM, Full SSIM, PSNR, Missing Contour Flag, Quality Note).
  - Evaluates against View A, View B, and dual-view average.

Stage E — Output Organization:
  - Populates results/phase_v11_local_contours/generated_36/S01/ ... S36/
  - Generates all_models_comparison_panel.png for each sample.
  - Compiles detailed CSV, summary CSV, and model ranking CSV in metrics/.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")
ALIGNED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")

V9_GEN_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")
AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
SDF_DIR = os.path.join(AUDIT_DIR, "gt_sdf")

OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v11_local_contours")
OUT_ORIG_CONTOURS = os.path.join(OUT_BASE, "original_contours")
OUT_GEN36 = os.path.join(OUT_BASE, "generated_36")
OUT_METRICS = os.path.join(OUT_BASE, "metrics")

for d in [OUT_ORIG_CONTOURS, OUT_GEN36, OUT_METRICS]:
    os.makedirs(d, exist_ok=True)

# =========================================================================
# Helper: Robust Contour Extraction from Grayscale Macrograph
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

# =========================================================================
# Helper: Mild Unsharp Masking for Visualization
# =========================================================================
def mild_unsharp_mask(rgb_img, alpha=0.8, sigma=1.0):
    blurred = cv2.GaussianBlur(rgb_img, (0, 0), sigma)
    sharp = rgb_img.astype(np.float32) + alpha * (rgb_img.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)

# =========================================================================
# Helper: Compute Clean Canny Edge Map
# =========================================================================
def compute_clean_canny(gray_img):
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    return cv2.Canny(blurred, lower, upper)

# =========================================================================
# Helper: Draw Clean Black-and-White Contour Image
# =========================================================================
def create_contour_image(pts, x_baseline=64):
    cnt_img = np.zeros((128, 128), dtype=np.uint8)
    for y, x in pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            cnt_img[y, x] = 255
    return cnt_img

# =========================================================================
# Helper: Create Diagnostic Contour Overlay on RGB Image
# =========================================================================
def create_contour_overlay(base_rgb, pts, apex_xy=None, x_baseline=64, line_color=(0, 255, 0)):
    overlay = base_rgb.copy()
    # Draw yellow dashed line at baseline X=64
    for y in range(0, 128, 4):
        overlay[y:y+2, x_baseline] = [255, 255, 0]

    # Draw contour
    for y, x in pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            overlay[y, x] = list(line_color)
            if x + 1 < 128: overlay[y, x + 1] = list(line_color)

    # Draw apex marker
    if apex_xy is not None:
        ax, ay = int(round(apex_xy[0])), int(round(apex_xy[1]))
        if 0 <= ay < 128 and 0 <= ax < 128:
            cv2.circle(overlay, (ax, ay), 2, (255, 0, 0), -1) # Red apex dot
    return overlay

# =========================================================================
# Helper: Create Dual-Contour Overlay (Original vs Generated)
# =========================================================================
def create_dual_contour_overlay(base_rgb, pts_real_a, pts_real_b, pts_gen, x_baseline=64):
    overlay = cv2.cvtColor(cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    # Dim background slightly for contrast
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

    # Generated (Lime Green - thicker for visibility)
    for y, x in pts_gen.items():
        if 0 <= y < 128 and 0 <= x < 128:
            overlay[y, x] = [0, 255, 0]
            if x - 1 >= 0: overlay[y, x - 1] = [0, 255, 0]

    return overlay

# =========================================================================
# Helper: Create Edge Overlay (Real vs Generated Canny Edges)
# =========================================================================
def create_edge_comparison_overlay(real_canny, gen_canny):
    overlay = np.zeros((128, 128, 3), dtype=np.uint8)
    # Red: Real edges
    overlay[:, :, 0] = real_canny
    # Green: Generated edges
    overlay[:, :, 1] = gen_canny
    # Where both are active -> Red + Green = Yellow [255, 255, 0]
    return overlay

# =========================================================================
# Baseline B1: PCA Contour Regressor (Leave-One-Out Fitting)
# =========================================================================
class PCAContourEngine:
    def __init__(self, n_components=4, alpha=1.0):
        self.n_components = n_components
        self.alpha = alpha

    def fit_and_predict_all_loo(self, df_clean, all_contours_dict):
        """
        Fits PCA + Ridge Regression in a strict leave-one-out manner for all 36 samples:
        For sample i, models are trained on all samples j != i.
        """
        predictions = {}
        samples = df_clean['sample'].tolist()

        f_vals = df_clean['feed_rate'].values
        s_vals = df_clean['speed'].values
        f_min, f_max = float(f_vals.min()), float(f_vals.max())
        s_min, s_max = float(s_vals.min()), float(s_vals.max())

        def encode(row):
            fn = (float(row['feed_rate']) - f_min) / (f_max - f_min + 1e-8)
            sn = (float(row['speed']) - s_min) / (s_max - s_min + 1e-8)
            return [
                fn, sn,
                1.0 if row['material'] == 'IN718' else 0.0,
                1.0 if row['material'] == 'SS304L' else 0.0,
                1.0 if row['process'] == 'ColdArc' else 0.0,
                1.0 if row['process'] == 'GMAW' else 0.0
            ]

        # Build design matrix and contour arrays
        C_all = np.array([encode(r) for _, r in df_clean.iterrows()]) # (36, 6)
        
        # Average contours between View A and View B for each sample
        X_contours = []
        for s in samples:
            cnt_a = all_contours_dict[s]["A"]["cnt"]["pts"]
            cnt_b = all_contours_dict[s]["B"]["cnt"]["pts"]
            vec = []
            for y in range(18, 113):
                xa = cnt_a.get(y, 64)
                xb = cnt_b.get(y, 64)
                vec.append((xa + xb) / 2.0)
            X_contours.append(vec)
        X_contours = np.array(X_contours) # (36, 95)

        for i, s_target in enumerate(samples):
            # Training set: all j != i
            train_mask = np.ones(len(samples), dtype=bool)
            train_mask[i] = False

            C_train = C_all[train_mask]
            X_train = X_contours[train_mask]

            mean_cnt = np.mean(X_train, axis=0)
            centered = X_train - mean_cnt

            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            components = Vt[:self.n_components] # (K, 95)
            W_train = np.dot(centered, components.T) # (35, K)

            N_tr = C_train.shape[0]
            C_bias = np.hstack([C_train, np.ones((N_tr, 1))])
            reg_I = self.alpha * np.eye(C_bias.shape[1])
            reg_I[-1, -1] = 0.0
            W_weights = np.linalg.solve(np.dot(C_bias.T, C_bias) + reg_I, np.dot(C_bias.T, W_train))

            # Predict target sample
            c_test_bias = np.append(C_all[i], 1.0)
            w_pred = np.dot(c_test_bias, W_weights)
            pred_cnt_vec = np.clip(mean_cnt + np.dot(w_pred, components), 15.0, 64.0)

            pts_pred = {18 + idx: int(round(val)) for idx, val in enumerate(pred_cnt_vec)}
            apex_y = min(pts_pred, key=pts_pred.get)
            apex_x = pts_pred[apex_y]

            # Rasterize mask
            mask = np.zeros((128, 128), dtype=np.uint8)
            for y, x_val in pts_pred.items():
                if x_val < 64:
                    mask[y, x_val:64] = 1

            # Render baseline profile RGB
            rgb = np.full((128, 128, 3), 160, dtype=np.uint8)
            rgb[mask > 0] = [80, 80, 80]
            rgb[:, 64:] = [120, 120, 120]

            predictions[s_target] = {
                "rgb": rgb,
                "mask": mask,
                "pts": pts_pred,
                "apex_x": float(apex_x),
                "apex_y": float(apex_y),
                "reinf_proj_px": float(64.0 - apex_x),
                "active_area_px": int(np.sum(mask)),
                "top_toe": (float(pts_pred[18]), 18.0),
                "bot_toe": (float(pts_pred[112]), 112.0)
            }

        return predictions

# =========================================================================
# Pixel-Based Metric Comparison Function
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

    # Contour MAE and RMSE along common rows
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

    # Toe Errors
    r_top = real_cnt.get("top_toe", (64.0, 64.0))
    g_top = gen_cnt.get("top_toe", (64.0, 64.0))
    top_toe_err = float(np.sqrt((r_top[0] - g_top[0])**2 + (r_top[1] - g_top[1])**2))

    r_bot = real_cnt.get("bot_toe", (64.0, 64.0))
    g_bot = gen_cnt.get("bot_toe", (64.0, 64.0))
    bot_toe_err = float(np.sqrt((r_bot[0] - g_bot[0])**2 + (r_bot[1] - g_bot[1])**2))

    # Area Error
    area_err_px2 = float(abs(real_cnt.get("active_area_px", 0) - gen_cnt.get("active_area_px", 0)))

    # Mask Dice & IoU
    r_mask = real_cnt.get("mask", np.zeros((128, 128), dtype=np.uint8))
    g_mask = gen_cnt.get("mask", np.zeros((128, 128), dtype=np.uint8))
    inter = np.sum((r_mask > 0) & (g_mask > 0))
    union = np.sum((r_mask > 0) | (g_mask > 0))
    mask_iou = float(inter / (union + 1e-8))
    mask_dice = float(2.0 * inter / (np.sum(r_mask > 0) + np.sum(g_mask > 0) + 1e-8))

    # Symmetric Distance and Hausdorff-95
    r_coords = np.array([[x, y] for y, x in r_pts.items()])
    g_coords = np.array([[x, y] for y, x in g_pts.items()])
    if len(r_coords) > 0 and len(g_coords) > 0:
        dists_rg = cdist(r_coords, g_coords)
        d_r2g = dists_rg.min(axis=1)
        d_g2r = dists_rg.min(axis=0)
        sym_dist = float((d_r2g.mean() + d_g2r.mean()) / 2.0)
        hd95 = float(np.percentile(np.concatenate([d_r2g, d_g2r]), 95))
    else:
        sym_dist, hd95 = 40.0, 50.0

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
        "top_toe_error_px": top_toe_err,
        "bottom_toe_error_px": bot_toe_err,
        "area_error_px2": area_err_px2,
        "symmetric_contour_distance_px": sym_dist,
        "hausdorff95_px": hd95,
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
# Main Execution Pipeline
# =========================================================================
def run_pipeline():
    t_start = time.time()
    print("=" * 75)
    print("PHASE V11: LOCAL CONTOUR EXTRACTION & BENCHMARKING (36 SAMPLES)")
    print("=" * 75)

    df_clean = pd.read_csv(CLEAN_CSV)
    samples = df_clean['sample'].tolist()

    # Pre-extract all original images & contours (Stage B)
    print("\n[STAGE B] Processing Original Experimental Images (Views A and B)...")
    orig_data = {}
    for s in samples:
        s_id = f"S{s:02d}"
        s_dir = os.path.join(OUT_GEN36, s_id)
        os.makedirs(s_dir, exist_ok=True)

        orig_data[s] = {}
        for view in ["A", "B"]:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(ALIGNED_DIR, f"{s}-{view}.png")), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            canny = compute_clean_canny(gray)
            cnt = extract_robust_crown_contour(gray)

            # Save stage B outputs
            cv2.imwrite(os.path.join(s_dir, f"original_{view}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(s_dir, f"original_{view}_canny.png"), canny)
            cnt_img = create_contour_image(cnt["pts"])
            cv2.imwrite(os.path.join(s_dir, f"original_{view}_contour.png"), cnt_img)
            overlay = create_contour_overlay(rgb, cnt["pts"], (cnt["apex_x"], cnt["apex_y"]))
            cv2.imwrite(os.path.join(s_dir, f"original_{view}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Also save to shared original_contours folder
            cv2.imwrite(os.path.join(OUT_ORIG_CONTOURS, f"{s_id}_{view}_contour.png"), cnt_img)
            cv2.imwrite(os.path.join(OUT_ORIG_CONTOURS, f"{s_id}_{view}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            orig_data[s][view] = {
                "rgb": rgb, "gray": gray, "canny": canny, "cnt": cnt
            }

    print(f"  Processed {len(samples) * 2} original experimental images and contours.")

    # Fit B1 Baseline (Leave-One-Out)
    print("\n[STAGE C] Fitting Baseline B1 (PCA-Contour Regressor)...")
    b1_engine = PCAContourEngine(n_components=4, alpha=1.0)
    b1_preds = b1_engine.fit_and_predict_all_loo(df_clean, orig_data)
    print("  B1 Baseline predictions generated for all 36 samples.")

    # Models to evaluate
    eval_models = [
        ("M1", "M1_DirectCNN", "Direct Conditional CNN"),
        ("M3", "M3_GCWPGv2", "GC-WPG-v2 (Multi-Scale)"),
        ("M5", "M5_cGAN", "Geometry-Conditioned cGAN"),
        ("B1", "B1_PCA", "PCA-Contour Baseline")
    ]

    all_detailed_metrics = []

    print("\n[STAGES C, D, E] Processing Generated Models, Computing Metrics & Saving...")
    for idx_row, row in df_clean.iterrows():
        s = int(row['sample'])
        s_id = f"S{s:02d}"
        s_dir = os.path.join(OUT_GEN36, s_id)
        v9_dir = os.path.join(V9_GEN_DIR, s_id)

        real_a = orig_data[s]["A"]
        real_b = orig_data[s]["B"]

        sample_model_cache = {}

        for m_tag, m_fn_prefix, m_desc in eval_models:
            # 1. Load or retrieve raw generated RGB
            if m_tag == "B1":
                gen_raw_rgb = b1_preds[s]["rgb"]
                gen_mask = b1_preds[s]["mask"]
                gen_cnt = b1_preds[s]
            else:
                raw_path = os.path.join(v9_dir, f"{m_fn_prefix}_raw.png")
                gen_raw_rgb = cv2.cvtColor(cv2.imread(raw_path), cv2.COLOR_BGR2RGB)
                gen_gray = cv2.cvtColor(gen_raw_rgb, cv2.COLOR_RGB2GRAY)
                gen_cnt = extract_robust_crown_contour(gen_gray)
                gen_mask = gen_cnt["mask"]

            gen_gray = cv2.cvtColor(gen_raw_rgb, cv2.COLOR_RGB2GRAY)
            gen_sharp = mild_unsharp_mask(gen_raw_rgb, alpha=0.8, sigma=1.0)
            gen_canny = compute_clean_canny(gen_gray)
            gen_cnt_img = create_contour_image(gen_cnt["pts"])
            gen_overlay = create_contour_overlay(gen_raw_rgb, gen_cnt["pts"], (gen_cnt["apex_x"], gen_cnt["apex_y"]))

            # Dual overlays
            dual_cnt_overlay = create_dual_contour_overlay(real_a["rgb"], real_a["cnt"]["pts"], real_b["cnt"]["pts"], gen_cnt["pts"])
            edge_overlay = create_edge_comparison_overlay(real_a["canny"], gen_canny)

            # Save explicitly separated files (Stage C & E)
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_raw_rgb.png"), cv2.cvtColor(gen_raw_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_sharpened_rgb.png"), cv2.cvtColor(gen_sharp, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_generated_canny.png"), gen_canny)
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_predicted_mask.png"), gen_mask * 255)
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_predicted_contour.png"), gen_cnt_img)
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_generated_overlay.png"), cv2.cvtColor(gen_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_overlay_original_vs_generated_contour.png"), cv2.cvtColor(dual_cnt_overlay, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(s_dir, f"{m_tag}_overlay_original_vs_generated_edges.png"), cv2.cvtColor(edge_overlay, cv2.COLOR_RGB2BGR))

            # Stage D: Metric Evaluation against View A and View B
            m_a = evaluate_metrics(real_a["rgb"], gen_raw_rgb, real_a["cnt"], gen_cnt, real_a["canny"], gen_canny)
            m_b = evaluate_metrics(real_b["rgb"], gen_raw_rgb, real_b["cnt"], gen_cnt, real_b["canny"], gen_canny)

            # Dual-view aggregate
            m_mean = {}
            for k in m_a.keys():
                if isinstance(m_a[k], float):
                    m_mean[k] = round((m_a[k] + m_b[k]) / 2.0, 4)
                elif isinstance(m_a[k], bool):
                    m_mean[k] = m_a[k] or m_b[k]
                else:
                    m_mean[k] = m_a[k]

            metrics_json_obj = {
                "sample_id": s_id,
                "model_tag": m_tag,
                "model_description": m_desc,
                "material": row['material'],
                "process": row['process'],
                "feed_rate": float(row['feed_rate']),
                "speed": float(row['speed']),
                "metrics_vs_view_A": m_a,
                "metrics_vs_view_B": m_b,
                "metrics_dual_view_mean": m_mean
            }

            with open(os.path.join(s_dir, f"{m_tag}_metrics.json"), "w") as f_json:
                json.dump(metrics_json_obj, f_json, indent=2)

            all_detailed_metrics.append({
                "sample_id": s_id,
                "sample_num": s,
                "material": row['material'],
                "process": row['process'],
                "feed_rate": float(row['feed_rate']),
                "speed": float(row['speed']),
                "model": m_tag,
                "model_desc": m_desc,
                **m_mean
            })

            sample_model_cache[m_tag] = {
                "overlay": gen_overlay,
                "raw": gen_raw_rgb,
                "cnt": gen_cnt,
                "mae": m_mean["contour_mae_px"],
                "f1": m_mean["boundary_f1_2px"],
                "ssim": m_mean["full_image_ssim"]
            }

        # ---------------------------------------------------------------------
        # Create all_models_comparison_panel.png for this sample
        # Panels: Original A | Original B | M1 | M3 | M5 | M6 (Missing) | B1
        # ---------------------------------------------------------------------
        fig_p, axes_p = plt.subplots(1, 7, figsize=(21, 3.5), dpi=150)
        
        # 1. Original A
        axes_p[0].imshow(real_a["rgb"])
        axes_p[0].axvline(64, color='yellow', linestyle='--', linewidth=0.8)
        axes_p[0].set_title(f"Original View A\n(Apex X={real_a['cnt']['apex_x']:.0f})", fontsize=9, fontweight='bold')

        # 2. Original B
        axes_p[1].imshow(real_b["rgb"])
        axes_p[1].axvline(64, color='yellow', linestyle='--', linewidth=0.8)
        axes_p[1].set_title(f"Original View B\n(Apex X={real_b['cnt']['apex_x']:.0f})", fontsize=9, fontweight='bold')

        # 3. M1
        c_m1 = sample_model_cache["M1"]
        axes_p[2].imshow(c_m1["overlay"])
        axes_p[2].set_title(f"M1: Direct CNN\nMAE: {c_m1['mae']:.1f}px | F1: {c_m1['f1']:.2f}", fontsize=9, fontweight='bold')

        # 4. M3
        c_m3 = sample_model_cache["M3"]
        axes_p[3].imshow(c_m3["overlay"])
        axes_p[3].set_title(f"M3: GC-WPG-v2\nMAE: {c_m3['mae']:.1f}px | F1: {c_m3['f1']:.2f}", fontsize=9, fontweight='bold')

        # 5. M5
        c_m5 = sample_model_cache["M5"]
        axes_p[4].imshow(c_m5["overlay"])
        axes_p[4].set_title(f"M5: cGAN\nMAE: {c_m5['mae']:.1f}px | F1: {c_m5['f1']:.2f}", fontsize=9, fontweight='bold')

        # 6. M6 (GF-WPG) -> Honestly show missing locally
        blank_m6 = np.full((128, 128, 3), 40, dtype=np.uint8)
        axes_p[5].imshow(blank_m6)
        axes_p[5].text(64, 50, "M6: GF-WPG", color='yellow', fontsize=9, fontweight='bold', ha='center')
        axes_p[5].text(64, 75, "Missing locally\n(Awaiting A100 GPU)", color='white', fontsize=8, ha='center')
        axes_p[5].set_title("M6: GF-WPG\n[Pending GPU]", fontsize=9, fontweight='bold', color='gray')

        # 7. B1 PCA Baseline
        c_b1 = sample_model_cache["B1"]
        axes_p[6].imshow(c_b1["overlay"])
        axes_p[6].set_title(f"B1: PCA Baseline\nMAE: {c_b1['mae']:.1f}px | F1: {c_b1['f1']:.2f}", fontsize=9, fontweight='bold')

        for ax in axes_p:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.suptitle(f"Sample {s_id} ({row['material']} / {row['process']} | Wire Feed: {row['feed_rate']} m/min, Travel Speed: {row['speed']} cm/min)",
                     fontsize=12, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(s_dir, "all_models_comparison_panel.png"), bbox_inches='tight')
        plt.close()

    print(f"  Completed per-sample image generation and comparison panels for all 36 samples.")

    # -------------------------------------------------------------------------
    # Save Metrics CSVs
    # -------------------------------------------------------------------------
    df_metrics = pd.DataFrame(all_detailed_metrics)
    all_metrics_csv = os.path.join(OUT_METRICS, "all_samples_detailed_metrics.csv")
    df_metrics.to_csv(all_metrics_csv, index=False)
    print(f"\nSaved detailed metrics to: {all_metrics_csv}")

    # Model Summary Metrics
    summary_list = []
    for m_tag, sub in df_metrics.groupby("model"):
        summary_list.append({
            "model": m_tag,
            "model_desc": sub['model_desc'].iloc[0],
            "mean_contour_mae_px": round(sub['contour_mae_px'].mean(), 2),
            "std_contour_mae_px": round(sub['contour_mae_px'].std(), 2),
            "mean_apex_proj_error_px": round(sub['apex_projection_error_px'].mean(), 2),
            "mean_boundary_f1_2px": round(sub['boundary_f1_2px'].mean(), 4),
            "mean_mask_dice": round(sub['mask_dice'].mean(), 4),
            "mean_mask_iou": round(sub['mask_iou'].mean(), 4),
            "mean_roi_ssim": round(sub['roi_ssim'].mean(), 4),
            "mean_full_ssim": round(sub['full_image_ssim'].mean(), 4),
            "mean_psnr_db": round(sub['psnr'].mean(), 2),
            "missing_contour_count": int(sub['missing_contour_flag'].sum())
        })

    df_summary = pd.DataFrame(summary_list)
    summary_csv = os.path.join(OUT_METRICS, "model_summary_metrics.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved model summary metrics to: {summary_csv}")

    # Model Ranking CSV (sorted by contour MAE)
    df_ranking = df_summary.sort_values(by="mean_contour_mae_px").copy()
    df_ranking.reset_index(drop=True, inplace=True)
    df_ranking['rank'] = range(1, len(df_ranking) + 1)
    ranking_csv = os.path.join(OUT_METRICS, "model_ranking.csv")
    df_ranking.to_csv(ranking_csv, index=False)
    print(f"Saved model ranking to: {ranking_csv}")

    print("\n" + "=" * 75)
    print("PHASE V11 LOCAL CONTOUR BENCHMARK RANKING:")
    print("=" * 75)
    print(df_ranking[['rank', 'model', 'mean_contour_mae_px', 'mean_boundary_f1_2px', 'mean_mask_dice', 'mean_full_ssim']].to_string(index=False))

    t_end = time.time() - t_start
    print(f"\nExecution finished in {t_end:.2f} seconds ({t_end/60:.2f} min).")

if __name__ == "__main__":
    run_pipeline()
