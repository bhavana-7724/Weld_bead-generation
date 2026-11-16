"""
Fair Scientific Evaluation of Pretrained Macrograph Models (Task 7).
Phase V15: Sharp Weld Macrograph Pretrained AI Study.

Follows the identical protocol and evaluates the exact 20 physical experiments (40 views)
established in Phase V14 against:
1. M1 Direct Conditional CNN (authentic Phase V9 out-of-fold CV predictions)
2. Nearest Reference Baseline (1-NN within material/process domain)
3. Direct Weighted Image Blending (linear alpha blend)
4. Phase V14 Shape-Aligned SDF Interpolation (SDF morphing + backward remap)
5. Pretrained RIFE Direct (Variant A)
6. Pretrained RIFE Geometry-Guided (Variant B)
7. Pretrained FILM Direct (Variant A)
8. Pretrained FILM Geometry-Guided (Variant B)

Computes paired metrics: Contour MAE, Boundary F1, Mask Dice & IoU, ROI SSIM, PSNR,
Laplacian Variance, Tenengrad Energy, Crown Apex Error, and Failure Rate.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, CURRENT_DIR)

from rife_inference import RIFEInterpolator
from film_inference import FILMInterpolator
from reference_selector import ReferenceSelector
from geometry_guided_interpolation import (
    synthesize_direct_pretrained,
    synthesize_geometry_guided_pretrained,
    blend_sdfs
)

CLEAN_CSV = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")
V14_CSV = os.path.join(
    BASE_DIR, "results", "phase_v14_sharp_interpolation", "FINAL_REVIEW_ADDENDUM", "FAIR_PAIRED_COMPARISON.csv"
)
REG_DIR = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation", "registered_images")
GT_MASK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit", "gt_masks")

OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v15_pretrained_models")
RIFE_OUT_DIR = os.path.join(OUT_BASE, "pretrained_rife")
FILM_OUT_DIR = os.path.join(OUT_BASE, "pretrained_film")
GUIDED_OUT_DIR = os.path.join(OUT_BASE, "geometry_guided")
EVAL_OUT_DIR = os.path.join(OUT_BASE, "heldout_evaluation")

os.makedirs(RIFE_OUT_DIR, exist_ok=True)
os.makedirs(FILM_OUT_DIR, exist_ok=True)
os.makedirs(GUIDED_OUT_DIR, exist_ok=True)
os.makedirs(EVAL_OUT_DIR, exist_ok=True)


def extract_contour_from_image(img_rgb, baseline_x=64):
    """
    Standardized contour extraction for generated images:
    Converts to grayscale, detects crown boundary using adaptive thresholding,
    and extracts outer crown perimeter points.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if len(img_rgb.shape) == 3 else img_rgb

    # Background estimate from upper-left corner
    bg_val = np.median(gray[10:110, :20])
    adaptive_thresh = max(135, min(175, bg_val - 12))
    metal_roi = (gray[:, :baseline_x] < adaptive_thresh).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(metal_roi)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        crown_cc = (labels == largest_label).astype(np.uint8)
    else:
        crown_cc = metal_roi

    pts = {}
    for y in range(15, 115):
        row_xs = np.where(crown_cc[y, :baseline_x] == 1)[0]
        if len(row_xs) > 0:
            pts[int(y)] = float(row_xs.min())

    is_failed = len(pts) < 10
    mask = np.zeros_like(gray, dtype=np.uint8)
    if not is_failed:
        mask[:, :baseline_x] = crown_cc * 255

    return pts, mask, is_failed


def compute_paired_metrics(pred_rgb, pred_mask, pred_pts, gt_rgb, gt_mask, gt_pts, baseline_x=64):
    """
    Computes all paired metrics with strict Phase V14 standardization.
    """
    # 1. PSNR & SSIM
    psnr_val = float(psnr_func(gt_rgb, pred_rgb, data_range=255))
    ssim_val = float(ssim_func(gt_rgb, pred_rgb, channel_axis=2, data_range=255))

    p_gray = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2GRAY)
    g_gray = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY)
    _, ssim_map = ssim_func(g_gray, p_gray, data_range=255, full=True)
    if np.sum(gt_mask > 0) > 10:
        roi_ssim = float(np.mean(ssim_map[gt_mask > 0]))
    else:
        roi_ssim = ssim_val

    # 2. Sharpness Metrics
    lap = cv2.Laplacian(p_gray, cv2.CV_64F, ksize=3)
    if np.sum(gt_mask > 0) > 10:
        lap_var = float(np.var(lap[gt_mask > 0]))
    else:
        lap_var = float(np.var(lap))

    sobel_x = cv2.Sobel(p_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(p_gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = sobel_x**2 + sobel_y**2
    if np.sum(gt_mask > 0) > 10:
        tenengrad_mean = float(np.mean(tenengrad[gt_mask > 0]))
    else:
        tenengrad_mean = float(np.mean(tenengrad))

    # 3. Mask Overlap: Dice & IoU
    p_bin = (pred_mask > 127).astype(np.uint8)
    g_bin = (gt_mask > 127).astype(np.uint8)
    inter = np.sum(p_bin & g_bin)
    union = np.sum(p_bin | g_bin)
    p_sum = np.sum(p_bin)
    g_sum = np.sum(g_bin)

    dice = float((2.0 * inter) / max(1.0, float(p_sum + g_sum)))
    iou = float(inter / max(1.0, float(union)))

    # 4. Contour MAE & Crown Apex Error
    common_ys = sorted(set(pred_pts.keys()).intersection(set(gt_pts.keys())))
    if len(common_ys) >= 8:
        mae = float(np.mean([abs(pred_pts[y] - gt_pts[y]) for y in common_ys]))
    else:
        mae = 35.0 # Standard penalty for failed contour

    if len(pred_pts) > 0 and len(gt_pts) > 0:
        pred_apex_x = min(pred_pts.values())
        gt_apex_x = min(gt_pts.values())
        apex_err = float(abs(pred_apex_x - gt_apex_x))
    else:
        apex_err = 35.0

    # 5. Boundary F1 (tolerance = 2 px)
    p_edge = np.zeros((128, 128), dtype=np.uint8)
    for y, x in pred_pts.items():
        if 0 <= y < 128 and 0 <= int(round(x)) < 128:
            p_edge[int(y), int(round(x))] = 1

    g_edge = np.zeros((128, 128), dtype=np.uint8)
    for y, x in gt_pts.items():
        if 0 <= y < 128 and 0 <= int(round(x)) < 128:
            g_edge[int(y), int(round(x))] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dil_g = cv2.dilate(g_edge, kernel)
    dil_p = cv2.dilate(p_edge, kernel)

    tp_p = np.sum((p_edge == 1) & (dil_g == 1))
    total_p = np.sum(p_edge == 1)
    prec = tp_p / max(1.0, float(total_p))

    tp_r = np.sum((g_edge == 1) & (dil_p == 1))
    total_g = np.sum(g_edge == 1)
    rec = tp_r / max(1.0, float(total_g))

    f1 = float((2.0 * prec * rec) / max(1e-4, prec + rec))

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        "roi_ssim": roi_ssim,
        "laplacian_var": lap_var,
        "tenengrad": tenengrad_mean,
        "dice": dice,
        "iou": iou,
        "mae": mae,
        "apex_err": apex_err,
        "boundary_f1": f1
    }


def run_evaluation():
    print("=" * 80)
    print("PHASE V15: FAIR PAIRED EVALUATION ACROSS 20 PHYSICAL EXPERIMENTS (40 VIEWS)")
    print("=" * 80)

    # 1. Load Phase V14 benchmark data
    if not os.path.exists(V14_CSV):
        raise FileNotFoundError(f"Phase V14 benchmark CSV not found: {V14_CSV}")
    v14_df = pd.read_csv(V14_CSV)
    print(f"Loaded {len(v14_df)} held-out instances from Phase V14 benchmark.")

    # 2. Instantiate interpolators
    print("\nInitializing Pretrained Models...")
    rife = RIFEInterpolator()
    print("  RIFE Interpolator loaded successfully.")
    film = FILMInterpolator()
    print("  FILM Interpolator loaded successfully.")
    ref_selector = ReferenceSelector()
    print("  Reference Selector initialized.")

    rows = []

    for idx, row in v14_df.iterrows():
        sid = int(row["sample_id"])
        scode = row["sample_code"]
        view = row["view"]
        mat = row["material"]
        proc = row["process"]
        f_rate = float(row["feed_rate"])
        spd = float(row["speed"])

        # Load Ground Truth Image & Mask
        gt_im_path = os.path.join(REG_DIR, f"{sid}-{view}_registered.png")
        gt_bgr = cv2.imread(gt_im_path)
        if gt_bgr is None:
            print(f"Warning: GT image not found: {gt_im_path}")
            continue
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)

        gt_mask_path = os.path.join(REG_DIR, f"{sid}-{view}_registered_mask.png")
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            # Fallback extract from gt_rgb
            _, gt_mask, _ = extract_contour_from_image(gt_rgb)

        gt_pts = {}
        for y in range(128):
            row_xs = np.where(gt_mask[y, :64] > 127)[0]
            if len(row_xs) > 0:
                gt_pts[int(y)] = float(row_xs.min())

        # Select donors strictly excluding target sample sid (LOOCV)
        donor_info = ref_selector.find_donors(mat, proc, f_rate, spd, view=view, exclude_sample=sid)
        donor_imgs_rgb = []
        for p in donor_info["images"]:
            d_bgr = cv2.imread(p)
            donor_imgs_rgb.append(cv2.cvtColor(d_bgr, cv2.COLOR_BGR2RGB))

        weights = donor_info["weights"]
        mask_paths = donor_info["masks"]

        # -------------------------------------------------------------
        # Method 5: RIFE Direct (Variant A)
        # -------------------------------------------------------------
        rife_dir_rgb = synthesize_direct_pretrained(rife, donor_imgs_rgb, weights)
        r_dir_pts, r_dir_mask, r_dir_failed = extract_contour_from_image(rife_dir_rgb)
        m_rife_dir = compute_paired_metrics(rife_dir_rgb, r_dir_mask, r_dir_pts, gt_rgb, gt_mask, gt_pts)
        cv2.imwrite(
            os.path.join(RIFE_OUT_DIR, f"{scode}_{view}_rife_direct.png"),
            cv2.cvtColor(rife_dir_rgb, cv2.COLOR_RGB2BGR)
        )

        # -------------------------------------------------------------
        # Method 6: RIFE Geometry-Guided (Variant B)
        # -------------------------------------------------------------
        rife_gui_rgb, rife_t_mask, rife_t_pts = synthesize_geometry_guided_pretrained(
            rife, donor_imgs_rgb, mask_paths, weights
        )
        r_gui_pts, r_gui_mask, r_gui_failed = extract_contour_from_image(rife_gui_rgb)
        m_rife_gui = compute_paired_metrics(rife_gui_rgb, r_gui_mask, r_gui_pts, gt_rgb, gt_mask, gt_pts)
        cv2.imwrite(
            os.path.join(RIFE_OUT_DIR, f"{scode}_{view}_rife_guided.png"),
            cv2.cvtColor(rife_gui_rgb, cv2.COLOR_RGB2BGR)
        )

        # -------------------------------------------------------------
        # Method 7: FILM Direct (Variant A)
        # -------------------------------------------------------------
        film_dir_rgb = synthesize_direct_pretrained(film, donor_imgs_rgb, weights)
        f_dir_pts, f_dir_mask, f_dir_failed = extract_contour_from_image(film_dir_rgb)
        m_film_dir = compute_paired_metrics(film_dir_rgb, f_dir_mask, f_dir_pts, gt_rgb, gt_mask, gt_pts)
        cv2.imwrite(
            os.path.join(FILM_OUT_DIR, f"{scode}_{view}_film_direct.png"),
            cv2.cvtColor(film_dir_rgb, cv2.COLOR_RGB2BGR)
        )

        # -------------------------------------------------------------
        # Method 8: FILM Geometry-Guided (Variant B)
        # -------------------------------------------------------------
        film_gui_rgb, film_t_mask, film_t_pts = synthesize_geometry_guided_pretrained(
            film, donor_imgs_rgb, mask_paths, weights
        )
        f_gui_pts, f_gui_mask, f_gui_failed = extract_contour_from_image(film_gui_rgb)
        m_film_gui = compute_paired_metrics(film_gui_rgb, f_gui_mask, f_gui_pts, gt_rgb, gt_mask, gt_pts)
        cv2.imwrite(
            os.path.join(FILM_OUT_DIR, f"{scode}_{view}_film_guided.png"),
            cv2.cvtColor(film_gui_rgb, cv2.COLOR_RGB2BGR)
        )

        # Build combined row preserving historical Phase V14 metrics verbatim
        out_row = {
            "sample_id": sid,
            "sample_code": scode,
            "view": view,
            "material": mat,
            "process": proc,
            "feed_rate": f_rate,
            "speed": spd,
            "condition_type": row["condition_type"],
            "donors": str(donor_info["donor_sids"]),
            "weights": str([round(w, 3) for w in weights]),
            # Historical baselines
            "m1_mae_px": row["m1_mae_px"],
            "m1_boundary_f1": row["m1_boundary_f1"],
            "m1_dice": row["m1_dice"],
            "m1_roi_ssim": row["m1_roi_ssim"],
            "m1_psnr_db": row["m1_psnr_db"],
            "m1_laplacian_var": row["m1_laplacian_var"],
            "nn_mae_px": row["nn_mae_px"],
            "nn_boundary_f1": row["nn_boundary_f1"],
            "nn_dice": row["nn_dice"],
            "nn_roi_ssim": row["nn_roi_ssim"],
            "nn_psnr_db": row["nn_psnr_db"],
            "nn_laplacian_var": row["nn_laplacian_var"],
            "dir_mae_px": row["dir_mae_px"],
            "dir_boundary_f1": row["dir_boundary_f1"],
            "dir_dice": row["dir_dice"],
            "dir_roi_ssim": row["dir_roi_ssim"],
            "dir_psnr_db": row["dir_psnr_db"],
            "dir_laplacian_var": row["dir_laplacian_var"],
            "v14_mae_px": row["prop_mae_px"],
            "v14_boundary_f1": row["prop_boundary_f1"],
            "v14_dice": row["prop_dice"],
            "v14_roi_ssim": row["prop_roi_ssim"],
            "v14_psnr_db": row["prop_psnr_db"],
            "v14_laplacian_var": row["prop_laplacian_var"],
            # Phase V15 Models
            "rife_dir_mae_px": m_rife_dir["mae"],
            "rife_dir_boundary_f1": m_rife_dir["boundary_f1"],
            "rife_dir_dice": m_rife_dir["dice"],
            "rife_dir_roi_ssim": m_rife_dir["roi_ssim"],
            "rife_dir_psnr_db": m_rife_dir["psnr"],
            "rife_dir_laplacian_var": m_rife_dir["laplacian_var"],
            "rife_dir_tenengrad": m_rife_dir["tenengrad"],
            "rife_dir_failed": r_dir_failed,
            "rife_gui_mae_px": m_rife_gui["mae"],
            "rife_gui_boundary_f1": m_rife_gui["boundary_f1"],
            "rife_gui_dice": m_rife_gui["dice"],
            "rife_gui_roi_ssim": m_rife_gui["roi_ssim"],
            "rife_gui_psnr_db": m_rife_gui["psnr"],
            "rife_gui_laplacian_var": m_rife_gui["laplacian_var"],
            "rife_gui_tenengrad": m_rife_gui["tenengrad"],
            "rife_gui_failed": r_gui_failed,
            "film_dir_mae_px": m_film_dir["mae"],
            "film_dir_boundary_f1": m_film_dir["boundary_f1"],
            "film_dir_dice": m_film_dir["dice"],
            "film_dir_roi_ssim": m_film_dir["roi_ssim"],
            "film_dir_psnr_db": m_film_dir["psnr"],
            "film_dir_laplacian_var": m_film_dir["laplacian_var"],
            "film_dir_tenengrad": m_film_dir["tenengrad"],
            "film_dir_failed": f_dir_failed,
            "film_gui_mae_px": m_film_gui["mae"],
            "film_gui_boundary_f1": m_film_gui["boundary_f1"],
            "film_gui_dice": m_film_gui["dice"],
            "film_gui_roi_ssim": m_film_gui["roi_ssim"],
            "film_gui_psnr_db": m_film_gui["psnr"],
            "film_gui_laplacian_var": m_film_gui["laplacian_var"],
            "film_gui_tenengrad": m_film_gui["tenengrad"],
            "film_gui_failed": f_gui_failed,
        }
        rows.append(out_row)

        if (idx + 1) % 5 == 0 or idx == len(v14_df) - 1:
            print(f"  Processed {idx + 1}/{len(v14_df)} views ({scode} {view})...")

    # Save detailed CSV
    df_eval = pd.DataFrame(rows)
    eval_csv_path = os.path.join(OUT_BASE, "FAIR_MODEL_COMPARISON.csv")
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"\nSaved 40-view Fair Model Comparison CSV to: {eval_csv_path}")

    # Build Summary Table
    models_spec = [
        ("M1 Direct Conditional CNN", "m1_mae_px", "m1_boundary_f1", "m1_dice", "m1_roi_ssim", "m1_psnr_db", "m1_laplacian_var", None),
        ("Nearest Reference Baseline", "nn_mae_px", "nn_boundary_f1", "nn_dice", "nn_roi_ssim", "nn_psnr_db", "nn_laplacian_var", None),
        ("Direct Weighted Blend", "dir_mae_px", "dir_boundary_f1", "dir_dice", "dir_roi_ssim", "dir_psnr_db", "dir_laplacian_var", None),
        ("Phase V14 Shape-Aligned SDF", "v14_mae_px", "v14_boundary_f1", "v14_dice", "v14_roi_ssim", "v14_psnr_db", "v14_laplacian_var", None),
        ("Pretrained RIFE Direct (Variant A)", "rife_dir_mae_px", "rife_dir_boundary_f1", "rife_dir_dice", "rife_dir_roi_ssim", "rife_dir_psnr_db", "rife_dir_laplacian_var", "rife_dir_failed"),
        ("Pretrained RIFE Geometry-Guided (Variant B)", "rife_gui_mae_px", "rife_gui_boundary_f1", "rife_gui_dice", "rife_gui_roi_ssim", "rife_gui_psnr_db", "rife_gui_laplacian_var", "rife_gui_failed"),
        ("Pretrained FILM Direct (Variant A)", "film_dir_mae_px", "film_dir_boundary_f1", "film_dir_dice", "film_dir_roi_ssim", "film_dir_psnr_db", "film_dir_laplacian_var", "film_dir_failed"),
        ("Pretrained FILM Geometry-Guided (Variant B)", "film_gui_mae_px", "film_gui_boundary_f1", "film_gui_dice", "film_gui_roi_ssim", "film_gui_psnr_db", "film_gui_laplacian_var", "film_gui_failed"),
    ]

    summary_rows = []
    for m_name, mae_col, f1_col, dice_col, ssim_col, psnr_col, lap_col, fail_col in models_spec:
        fail_rate = 0.0
        if fail_col and fail_col in df_eval.columns:
            fail_rate = float(df_eval[fail_col].mean() * 100.0)

        summary_rows.append({
            "model_name": m_name,
            "contour_mae_px_mean": round(df_eval[mae_col].mean(), 2),
            "contour_mae_px_std": round(df_eval[mae_col].std(), 2),
            "contour_mae_px_median": round(df_eval[mae_col].median(), 2),
            "boundary_f1_mean": round(df_eval[f1_col].mean(), 2),
            "mask_dice_mean": round(df_eval[dice_col].mean(), 2),
            "crown_roi_ssim_mean": round(df_eval[ssim_col].mean(), 2),
            "psnr_db_mean": round(df_eval[psnr_col].mean(), 2),
            "laplacian_variance_mean": round(df_eval[lap_col].mean(), 1),
            "failure_rate_pct": round(fail_rate, 1)
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUT_BASE, "MODEL_COMPARISON_SUMMARY.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved Model Comparison Summary CSV to: {summary_csv_path}")

    print("\n" + "=" * 80)
    print("PHASE V15 BENCHMARK RESULTS SUMMARY (40 HELD-OUT VIEWS):")
    print("=" * 80)
    print(df_summary[["model_name", "contour_mae_px_mean", "boundary_f1_mean", "mask_dice_mean", "crown_roi_ssim_mean", "psnr_db_mean", "laplacian_variance_mean"]].to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    run_evaluation()
