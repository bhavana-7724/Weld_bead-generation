"""
audit_segmentation_methods.py

Performs a rigorous comparative benchmark of three crown contour extraction methods
for known-condition specimens S04 and S33:
1. Method 1: Original Scanline Thresholding (Phase V12 baseline)
2. Method 2: Unconstrained 95-Row Refined Extractor (fixed sweep Y=18..112)
3. Method 3: Physically Toe-Bounded Refined Extractor (adaptive support & subpixel refinement)

Evaluates against:
- Experimental View A
- Experimental View B
- Dual-View Average (A & B)

Saves results to:
results/phase_v12_finalized_project_stage/SEGMENTATION_METHOD_COMPARISON.csv
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v12.run_model_36_panels import (
    extract_robust_crown_contour,
    evaluate_metrics,
    create_contour_image,
    compute_clean_canny
)
from scripts.phase_v12.generate_unseen_interactive import (
    extract_refined_mask_and_contour
)

STAGE_DIR = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage")
V9_GEN_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")
GT_MASK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit", "gt_masks")
CSV_OUT_PATH = os.path.join(STAGE_DIR, "SEGMENTATION_METHOD_COMPARISON.csv")

# -------------------------------------------------------------------------
# Method 3: Physically Toe-Bounded Refined Extractor
# -------------------------------------------------------------------------
def extract_method3_toe_bounded(gray_img, x_baseline=64, y_min=18, y_max=112):
    bg_val = np.median(gray_img[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))
    
    metal_roi = (gray_img[:, :x_baseline] < adaptive_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    metal_cleaned = cv2.morphologyEx(metal_roi, cv2.MORPH_OPEN, kernel)
    
    # Connected component to isolate real crown deposit
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(metal_cleaned)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        crown_cc = (labels == largest_label).astype(np.uint8)
    else:
        crown_cc = metal_cleaned

    # Extract scanline boundaries from connected component
    pts_cc = {}
    for y in range(y_min, y_max + 1):
        idx = np.where(crown_cc[y, :x_baseline] == 1)[0]
        if len(idx) > 0:
            x_min = int(idx[0])
            if x_min < x_baseline:
                pts_cc[y] = x_min

    if len(pts_cc) < 5:
        return extract_robust_crown_contour(gray_img, x_baseline, y_min, y_max)

    # Subpixel gradient refinement within local window
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)[:, :x_baseline]

    refined_pts = {}
    for y, x_init in pts_cc.items():
        x_start = max(18, x_init - 3)
        x_end = min(x_baseline - 1, x_init + 4)
        if x_end > x_start:
            win_grad = sobelx[y, x_start:x_end]
            min_local = np.argmin(win_grad)
            if win_grad[min_local] < -20.0:
                refined_pts[y] = float(x_start + min_local)
            else:
                refined_pts[y] = float(x_init)
        else:
            refined_pts[y] = float(x_init)

    ys = sorted(refined_pts.keys())
    xs = [refined_pts[y] for y in ys]
    if len(xs) >= 11:
        xs_smooth = savgol_filter(xs, window_length=11, polyorder=2)
    elif len(xs) >= 7:
        xs_smooth = savgol_filter(xs, window_length=7, polyorder=2)
    elif len(xs) >= 5:
        xs_smooth = savgol_filter(xs, window_length=5, polyorder=2)
    else:
        xs_smooth = xs

    final_pts = {int(y): int(round(np.clip(x_val, 15.0, float(x_baseline - 1)))) for y, x_val in zip(ys, xs_smooth)}

    mask = np.zeros((128, 128), dtype=np.uint8)
    for y, x_val in final_pts.items():
        x_int = int(round(x_val))
        if x_int < x_baseline:
            mask[y, x_int:x_baseline] = 1

    apex_y = min(final_pts, key=final_pts.get)
    apex_x = final_pts[apex_y]
    reinf_proj = 64.0 - apex_x
    top_toe = (final_pts[ys[0]], float(ys[0]))
    bot_toe = (final_pts[ys[-1]], float(ys[-1]))

    return {
        'pts': final_pts,
        'mask': mask,
        'apex_x': float(apex_x),
        'apex_y': float(apex_y),
        'reinf_proj_px': float(reinf_proj),
        'top_toe': top_toe,
        'bot_toe': bot_toe,
        'active_area_px': int(np.sum(mask)),
        'support_span': len(ys),
        'y_range': (ys[0], ys[-1]),
        'confidence_flag': "HIGH_CONFIDENCE_TOE_BOUNDED"
    }

def wrap_method2_output(gray_img):
    mask, cnt_img, final_pts, apex_x, apex_y, reinf_px, active_area = extract_refined_mask_and_contour(gray_img)
    final_pts_int = {int(y): int(round(x)) for y, x in final_pts.items()}
    ys = sorted(final_pts_int.keys())
    top_toe = (float(final_pts_int[ys[0]]), float(ys[0])) if len(ys) > 0 else (64.0, 64.0)
    bot_toe = (float(final_pts_int[ys[-1]]), float(ys[-1])) if len(ys) > 0 else (64.0, 64.0)
    return {
        'pts': final_pts_int,
        'mask': (mask > 0).astype(np.uint8),
        'apex_x': float(apex_x),
        'apex_y': float(apex_y),
        'reinf_proj_px': float(reinf_px),
        'top_toe': top_toe,
        'bot_toe': bot_toe,
        'active_area_px': active_area,
        'support_span': len(ys),
        'y_range': (ys[0], ys[-1]) if len(ys) > 0 else (0, 0),
        'confidence_flag': "UNCONSTRAINED_SWEEP_LOW_SELECTIVITY"
    }

def get_gt_contour_from_mask(mask_img, x_baseline=64):
    ys, xs = np.where(mask_img > 127)
    if len(ys) == 0:
        return {'pts': {}, 'apex_x': 64.0, 'apex_y': 64, 'reinf_proj_px': 0.0,
                'top_toe': (64.0, 64.0), 'bot_toe': (64.0, 64.0), 'mask': np.zeros((128, 128), dtype=np.uint8)}
    pts = {}
    for y in sorted(np.unique(ys)):
        row_xs = np.where(mask_img[y, :x_baseline] > 127)[0]
        if len(row_xs) > 0:
            pts[int(y)] = int(row_xs[0])
    
    apex_y = min(pts, key=pts.get)
    apex_x = pts[apex_y]
    sorted_ys = sorted(pts.keys())
    top_toe = (pts[sorted_ys[0]], float(sorted_ys[0]))
    bot_toe = (pts[sorted_ys[-1]], float(sorted_ys[-1]))
    
    return {
        'pts': pts,
        'apex_x': float(apex_x),
        'apex_y': float(apex_y),
        'reinf_proj_px': float(x_baseline - apex_x),
        'top_toe': top_toe,
        'bot_toe': bot_toe,
        'active_area_px': int(len(ys)),
        'mask': (mask_img > 127).astype(np.uint8),
        'support_span': len(sorted_ys),
        'y_range': (sorted_ys[0], sorted_ys[-1])
    }

def compute_toe_errors(cnt_ref, cnt_pred):
    t_ref_top = cnt_ref['top_toe']
    t_pred_top = cnt_pred['top_toe']
    err_top = np.sqrt((t_ref_top[0] - t_pred_top[0])**2 + (t_ref_top[1] - t_pred_top[1])**2)

    t_ref_bot = cnt_ref['bot_toe']
    t_pred_bot = cnt_pred['bot_toe']
    err_bot = np.sqrt((t_ref_bot[0] - t_pred_bot[0])**2 + (t_ref_bot[1] - t_pred_bot[1])**2)
    return float(err_top), float(err_bot)

def run_evaluation():
    samples = [
        {"sample": "S04", "num": 4, "cond": "KNOWN_EXPERIMENTAL_CONDITION"},
        {"sample": "S33", "num": 33, "cond": "KNOWN_EXPERIMENTAL_CONDITION"}
    ]

    records = []

    for s_info in samples:
        s_code = s_info["sample"]
        s_num = s_info["num"]
        cond = s_info["cond"]
        s_dir = os.path.join(V9_GEN_DIR, s_code)

        # Real images & masks
        real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_A.png")), cv2.COLOR_BGR2RGB)
        real_b_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_B.png")), cv2.COLOR_BGR2RGB)
        
        gt_mask_a = cv2.imread(os.path.join(GT_MASK_DIR, f"{s_num}-A.png"), cv2.IMREAD_GRAYSCALE)
        gt_mask_b = cv2.imread(os.path.join(GT_MASK_DIR, f"{s_num}-B.png"), cv2.IMREAD_GRAYSCALE)

        cnt_ref_a = get_gt_contour_from_mask(gt_mask_a)
        cnt_ref_b = get_gt_contour_from_mask(gt_mask_b)

        # Generated image from M1
        gen_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "M1_DirectCNN_raw.png")), cv2.COLOR_BGR2RGB)
        gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)
        canny_a = compute_clean_canny(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY))
        canny_b = compute_clean_canny(cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY))
        canny_gen = compute_clean_canny(gen_gray)

        # 3 Methods
        cnt_m1 = extract_robust_crown_contour(gen_gray)
        cnt_m2 = wrap_method2_output(gen_gray)
        cnt_m3 = extract_method3_toe_bounded(gen_gray)

        methods = [
            ("Method 1: Original Scanline Thresholding", cnt_m1),
            ("Method 2: Unconstrained 95-Row Refined", cnt_m2),
            ("Method 3: Physically Toe-Bounded Refined", cnt_m3)
        ]

        for m_name, cnt_pred in methods:
            # Eval vs View A
            mA = evaluate_metrics(real_a_rgb, gen_rgb, cnt_ref_a, cnt_pred, canny_a, canny_gen)
            top_err_a, bot_err_a = compute_toe_errors(cnt_ref_a, cnt_pred)
            
            rec_a = {
                "sample": s_code,
                "condition_type": cond,
                "method": m_name,
                "view_evaluated": "View_A",
                "contour_mae_px": round(mA["contour_mae_px"], 2),
                "boundary_f1_2px": round(mA["boundary_f1_2px"], 3),
                "mask_dice": round(mA["mask_dice"], 3),
                "mask_iou": round(mA["mask_iou"], 3),
                "apex_projection_error_px": round(mA["apex_projection_error_px"], 2),
                "top_toe_error_px": round(top_err_a, 2),
                "bot_toe_error_px": round(bot_err_a, 2),
                "detected_vertical_span_px": len(cnt_pred["pts"]),
                "real_vertical_span_px": len(cnt_ref_a["pts"]),
                "failure_flag": mA["missing_contour_flag"]
            }
            records.append(rec_a)

            # Eval vs View B
            mB = evaluate_metrics(real_b_rgb, gen_rgb, cnt_ref_b, cnt_pred, canny_b, canny_gen)
            top_err_b, bot_err_b = compute_toe_errors(cnt_ref_b, cnt_pred)
            
            rec_b = {
                "sample": s_code,
                "condition_type": cond,
                "method": m_name,
                "view_evaluated": "View_B",
                "contour_mae_px": round(mB["contour_mae_px"], 2),
                "boundary_f1_2px": round(mB["boundary_f1_2px"], 3),
                "mask_dice": round(mB["mask_dice"], 3),
                "mask_iou": round(mB["mask_iou"], 3),
                "apex_projection_error_px": round(mB["apex_projection_error_px"], 2),
                "top_toe_error_px": round(top_err_b, 2),
                "bot_toe_error_px": round(bot_err_b, 2),
                "detected_vertical_span_px": len(cnt_pred["pts"]),
                "real_vertical_span_px": len(cnt_ref_b["pts"]),
                "failure_flag": mB["missing_contour_flag"]
            }
            records.append(rec_b)

            # Dual-View Average
            rec_avg = {
                "sample": s_code,
                "condition_type": cond,
                "method": m_name,
                "view_evaluated": "Dual_View_Average",
                "contour_mae_px": round(0.5 * (mA["contour_mae_px"] + mB["contour_mae_px"]), 2),
                "boundary_f1_2px": round(0.5 * (mA["boundary_f1_2px"] + mB["boundary_f1_2px"]), 3),
                "mask_dice": round(0.5 * (mA["mask_dice"] + mB["mask_dice"]), 3),
                "mask_iou": round(0.5 * (mA["mask_iou"] + mB["mask_iou"]), 3),
                "apex_projection_error_px": round(0.5 * (mA["apex_projection_error_px"] + mB["apex_projection_error_px"]), 2),
                "top_toe_error_px": round(0.5 * (top_err_a + top_err_b), 2),
                "bot_toe_error_px": round(0.5 * (bot_err_a + bot_err_b), 2),
                "detected_vertical_span_px": len(cnt_pred["pts"]),
                "real_vertical_span_px": round(0.5 * (len(cnt_ref_a["pts"]) + len(cnt_ref_b["pts"])), 1),
                "failure_flag": mA["missing_contour_flag"] or mB["missing_contour_flag"]
            }
            records.append(rec_avg)

    df_res = pd.DataFrame(records)
    df_res.to_csv(CSV_OUT_PATH, index=False)
    print(f"Saved benchmark results to {CSV_OUT_PATH}")
    print("\nBenchmark Summary Table:")
    print(df_res.to_string())

if __name__ == "__main__":
    run_evaluation()
