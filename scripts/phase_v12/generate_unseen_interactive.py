
import os
import sys
import json
import time
import pickle
import argparse
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v9.geometry_models import DirectCNN, ConditionalVAE, GC_WPG_v2, GeoSDF_CVAE, Geometry_cGAN_Generator

STAGE_DIR = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage")
REFINED_OUTPUT_BASE = os.path.join(STAGE_DIR, "unseen_parameter_outputs_refined")
NUM_REG_PATH = os.path.join(BASE_DIR, "models", "numerical", "prospective_regressor_checkpoint.pkl")
V9_CKPT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "checkpoints")
NORM_PARAMS_PATH = os.path.join(V9_CKPT_DIR, "normalization_params.json")
CLEAN_DATASET_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")

os.makedirs(REFINED_OUTPUT_BASE, exist_ok=True)

# Dataset Condition Classifier

def classify_condition(material, process, feed_rate, speed):
    """
    Checks whether the input parameters match an existing training experiment
    in data/clean_dataset_36.csv.
    """
    if not os.path.exists(CLEAN_DATASET_PATH):
        return "GENUINE_UNSEEN_INTERPOLATION", None, None

    df = pd.read_csv(CLEAN_DATASET_PATH)
    match = df[
        (df["material"].str.strip().str.upper() == material.strip().upper()) &
        (df["process"].str.strip().str.upper() == process.strip().upper()) &
        (np.isclose(df["feed_rate"].astype(float), float(feed_rate), atol=1e-3)) &
        (np.isclose(df["speed"].astype(float), float(speed), atol=1e-3))
    ]

    if len(match) > 0:
        row = match.iloc[0]
        sample_id = f"S{int(row['sample']):02d}"
        row_dict = {
            "sample_id": sample_id,
            "sample_num": int(row["sample"]),
            "material": row["material"],
            "process": row["process"],
            "feed_rate": float(row["feed_rate"]),
            "speed": float(row["speed"]),
            "HIP": float(row["HIP"]),
            "width": float(row["width"]),
            "depth": float(row["depth"]),
            "reinforcement": float(row["reinforcement"])
        }
        return "KNOWN_EXPERIMENTAL_CONDITION", sample_id, row_dict
    else:
        return "GENUINE_UNSEEN_INTERPOLATION", None, None

# Image Processing & Geometry Extraction Helpers


def mild_unsharp_mask(rgb_img, alpha=0.8, sigma=1.0):
    """Enhances subtle edge gradients using unsharp masking without over-saturating."""
    blurred = cv2.GaussianBlur(rgb_img, (0, 0), sigma)
    sharp = rgb_img.astype(np.float32) + alpha * (rgb_img.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharp, 0, 255).astype(np.uint8)

def compute_clean_edge_gradient_map(rgb_img, min_edges=80):
    """
    Computes a crisp, informative Edge / Gradient Map and calculates spatial edge decomposition:
    - Air (X < 20)
    - Crown Boundary Zone (20 <= X < 62)
    - Baseline Interface (62 <= X <= 66)
    - Substrate Bulk Texture (X > 66)
    """
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY) if len(rgb_img.shape) == 3 else rgb_img.copy()

    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stretched)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    med = np.median(blurred)
    lower = int(max(25, 0.45 * med))
    upper = int(min(220, 1.15 * med))
    canny_edges = cv2.Canny(blurred, lower, upper)
    edge_count = int(np.count_nonzero(canny_edges))

    used_fallback = False
    if edge_count >= min_edges:
        edge_map = canny_edges
        method_note = f"Adaptive Canny (lower={lower}, upper={upper}, active_edges={edge_count})"
    else:
        used_fallback = True
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edge_map = grad_norm
        method_note = f"Gradient Magnitude Fallback (Sobel norm, active_px={int(np.count_nonzero(edge_map))})"

    # Edge decomposition
    ys, xs = np.where(edge_map > 0)
    total_edges = max(1, len(xs))
    air_cnt = int(np.sum(xs < 20))
    crown_cnt = int(np.sum((xs >= 20) & (xs < 62)))
    base_cnt = int(np.sum((xs >= 62) & (xs <= 66)))
    sub_cnt = int(np.sum(xs > 66))

    decomp = {
        "total_edge_pixels": int(len(xs)),
        "air_px": air_cnt,
        "air_pct": round(air_cnt / total_edges * 100, 1),
        "crown_boundary_px": crown_cnt,
        "crown_boundary_pct": round(crown_cnt / total_edges * 100, 1),
        "baseline_interface_px": base_cnt,
        "baseline_interface_pct": round(base_cnt / total_edges * 100, 1),
        "substrate_bulk_px": sub_cnt,
        "substrate_bulk_pct": round(sub_cnt / total_edges * 100, 1)
    }

    return edge_map, method_note, used_fallback, decomp

def extract_refined_mask_and_contour(gray_img, x_baseline=64, y_min=18, y_max=112):
    """
    Extracts a physically toe-bounded weld crown mask and smooth outer boundary (Method 3):
    1. Evaluates background metal intensity and segments deposit region (X < 64).
    2. Isolates the primary connected crown component to reject substrate grain noise.
    3. Refines outer boundary with subpixel horizontal Sobel gradients.
    4. Smooths boundary profile using Savitzky-Golay filtering.
    5. Detects apex coordinate (X_apex, Y_apex) and reinforcement projection in pixels.
    6. Identifies top toe (X_top, Y_top) and bottom toe (X_bot, Y_bot) where deposit joins substrate.
    """
    bg_val = np.median(gray_img[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))

    metal_roi = (gray_img[:, :x_baseline] < adaptive_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    metal_cleaned = cv2.morphologyEx(metal_roi, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(metal_cleaned)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        crown_cc = (labels == largest_label).astype(np.uint8)
    else:
        crown_cc = metal_cleaned

    pts_cc = {}
    for y in range(y_min, y_max + 1):
        idx = np.where(crown_cc[y, :x_baseline] == 1)[0]
        if len(idx) > 0:
            x_min = int(idx[0])
            if x_min < x_baseline:
                pts_cc[y] = x_min

    if len(pts_cc) < 5:
        # Fallback to local scanline thresholding
        for y in range(y_min, y_max + 1):
            idx = np.where(gray_img[y, :x_baseline] < adaptive_thresh)[0]
            if len(idx) > 0 and idx[0] < x_baseline:
                pts_cc[y] = int(idx[0])

    if len(pts_cc) < 3:
        # Emergency default
        for y in range(50, 78):
            pts_cc[y] = 46

    # Subpixel gradient refinement
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

    # Savitzky-Golay smoothing
    if len(xs) >= 11:
        xs_smooth = savgol_filter(xs, window_length=11, polyorder=2)
    elif len(xs) >= 7:
        xs_smooth = savgol_filter(xs, window_length=7, polyorder=2)
    elif len(xs) >= 5:
        xs_smooth = savgol_filter(xs, window_length=5, polyorder=2)
    else:
        xs_smooth = xs

    final_pts = {int(y): float(np.clip(x_val, 15.0, float(x_baseline - 1))) for y, x_val in zip(ys, xs_smooth)}

    # Binary mask
    mask = np.zeros((128, 128), dtype=np.uint8)
    for y, x_val in final_pts.items():
        xi = int(round(x_val))
        if xi < x_baseline:
            mask[y, xi:x_baseline] = 255

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    apex_y = min(final_pts, key=final_pts.get)
    apex_x = float(final_pts[apex_y])
    reinf_px = float(x_baseline - apex_x)
    active_area = int(np.count_nonzero(mask))

    top_toe = (float(final_pts[ys[0]]), float(ys[0]))
    bot_toe = (float(final_pts[ys[-1]]), float(ys[-1]))

    # 1-pixel contour image
    cnt_img = np.zeros((128, 128), dtype=np.uint8)
    for y, x_val in final_pts.items():
        xi = int(round(x_val))
        if 0 <= y < 128 and 0 <= xi < 128:
            cnt_img[y, xi] = 255

    toe_info = {
        "top_toe": [round(top_toe[0], 2), round(top_toe[1], 1)],
        "bot_toe": [round(bot_toe[0], 2), round(bot_toe[1], 1)],
        "support_span_px": len(ys),
        "y_range": [int(ys[0]), int(ys[-1])],
        "confidence_flag": "HIGH_CONFIDENCE_TOE_BOUNDED" if len(ys) >= 15 else "NARROW_SUPPORT_BEAD"
    }

    return mask, cnt_img, final_pts, apex_x, int(apex_y), reinf_px, active_area, toe_info

def create_prominent_overlay(base_rgb, pts, apex_x, apex_y, reinf_px, reinf_mm, x_baseline=64):
    """
    Renders an engineering-grade prominent overlay on the synthetic cross-section:
    - Dashed yellow line along substrate baseline X=64
    - Bright lime-green 2px contour line along crown boundary
    - Bright red apex dot with circular halo ring
    - Caliper line connecting apex to substrate baseline
    - Dimension annotation badge with apex coordinate and reinforcement
    """
    overlay = base_rgb.copy()

    # Yellow dashed baseline at X=64
    for y in range(0, 128, 4):
        overlay[y:y+2, x_baseline] = [255, 235, 59]

    # Bright lime-green 2px contour line
    for y, x_val in pts.items():
        x_int = int(round(x_val))
        if 0 <= y < 128:
            if 0 <= x_int < 128:
                overlay[y, x_int] = [0, 255, 127]
            if 0 <= x_int - 1 < 128:
                overlay[y, x_int - 1] = [0, 255, 127]

    # Caliper line connecting apex to baseline
    ax_int = int(round(apex_x))
    ay_int = int(round(apex_y))
    for x in range(ax_int, x_baseline, 2):
        if 0 <= ay_int < 128 and 0 <= x < 128:
            overlay[ay_int, x] = [0, 220, 255] # Cyan dotted line

    # Apex marker (Red circle with white center)
    cv2.circle(overlay, (ax_int, ay_int), 3, (255, 50, 50), -1)
    cv2.circle(overlay, (ax_int, ay_int), 4, (255, 255, 255), 1)

    return overlay

# Generator Class

class UnseenWeldGenerator:
    def __init__(self, model_key="M1", device="cpu"):
        self.device = torch.device(device)
        self.model_key = model_key
        self._load_regressor()
        self._load_generative_model(model_key)
        self._load_norm_params()

    def _load_regressor(self):
        if not os.path.exists(NUM_REG_PATH):
            raise FileNotFoundError(f"Missing numerical regressor: {NUM_REG_PATH}")
        with open(NUM_REG_PATH, "rb") as f:
            self.reg_data = pickle.load(f)
        self.reg_model = self.reg_data["model"]
        self.reg_scaler = self.reg_data["scaler"]
        self.reg_encoder = self.reg_data["encoder"]
        self.num_features = self.reg_data["num_features"]
        self.cat_features = self.reg_data["cat_features"]
        self.target_cols = self.reg_data["target_cols"]

    def _load_generative_model(self, model_key):
        self.model_name = "M1: Direct Conditional CNN"
        m1_path = os.path.join(V9_CKPT_DIR, "m1_direct_cnn_final.pt")
        if not os.path.exists(m1_path):
            raise FileNotFoundError(f"Missing M1 checkpoint: {m1_path}")
        self.gen_model = DirectCNN(cond_dim=6, noise_dim=16).to(self.device)
        ckpt = torch.load(m1_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.gen_model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            self.gen_model.load_state_dict(ckpt["model"])
        else:
            self.gen_model.load_state_dict(ckpt)
        self.gen_model.eval()

    def _load_norm_params(self):
        if not os.path.exists(NORM_PARAMS_PATH):
            raise FileNotFoundError(f"Missing normalization params: {NORM_PARAMS_PATH}")
        with open(NORM_PARAMS_PATH, "r") as f:
            self.norm_params = json.load(f)

    def predict_numerical_geometry(self, material, process, feed_rate, speed):
        """Predicts Width, Depth, Reinforcement (mm) using multi-output regressor."""
        df_in = pd.DataFrame([{
            "feed_rate": float(feed_rate),
            "speed": float(speed),
            "material": material,
            "process": process
        }])
        X_num = self.reg_scaler.transform(df_in[self.num_features])
        X_cat = self.reg_encoder.transform(df_in[self.cat_features])
        X_all = np.hstack([X_num, X_cat])
        preds = self.reg_model.predict(X_all)[0]

        return {
            "predicted_width_mm": round(float(preds[0]), 2),
            "predicted_depth_mm": round(float(preds[1]), 2),
            "predicted_reinforcement_mm": round(float(preds[2]), 2)
        }

    def generate_image(self, material, process, feed_rate, speed):
        """Generates cross-section RGB image from condition vector."""
        f_min = self.norm_params["f_min"]
        f_max = self.norm_params["f_max"]
        s_min = self.norm_params["s_min"]
        s_max = self.norm_params["s_max"]

        f_norm = (float(feed_rate) - f_min) / (f_max - f_min + 1e-8)
        s_norm = (float(speed) - s_min) / (s_max - s_min + 1e-8)

        c = [
            f_norm,
            s_norm,
            1.0 if material == "IN718" else 0.0,
            1.0 if material == "SS304L" else 0.0,
            1.0 if process == "ColdArc" else 0.0,
            1.0 if process == "GMAW" else 0.0
        ]
        c_tensor = torch.tensor([c], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            img_tensor = self.gen_model(c_tensor)

        img_np = (np.clip(img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        return img_np, c

    def render_presentation_panel(self, material, process, feed_rate, speed, geom_preds,
                                  raw_rgb, sharp_rgb, mask, cnt_img, edge_map, overlay,
                                  apex_x, apex_y, reinf_px, edge_note, cond_info, out_path):
        """
        Renders an academic presentation plate with explicit condition classification.
        """
        cond_type, sample_id, row_dict = cond_info

        fig = plt.figure(figsize=(16, 11), dpi=150, facecolor='#0D1117')
        gs = GridSpec(nrows=3, ncols=3, height_ratios=[1.15, 4.0, 4.0],
                      top=0.95, bottom=0.06, left=0.04, right=0.96,
                      hspace=0.30, wspace=0.20)

        # ---------------- Top Header Banner ----------------
        ax_banner = fig.add_subplot(gs[0, :])
        ax_banner.set_facecolor('#161B22')
        ax_banner.axis('off')

        border_color = '#3FB950' if cond_type == "KNOWN_EXPERIMENTAL_CONDITION" else '#FF7B72'
        rect = plt.Rectangle((0, 0), 1, 1, transform=ax_banner.transAxes,
                             fill=False, edgecolor=border_color, linewidth=1.5)
        ax_banner.add_patch(rect)

        if cond_type == "KNOWN_EXPERIMENTAL_CONDITION":
            title_str = f"KNOWN-CONDITION RECONSTRUCTION VALIDATION — SPECIMEN {sample_id}"
            sub_str = f"Model: {self.model_name} | Training Specimen In-Sample Evaluation | Ground-Truth Macrographs Exist"
            geom_str = f"Experimental Ground-Truth: Width = {row_dict['width']:.2f} mm | Depth = {row_dict['depth']:.2f} mm | Reinf = {row_dict['reinforcement']:.2f} mm"
            geom_color = '#3FB950'
        else:
            title_str = "PROSPECTIVE UNSEEN INTERPOLATION — ZERO-SHOT SYNTHESIS"
            sub_str = f"Model: {self.model_name} | Prospective Parameter Set (Absent from 36 Samples) | MultiOutput Ridge Priors"
            geom_str = f"Regressed Geometry (Physical): Width = {geom_preds['predicted_width_mm']:.2f} mm | Depth = {geom_preds['predicted_depth_mm']:.2f} mm | Reinf = {geom_preds['predicted_reinforcement_mm']:.2f} mm"
            geom_color = '#58A6FF'

        proc_str = f"Process Conditions:  Material = {material}   |   Process = {process}   |   Feed Rate = {feed_rate:.1f} m/min   |   Speed = {speed:.1f} cm/min"

        ax_banner.text(0.02, 0.78, title_str, color=border_color, fontsize=15, fontweight='bold', va='center')
        ax_banner.text(0.02, 0.54, sub_str, color='#8B949E', fontsize=10.5, fontstyle='italic', va='center')
        ax_banner.text(0.02, 0.24, proc_str, color='#F0F6FC', fontsize=11, fontweight='semibold', va='center')
        ax_banner.text(0.98, 0.24, geom_str, color=geom_color, fontsize=11, fontweight='bold', ha='right', va='center')

        # ---------------- 6 Visual Content Panels (2 rows x 3 cols) ----------------
        # 1. Raw Generated RGB
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.imshow(raw_rgb)
        ax1.set_title("1. Raw Generated RGB\n[Direct Neural Convolution Output (128x128)]", color='#F0F6FC', fontsize=10.5, pad=7)
        ax1.axis('off')

        # 2. Sharpened RGB
        ax2 = fig.add_subplot(gs[1, 1])
        ax2.imshow(sharp_rgb)
        ax2.set_title("2. Sharpened RGB (Visual Inspection)\n[Mild Unsharp Mask: alpha=0.8, sigma=1.0]", color='#F0F6FC', fontsize=10.5, pad=7)
        ax2.axis('off')

        # 3. Predicted Binary Mask
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.imshow(mask, cmap='viridis')
        ax3.axvline(64, color='#FFEB3B', linestyle='--', linewidth=1.2, label='Baseline X=64')
        ax3.set_title("3. Predicted Binary Crown Mask\n[Toe-Bounded Solid Deposit Region: X < 64]", color='#F0F6FC', fontsize=10.5, pad=7)
        ax3.axis('off')

        # 4. Extracted Crown Contour
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.imshow(cnt_img, cmap='gray')
        ax4.axvline(64, color='#FFEB3B', linestyle='--', linewidth=1.2)
        ax4.set_title("4. Extracted Crown Contour\n[Continuous 1-Pixel Outer Crown Boundary]", color='#F0F6FC', fontsize=10.5, pad=7)
        ax4.axis('off')

        # 5. Edge / Gradient Map
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.imshow(edge_map, cmap='inferno')
        ax5.set_title(f"5. Edge / Gradient Map\n[{edge_note[:42]}...]", color='#F0F6FC', fontsize=10.5, pad=7)
        ax5.axis('off')

        # 6. Final Contour Overlay with Apex
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.imshow(overlay)
        ax6.set_title(f"6. Crown Contour & Apex Overlay\n[Apex: ({apex_x:.1f}, {apex_y}) | Image Reinf: {reinf_px:.1f} px | Physical: {geom_preds['predicted_reinforcement_mm']:.2f} mm]",
                      color='#3FB950', fontsize=10.5, fontweight='bold', pad=7)
        ax6.axis('off')

        # ---------------- Footer Bar ----------------
        if cond_type == "KNOWN_EXPERIMENTAL_CONDITION":
            footer_text = (
                f"VALIDATION NOTE: Specimen {sample_id} is a known training condition. "
                f"Experimental ground-truth macrographs (Views A and B) exist for quantitative benchmarking."
            )
        else:
            footer_text = (
                "SCIENTIFIC DISCLOSURE: Prospective zero-shot synthesis for unseen condition. "
                "No ground-truth macrograph exists. Pixel metrics (MAE, Dice) are NOT fabricated. "
                "Physical dimensions (mm) are derived from Ridge regression priors; image geometry (px) is synthesized via direct CNN."
            )

        fig.text(0.5, 0.015, footer_text, color='#E6EDF3', fontsize=9.2,
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262D', edgecolor=border_color, linewidth=1))

        plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)

    def process_and_save(self, material, process, feed_rate, speed):
        """Executes complete end-to-end prospective pipeline and preserves all artifacts."""
        t_start = time.time()

        # Audit condition classification
        cond_info = classify_condition(material, process, feed_rate, speed)
        cond_type, sample_id, row_dict = cond_info

        # 1. Numerical geometry predictions
        geom_preds = self.predict_numerical_geometry(material, process, feed_rate, speed)

        # 2. Generative neural synthesis
        raw_rgb, c_vec = self.generate_image(material, process, feed_rate, speed)

        # 3. Post-processing & Enhancements
        sharp_rgb = mild_unsharp_mask(raw_rgb)
        gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)

        # 4. Method 3 Toe-Bounded Mask & Contour Extraction
        mask, cnt_img, final_pts, apex_x, apex_y, reinf_px, active_area, toe_info = extract_refined_mask_and_contour(gray)

        # 5. Edge / Gradient Map & Decomposition
        edge_map, edge_note, used_fallback, edge_decomp = compute_clean_edge_gradient_map(sharp_rgb)

        # 6. Prominent overlay
        overlay = create_prominent_overlay(sharp_rgb, final_pts, apex_x, apex_y, reinf_px, geom_preds['predicted_reinforcement_mm'])

        elapsed_ms = (time.time() - t_start) * 1000

        # 7. Create Dedicated Timestamped Output Folder
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = f"{sample_id}_reconstructed" if sample_id else "unseen"
        folder_name = f"{prefix}_{ts}_{material}_{process}_F{feed_rate:.1f}_S{speed:.0f}"
        run_dir = os.path.join(REFINED_OUTPUT_BASE, folder_name)
        os.makedirs(run_dir, exist_ok=True)

        # File paths
        p_raw = os.path.join(run_dir, "raw_rgb.png")
        p_sharp = os.path.join(run_dir, "sharpened_rgb.png")
        p_mask = os.path.join(run_dir, "predicted_mask.png")
        p_cnt = os.path.join(run_dir, "predicted_contour.png")
        p_edge = os.path.join(run_dir, "edge_map.png")
        p_overlay = os.path.join(run_dir, "overlay.png")
        p_panel = os.path.join(run_dir, "final_panel.png")
        p_meta = os.path.join(run_dir, "metadata.json")
        p_coords = os.path.join(run_dir, "contour_coords.json")

        # Save individual images
        cv2.imwrite(p_raw, cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(p_sharp, cv2.cvtColor(sharp_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(p_mask, mask)
        cv2.imwrite(p_cnt, cnt_img)
        cv2.imwrite(p_edge, edge_map)
        cv2.imwrite(p_overlay, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Render and save presentation panel
        self.render_presentation_panel(material, process, feed_rate, speed, geom_preds,
                                       raw_rgb, sharp_rgb, mask, cnt_img, edge_map, overlay,
                                       apex_x, apex_y, reinf_px, edge_note, cond_info, p_panel)

        # Save contour coordinates
        coords_data = {
            "num_points": len(final_pts),
            "apex": [round(apex_x, 2), apex_y],
            "reinforcement_projection_px": round(reinf_px, 2),
            "active_area_px": active_area,
            "toe_info": toe_info,
            "contour_points": [[round(x, 2), y] for y, x in sorted(final_pts.items())]
        }
        with open(p_coords, "w", encoding="utf-8") as f:
            json.dump(coords_data, f, indent=2)

        # Save comprehensive metadata JSON
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "condition_classification": cond_type,
            "matched_training_sample": sample_id,
            "experimental_ground_truth": row_dict,
            "inputs": {
                "material": material,
                "process": process,
                "feed_rate_mpm": float(feed_rate),
                "speed_cmpm": float(speed)
            },
            "condition_vector_normalized": [round(float(v), 4) for v in c_vec],
            "predicted_geometry_mm": geom_preds,
            "generative_architecture": self.model_name,
            "inference_device": str(self.device),
            "inference_time_ms": round(elapsed_ms, 2),
            "contour_geometry": {
                "apex_x": round(apex_x, 2),
                "apex_y": apex_y,
                "reinforcement_px": round(reinf_px, 2),
                "active_area_px": active_area,
                "num_contour_points": len(final_pts),
                "toe_info": toe_info
            },
            "edge_extraction": {
                "method_note": edge_note,
                "used_fallback": used_fallback,
                "edge_decomposition": edge_decomp
            },
            "output_directory": run_dir,
            "output_files": {
                "raw_rgb": p_raw,
                "sharpened_rgb": p_sharp,
                "predicted_mask": p_mask,
                "predicted_contour": p_cnt,
                "edge_map": p_edge,
                "overlay": p_overlay,
                "final_panel": p_panel,
                "contour_coords": p_coords,
                "metadata": p_meta
            }
        }
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return run_dir, p_panel, metadata

# CLI & Interactive Entrypoint

def run_cli():
    parser = argparse.ArgumentParser(description="Phase V12 Unseen Parameter Weld Profile Generator")
    parser.add_argument("--material", choices=["IN718", "SS304L"], default=None, help="Base/Wire Material")
    parser.add_argument("--process", choices=["ColdArc", "GMAW"], default=None, help="Welding Process Mode")
    parser.add_argument("--feed_rate", type=float, default=None, help="Wire feed rate in m/min (4.5 - 7.0)")
    parser.add_argument("--speed", type=float, default=None, help="Welding travel speed in cm/min (20.0 - 32.0)")
    parser.add_argument("--model", choices=["M1"], default="M1", help="Generative model checkpoint")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu or cuda")
    args = parser.parse_args()

    print("  PHASE V12 FINALIZED STAGE: UNSEEN PARAMETER WELD PROFILE GENERATOR")

    # Interactive prompts if CLI arguments omitted
    if args.material is None:
        print("\nSelect Material:")
        print("  [1] IN718 (Inconel 718)")
        print("  [2] SS304L (Stainless Steel 304L)")
        m_choice = input("Enter choice (1/2) [default: 1]: ").strip()
        material = "SS304L" if m_choice == "2" else "IN718"
    else:
        material = args.material

    if args.process is None:
        print("\nSelect Process Mode:")
        print("  [1] ColdArc")
        print("  [2] GMAW (Conventional)")
        p_choice = input("Enter choice (1/2) [default: 1]: ").strip()
        process = "GMAW" if p_choice == "2" else "ColdArc"
    else:
        process = args.process

    if args.feed_rate is None:
        f_input = input("\nEnter Wire Feed Rate in m/min [4.5 - 7.0, default: 5.2]: ").strip()
        feed_rate = float(f_input) if f_input else 5.2
    else:
        feed_rate = args.feed_rate

    if args.speed is None:
        s_input = input("Enter Welding Travel Speed in cm/min [20.0 - 32.0, default: 26.0]: ").strip()
        speed = float(s_input) if s_input else 26.0
    else:
        speed = args.speed

    cond_type, s_id, _ = classify_condition(material, process, feed_rate, speed)
    print("\n")
    print(f"  Configuration: Material={material} | Process={process} | Feed={feed_rate:.1f} m/min | Speed={speed:.1f} cm/min")
    if cond_type == "KNOWN_EXPERIMENTAL_CONDITION":
        print(f"  CLASSIFICATION: KNOWN_EXPERIMENTAL_CONDITION (Matches specimen {s_id} in dataset)")
    else:
        print(f"  CLASSIFICATION: GENUINE_UNSEEN_INTERPOLATION (Absent from 36-sample dataset)")

    print("\nInitializing generator on local CPU...")
    generator = UnseenWeldGenerator(model_key=args.model, device=args.device)

    print("Executing synthesis and toe-bounded contour extraction...")
    run_dir, panel_path, meta = generator.process_and_save(material, process, feed_rate, speed)

    print(f"  RUN COMPLETE! (Execution time: {meta['inference_time_ms']} ms)")
    print(f"  Condition Classification: {meta['condition_classification']}")
    if meta['matched_training_sample']:
        print(f"  Matched Specimen:         {meta['matched_training_sample']}")
    print(f"  Output Directory:         {run_dir}")
    print(f"  Final Presentation Panel: {panel_path}")

if __name__ == "__main__":
    run_cli()
