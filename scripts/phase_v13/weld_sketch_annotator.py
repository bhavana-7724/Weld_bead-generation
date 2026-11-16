"""
weld_sketch_annotator.py

Phase V13: Clean Weld Bead Sketch & Dimension-Annotation Stage
Project: Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning

Features:
1. Extracts physically toe-bounded crown contours from predicted masks / generated RGB.
2. Constructs a clean, publication-ready engineering-style sketch on a pure white background:
   - Clear black bead outline (crown + fusion penetration zone)
   - Substrate baseline (plate surface)
   - Left Toe, Right Toe, Crown Apex, Deepest Penetration Point landmarks
3. Generates high-contrast dimension annotations:
   - Width (W): horizontal toe-to-toe distance
   - Depth (D): vertical distance from baseline to deepest penetration point
   - Reinforcement (R): vertical distance from baseline to crown apex
   - Shows Exp vs Pred values in mm for known samples, Pred values for unseen samples.
4. Generates all 6 required artifacts per sample:
   - 1. raw_generated_rgb.png
   - 2. predicted_mask.png
   - 3. predicted_contour.png
   - 4. weld_bead_sketch.png
   - 5. annotated_weld_bead_sketch.png
   - 6. final_annotated_panel.png
"""

import os
import sys
import json
import time
import pickle
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v9.geometry_models import DirectCNN
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV_PATH = os.path.join(DATA_DIR, "clean_dataset_36.csv")
NUM_REG_PATH = os.path.join(BASE_DIR, "models", "numerical", "prospective_regressor_checkpoint.pkl")
V9_CKPT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "checkpoints")
NORM_PARAMS_PATH = os.path.join(V9_CKPT_DIR, "normalization_params.json")
M1_CKPT_PATH = os.path.join(V9_CKPT_DIR, "m1_direct_cnn_final.pt")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
GEN36_FALLBACK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")


# =============================================================================
# Helper: Prospective Dimension Predictor
# =============================================================================
class ProspectiveDimensionPredictor:
    """Predicts Width, Depth, Reinforcement (mm) using prospective multi-output regressor."""
    def __init__(self):
        if not os.path.exists(NUM_REG_PATH):
            raise FileNotFoundError(f"Missing numerical regressor: {NUM_REG_PATH}")
        with open(NUM_REG_PATH, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.encoder = data["encoder"]
        self.num_features = data["num_features"]
        self.cat_features = data["cat_features"]

    def predict(self, material, process, feed_rate, speed):
        df_in = pd.DataFrame([{
            "feed_rate": float(feed_rate),
            "speed": float(speed),
            "material": str(material),
            "process": str(process)
        }])
        X_num = self.scaler.transform(df_in[self.num_features])
        X_cat = self.encoder.transform(df_in[self.cat_features])
        X_all = np.hstack([X_num, X_cat])
        preds = self.model.predict(X_all)[0]
        return {
            "width": round(float(preds[0]), 2),
            "depth": round(float(preds[1]), 2),
            "reinforcement": round(float(preds[2]), 2)
        }


# =============================================================================
# Helper: Generative Neural Inference
# =============================================================================
class GenerativeImageInference:
    """Infers raw 128x128 weld cross-section RGB using trained DirectCNN."""
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        with open(NORM_PARAMS_PATH, "r") as f:
            self.norm_params = json.load(f)
        self.model = DirectCNN(cond_dim=6, noise_dim=16).to(self.device)
        ckpt = torch.load(M1_CKPT_PATH, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict) and "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()

    def generate(self, material, process, feed_rate, speed):
        f_min, f_max = self.norm_params["f_min"], self.norm_params["f_max"]
        s_min, s_max = self.norm_params["s_min"], self.norm_params["s_max"]
        f_norm = (float(feed_rate) - f_min) / (f_max - f_min + 1e-8)
        s_norm = (float(speed) - s_min) / (s_max - s_min + 1e-8)

        c = [
            f_norm, s_norm,
            1.0 if material == "IN718" else 0.0,
            1.0 if material == "SS304L" else 0.0,
            1.0 if process == "ColdArc" else 0.0,
            1.0 if process == "GMAW" else 0.0
        ]
        c_tensor = torch.tensor([c], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            img_t = self.model(c_tensor)
        img_np = (np.clip(img_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        return img_np


# =============================================================================
# Helper: Refined Crown Contour & Mask Extraction
# =============================================================================
def extract_refined_crown_geometry(gray_img, x_baseline=64, y_min=18, y_max=112):
    """
    Extracts a physically toe-bounded crown mask and smooth contour profile:
    1. Isolates primary crown deposit via adaptive thresholding and connected components.
    2. Subpixel horizontal Sobel refinement.
    3. Savitzky-Golay boundary smoothing.
    4. Identifies apex and toe coordinates.
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
        if len(idx) > 0 and idx[0] < x_baseline:
            pts_cc[y] = int(idx[0])

    if len(pts_cc) < 5:
        for y in range(y_min, y_max + 1):
            idx = np.where(gray_img[y, :x_baseline] < adaptive_thresh)[0]
            if len(idx) > 0 and idx[0] < x_baseline:
                pts_cc[y] = int(idx[0])

    if len(pts_cc) < 3:
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

    top_toe = (float(final_pts[ys[0]]), float(ys[0]))
    bot_toe = (float(final_pts[ys[-1]]), float(ys[-1]))

    # 1-pixel contour image
    cnt_img = np.zeros((128, 128), dtype=np.uint8)
    for y, x_val in final_pts.items():
        xi = int(round(x_val))
        if 0 <= y < 128 and 0 <= xi < 128:
            cnt_img[y, xi] = 255

    return {
        "pts": final_pts,
        "mask": mask,
        "cnt_img": cnt_img,
        "apex_x": apex_x,
        "apex_y": float(apex_y),
        "reinf_px": reinf_px,
        "top_toe": top_toe,
        "bot_toe": bot_toe,
        "height_span_px": float(ys[-1] - ys[0] + 1)
    }


# =============================================================================
# Core Function: Draw Engineering Weld Bead Sketch (Clean or Annotated)
# =============================================================================
def draw_engineering_weld_sketch(pts_dict, width_mm, depth_mm, reinf_mm,
                                 exp_dims=None, is_annotated=True,
                                 title=None, out_path=None):
    """
    Renders an engineering-style weld bead cross-section sketch on a pure white background:
    - Workpiece plate surface baseline: horizontal line at Y = 0
    - Crown reinforcement: rises above baseline (Y >= 0) to peak height R
    - Penetration fusion boundary: curves below baseline (Y <= 0) to deepest point D
    - Left Toe (-W/2, 0) and Right Toe (+W/2, 0)
    - If is_annotated=True, adds W, D, R dimension lines, extension lines, arrows, and Exp vs Pred values.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.0), dpi=180)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    W = float(width_mm)
    D = float(depth_mm)
    R = float(reinf_mm)

    # 1. Plate Substrate Geometry
    plate_margin = max(3.0, W * 0.35)
    x_min = -W / 2.0 - plate_margin
    x_max = W / 2.0 + plate_margin
    y_plate_bottom = -D - max(2.5, D * 0.8)

    # Base metal fill
    ax.fill_between([x_min, x_max], [y_plate_bottom, y_plate_bottom], [0, 0],
                    color='#F8FAFC', edgecolor='#94A3B8', linewidth=1.5, linestyle='-', zorder=1)

    # Substrate Baseline (solid dark line)
    ax.plot([x_min, x_max], [0, 0], color='#0F172A', linewidth=2.0, zorder=3)

    # 2. Extract and Map Crown Geometry from Predicted Contour
    if pts_dict and len(pts_dict) >= 5:
        ys_raw = sorted(pts_dict.keys())
        xs_raw = [pts_dict[y] for y in ys_raw]
        y_min_val, y_max_val = ys_raw[0], ys_raw[-1]
        y_span = max(1.0, float(y_max_val - y_min_val))
        y_mid = float(y_min_val + y_max_val) / 2.0

        # Protrusion from baseline x=64
        protrusions = [max(0.0, 64.0 - x_val) for x_val in xs_raw]
        p_max = max(1.0, max(protrusions))

        # Map to physical horizontal sketch coordinates
        # -W/2 at y_min_val, +W/2 at y_max_val
        x_crown_mapped = [((y - y_mid) / (y_span / 2.0)) * (W / 2.0) for y in ys_raw]
        y_crown_mapped = [(p / p_max) * R for p in protrusions]

        # Force exact toe closure at baseline (Y=0)
        x_crown_mapped[0] = -W / 2.0
        y_crown_mapped[0] = 0.0
        x_crown_mapped[-1] = W / 2.0
        y_crown_mapped[-1] = 0.0

        xs_crown = np.array(x_crown_mapped)
        ys_crown = np.array(y_crown_mapped)
    else:
        # Graceful elliptical fallback
        xs_crown = np.linspace(-W / 2.0, W / 2.0, 100)
        ys_crown = R * (1.0 - (xs_crown / (W / 2.0)) ** 2) ** 0.75
        ys_crown[0] = 0.0
        ys_crown[-1] = 0.0

    # 3. Fusion Zone (Penetration) Geometry below Baseline
    xs_pen = np.linspace(-W / 2.0, W / 2.0, 120)
    ys_pen = -D * (1.0 - (xs_pen / (W / 2.0)) ** 2) ** 1.25

    # 4. Assemble Closed Weld Metal Polygon
    x_bead = np.concatenate([xs_crown, xs_pen[::-1]])
    y_bead = np.concatenate([ys_crown, ys_pen[::-1]])

    # Weld metal fill (metallic light steel blue)
    ax.fill(x_bead, y_bead, color='#E2E8F0', alpha=0.9, zorder=4)

    # Clean crisp black outline
    ax.plot(xs_crown, ys_crown, color='#0F172A', linewidth=2.4, zorder=5)
    ax.plot(xs_pen, ys_pen, color='#0F172A', linewidth=2.2, linestyle='-', zorder=5)

    # Baseline continuation inside bead as subtle dashed line
    ax.plot([-W / 2.0, W / 2.0], [0, 0], color='#64748B', linewidth=1.2, linestyle='--', zorder=6)

    # 5. Landmark Points
    apex_idx = np.argmax(ys_crown)
    apex_x_sketch = xs_crown[apex_idx]
    apex_y_sketch = ys_crown[apex_idx]

    ax.scatter([-W / 2.0, W / 2.0], [0, 0], color='#2563EB', s=55, zorder=7, edgecolor='#0F172A', linewidth=1.0)
    ax.scatter([apex_x_sketch], [apex_y_sketch], color='#DC2626', s=55, zorder=7, edgecolor='#0F172A', linewidth=1.0)
    ax.scatter([0], [-D], color='#059669', s=55, zorder=7, edgecolor='#0F172A', linewidth=1.0)

    # 6. Landmark Callout Annotations
    ax.annotate('Left Toe', xy=(-W / 2.0, 0), xytext=(-W / 2.0 - 2.8, max(1.2, R * 0.35)),
                arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.3),
                fontsize=9.5, fontweight='bold', color='#1E40AF', zorder=8)

    ax.annotate('Right Toe', xy=(W / 2.0, 0), xytext=(W / 2.0 + 1.0, max(1.2, R * 0.35)),
                arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.3),
                fontsize=9.5, fontweight='bold', color='#1E40AF', zorder=8)

    ax.annotate('Crown Apex', xy=(apex_x_sketch, apex_y_sketch),
                xytext=(apex_x_sketch - 3.8, apex_y_sketch + max(0.9, R * 0.25)),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.3),
                fontsize=9.5, fontweight='bold', color='#991B1B', zorder=8)

    ax.annotate('Deepest Penetration Point', xy=(0, -D),
                xytext=(-4.8, -D - max(1.2, D * 0.6)),
                arrowprops=dict(arrowstyle='->', color='#059669', lw=1.3),
                fontsize=9.5, fontweight='bold', color='#065F46', zorder=8)

    # Substrate Baseline Label
    ax.text(x_min + 0.4, -0.45, 'Substrate Baseline (Plate Surface)',
            fontsize=9.0, fontstyle='italic', color='#475569', zorder=8)

    # 7. Dimension Lines & Numerical Annotations (if requested)
    if is_annotated:
        # A. Width (W) on top of the bead
        y_w = R + max(1.8, R * 0.45)
        ax.plot([-W / 2.0, -W / 2.0], [0, y_w + 0.4], color='#64748B', linestyle='--', linewidth=1.0, zorder=6)
        ax.plot([W / 2.0, W / 2.0], [0, y_w + 0.4], color='#64748B', linestyle='--', linewidth=1.0, zorder=6)
        ax.annotate('', xy=(-W / 2.0, y_w), xytext=(W / 2.0, y_w),
                    arrowprops=dict(arrowstyle='<->', color='#1E3A8A', lw=1.8))

        if exp_dims and "width" in exp_dims:
            w_label = f"Width (W): Exp {exp_dims['width']:.2f} mm | Pred {W:.2f} mm"
        else:
            w_label = f"Width (W): Pred {W:.2f} mm"

        ax.text(0, y_w + 0.35, w_label, ha='center', va='bottom', fontsize=10.0, fontweight='bold',
                color='#1E3A8A', bbox=dict(boxstyle='round,pad=0.25', facecolor='#EFF6FF', edgecolor='#93C5FD', lw=1.0),
                zorder=9)

        # B. Reinforcement (R) and Depth (D) on the Right Side
        x_dim = W / 2.0 + max(2.8, W * 0.28)
        
        # Extension lines for R
        ax.plot([W / 2.0, x_dim + 0.6], [0, 0], color='#64748B', linestyle='--', linewidth=1.0, zorder=6)
        ax.plot([apex_x_sketch, x_dim + 0.6], [R, R], color='#64748B', linestyle='--', linewidth=1.0, zorder=6)
        ax.annotate('', xy=(x_dim, 0), xytext=(x_dim, R),
                    arrowprops=dict(arrowstyle='<->', color='#991B1B', lw=1.8))

        if exp_dims and "reinforcement" in exp_dims:
            r_label = f"Reinforcement (R):\nExp {exp_dims['reinforcement']:.2f} mm\nPred {R:.2f} mm"
        else:
            r_label = f"Reinforcement (R):\nPred {R:.2f} mm"

        ax.text(x_dim + 0.4, R / 2.0, r_label, ha='left', va='center', fontsize=9.5, fontweight='bold',
                color='#991B1B', bbox=dict(boxstyle='round,pad=0.25', facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=1.0),
                zorder=9)

        # Extension lines for D
        ax.plot([0, x_dim + 0.6], [-D, -D], color='#64748B', linestyle='--', linewidth=1.0, zorder=6)
        ax.annotate('', xy=(x_dim, -D), xytext=(x_dim, 0),
                    arrowprops=dict(arrowstyle='<->', color='#065F46', lw=1.8))

        if exp_dims and "depth" in exp_dims:
            d_label = f"Depth (D):\nExp {exp_dims['depth']:.2f} mm\nPred {D:.2f} mm"
        else:
            d_label = f"Depth (D):\nPred {D:.2f} mm"

        ax.text(x_dim + 0.4, -D / 2.0, d_label, ha='left', va='center', fontsize=9.5, fontweight='bold',
                color='#065F46', bbox=dict(boxstyle='round,pad=0.25', facecolor='#ECFDF5', edgecolor='#6EE7B7', lw=1.0),
                zorder=9)

    # Frame bounds
    x_right_bound = (W / 2.0 + max(7.5, W * 0.75)) if is_annotated else (x_max + 1.0)
    ax.set_xlim(x_min - 0.5, x_right_bound)
    ax.set_ylim(y_plate_bottom - 1.0, R + max(3.5, R * 0.75))
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        plt.title(title, fontsize=12.0, fontweight='bold', pad=15, color='#0F172A')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig, ax


# =============================================================================
# Core Function: Render Complete Academic Final Annotated Panel
# =============================================================================
def render_final_academic_panel(condition_info, pred_dims, exp_dims=None,
                                real_a_rgb=None, real_b_rgb=None,
                                gen_rgb=None, mask=None, cnt_img=None,
                                sketch_clean_path=None, sketch_annotated_path=None,
                                out_panel_path=None):
    """
    Renders the complete publication-ready final academic panel:
    - Clean white background (#FFFFFF)
    - Original experimental images (Views A and B) if available
    - Generated image (labeled as generated/predicted)
    - Clean sketch
    - Annotated sketch with W, D, R
    - Summary metric box
    """
    is_known = exp_dims is not None
    cond_name = condition_info.get("sample_code", "Unseen Setpoint")
    material = condition_info.get("material", "IN718")
    process = condition_info.get("process", "ColdArc")
    feed_rate = float(condition_info.get("feed_rate", 5.0))
    speed = float(condition_info.get("speed", 25.0))

    fig = plt.figure(figsize=(20, 11), dpi=160, facecolor='white')
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[1.0, 1.25],
                  top=0.92, bottom=0.05, left=0.04, right=0.96,
                  hspace=0.28, wspace=0.22)

    # ---------------- Top Row: Photographs & Visualizations ----------------
    if is_known and real_a_rgb is not None and real_b_rgb is not None:
        # Col 0, Row 0: Macrograph View A
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(real_a_rgb)
        ax0.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax0.set_title(f"(a) Experimental Macrograph View A\n({material}, {process}, F={feed_rate}, S={speed})",
                      fontsize=11, fontweight='bold', color='#0F172A')
        ax0.axis('off')

        # Col 1, Row 0: Macrograph View B
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(real_b_rgb)
        ax1.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax1.set_title(f"(b) Experimental Macrograph View B\n(Replicate Metallographic Section)",
                      fontsize=11, fontweight='bold', color='#0F172A')
        ax1.axis('off')
    else:
        # For unseen: Col 0: Predicted Mask, Col 1: Extracted Contour
        ax0 = fig.add_subplot(gs[0, 0])
        mask_vis = np.zeros((128, 128, 3), dtype=np.uint8)
        if mask is not None:
            mask_vis[mask > 0] = [37, 99, 235] # Royal blue
        ax0.imshow(mask_vis)
        ax0.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax0.set_title(f"(a) Predicted Binary Crown Mask\n(Differentiable / Segmented)",
                      fontsize=11, fontweight='bold', color='#0F172A')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])
        cnt_vis = np.ones((128, 128, 3), dtype=np.uint8) * 248
        if cnt_img is not None:
            cnt_vis[cnt_img > 0] = [220, 38, 38] # Red contour
        ax1.imshow(cnt_vis)
        ax1.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax1.set_title(f"(b) Extracted Crown Boundary Contour\n(Physical Toe-Bounded Profile)",
                      fontsize=11, fontweight='bold', color='#0F172A')
        ax1.axis('off')

    # Col 2, Row 0: Generated Image (Explicitly labeled as generated / predicted)
    ax2 = fig.add_subplot(gs[0, 2])
    if gen_rgb is not None:
        ax2.imshow(gen_rgb)
        ax2.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
    ax2.set_title(f"(c) Raw Model-Generated Cross-Section\n[PREDICTED / SYNTHETIC NEURAL RGB]",
                  fontsize=11, fontweight='bold', color='#B91C1C')
    ax2.axis('off')

    # ---------------- Bottom Row: Sketches & Metrics ----------------
    # Col 0, Row 1: Clean Engineering Sketch
    ax3 = fig.add_subplot(gs[1, 0])
    if sketch_clean_path and os.path.exists(sketch_clean_path):
        sk_clean_img = cv2.cvtColor(cv2.imread(sketch_clean_path), cv2.COLOR_BGR2RGB)
        ax3.imshow(sk_clean_img)
    ax3.set_title("(d) Clean Engineering Weld Bead Sketch\n(Outlines, Substrate Baseline & Landmarks)",
                  fontsize=11, fontweight='bold', color='#0F172A')
    ax3.axis('off')

    # Col 1, Row 1: Dimension-Annotated Sketch
    ax4 = fig.add_subplot(gs[1, 1])
    if sketch_annotated_path and os.path.exists(sketch_annotated_path):
        sk_ann_img = cv2.cvtColor(cv2.imread(sketch_annotated_path), cv2.COLOR_BGR2RGB)
        ax4.imshow(sk_ann_img)
    ax4.set_title("(e) Dimension-Annotated Weld Bead Sketch\n(Width W, Depth D, Reinforcement R in mm)",
                  fontsize=11, fontweight='bold', color='#0F172A')
    ax4.axis('off')

    # Col 2, Row 1: Summary Metric Box
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    # Construct metric card content
    if is_known:
        w_err = abs(exp_dims["width"] - pred_dims["width"])
        d_err = abs(exp_dims["depth"] - pred_dims["depth"])
        r_err = abs(exp_dims["reinforcement"] - pred_dims["reinforcement"])
        w_pct = (w_err / (exp_dims["width"] + 1e-5)) * 100.0
        d_pct = (d_err / (exp_dims["depth"] + 1e-5)) * 100.0
        r_pct = (r_err / (exp_dims["reinforcement"] + 1e-5)) * 100.0

        card_lines = [
            f"SPECIMEN VALIDATION SUMMARY: {cond_name}",
            "=" * 44,
            f"Material:     {material}",
            f"Process:      {process}",
            f"Feed Rate:    {feed_rate:.1f} m/min",
            f"Travel Speed: {speed:.1f} cm/min",
            "-" * 44,
            "DIMENSIONAL ACCURACY COMPARISON:",
            f"  Width (W):",
            f"    Exp: {exp_dims['width']:.2f} mm | Pred: {pred_dims['width']:.2f} mm",
            f"    Error: {w_err:.2f} mm ({w_pct:.1f}%)",
            f"  Depth (D):",
            f"    Exp: {exp_dims['depth']:.2f} mm | Pred: {pred_dims['depth']:.2f} mm",
            f"    Error: {d_err:.2f} mm ({d_pct:.1f}%)",
            f"  Reinforcement (R):",
            f"    Exp: {exp_dims['reinforcement']:.2f} mm | Pred: {pred_dims['reinforcement']:.2f} mm",
            f"    Error: {r_err:.2f} mm ({r_pct:.1f}%)",
            "-" * 44,
            "STATUS: Verified Known Experimental Condition",
            "MODELS: M1 DirectCNN + Prospective Regressor"
        ]
    else:
        card_lines = [
            f"PROSPECTIVE UNSEEN INTERPOLATION",
            "=" * 44,
            f"Material:     {material}",
            f"Process:      {process}",
            f"Feed Rate:    {feed_rate:.1f} m/min",
            f"Travel Speed: {speed:.1f} cm/min",
            "-" * 44,
            "REGRESSED PHYSICAL DIMENSIONS:",
            f"  Width (W):         Pred {pred_dims['width']:.2f} mm",
            f"  Depth (D):         Pred {pred_dims['depth']:.2f} mm",
            f"  Reinforcement (R): Pred {pred_dims['reinforcement']:.2f} mm",
            "-" * 44,
            "GEOMETRIC RATIOS:",
            f"  Aspect Ratio (W/D): {pred_dims['width'] / max(0.1, pred_dims['depth']):.2f}",
            f"  Form Factor (W/R):  {pred_dims['width'] / max(0.1, pred_dims['reinforcement']):.2f}",
            "-" * 44,
            "STATUS: Prospective Zero-Shot Synthesis",
            "MODELS: M1 DirectCNN + Prospective Regressor"
        ]

    card_text = "\n".join(card_lines)
    ax5.text(0.05, 0.50, card_text, fontsize=9.2, family='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.9', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5))
    ax5.set_title("(f) Quantitative Dimension & Verification Box", fontsize=11, fontweight='bold', color='#0F172A')

    # Main Supertitle
    fig_title = (
        f"Weld Bead Profile & Explicit Dimension Annotations — {cond_name} "
        f"({material}, {process}, F={feed_rate} m/min, S={speed} cm/min)"
    )
    plt.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.97, color='#0F172A')

    if out_panel_path:
        os.makedirs(os.path.dirname(out_panel_path), exist_ok=True)
        plt.savefig(out_panel_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig


# =============================================================================
# Core Pipeline: Process a Single Condition and Save All 6 Artifacts
# =============================================================================
def process_and_save_sample(sample_id_str, row_info, pred_dims, exp_dims,
                            gen_rgb, out_dir, real_a_rgb=None, real_b_rgb=None):
    """
    Generates and saves the 6 mandatory deliverables for a sample:
    1. raw_generated_rgb.png
    2. predicted_mask.png
    3. predicted_contour.png
    4. weld_bead_sketch.png
    5. annotated_weld_bead_sketch.png
    6. final_annotated_panel.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. raw_generated_rgb.png
    raw_rgb_path = os.path.join(out_dir, "raw_generated_rgb.png")
    cv2.imwrite(raw_rgb_path, cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR))

    # Extract refined geometry from generated RGB
    gray_gen = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)
    geom_res = extract_refined_crown_geometry(gray_gen, x_baseline=64)
    mask = geom_res["mask"]
    cnt_img = geom_res["cnt_img"]
    pts_dict = geom_res["pts"]

    # 2. predicted_mask.png
    mask_path = os.path.join(out_dir, "predicted_mask.png")
    cv2.imwrite(mask_path, mask)

    # 3. predicted_contour.png (Contour visualized clearly on white background)
    cnt_display = np.ones((128, 128, 3), dtype=np.uint8) * 255
    for y, x_val in pts_dict.items():
        xi = int(round(x_val))
        if 0 <= y < 128 and 0 <= xi < 128:
            cnt_display[y, xi] = [30, 41, 59] # Dark slate line
            if xi + 1 < 128:
                cnt_display[y, xi + 1] = [30, 41, 59]
    # Draw baseline
    cnt_display[:, 64] = [234, 179, 8] # Yellow baseline
    cnt_path = os.path.join(out_dir, "predicted_contour.png")
    cv2.imwrite(cnt_path, cv2.cvtColor(cnt_display, cv2.COLOR_RGB2BGR))

    # 4. weld_bead_sketch.png (Clean engineering sketch without dimension numbers)
    sketch_path = os.path.join(out_dir, "weld_bead_sketch.png")
    draw_engineering_weld_sketch(
        pts_dict=pts_dict,
        width_mm=pred_dims["width"],
        depth_mm=pred_dims["depth"],
        reinf_mm=pred_dims["reinforcement"],
        exp_dims=exp_dims,
        is_annotated=False,
        title=f"Weld Bead Profile Sketch — {sample_id_str}",
        out_path=sketch_path
    )

    # 5. annotated_weld_bead_sketch.png (Annotated with W, D, R dimensions & values)
    ann_sketch_path = os.path.join(out_dir, "annotated_weld_bead_sketch.png")
    draw_engineering_weld_sketch(
        pts_dict=pts_dict,
        width_mm=pred_dims["width"],
        depth_mm=pred_dims["depth"],
        reinf_mm=pred_dims["reinforcement"],
        exp_dims=exp_dims,
        is_annotated=True,
        title=f"Dimension-Annotated Weld Bead Sketch — {sample_id_str}",
        out_path=ann_sketch_path
    )

    # 6. final_annotated_panel.png (Academic final presentation panel)
    final_panel_path = os.path.join(out_dir, "final_annotated_panel.png")
    render_final_academic_panel(
        condition_info=row_info,
        pred_dims=pred_dims,
        exp_dims=exp_dims,
        real_a_rgb=real_a_rgb,
        real_b_rgb=real_b_rgb,
        gen_rgb=gen_rgb,
        mask=mask,
        cnt_img=cnt_img,
        sketch_clean_path=sketch_path,
        sketch_annotated_path=ann_sketch_path,
        out_panel_path=final_panel_path
    )

    # Save metadata JSON
    meta = {
        "sample": sample_id_str,
        "condition": row_info,
        "predicted_dimensions_mm": pred_dims,
        "experimental_dimensions_mm": exp_dims,
        "geometry_extraction": {
            "apex_x": geom_res["apex_x"],
            "apex_y": geom_res["apex_y"],
            "reinf_px": geom_res["reinf_px"],
            "top_toe": geom_res["top_toe"],
            "bot_toe": geom_res["bot_toe"],
            "height_span_px": geom_res["height_span_px"],
            "num_contour_points": len(pts_dict)
        },
        "artifacts": {
            "raw_generated_rgb": "raw_generated_rgb.png",
            "predicted_mask": "predicted_mask.png",
            "predicted_contour": "predicted_contour.png",
            "weld_bead_sketch": "weld_bead_sketch.png",
            "annotated_weld_bead_sketch": "annotated_weld_bead_sketch.png",
            "final_annotated_panel": "final_annotated_panel.png"
        }
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta
