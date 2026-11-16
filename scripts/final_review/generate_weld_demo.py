"""
generate_weld_demo.py

Interactive Weld Bead Profile Generator & Demonstration
Project: Geometry-Aware Generative AI Framework for Weld Bead Cross-Section Prediction and Visualization
Review Date: 16 November 2026

Features:
- Pure offline Windows CPU execution (zero remote SSH / GPU requirement)
- Interactive prompts or CLI arguments
- Automatic Known Condition vs Unseen In-Domain vs Out-of-Domain classification
- Dual-stream pipeline: Multi-Output RBF-SVR + M1 Direct Conditional CNN
- Output of all 7 mandatory artifacts per execution
- Auto-display of the final publication-ready panel
"""

import os
import sys
import json
import time
import argparse
import pickle
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")
ALIGNED_IMG_DIR = os.path.join(BASE_DIR, "data", "processed", "aligned_128")
SVR_CKPT_PATH = os.path.join(BASE_DIR, "models", "numerical", "prospective_regressor_checkpoint.pkl")
M1_CKPT_PATH = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage", "checkpoints_used", "m1_direct_cnn_final.pt")
V9_CKPT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "checkpoints")
NORM_PARAMS_PATH = os.path.join(V9_CKPT_DIR, "normalization_params.json")

DEMO_OUTPUTS_BASE = os.path.join(BASE_DIR, "results", "interactive_demo_outputs")
os.makedirs(DEMO_OUTPUTS_BASE, exist_ok=True)

# DirectCNN Architecture Definition
from scripts.phase_v9.geometry_models import DirectCNN

# =============================================================================
# Helper: Extract Crown Geometry
# =============================================================================
def extract_crown_geometry(gray_img, x_baseline=64):
    bg_val = np.median(gray_img[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))
    metal_roi = (gray_img[:, :x_baseline] < adaptive_thresh).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(metal_roi)
    if num_labels > 1:
        crown_cc = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    else:
        crown_cc = metal_roi
    pts = {}
    for y in range(18, 113):
        idx = np.where(crown_cc[y, :x_baseline] == 1)[0]
        if len(idx) > 0 and idx[0] < x_baseline:
            pts[y] = float(idx[0])
    if len(pts) < 5:
        for y in range(40, 88):
            pts[y] = 45.0

    mask = np.zeros((128, 128), dtype=np.uint8)
    cnt_img = np.zeros((128, 128), dtype=np.uint8)
    for y, x_val in pts.items():
        xi = int(round(x_val))
        if xi < x_baseline:
            mask[y, xi:x_baseline] = 255
            cnt_img[y, xi] = 255
    return {"pts": pts, "mask": mask, "cnt_img": cnt_img}

# =============================================================================
# Helper: Engineering Weld Bead Sketch
# =============================================================================
def render_demo_sketch(pts_dict, W, D, R, exp_dims=None, is_annotated=True, title=None, out_path=None):
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=180)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plate_margin = max(4.0, W * 0.35)
    x_min = -W / 2.0 - plate_margin
    x_max = W / 2.0 + plate_margin
    y_plate_bottom = -D - max(2.8, D * 0.8)

    ax.fill_between([x_min, x_max], [y_plate_bottom, y_plate_bottom], [0, 0],
                    color='#F8FAFC', edgecolor='#94A3B8', linewidth=1.5, linestyle='-', zorder=1)
    ax.plot([x_min, x_max], [0, 0], color='#0F172A', linewidth=2.4, zorder=4)

    if pts_dict and len(pts_dict) >= 5:
        ys_raw = sorted(pts_dict.keys())
        xs_raw = [pts_dict[y] for y in ys_raw]
        y_min_val, y_max_val = ys_raw[0], ys_raw[-1]
        y_span = max(1.0, float(y_max_val - y_min_val))
        y_mid = float(y_min_val + y_max_val) / 2.0
        protrusions = [max(0.0, 64.0 - x_val) for x_val in xs_raw]
        p_max = max(1.0, max(protrusions))

        x_crown = [((y - y_mid) / (y_span / 2.0)) * (W / 2.0) for y in ys_raw]
        y_crown = [(p / p_max) * R for p in protrusions]
        x_crown[0] = -W / 2.0
        y_crown[0] = 0.0
        x_crown[-1] = W / 2.0
        y_crown[-1] = 0.0
        xs_crown = np.array(x_crown)
        ys_crown = np.array(y_crown)
    else:
        xs_crown = np.linspace(-W / 2.0, W / 2.0, 100)
        ys_crown = R * (1.0 - (xs_crown / (W / 2.0)) ** 2) ** 0.75
        ys_crown[0] = 0.0
        ys_crown[-1] = 0.0

    ax.fill_between(xs_crown, 0, ys_crown, color='#E2E8F0', alpha=0.9, zorder=2)
    ax.plot(xs_crown, ys_crown, color='#0F172A', linewidth=2.4, zorder=5)

    # Schematic depth indicator line (NO FAKE FUSION BOUNDARY)
    ax.plot([-W / 2.0, W / 2.0], [-D, -D], color='#059669', linewidth=1.6, linestyle='--', zorder=3)
    ax.scatter([0], [-D], color='#059669', s=45, zorder=5, edgecolor='#0F172A', linewidth=1.0)

    apex_idx = np.argmax(ys_crown)
    apex_x = xs_crown[apex_idx]
    apex_y = ys_crown[apex_idx]

    ax.scatter([-W / 2.0, W / 2.0], [0, 0], color='#2563EB', s=60, zorder=6, edgecolor='#0F172A', linewidth=1.2)
    ax.scatter([apex_x], [apex_y], color='#DC2626', s=60, zorder=6, edgecolor='#0F172A', linewidth=1.2)

    ax.annotate('Left Weld Toe', xy=(-W / 2.0, 0), xytext=(-W / 2.0 - 3.2, max(1.2, R * 0.35)),
                arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.3),
                fontsize=9.0, fontweight='bold', color='#1E40AF', zorder=8)

    ax.annotate('Right Weld Toe', xy=(W / 2.0, 0), xytext=(W / 2.0 + 1.2, max(1.2, R * 0.35)),
                arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.3),
                fontsize=9.0, fontweight='bold', color='#1E40AF', zorder=8)

    ax.annotate('Crown Apex', xy=(apex_x, apex_y), xytext=(apex_x - 3.8, apex_y + max(0.9, R * 0.25)),
                arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.3),
                fontsize=9.0, fontweight='bold', color='#991B1B', zorder=8)

    ax.annotate('Predicted Penetration Depth Indicator (SVR)', xy=(0, -D),
                xytext=(-5.2, -D - max(1.0, D * 0.45)),
                arrowprops=dict(arrowstyle='->', color='#059669', lw=1.3),
                fontsize=8.8, fontweight='bold', color='#065F46', zorder=8)

    ax.text(x_min + 0.4, -0.42, 'Substrate Baseline (Plate Surface Y = 0) [Rotated from macrograph baseline X = 64]',
            fontsize=8.5, fontstyle='italic', color='#475569', zorder=8)

    if is_annotated:
        # W — Bead Width
        y_w = R + max(1.8, R * 0.42)
        ax.plot([-W / 2.0, -W / 2.0], [0, y_w + 0.35], color='#64748B', linestyle='--', linewidth=1.1, zorder=4)
        ax.plot([W / 2.0, W / 2.0], [0, y_w + 0.35], color='#64748B', linestyle='--', linewidth=1.1, zorder=4)
        ax.annotate('', xy=(-W / 2.0, y_w), xytext=(W / 2.0, y_w),
                    arrowprops=dict(arrowstyle='<->', color='#1E3A8A', lw=2.2))
        w_text = f"W — Bead Width: {W:.2f} mm"
        if exp_dims and "width" in exp_dims: w_text += f"\n(Exp: {exp_dims['width']:.2f} mm)"
        ax.text(0, y_w + 0.30, w_text, ha='center', va='bottom', fontsize=9.8, fontweight='bold',
                color='#1E3A8A', bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor='#93C5FD', lw=1.2),
                zorder=9)

        # R — Reinforcement Height
        x_dim = W / 2.0 + max(2.8, W * 0.28)
        ax.plot([W / 2.0, x_dim + 0.6], [0, 0], color='#64748B', linestyle='--', linewidth=1.1, zorder=4)
        ax.plot([apex_x, x_dim + 0.6], [R, R], color='#64748B', linestyle='--', linewidth=1.1, zorder=4)
        ax.annotate('', xy=(x_dim, 0), xytext=(x_dim, R),
                    arrowprops=dict(arrowstyle='<->', color='#991B1B', lw=2.2))
        r_text = f"R — Reinforcement Height: {R:.2f} mm"
        if exp_dims and "reinforcement" in exp_dims: r_text += f"\n(Exp: {exp_dims['reinforcement']:.2f} mm)"
        ax.text(x_dim + 0.5, R / 2.0, r_text, ha='left', va='center', fontsize=9.2, fontweight='bold',
                color='#991B1B', bbox=dict(boxstyle='round,pad=0.35', facecolor='#FEF2F2', edgecolor='#FCA5A5', lw=1.2),
                zorder=9)

        # D — Predicted Penetration Depth
        ax.plot([0, x_dim + 0.6], [-D, -D], color='#64748B', linestyle='--', linewidth=1.1, zorder=4)
        ax.annotate('', xy=(x_dim, -D), xytext=(x_dim, 0),
                    arrowprops=dict(arrowstyle='<->', color='#065F46', lw=2.2))
        d_text = f"D — Predicted Penetration Depth: {D:.2f} mm"
        if exp_dims and "depth" in exp_dims: d_text += f"\n(Exp: {exp_dims['depth']:.2f} mm)"
        ax.text(x_dim + 0.5, -D / 2.0, d_text, ha='left', va='center', fontsize=9.2, fontweight='bold',
                color='#065F46', bbox=dict(boxstyle='round,pad=0.35', facecolor='#ECFDF5', edgecolor='#6EE7B7', lw=1.2),
                zorder=9)

    disclaimer = "Engineering sketch is schematic; millimetre dimensions are numerical SVR predictions."
    ax.text(0.5, 0.02, disclaimer, transform=ax.transAxes, ha='center', va='bottom',
            fontsize=8.8, fontweight='bold', fontstyle='italic', color='#475569',
            bbox=dict(boxstyle='square,pad=0.4', facecolor='#F8FAFC', edgecolor='#CBD5E1', lw=0.8))

    x_right_bound = (W / 2.0 + max(8.0, W * 0.78)) if is_annotated else (x_max + 1.0)
    ax.set_xlim(x_min - 0.5, x_right_bound)
    ax.set_ylim(y_plate_bottom - 1.2, R + max(3.2, R * 0.70))
    ax.set_aspect('equal')
    ax.axis('off')

    if title: plt.title(title, fontsize=11.5, fontweight='bold', pad=14, color='#0F172A')
    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig, ax

# =============================================================================
# Helper: Render Final Demo Panel
# =============================================================================
def render_demo_panel(mat, proc, feed, speed, W, D, R,
                      gen_rgb, mask, cnt_img, sketch_path,
                      status_label, matched_sample=None, exp_dims=None,
                      real_a_rgb=None, real_b_rgb=None, out_panel_path=None):
    fig = plt.figure(figsize=(19, 10.5), dpi=160, facecolor='white')
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[1.0, 1.25],
                  top=0.91, bottom=0.05, left=0.04, right=0.96,
                  hspace=0.28, wspace=0.22)

    is_known = matched_sample is not None and real_a_rgb is not None

    # Row 1
    if is_known:
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(real_a_rgb)
        ax0.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax0.set_title(f"(a) Experimental Macrograph — View A\n(Matched: {matched_sample})", fontsize=11, fontweight='bold', color='#0F172A')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(real_b_rgb)
        ax1.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax1.set_title(f"(b) Experimental Macrograph — View B\n(Replicate Metallographic Section)", fontsize=11, fontweight='bold', color='#0F172A')
        ax1.axis('off')
    else:
        ax0 = fig.add_subplot(gs[0, 0])
        mask_vis = np.zeros((128, 128, 3), dtype=np.uint8)
        mask_vis[mask > 0] = [37, 99, 235]
        ax0.imshow(mask_vis)
        ax0.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax0.set_title(f"(a) Model-Predicted Binary Crown Mask\n[Differentiable Geometric Mask]", fontsize=11, fontweight='bold', color='#0F172A')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])
        cnt_vis = np.ones((128, 128, 3), dtype=np.uint8) * 248
        cnt_vis[cnt_img > 0] = [220, 38, 38]
        ax1.imshow(cnt_vis)
        ax1.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        ax1.set_title(f"(b) Extracted Crown Boundary Contour\n[Physical Toe-Bounded Profile]", fontsize=11, fontweight='bold', color='#0F172A')
        ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(gen_rgb)
    ax2.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
    ax2.set_title(f"(c) Model-Generated Cross-Section Image\n[M1: Direct Conditional CNN]", fontsize=11, fontweight='bold', color='#B91C1C')
    ax2.axis('off')

    # Row 2
    ax3 = fig.add_subplot(gs[1, 0])
    if os.path.exists(sketch_path):
        sk = cv2.cvtColor(cv2.imread(sketch_path), cv2.COLOR_BGR2RGB)
        ax3.imshow(sk)
    ax3.set_title("(d) Clean Engineering Weld Bead Sketch\n(Outlines, Plate Baseline & Landmarks)", fontsize=11, fontweight='bold', color='#0F172A')
    ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 1])
    # Visual overlay or contour
    if is_known:
        overlay = (cv2.cvtColor(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB).astype(np.float32) * 0.8).astype(np.uint8)
        for y in range(0, 128, 4): overlay[y:y+2, 64] = [255, 255, 0]
        cnt_pts = np.where(cnt_img > 0)
        for cy, cx in zip(cnt_pts[0], cnt_pts[1]):
            overlay[cy, cx] = [0, 255, 0]
            if cx - 1 >= 0: overlay[cy, cx - 1] = [0, 255, 0]
        ax4.imshow(overlay)
        ax4.set_title("(e) Experimental vs Model-Predicted Crown Contour\n[Background: View A | Lime Green: Predicted Crown]", fontsize=11, fontweight='bold', color='#0F172A')
    else:
        # Show predicted RGB with overlaid green contour
        comp = gen_rgb.copy()
        for y in range(0, 128, 4): comp[y:y+2, 64] = [255, 255, 0]
        cnt_pts = np.where(cnt_img > 0)
        for cy, cx in zip(cnt_pts[0], cnt_pts[1]):
            comp[cy, cx] = [0, 255, 0]
            if cx - 1 >= 0: comp[cy, cx - 1] = [0, 255, 0]
        ax4.imshow(comp)
        ax4.set_title("(e) Synthetic Macrograph with Predicted Crown\n[Yellow: Baseline X=64 | Green: Crown Contour]", fontsize=11, fontweight='bold', color='#0F172A')
    ax4.axis('off')

    # Metric card
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')

    card_lines = [
        f"INSPECTION & VERIFICATION CARD",
        "=" * 50,
        f"Status:       {status_label}",
        f"Material:     {mat}",
        f"Process:      {proc}",
        f"Feed Rate:    {feed:.2f} m/min",
        f"Travel Speed: {speed:.2f} cm/min",
        "-" * 50,
        "PREDICTED BEAD DIMENSIONS (Multi-Output SVR):",
        f"  Width (W):         {W:.2f} mm",
        f"  Depth (D):         {D:.2f} mm",
        f"  Reinforcement (R): {R:.2f} mm",
        "-" * 50
    ]

    if is_known and exp_dims:
        w_err = abs(exp_dims['width'] - W)
        d_err = abs(exp_dims['depth'] - D)
        r_err = abs(exp_dims['reinforcement'] - R)
        card_lines.extend([
            f"EXPERIMENTAL GROUND TRUTH ({matched_sample}):",
            f"  Width (W):         {exp_dims['width']:.2f} mm (Err: {w_err:.2f} mm)",
            f"  Depth (D):         {exp_dims['depth']:.2f} mm (Err: {d_err:.2f} mm)",
            f"  Reinforcement (R): {exp_dims['reinforcement']:.2f} mm (Err: {r_err:.2f} mm)",
            "-" * 50
        ])
    else:
        card_lines.extend([
            "GEOMETRIC RATIOS:",
            f"  Aspect Ratio (W/D): {W / max(0.1, D):.2f}",
            f"  Form Factor (W/R):  {W / max(0.1, R):.2f}",
            "-" * 50,
            "NOTE: Unseen prospective condition.",
            "No experimental ground truth available for this condition."
        ])

    card_lines.extend([
        "MODELS DEPLOYED:",
        "  - Numerical: RBF-SVR (C=10, epsilon=0.10)",
        "  - Generative: M1 Direct Conditional CNN",
        "=" * 50
    ])

    ax5.text(0.04, 0.50, "\n".join(card_lines), fontsize=8.8, family='monospace', verticalalignment='center',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.5))
    ax5.set_title("(f) Quantitative Dimension & Verification Summary", fontsize=11, fontweight='bold', color='#0F172A')

    plt.suptitle(f"Weld Bead Profile Reconstruction and Dimension Comparison — {status_label}\n({mat}, {proc}, F={feed:.1f} m/min, S={speed:.1f} cm/min)",
                 fontsize=13.0, fontweight='bold', y=0.97, color='#0F172A')

    if out_panel_path:
        os.makedirs(os.path.dirname(out_panel_path), exist_ok=True)
        plt.savefig(out_panel_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig

# =============================================================================
# Core Generator Function
# =============================================================================
def generate_weld_cross_section(material, process, feed_rate, speed, auto_open=False, custom_out_dir=None):
    mat = str(material).strip()
    proc = str(process).strip()
    feed = float(feed_rate)
    speed = float(speed)

    print(f"\n[DEMO] Processing input: Material={mat}, Process={proc}, Feed={feed:.2f} m/min, Speed={speed:.2f} cm/min")

    # 1. Load Experimental Dataset & Check Classification
    df = pd.read_csv(DATA_PATH)
    matched = df[(df['material'] == mat) &
                 (df['process'] == proc) &
                 (np.isclose(df['feed_rate'], feed, atol=0.01)) &
                 (np.isclose(df['speed'], speed, atol=0.01))]

    # Domain limits
    f_min, f_max = df['feed_rate'].min(), df['feed_rate'].max()
    s_min, s_max = df['speed'].min(), df['speed'].max()

    is_known = len(matched) > 0
    in_domain = (f_min <= feed <= f_max) and (s_min <= speed <= s_max) and (mat in ["IN718", "SS304L"]) and (proc in ["ColdArc", "GMAW"])

    matched_sample = None
    exp_dims = None
    real_a_rgb = None
    real_b_rgb = None

    if is_known:
        row = matched.iloc[0]
        s_id = int(row['sample'])
        matched_sample = f"S{s_id:02d}"
        status_label = f"KNOWN EXPERIMENTAL CONDITION — RECONSTRUCTION ({matched_sample})"
        exp_dims = {"width": float(row['width']), "depth": float(row['depth']), "reinforcement": float(row['reinforcement'])}

        img_a = os.path.join(ALIGNED_IMG_DIR, f"{s_id}-A.png")
        img_b = os.path.join(ALIGNED_IMG_DIR, f"{s_id}-B.png")
        if os.path.exists(img_a): real_a_rgb = cv2.cvtColor(cv2.imread(img_a), cv2.COLOR_BGR2RGB)
        if os.path.exists(img_b): real_b_rgb = cv2.cvtColor(cv2.imread(img_b), cv2.COLOR_BGR2RGB)
        print(f"[STATUS] {status_label}")
    elif in_domain:
        status_label = "UNSEEN IN-DOMAIN PARAMETER INTERPOLATION"
        print(f"[STATUS] {status_label}")
    else:
        status_label = "OUT-OF-DOMAIN EXTRAPOLATION (CAUTION: Unsupported Parameters)"
        print(f"[WARN] {status_label}")

    # 2. Predict Dimensions via SVR
    with open(SVR_CKPT_PATH, "rb") as f:
        svr_bundle = pickle.load(f)
    svr_model = svr_bundle["model"]
    scaler = svr_bundle["scaler"]
    encoder = svr_bundle["encoder"]

    df_in = pd.DataFrame([{"feed_rate": feed, "speed": speed, "material": mat, "process": proc}])
    X_num = scaler.transform(df_in[["feed_rate", "speed"]])
    X_cat = encoder.transform(df_in[["material", "process"]])
    X_all = np.hstack([X_num, X_cat])
    svr_preds = svr_model.predict(X_all)[0]

    pred_w = round(float(svr_preds[0]), 2)
    pred_d = round(float(svr_preds[1]), 2)
    pred_r = round(float(svr_preds[2]), 2)
    print(f"[SVR PREDICTIONS] Width: {pred_w:.2f} mm | Depth: {pred_d:.2f} mm | Reinforcement: {pred_r:.2f} mm")

    # 3. Generate Image via M1 Direct CNN
    with open(NORM_PARAMS_PATH, "r") as f:
        norm_params = json.load(f)

    f_norm = (feed - norm_params["f_min"]) / (norm_params["f_max"] - norm_params["f_min"] + 1e-8)
    s_norm = (speed - norm_params["s_min"]) / (norm_params["s_max"] - norm_params["s_min"] + 1e-8)
    c = [
        f_norm, s_norm,
        1.0 if mat == "IN718" else 0.0,
        1.0 if mat == "SS304L" else 0.0,
        1.0 if proc == "ColdArc" else 0.0,
        1.0 if proc == "GMAW" else 0.0
    ]
    c_tensor = torch.tensor([c], dtype=torch.float32)

    model = DirectCNN(cond_dim=6, noise_dim=16)
    ckpt = torch.load(M1_CKPT_PATH, map_location='cpu')
    if isinstance(ckpt, dict) and "state_dict" in ckpt: model.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict) and "model" in ckpt: model.load_state_dict(ckpt["model"])
    else: model.load_state_dict(ckpt)
    model.eval()

    with torch.no_grad():
        gen_t = model(c_tensor)
    gen_rgb = (np.clip(gen_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

    # 4. Extract Geometry & Produce Deliverables
    gray_gen = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)
    geom = extract_crown_geometry(gray_gen, x_baseline=64)
    mask = geom["mask"]
    cnt_img = geom["cnt_img"]
    pts = geom["pts"]

    # Destination directory
    if custom_out_dir:
        run_dir = custom_out_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(DEMO_OUTPUTS_BASE, f"run_{mat}_{proc}_F{feed}_S{speed}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save 7 required deliverables:
    # 1. generated_rgb.png
    cv2.imwrite(os.path.join(run_dir, "generated_rgb.png"), cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR))
    # 2. predicted_crown_mask.png
    cv2.imwrite(os.path.join(run_dir, "predicted_crown_mask.png"), mask)
    # 3. predicted_crown_contour.png
    cnt_vis = np.ones((128, 128, 3), dtype=np.uint8) * 255
    cnt_vis[cnt_img > 0] = [30, 41, 59]
    cnt_vis[:, 64] = [234, 179, 8] # Yellow baseline
    cv2.imwrite(os.path.join(run_dir, "predicted_crown_contour.png"), cv2.cvtColor(cnt_vis, cv2.COLOR_RGB2BGR))

    # 4. engineering_weld_sketch.png
    clean_sketch_path = os.path.join(run_dir, "engineering_weld_sketch.png")
    render_demo_sketch(pts, pred_w, pred_d, pred_r, exp_dims=exp_dims, is_annotated=False,
                       title=f"Engineering Weld Bead Sketch — {mat} {proc}", out_path=clean_sketch_path)

    # 5. annotated_weld_sketch.png
    ann_sketch_path = os.path.join(run_dir, "annotated_weld_sketch.png")
    render_demo_sketch(pts, pred_w, pred_d, pred_r, exp_dims=exp_dims, is_annotated=True,
                       title=f"Dimension-Annotated Weld Bead Sketch — {mat} {proc}", out_path=ann_sketch_path)

    # 6. final_demo_panel.png
    final_panel_path = os.path.join(run_dir, "final_demo_panel.png")
    render_demo_panel(mat, proc, feed, speed, pred_w, pred_d, pred_r,
                      gen_rgb, mask, cnt_img, ann_sketch_path,
                      status_label, matched_sample=matched_sample, exp_dims=exp_dims,
                      real_a_rgb=real_a_rgb, real_b_rgb=real_b_rgb,
                      out_panel_path=final_panel_path)

    # 7. prediction_metadata.json
    metadata = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status_label,
        "is_known_experiment": is_known,
        "matched_sample_id": matched_sample,
        "input_parameters": {
            "material": mat,
            "process": proc,
            "wire_feed_rate_m_min": feed,
            "travel_speed_cm_min": speed
        },
        "predicted_dimensions_mm": {
            "width": pred_w,
            "depth": pred_d,
            "reinforcement": pred_r
        },
        "experimental_ground_truth_mm": exp_dims,
        "models_used": {
            "numerical_regressor": "Multi-Output RBF-SVR (C=10, epsilon=0.10)",
            "generative_cross_section": "M1 Direct Conditional CNN"
        },
        "artifacts_generated": [
            "generated_rgb.png",
            "predicted_crown_mask.png",
            "predicted_crown_contour.png",
            "engineering_weld_sketch.png",
            "annotated_weld_sketch.png",
            "final_demo_panel.png",
            "prediction_metadata.json"
        ]
    }
    with open(os.path.join(run_dir, "prediction_metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"\n[SUCCESS] Generated all 7 artifacts in: {run_dir}")
    print(f"Final presentation panel: {final_panel_path}")

    # Automatically open image if on Windows and requested
    if auto_open and sys.platform.startswith('win'):
        try:
            os.startfile(final_panel_path)
            print("[DISPLAY] Opened final_demo_panel.png automatically.")
        except Exception as e:
            print(f"[DISPLAY] Note: could not launch auto-open: {e}")

    return run_dir, final_panel_path

# =============================================================================
# CLI & Interactive Menu
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Interactive Weld Bead Cross-Section Generator")
    parser.add_argument("--material", type=str, choices=["IN718", "SS304L"], help="Base material")
    parser.add_argument("--process", type=str, choices=["ColdArc", "GMAW"], help="Welding process")
    parser.add_argument("--feed", type=float, help="Wire feed rate in m/min")
    parser.add_argument("--speed", type=float, help="Travel speed in cm/min")
    parser.add_argument("--auto-open", action="store_true", help="Automatically open final panel")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    if args.material and args.process and args.feed and args.speed:
        generate_weld_cross_section(args.material, args.process, args.feed, args.speed,
                                    auto_open=args.auto_open, custom_out_dir=args.out_dir)
        return

    # Interactive Console Mode
    print("=" * 70)
    print("WELDING GENERATIVE AI — INTERACTIVE CROSS-SECTION GENERATOR")
    print("=" * 70)
    print("Supported Materials: IN718, SS304L")
    print("Supported Processes: ColdArc, GMAW")
    print("Recommended Ranges:  Feed rate: 4.5 - 6.5 m/min | Speed: 20 - 32 cm/min\n")

    # Material
    while True:
        m_in = input("Select Material (1: IN718, 2: SS304L) [default: 1]: ").strip()
        if m_in in ["", "1", "IN718", "in718"]: mat = "IN718"; break
        elif m_in in ["2", "SS304L", "ss304l"]: mat = "SS304L"; break
        print("Invalid selection. Enter 1 or 2.")

    # Process
    while True:
        p_in = input("Select Process  (1: ColdArc, 2: GMAW)  [default: 1]: ").strip()
        if p_in in ["", "1", "ColdArc", "coldarc"]: proc = "ColdArc"; break
        elif p_in in ["2", "GMAW", "gmaw"]: proc = "GMAW"; break
        print("Invalid selection. Enter 1 or 2.")

    # Feed rate
    while True:
        f_in = input("Enter Wire Feed Rate in m/min (e.g. 5.2) [default: 5.2]: ").strip()
        if f_in == "": feed = 5.2; break
        try:
            feed = float(f_in)
            break
        except ValueError:
            print("Please enter a valid numeric value.")

    # Speed
    while True:
        s_in = input("Enter Travel Speed in cm/min (e.g. 26.0)   [default: 26.0]: ").strip()
        if s_in == "": speed = 26.0; break
        try:
            speed = float(s_in)
            break
        except ValueError:
            print("Please enter a valid numeric value.")

    generate_weld_cross_section(mat, proc, feed, speed, auto_open=True, custom_out_dir=args.out_dir)

if __name__ == "__main__":
    main()
