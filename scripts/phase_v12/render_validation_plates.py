"""
render_validation_plates.py

Renders publication-ready, professor-grade scientific validation plates:
1. S04_validation_panel.png (Known Experimental Condition S04: IN718, ColdArc, F=5.0, S=25)
2. S33_validation_panel.png (Known Experimental Condition S33: SS304L, GMAW, F=5.5, S=27)
3. UNSEEN_INTERPOLATION_PANEL.png (Genuine Unseen Interpolation: IN718, ColdArc, F=5.2, S=26)

Adheres strictly to scientific integrity:
- Preserves distinct modalities (physical mm regression vs. synthesized px geometry).
- Accurately reports View A, View B, and Dual-View Average metrics for known specimens.
- Explicitly labels unseen interpolation as prospective with ZERO fabricated ground-truth scores.
"""

import os
import sys
import json
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
    UnseenWeldGenerator,
    compute_clean_edge_gradient_map,
    create_prominent_overlay
)

STAGE_DIR = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage")
V9_GEN_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")
GT_MASK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit", "gt_masks")
CLEAN_CSV = os.path.join(BASE_DIR, "data", "clean_dataset_36.csv")


def render_known_specimen_panel(s_code, s_num, out_path):
    df_clean = pd.read_csv(CLEAN_CSV)
    row = df_clean[df_clean["sample"] == s_num].iloc[0]

    mat = row["material"]
    proc = row["process"]
    f_rate = float(row["feed_rate"])
    speed = float(row["speed"])
    hip = float(row["HIP"])
    width_mm = float(row["width"])
    depth_mm = float(row["depth"])
    reinf_mm = float(row["reinforcement"])

    s_dir = os.path.join(V9_GEN_DIR, s_code)
    real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_A.png")), cv2.COLOR_BGR2RGB)
    real_b_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "Real_B.png")), cv2.COLOR_BGR2RGB)
    gen_rgb = cv2.cvtColor(cv2.imread(os.path.join(s_dir, "M1_DirectCNN_raw.png")), cv2.COLOR_BGR2RGB)
    sharp_rgb = mild_unsharp_mask(gen_rgb)
    gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)

    gt_mask_a = cv2.imread(os.path.join(GT_MASK_DIR, f"{s_num}-A.png"), cv2.IMREAD_GRAYSCALE)
    gt_mask_b = cv2.imread(os.path.join(GT_MASK_DIR, f"{s_num}-B.png"), cv2.IMREAD_GRAYSCALE)
    cnt_a = get_gt_contour_from_mask(gt_mask_a)
    cnt_b = get_gt_contour_from_mask(gt_mask_b)

    # Method 3 Toe-Bounded Contour
    cnt_gen = extract_method3_toe_bounded(gen_gray)
    canny_a = compute_clean_canny(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY))
    canny_b = compute_clean_canny(cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY))
    canny_gen = compute_clean_canny(gen_gray)

    mA = evaluate_metrics(real_a_rgb, gen_rgb, cnt_a, cnt_gen, canny_a, canny_gen)
    mB = evaluate_metrics(real_b_rgb, gen_rgb, cnt_b, cnt_gen, canny_b, canny_gen)
    top_err_a, bot_err_a = compute_toe_errors(cnt_a, cnt_gen)
    top_err_b, bot_err_b = compute_toe_errors(cnt_b, cnt_gen)

    edge_map, edge_note, _ = compute_clean_edge_gradient_map(sharp_rgb)

    # 1800x1200 plate
    fig = plt.figure(figsize=(18, 12), dpi=150, facecolor='#0D1117')
    gs = GridSpec(nrows=3, ncols=4, height_ratios=[0.95, 3.8, 3.8],
                  top=0.96, bottom=0.05, left=0.035, right=0.965,
                  hspace=0.28, wspace=0.18)

    # Header Banner
    ax_b = fig.add_subplot(gs[0, :])
    ax_b.set_facecolor('#161B22')
    ax_b.axis('off')
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax_b.transAxes, fill=False, edgecolor='#30363D', linewidth=1.5)
    ax_b.add_patch(rect)

    title_str = f"KNOWN-CONDITION RECONSTRUCTION VALIDATION — SPECIMEN {s_code}"
    sub_str = f"Model: M1 Direct Conditional CNN | Training Specimen In-Sample Evaluation | Architecture: Geometry-Aware Transposed Conv"
    proc_str = f"Process Conditions:  Material = {mat}  |  Process = {proc}  |  Feed Rate = {f_rate:.1f} m/min  |  Speed = {speed:.1f} cm/min  |  HIP = {hip:.2f} J/mm"
    exp_str = f"Experimental Ground-Truth: Width = {width_mm:.2f} mm | Depth = {depth_mm:.2f} mm | Reinforcement = {reinf_mm:.2f} mm"

    ax_b.text(0.015, 0.78, title_str, color='#58A6FF', fontsize=16, fontweight='bold', va='center')
    ax_b.text(0.015, 0.52, sub_str, color='#8B949E', fontsize=11, fontstyle='italic', va='center')
    ax_b.text(0.015, 0.22, proc_str, color='#F0F6FC', fontsize=11, fontweight='semibold', va='center')
    ax_b.text(0.985, 0.22, exp_str, color='#3FB950', fontsize=11, fontweight='bold', ha='right', va='center')

    # Visual Panels (2 rows x 4 cols)
    # 1. Real A
    ax1 = fig.add_subplot(gs[1, 0])
    ov_a = create_contour_overlay(real_a_rgb, cnt_a['pts'], (cnt_a['apex_x'], cnt_a['apex_y']), line_color=(0, 220, 255))
    ax1.imshow(ov_a)
    ax1.set_title(f"1. Experimental View A (Macrograph)\n[Cyan: GT Contour | Span: {len(cnt_a['pts'])} px]", color='#F0F6FC', fontsize=10, pad=6)
    ax1.axis('off')

    # 2. Real B
    ax2 = fig.add_subplot(gs[1, 1])
    ov_b = create_contour_overlay(real_b_rgb, cnt_b['pts'], (cnt_b['apex_x'], cnt_b['apex_y']), line_color=(255, 0, 220))
    ax2.imshow(ov_b)
    ax2.set_title(f"2. Experimental View B (Reference)\n[Magenta: GT Contour | Span: {len(cnt_b['pts'])} px]", color='#F0F6FC', fontsize=10, pad=6)
    ax2.axis('off')

    # 3. Raw Generated RGB
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(gen_rgb)
    ax3.set_title("3. Raw Generated RGB\n[Direct Neural Convolution Output (128x128)]", color='#F0F6FC', fontsize=10, pad=6)
    ax3.axis('off')

    # 4. Sharpened RGB
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.imshow(sharp_rgb)
    ax4.set_title("4. Sharpened RGB (Visual Inspection)\n[Mild Unsharp Mask: alpha=0.8, sigma=1.0]", color='#F0F6FC', fontsize=10, pad=6)
    ax4.axis('off')

    # 5. Predicted Binary Mask
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.imshow(cnt_gen['mask'], cmap='viridis')
    ax5.axvline(64, color='#FFEB3B', linestyle='--', linewidth=1.1)
    ax5.set_title(f"5. Predicted Binary Crown Mask\n[Toe-Bounded Support: {cnt_gen['support_span']} rows | Area: {cnt_gen['active_area_px']} px]", color='#F0F6FC', fontsize=10, pad=6)
    ax5.axis('off')

    # 6. Extracted Crown Contour
    ax6 = fig.add_subplot(gs[2, 1])
    cnt_display = np.zeros((128, 128), dtype=np.uint8)
    for y, x in cnt_gen['pts'].items():
        cnt_display[int(y), int(round(x))] = 255
    ax6.imshow(cnt_display, cmap='gray')
    ax6.axvline(64, color='#FFEB3B', linestyle='--', linewidth=1.1)
    ax6.set_title(f"6. Physical Crown Contour\n[Continuous Outer Boundary | Span: Y={cnt_gen['y_range'][0]}..{cnt_gen['y_range'][1]}]", color='#F0F6FC', fontsize=10, pad=6)
    ax6.axis('off')

    # 7. Edge / Gradient Map
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.imshow(edge_map, cmap='inferno')
    ax7.set_title(f"7. Edge / Gradient Diagnostic Map\n[{edge_note[:40]}...]", color='#F0F6FC', fontsize=10, pad=6)
    ax7.axis('off')

    # 8. Triple Contour Overlay
    ax8 = fig.add_subplot(gs[2, 3])
    tri_overlay = create_dual_contour_overlay(sharp_rgb, cnt_a['pts'], cnt_b['pts'], cnt_gen['pts'])
    ax8.imshow(tri_overlay)
    ax8.set_title("8. Multi-View Overlay\n[Cyan: View A | Magenta: View B | Green: Gen M1]", color='#3FB950', fontsize=10, fontweight='bold', pad=6)
    ax8.axis('off')

    # Bottom Scorecard Bar
    avg_mae = 0.5 * (mA['contour_mae_px'] + mB['contour_mae_px'])
    avg_f1 = 0.5 * (mA['boundary_f1_2px'] + mB['boundary_f1_2px'])
    avg_dice = 0.5 * (mA['mask_dice'] + mB['mask_dice'])
    avg_iou = 0.5 * (mA['mask_iou'] + mB['mask_iou'])
    avg_apex = 0.5 * (mA['apex_projection_error_px'] + mB['apex_projection_error_px'])
    avg_top_toe = 0.5 * (top_err_a + top_err_b)
    avg_bot_toe = 0.5 * (bot_err_a + bot_err_b)

    score_text = (
        f"SCIENTIFIC INTEGRITY SCORECARD:  "
        f"View A MAE = {mA['contour_mae_px']:.2f} px  |  "
        f"View B MAE = {mB['contour_mae_px']:.2f} px  |  "
        f"Dual-View Avg MAE = {avg_mae:.2f} px  |  "
        f"Boundary F1 (2px) = {avg_f1:.3f}  |  "
        f"Mask Dice = {avg_dice:.3f}  |  "
        f"Mask IoU = {avg_iou:.3f}  |  "
        f"Apex Error = {avg_apex:.2f} px  |  "
        f"Toe Errors (Top/Bot) = {avg_top_toe:.1f}/{avg_bot_toe:.1f} px  |  "
        f"Classification: KNOWN_EXPERIMENTAL_CONDITION"
    )
    fig.text(0.5, 0.014, score_text, color='#E6EDF3', fontsize=9.2,
             ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262D', edgecolor='#30363D', linewidth=1))

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Rendered: {out_path}")


def render_unseen_interpolation_panel(out_path):
    gen = UnseenWeldGenerator(device="cpu")
    mat, proc, f_rate, speed = "IN718", "ColdArc", 5.2, 26.0
    geom_preds = gen.predict_numerical_geometry(mat, proc, f_rate, speed)
    raw_rgb, c_vec = gen.generate_image(mat, proc, f_rate, speed)
    sharp_rgb = mild_unsharp_mask(raw_rgb)
    gen_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)

    cnt_gen = extract_method3_toe_bounded(gen_gray)
    edge_map, edge_note, _ = compute_clean_edge_gradient_map(sharp_rgb)

    overlay = create_prominent_overlay(
        sharp_rgb,
        cnt_gen['pts'],
        float(cnt_gen['apex_x']),
        int(round(cnt_gen['apex_y'])),
        float(cnt_gen['reinf_proj_px']),
        geom_preds['predicted_reinforcement_mm']
    )

    fig = plt.figure(figsize=(16, 11), dpi=150, facecolor='#0D1117')
    gs = GridSpec(nrows=3, ncols=3, height_ratios=[1.15, 4.0, 4.0],
                  top=0.95, bottom=0.06, left=0.04, right=0.96,
                  hspace=0.30, wspace=0.20)

    # Banner
    ax_b = fig.add_subplot(gs[0, :])
    ax_b.set_facecolor('#161B22')
    ax_b.axis('off')
    rect = plt.Rectangle((0, 0), 1, 1, transform=ax_b.transAxes, fill=False, edgecolor='#FF7B72', linewidth=1.5)
    ax_b.add_patch(rect)

    title_str = "PROSPECTIVE UNSEEN INTERPOLATION — ZERO-SHOT SYNTHESIS"
    sub_str = "Model: M1 Direct Conditional CNN | Prospective Parameter Combination (Absent from 36-Sample Training Set) | MultiOutput Ridge Priors"
    proc_str = f"Target Conditions:  Material = {mat}   |   Process = {proc}   |   Feed Rate = {f_rate:.1f} m/min   |   Speed = {speed:.1f} cm/min"
    geom_str = f"Regressed Geometry (Physical):  Width = {geom_preds['predicted_width_mm']:.2f} mm   |   Depth = {geom_preds['predicted_depth_mm']:.2f} mm   |   Reinforcement = {geom_preds['predicted_reinforcement_mm']:.2f} mm"

    ax_b.text(0.02, 0.78, title_str, color='#FF7B72', fontsize=15, fontweight='bold', va='center')
    ax_b.text(0.02, 0.54, sub_str, color='#8B949E', fontsize=10.5, fontstyle='italic', va='center')
    ax_b.text(0.02, 0.24, proc_str, color='#F0F6FC', fontsize=11, fontweight='semibold', va='center')
    ax_b.text(0.98, 0.24, geom_str, color='#3FB950', fontsize=11, fontweight='bold', ha='right', va='center')

    # 6 Panels
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
    ax3.imshow(cnt_gen['mask'], cmap='viridis')
    ax3.axvline(64, color='#FFEB3B', linestyle='--', linewidth=1.2)
    ax3.set_title(f"3. Predicted Binary Crown Mask\n[Toe-Bounded Support: {cnt_gen['support_span']} rows | Area: {cnt_gen['active_area_px']} px]", color='#F0F6FC', fontsize=10.5, pad=7)
    ax3.axis('off')

    # 4. Extracted Crown Contour
    ax4 = fig.add_subplot(gs[2, 0])
    cnt_display = np.zeros((128, 128), dtype=np.uint8)
    for y, x in cnt_gen['pts'].items():
        cnt_display[int(y), int(round(x))] = 255
    ax4.imshow(cnt_display, cmap='gray')
    ax4.axvline(64, color='#FFEB3B', linestyle='--', linewidth=1.2)
    ax4.set_title(f"4. Physical Crown Contour\n[Support Span: Y={cnt_gen['y_range'][0]}..{cnt_gen['y_range'][1]} | Continuous Boundary]", color='#F0F6FC', fontsize=10.5, pad=7)
    ax4.axis('off')

    # 5. Edge / Gradient Map
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.imshow(edge_map, cmap='inferno')
    ax5.set_title(f"5. Edge / Gradient Diagnostic Map\n[{edge_note[:42]}...]", color='#F0F6FC', fontsize=10.5, pad=7)
    ax5.axis('off')

    # 6. Final Contour Overlay
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.imshow(overlay)
    ax6.set_title(f"6. Crown Contour & Apex Overlay\n[Apex: ({cnt_gen['apex_x']:.1f}, {cnt_gen['apex_y']}) | Image Reinf: {cnt_gen['reinf_proj_px']:.1f} px | Physical: {geom_preds['predicted_reinforcement_mm']:.2f} mm]",
                  color='#3FB950', fontsize=10.5, fontweight='bold', pad=7)
    ax6.axis('off')

    # Notice Footer
    notice_str = (
        "SCIENTIFIC DISCLOSURE: Prospective zero-shot synthesis for unseen condition (F=5.2 m/min, S=26 cm/min). "
        "No ground-truth macrograph exists. Pixel metrics (MAE, Dice, IoU) are NOT fabricated. "
        "Physical dimensions (mm) are derived from Ridge regression priors; image geometry (px) is synthesized via direct CNN."
    )
    fig.text(0.5, 0.015, notice_str, color='#E6EDF3', fontsize=9.2,
             ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#21262D', edgecolor='#FF7B72', linewidth=1))

    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"Rendered: {out_path}")


def main():
    p_s04 = os.path.join(STAGE_DIR, "S04_validation_panel.png")
    p_s33 = os.path.join(STAGE_DIR, "S33_validation_panel.png")
    p_unseen = os.path.join(STAGE_DIR, "UNSEEN_INTERPOLATION_PANEL.png")

    print("Rendering S04 validation panel...")
    render_known_specimen_panel("S04", 4, p_s04)

    print("Rendering S33 validation panel...")
    render_known_specimen_panel("S33", 33, p_s33)

    print("Rendering Unseen Interpolation panel...")
    render_unseen_interpolation_panel(p_unseen)

if __name__ == "__main__":
    main()
