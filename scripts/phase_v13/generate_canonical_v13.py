"""
generate_canonical_v13.py

Phase V13 Stage 3: Canonical 36-Sample Generation & Academic Panel Production
Target System: Single NVIDIA A100 GPU (CUDA_VISIBLE_DEVICES=0) / Fallback CPU
Model: Enhanced Geometry-First Weld Profile Generator v2 (GF-WPG v2)

Generates:
1. Individual sample outputs for all 36 physical conditions S01 to S36:
   - GFWPGv2_raw.png (Raw un-retouched neural RGB)
   - GFWPGv2_mask.png (Differentiably rasterized binary geometry mask)
   - GFWPGv2_contour.json (Predicted contour coordinates, apex, toes)
   - Real_A.png and Real_B.png (Experimental ground truth references)
2. Professional Academic Validation Panels (S01_academic_panel.png to S36_academic_panel.png)
   - Clean white background, publication typography, no overlapping text.
   - Real View A | Real View B | Raw GF-WPG v2 | Mask & Contour | Contour Overlay | Metrics Card
3. Master Evaluation Table:
   - results/phase_v13_a100/metrics/GFWPGv2_ALL_36_METRICS.csv
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v13.models_v13 import GF_WPG_v2
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")

AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
GEN36_FALLBACK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v13_a100")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
GEN36_DIR = os.path.join(RESULTS_DIR, "generated_36")
PANELS_DIR = os.path.join(RESULTS_DIR, "panels")

for d in [GEN36_DIR, PANELS_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)


def compute_metrics_pair(real_rgb, gen_rgb, real_mask, gen_mask, real_cnt, pred_pts_dict, pred_apex, pred_toes):
    ssim = float(ssim_func(real_rgb, gen_rgb, channel_axis=2, data_range=255))
    psnr = float(psnr_func(real_rgb, gen_rgb, data_range=255))

    roi_real = real_rgb[:, :64, :].copy()
    roi_gen = gen_rgb[:, :64, :].copy()
    roi_ssim = float(ssim_func(roi_real, roi_gen, channel_axis=2, data_range=255))

    r_pts = real_cnt.get("pts", {})
    y_common = sorted(list(set(r_pts.keys()).intersection(set(pred_pts_dict.keys()))))
    if len(y_common) >= 5:
        diffs = [abs(r_pts[y] - pred_pts_dict[y]) for y in y_common]
        contour_mae_px = float(np.mean(diffs))
    else:
        contour_mae_px = 35.0

    real_apex_x = real_cnt.get("apex_x", 64.0)
    real_apex_y = real_cnt.get("apex_y", 64.0)
    apex_error_px = float(np.sqrt((real_apex_x - pred_apex[0]) ** 2 + (real_apex_y - pred_apex[1]) ** 2))

    if len(r_pts) > 0:
        real_toe_top = min(r_pts.keys())
        real_toe_bot = max(r_pts.keys())
    else:
        real_toe_top = 38.0
        real_toe_bot = 98.0
    toe_top_err_px = float(abs(real_toe_top - pred_toes[0]))
    toe_bot_err_px = float(abs(real_toe_bot - pred_toes[1]))
    mean_toe_err_px = float((toe_top_err_px + toe_bot_err_px) / 2.0)

    inter = np.sum((real_mask > 0) & (gen_mask > 0))
    union = np.sum((real_mask > 0) | (gen_mask > 0))
    iou = float(inter / (union + 1e-8))
    dice = float(2.0 * inter / (np.sum(real_mask > 0) + np.sum(gen_mask > 0) + 1e-8))

    real_edge = np.zeros((128, 128), dtype=np.uint8)
    for y, x in r_pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            real_edge[y, int(round(x))] = 1

    gen_edge = np.zeros((128, 128), dtype=np.uint8)
    for y, x in pred_pts_dict.items():
        if 0 <= y < 128 and 0 <= x < 128:
            gen_edge[y, int(round(x))] = 1

    kernel_2px = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_real_2px = cv2.dilate(real_edge, kernel_2px)
    dilated_gen_2px = cv2.dilate(gen_edge, kernel_2px)

    gen_pts_count = np.sum(gen_edge)
    real_pts_count = np.sum(real_edge)

    precision = np.sum((gen_edge == 1) & (dilated_real_2px == 1)) / (gen_pts_count + 1e-8)
    recall = np.sum((real_edge == 1) & (dilated_gen_2px == 1)) / (real_pts_count + 1e-8)
    f1_2px = float(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "ssim": ssim,
        "psnr": psnr,
        "roi_ssim": roi_ssim,
        "contour_mae_px": contour_mae_px,
        "apex_error_px": apex_error_px,
        "toe_top_err_px": toe_top_err_px,
        "toe_bot_err_px": toe_bot_err_px,
        "mean_toe_err_px": mean_toe_err_px,
        "mask_iou": iou,
        "mask_dice": dice,
        "boundary_f1": f1_2px
    }


def render_academic_panel(sample_id, row, real_a_rgb, real_b_rgb, gen_rgb, gen_mask,
                          cnt_a, cnt_b, pred_pts_dict, pred_apex, pred_toes,
                          m_a, m_b, m_avg, out_panel_path):
    """
    Renders an individual high-resolution 1x6 academic validation panel.
    """
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4), dpi=150)
    fig.patch.set_facecolor('white')

    # Substrate baseline line
    def draw_baseline(ax):
        ax.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2, alpha=0.9)

    # 1. Real View A
    axes[0].imshow(real_a_rgb)
    draw_baseline(axes[0])
    axes[0].set_title("Experimental Macrograph A\n(Real Microstructure)", fontsize=10, fontweight='bold', color='#1E293B')
    axes[0].axis('off')

    # 2. Real View B
    axes[1].imshow(real_b_rgb)
    draw_baseline(axes[1])
    axes[1].set_title("Experimental Macrograph B\n(Replicate Repeat)", fontsize=10, fontweight='bold', color='#1E293B')
    axes[1].axis('off')

    # 3. Raw Generated RGB
    axes[2].imshow(gen_rgb)
    draw_baseline(axes[2])
    axes[2].set_title("Raw Generated Cross-Section\n(GF-WPG v2 Neural)", fontsize=10, fontweight='bold', color='#1E293B')
    axes[2].axis('off')

    # 4. Predicted Binary Mask & Extracted Contour
    mask_display = np.zeros((128, 128, 3), dtype=np.uint8)
    mask_display[gen_mask > 0] = [37, 99, 235] # Royal blue
    # Draw contour
    for y, x in pred_pts_dict.items():
        if 0 <= y < 128 and 0 <= x < 128:
            mask_display[y, int(round(x))] = [239, 68, 68] # Red contour
    axes[3].imshow(mask_display)
    draw_baseline(axes[3])
    axes[3].set_title("Predicted Geometry Mask\n(Differentiable Rasterizer)", fontsize=10, fontweight='bold', color='#1E293B')
    axes[3].axis('off')

    # 5. High-Resolution Contour Overlay Comparison
    overlay_img = np.ones((128, 128, 3), dtype=np.uint8) * 248
    axes[4].imshow(overlay_img)
    draw_baseline(axes[4])

    # Plot Real A contour
    pts_a = cnt_a.get("pts", {})
    if len(pts_a) > 0:
        ys_a = sorted(pts_a.keys())
        xs_a = [pts_a[y] for y in ys_a]
        axes[4].plot(xs_a, ys_a, color='#2563EB', linestyle=':', linewidth=1.5, label='Real View A')

    # Plot Real B contour
    pts_b = cnt_b.get("pts", {})
    if len(pts_b) > 0:
        ys_b = sorted(pts_b.keys())
        xs_b = [pts_b[y] for y in ys_b]
        axes[4].plot(xs_b, ys_b, color='#059669', linestyle=':', linewidth=1.5, label='Real View B')

    # Plot GF-WPG v2 predicted contour
    if len(pred_pts_dict) > 0:
        ys_p = sorted(pred_pts_dict.keys())
        xs_p = [pred_pts_dict[y] for y in ys_p]
        axes[4].plot(xs_p, ys_p, color='#DC2626', linestyle='-', linewidth=2.0, label='GF-WPG v2 (Ours)')

    # Plot Apex
    axes[4].scatter([pred_apex[0]], [pred_apex[1]], color='#7C3AED', s=35, zorder=5, marker='x', label='Pred Apex')
    axes[4].set_xlim(0, 128)
    axes[4].set_ylim(128, 0)
    axes[4].set_title("Contour Overlay Comparison\n(Real A/B vs Predicted)", fontsize=10, fontweight='bold', color='#1E293B')
    axes[4].axis('off')
    axes[4].legend(loc='lower left', fontsize=7, framealpha=0.85)

    # 6. Parameter & Verification Metrics Card
    axes[5].axis('off')
    card_text = (
        f"Sample: S{sample_id:02d} | {row['material']} | {row['process']}\n"
        f"Feed Rate: {row['feed_rate']:.1f} m/min | Speed: {row['speed']:.1f} cm/min\n"
        f"----------------------------------------\n"
        f"Geometric Accuracy (Dual-View Avg):\n"
        f"  Contour MAE:       {m_avg['contour_mae_px']:.2f} px\n"
        f"  Boundary F1 (@2px): {m_avg['boundary_f1']:.3f}\n"
        f"  Mask Dice:          {m_avg['mask_dice']:.3f}\n"
        f"  Mask IoU:           {m_avg['mask_iou']:.3f}\n"
        f"  Apex Error:         {m_avg['apex_error_px']:.2f} px\n"
        f"  Mean Toe Error:     {m_avg['mean_toe_err_px']:.2f} px\n"
        f"----------------------------------------\n"
        f"Photometric Quality:\n"
        f"  SSIM (Full/ROI):    {m_avg['ssim']:.3f} / {m_avg['roi_ssim']:.3f}\n"
        f"  PSNR:               {m_avg['psnr']:.1f} dB"
    )
    axes[5].text(
        0.05, 0.5, card_text,
        fontsize=9, family='monospace', verticalalignment='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.2)
    )

    plt.suptitle(
        f"Phase V13 Academic Validation: Specimen S{sample_id:02d} ({row['material']}, {row['process']}, F={row['feed_rate']}, S={row['speed']})",
        fontsize=13, fontweight='bold', y=0.98, color='#0F172A'
    )
    plt.tight_layout()
    plt.savefig(out_panel_path, bbox_inches='tight', facecolor='white')
    plt.close()


def run_canonical_generation(smoke_test=False, target_device=None):
    t_start = time.time()
    if target_device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(target_device)

    print("=" * 80)
    print("PHASE V13 [STAGE 3]: CANONICAL 36-SAMPLE GENERATION & ACADEMIC PANELS")
    print(f"Device:          {device}")
    print(f"Smoke Test Mode: {smoke_test}")
    print("=" * 80)

    # Load Normalization params
    norm_path = os.path.join(CKPT_DIR, "normalization_params.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing normalization parameters at {norm_path}. Run Stage 2 first.")
    with open(norm_path, "r") as f:
        norm_params = json.load(f)

    f_min, f_max = norm_params["f_min"], norm_params["f_max"]
    s_min, s_max = norm_params["s_min"], norm_params["s_max"]

    def encode_condition(row):
        f_norm = (float(row['feed_rate']) - f_min) / (f_max - f_min + 1e-8)
        s_norm = (float(row['speed']) - s_min) / (s_max - s_min + 1e-8)
        return [
            f_norm, s_norm,
            1.0 if row['material'] == 'IN718' else 0.0,
            1.0 if row['material'] == 'SS304L' else 0.0,
            1.0 if row['process'] == 'ColdArc' else 0.0,
            1.0 if row['process'] == 'GMAW' else 0.0
        ]

    # Load Model
    ckpt_path = os.path.join(CKPT_DIR, "gfwpg_v2_final.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Production checkpoint not found at {ckpt_path}. Run Stage 2 first.")
    print(f"Loading production checkpoint from: {ckpt_path}")
    model = GF_WPG_v2(cond_dim=6, num_contour_pts=95).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    df_clean = pd.read_csv(CLEAN_CSV)
    num_samples = 3 if smoke_test else len(df_clean)
    all_36_records = []

    print(f"\nSynthesizing canonical outputs and academic panels for {num_samples} samples...")
    for idx in range(num_samples):
        row = df_clean.iloc[idx]
        s_id = int(row['sample'])
        s_code = f"S{s_id:02d}"
        s_dir = os.path.join(GEN36_DIR, s_code)
        os.makedirs(s_dir, exist_ok=True)

        c_vec = encode_condition(row)
        c_t = torch.tensor([c_vec], dtype=torch.float32, device=device)

        with torch.no_grad():
            p_rgb, p_mask, p_pts, p_apex, p_toes = model.generate(c_t)

        gen_rgb = (np.clip(p_rgb.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
        gen_mask = (p_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        pred_pts_np = p_pts.squeeze().cpu().numpy()
        pred_apex_np = p_apex.squeeze().cpu().numpy()
        pred_toes_np = p_toes.squeeze().cpu().numpy()

        pred_pts_dict = {18 + i: float(pred_pts_np[i]) for i in range(95)}

        # Save individual raw artifacts
        raw_rgb_path = os.path.join(s_dir, "GFWPGv2_raw.png")
        cv2.imwrite(raw_rgb_path, cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2BGR))

        mask_path = os.path.join(s_dir, "GFWPGv2_mask.png")
        cv2.imwrite(mask_path, (gen_mask * 255).astype(np.uint8))

        contour_json_path = os.path.join(s_dir, "GFWPGv2_contour.json")
        contour_meta = {
            "sample": s_id,
            "sample_code": s_code,
            "apex_x": float(pred_apex_np[0]),
            "apex_y": float(pred_apex_np[1]),
            "reinf_proj_px": float(64.0 - pred_apex_np[0]),
            "toe_top_y": float(pred_toes_np[0]),
            "toe_bot_y": float(pred_toes_np[1]),
            "height_span_px": float(pred_toes_np[1] - pred_toes_np[0]),
            "pts": {str(k): float(v) for k, v in pred_pts_dict.items()}
        }
        with open(contour_json_path, "w") as f:
            json.dump(contour_meta, f, indent=2)

        # Load real views A and B
        p_real_a = os.path.join(PROCESSED_DIR, f"{s_id}-A.png")
        if not os.path.exists(p_real_a):
            p_real_a = os.path.join(GEN36_FALLBACK_DIR, s_code, "Real_A.png")
        real_a_rgb = cv2.cvtColor(cv2.imread(p_real_a), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(s_dir, "Real_A.png"), cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2BGR))

        p_real_b = os.path.join(PROCESSED_DIR, f"{s_id}-B.png")
        if not os.path.exists(p_real_b):
            p_real_b = os.path.join(GEN36_FALLBACK_DIR, s_code, "Real_B.png")
        real_b_rgb = cv2.cvtColor(cv2.imread(p_real_b), cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(s_dir, "Real_B.png"), cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2BGR))

        # Ground truth contours
        cnt_a = extract_robust_crown_contour(cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2GRAY), x_baseline=64)
        cnt_b = extract_robust_crown_contour(cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2GRAY), x_baseline=64)

        mask_a_path = os.path.join(MASKS_DIR, f"{s_id}-A.png")
        mask_a = (cv2.imread(mask_a_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8) if os.path.exists(mask_a_path) else cnt_a["mask"]
        mask_b_path = os.path.join(MASKS_DIR, f"{s_id}-B.png")
        mask_b = (cv2.imread(mask_b_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8) if os.path.exists(mask_b_path) else cnt_b["mask"]

        # Compute metrics vs View A and View B
        m_a = compute_metrics_pair(real_a_rgb, gen_rgb, mask_a, gen_mask, cnt_a, pred_pts_dict, pred_apex_np, pred_toes_np)
        m_b = compute_metrics_pair(real_b_rgb, gen_rgb, mask_b, gen_mask, cnt_b, pred_pts_dict, pred_apex_np, pred_toes_np)

        m_avg = {}
        for k in m_a:
            m_avg[k] = float((m_a[k] + m_b[k]) / 2.0)

        # Render Academic Panel
        panel_path = os.path.join(PANELS_DIR, f"{s_code}_academic_panel.png")
        render_academic_panel(
            s_id, row, real_a_rgb, real_b_rgb, gen_rgb, gen_mask,
            cnt_a, cnt_b, pred_pts_dict, pred_apex_np, pred_toes_np,
            m_a, m_b, m_avg, panel_path
        )

        # Also copy panel into sample folder
        cv2.imwrite(os.path.join(s_dir, f"{s_code}_academic_panel.png"), cv2.imread(panel_path))

        rec = {
            "sample": s_id,
            "sample_code": s_code,
            "material": row['material'],
            "process": row['process'],
            "feed_rate": row['feed_rate'],
            "speed": row['speed'],
            "exp_reinf_mm": row['reinforcement'],
            "exp_width_mm": row['width'],
            "exp_depth_mm": row['depth'],
            "pred_apex_x": float(pred_apex_np[0]),
            "pred_apex_y": float(pred_apex_np[1]),
            "pred_reinf_px": float(64.0 - pred_apex_np[0]),
            "contour_mae_px": m_avg["contour_mae_px"],
            "boundary_f1": m_avg["boundary_f1"],
            "mask_dice": m_avg["mask_dice"],
            "mask_iou": m_avg["mask_iou"],
            "apex_error_px": m_avg["apex_error_px"],
            "mean_toe_err_px": m_avg["mean_toe_err_px"],
            "ssim": m_avg["ssim"],
            "psnr": m_avg["psnr"],
            "roi_ssim": m_avg["roi_ssim"],
            "mae_view_a": m_a["contour_mae_px"],
            "mae_view_b": m_b["contour_mae_px"]
        }
        all_36_records.append(rec)
        print(f"  [{idx+1:02d}/{num_samples:02d}] {s_code} ({row['material']}, {row['process']}) | MAE: {m_avg['contour_mae_px']:.2f} px | F1: {m_avg['boundary_f1']:.3f} | Dice: {m_avg['mask_dice']:.3f}")

    df_metrics = pd.DataFrame(all_36_records)
    metrics_csv_path = os.path.join(METRICS_DIR, "GFWPGv2_ALL_36_METRICS.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\nAll-36 canonical metrics saved to: {metrics_csv_path}")

    # Console Summary
    print("\n" + "=" * 80)
    print("PHASE V13: GF-WPG v2 CANONICAL RECONSTRUCTION BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Mean Contour MAE:     {df_metrics['contour_mae_px'].mean():.2f} px (std: {df_metrics['contour_mae_px'].std():.2f})")
    print(f"Mean Boundary F1:     {df_metrics['boundary_f1'].mean():.3f}")
    print(f"Mean Mask Dice:       {df_metrics['mask_dice'].mean():.3f}")
    print(f"Mean Mask IoU:        {df_metrics['mask_iou'].mean():.3f}")
    print(f"Mean Apex Error:      {df_metrics['apex_error_px'].mean():.2f} px")
    print(f"Mean Toe Error:       {df_metrics['mean_toe_err_px'].mean():.2f} px")
    print(f"Mean Full SSIM:       {df_metrics['ssim'].mean():.3f}")
    print(f"Best Sample:          {df_metrics.loc[df_metrics['contour_mae_px'].idxmin(), 'sample_code']} ({df_metrics['contour_mae_px'].min():.2f} px)")
    print(f"Worst Sample:         {df_metrics.loc[df_metrics['contour_mae_px'].idxmax(), 'sample_code']} ({df_metrics['contour_mae_px'].max():.2f} px)")
    print("=" * 80)

    t_total = time.time() - t_start
    print(f"Stage 3 canonical generation finished in {t_total:.1f} s ({t_total/60:.2f} min)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase V13 Canonical 36-Sample Generation")
    parser.add_argument("--smoke-test", action="store_true", help="Generate first 3 samples only")
    parser.add_argument("--device", type=str, default=None, help="Target device (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    sys.exit(run_canonical_generation(
        smoke_test=args.smoke_test,
        target_device=args.device
    ))
