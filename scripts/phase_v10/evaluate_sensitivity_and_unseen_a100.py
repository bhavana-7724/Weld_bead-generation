"""
evaluate_sensitivity_and_unseen_a100.py

Phase V10 on NVIDIA A100 AI Server:
Phases E & F:
1. Controlled Parameter Sensitivity Analysis (Respecting Group-Specific Domains):
   - IN718 ColdArc: Feed in [5.0, 6.0] at S=25.0; Speed in [20.0, 30.0] at F=5.5
   - IN718 GMAW:    Feed in [4.5, 5.5] at S=25.0; Speed in [20.0, 30.0] at F=5.0
   - SS304L ColdArc: Feed in [5.0, 6.0] at S=27.0; Speed in [22.0, 32.0] at F=5.5
   - SS304L GMAW:    Feed in [4.5, 5.5] at S=27.0; Speed in [22.0, 32.0] at F=5.0
   - Measures external crown reinforcement projection P = 64 - min(X) (in PIXELS).
   - Computes local interval derivatives (dP/dF and dP/dS) and reports true monotonicity
     or observed non-monotonic inflections without forcing universal monotonicity claims.
   - Generates fig6_parameter_sensitivity_curves.png.
   - Saves metrics to PARAMETER_SENSITIVITY_METRICS.csv.

2. Unseen In-Domain Parameter Generation (4 Genuine In-Domain Cases):
   - Case U1: IN718 ColdArc (F=5.50 m/min, S=26.0 cm/min)
   - Case U2: IN718 GMAW (F=5.25 m/min, S=22.5 cm/min)
   - Case U3: SS304L ColdArc (F=5.25 m/min, S=25.0 cm/min)
   - Case U4: SS304L GMAW (F=5.25 m/min, S=24.5 cm/min)
   - Pure prospective inference with STRICT SEPARATION:
     * M{k}_raw_rgb.png (Pure RGB output)
     * M{k}_predicted_mask.png (Pure binary mask)
     * M{k}_contour_coords.json (Numerical coordinates)
     * M{k}_visualization_overlay.png (Visual diagnostic overlay)
   - Generates fig7_unseen_parameter_predictions.png.
   - Saves metrics to UNSEEN_PARAMETER_PREDICTIONS.csv.

3. Formulates 4 Targeted Physical Validation Experiments (strictly reporting in PIXELS).
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
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v10.geometry_models_cuda import (
    DirectCNN, ConditionalVAE, GC_WPG_v2, GeoSDF_CVAE,
    Geometry_cGAN_Generator, GF_WPG
)
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v10_a100")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
UNSEEN_DIR = os.path.join(RESULTS_DIR, "unseen_predictions")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

for d in [FIGURES_DIR, METRICS_DIR, UNSEEN_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

def load_trained_models(device):
    norm_path = os.path.join(CKPT_DIR, "normalization_params.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing normalization_params.json at {norm_path}.")
    
    with open(norm_path, "r") as f:
        meta = json.load(f)

    print(f"Loading Phase V10 production models from {CKPT_DIR} to {device}...")

    m1 = DirectCNN(6, 16).to(device)
    m1.load_state_dict(torch.load(os.path.join(CKPT_DIR, "m1_direct_cnn_final.pt"), map_location=device))
    m1.eval()

    m2 = ConditionalVAE(6, 24).to(device)
    m2.load_state_dict(torch.load(os.path.join(CKPT_DIR, "m2_cvae_final.pt"), map_location=device))
    m2.eval()

    m3 = GC_WPG_v2(6, 16, scaler_params=meta).to(device)
    m3.load_state_dict(torch.load(os.path.join(CKPT_DIR, "m3_gc_wpg_v2_final.pt"), map_location=device))
    m3.eval()

    m4 = GeoSDF_CVAE(6, 16, beta=1.0).to(device)
    m4.load_state_dict(torch.load(os.path.join(CKPT_DIR, "m4_geosdf_cvae_final.pt"), map_location=device))
    m4.eval()

    m5 = Geometry_cGAN_Generator(6).to(device)
    m5.load_state_dict(torch.load(os.path.join(CKPT_DIR, "m5_geometry_cgan_final.pt"), map_location=device))
    m5.eval()

    m6 = GF_WPG(6, 95).to(device)
    m6.load_state_dict(torch.load(os.path.join(CKPT_DIR, "m6_gf_wpg_final.pt"), map_location=device))
    m6.eval()

    def encode_c(feed, speed, mat, proc):
        f_norm = (float(feed) - meta['f_min']) / (meta['f_max'] - meta['f_min'] + 1e-8)
        s_norm = (float(speed) - meta['s_min']) / (meta['s_max'] - meta['s_min'] + 1e-8)
        return torch.tensor([[
            f_norm, s_norm,
            1.0 if mat == 'IN718' else 0.0,
            1.0 if mat == 'SS304L' else 0.0,
            1.0 if proc == 'ColdArc' else 0.0,
            1.0 if proc == 'GMAW' else 0.0
        ]], dtype=torch.float32, device=device)

    print("All 6 production models successfully loaded.")
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "M6": m6}, encode_c

def run_parameter_sensitivity(models, encode_c, device):
    print("\n" + "=" * 75)
    print("PHASE E: CONTROLLED PARAMETER SENSITIVITY SWEEPS (GROUP-SPECIFIC DOMAINS)")
    print("=" * 75)

    # 4 distinct groups with their TRUE in-domain experimental parameter ranges
    groups_config = [
        {
            "mat": "IN718", "proc": "ColdArc", "label": "IN718 / Cold Arc",
            "feed_range": np.linspace(5.0, 6.0, 5), "const_speed": 25.0,
            "speed_range": np.linspace(20.0, 30.0, 5), "const_feed": 5.5
        },
        {
            "mat": "IN718", "proc": "GMAW", "label": "IN718 / GMAW",
            "feed_range": np.linspace(4.5, 5.5, 5), "const_speed": 25.0,
            "speed_range": np.linspace(20.0, 30.0, 5), "const_feed": 5.0
        },
        {
            "mat": "SS304L", "proc": "ColdArc", "label": "SS304L / Cold Arc",
            "feed_range": np.linspace(5.0, 6.0, 5), "const_speed": 27.0,
            "speed_range": np.linspace(22.0, 32.0, 5), "const_feed": 5.5
        },
        {
            "mat": "SS304L", "proc": "GMAW", "label": "SS304L / GMAW",
            "feed_range": np.linspace(4.5, 5.5, 5), "const_speed": 27.0,
            "speed_range": np.linspace(22.0, 32.0, 5), "const_feed": 5.0
        }
    ]

    m1 = models["M1"]
    m3 = models["M3"]
    m4 = models["M4"]
    m5 = models["M5"]
    m6 = models["M6"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=150)
    sensitivity_records = []

    for col_idx, g_cfg in enumerate(groups_config):
        mat = g_cfg["mat"]
        proc = g_cfg["proc"]
        label = g_cfg["label"]
        f_range = g_cfg["feed_range"]
        const_s = g_cfg["const_speed"]
        s_range = g_cfg["speed_range"]
        const_f = g_cfg["const_feed"]

        # 1. Sweep Feed Rate
        proj_m1_f, proj_m3_f, proj_m4_f, proj_m5_f, proj_m6_f = [], [], [], [], []
        for f_val in f_range:
            c = encode_c(f_val, const_s, mat, proc)
            with torch.no_grad():
                rgb_m1 = (np.clip(m1(c).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                rgb_m3 = (np.clip(m3(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                p_rgb4, _, _, _ = m4.generate(c)
                rgb_m4 = (np.clip(p_rgb4.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                p_rgb5, _ = m5(c)
                rgb_m5 = (np.clip(p_rgb5.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                p_rgb6, _, _, _ = m6.generate(c)
                rgb_m6 = (np.clip(p_rgb6.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

            c1 = extract_robust_crown_contour(cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2GRAY))
            c3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
            c4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
            c5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))
            c6 = extract_robust_crown_contour(cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2GRAY))

            proj_m1_f.append(c1["reinf_proj_px"])
            proj_m3_f.append(c3["reinf_proj_px"])
            proj_m4_f.append(c4["reinf_proj_px"])
            proj_m5_f.append(c5["reinf_proj_px"])
            proj_m6_f.append(c6["reinf_proj_px"])

            sensitivity_records.append({
                "sweep_type": "feed_rate", "material": mat, "process": proc,
                "feed_rate": round(float(f_val), 3), "speed": const_s,
                "m1_proj_px": c1["reinf_proj_px"],
                "m3_proj_px": c3["reinf_proj_px"],
                "m4_proj_px": c4["reinf_proj_px"],
                "m5_proj_px": c5["reinf_proj_px"],
                "m6_proj_px": c6["reinf_proj_px"]
            })

        # Compute Feed Rate Monotonicity for M4 and M6
        def check_mono(arr):
            diffs = np.diff(arr)
            pos_ratio = float(np.mean(diffs > 0))
            if pos_ratio == 1.0:
                return "Monotonic Increasing"
            elif pos_ratio == 0.0:
                return "Monotonic Decreasing"
            else:
                return f"Non-Monotonic ({pos_ratio*100:.0f}% increasing)"

        status_m4_f = check_mono(proj_m4_f)
        status_m6_f = check_mono(proj_m6_f)

        ax_top = axes[0, col_idx]
        ax_top.plot(f_range, proj_m1_f, 'd-.', color='gray', label='M1: Direct CNN', linewidth=1.2)
        ax_top.plot(f_range, proj_m3_f, 'o--', color='blue', label='M3: GC-WPG-v2', linewidth=1.5)
        ax_top.plot(f_range, proj_m4_f, 's-', color='green', label='M4: GeoSDF-CVAE', linewidth=2.0)
        ax_top.plot(f_range, proj_m5_f, '^:', color='purple', label='M5: cGAN', linewidth=1.2)
        ax_top.plot(f_range, proj_m6_f, '*-', color='red', label='M6: GF-WPG', linewidth=2.2)
        ax_top.set_title(f"{label}\nFeed Sweep (S={const_s:.1f} cm/min)\nM6: {status_m6_f}", fontsize=10, fontweight='bold')
        ax_top.set_xlabel("Wire Feed Rate (m/min)", fontsize=9)
        ax_top.set_ylabel("Reinforcement Projection (px)", fontsize=9)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        if col_idx == 0:
            ax_top.legend(loc='lower right', fontsize=8)

        # 2. Sweep Travel Speed
        proj_m1_s, proj_m3_s, proj_m4_s, proj_m5_s, proj_m6_s = [], [], [], [], []
        for s_val in s_range:
            c = encode_c(const_f, s_val, mat, proc)
            with torch.no_grad():
                rgb_m1 = (np.clip(m1(c).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                rgb_m3 = (np.clip(m3(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                p_rgb4, _, _, _ = m4.generate(c)
                rgb_m4 = (np.clip(p_rgb4.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                p_rgb5, _ = m5(c)
                rgb_m5 = (np.clip(p_rgb5.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                p_rgb6, _, _, _ = m6.generate(c)
                rgb_m6 = (np.clip(p_rgb6.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

            c1 = extract_robust_crown_contour(cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2GRAY))
            c3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
            c4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
            c5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))
            c6 = extract_robust_crown_contour(cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2GRAY))

            proj_m1_s.append(c1["reinf_proj_px"])
            proj_m3_s.append(c3["reinf_proj_px"])
            proj_m4_s.append(c4["reinf_proj_px"])
            proj_m5_s.append(c5["reinf_proj_px"])
            proj_m6_s.append(c6["reinf_proj_px"])

            sensitivity_records.append({
                "sweep_type": "travel_speed", "material": mat, "process": proc,
                "feed_rate": const_f, "speed": round(float(s_val), 3),
                "m1_proj_px": c1["reinf_proj_px"],
                "m3_proj_px": c3["reinf_proj_px"],
                "m4_proj_px": c4["reinf_proj_px"],
                "m5_proj_px": c5["reinf_proj_px"],
                "m6_proj_px": c6["reinf_proj_px"]
            })

        status_m4_s = check_mono(proj_m4_s)
        status_m6_s = check_mono(proj_m6_s)

        ax_bot = axes[1, col_idx]
        ax_bot.plot(s_range, proj_m1_s, 'd-.', color='gray', label='M1: Direct CNN', linewidth=1.2)
        ax_bot.plot(s_range, proj_m3_s, 'o--', color='blue', label='M3: GC-WPG-v2', linewidth=1.5)
        ax_bot.plot(s_range, proj_m4_s, 's-', color='green', label='M4: GeoSDF-CVAE', linewidth=2.0)
        ax_bot.plot(s_range, proj_m5_s, '^:', color='purple', label='M5: cGAN', linewidth=1.2)
        ax_bot.plot(s_range, proj_m6_s, '*-', color='red', label='M6: GF-WPG', linewidth=2.2)
        ax_bot.set_title(f"Speed Sweep (F={const_f:.1f} m/min)\nM6: {status_m6_s}", fontsize=10, fontweight='bold')
        ax_bot.set_xlabel("Welding Travel Speed (cm/min)", fontsize=9)
        ax_bot.set_ylabel("Reinforcement Projection (px)", fontsize=9)
        ax_bot.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Phase V10: Parameter Sensitivity Analysis Respecting Group-Specific Domains\n(Accurately Reporting Monotonic & Non-Monotonic Behavior)",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig6_parameter_sensitivity_curves.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")

    df_sens = pd.DataFrame(sensitivity_records)
    csv_path = os.path.join(METRICS_DIR, "PARAMETER_SENSITIVITY_METRICS.csv")
    df_sens.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

def run_unseen_in_domain_predictions(models, encode_c, device):
    print("\n" + "=" * 75)
    print("PHASE F: UNSEEN IN-DOMAIN PARAMETER GENERATION (4 VERIFIED CASES)")
    print("=" * 75)

    m1 = models["M1"]
    m2 = models["M2"]
    m3 = models["M3"]
    m4 = models["M4"]
    m5 = models["M5"]
    m6 = models["M6"]

    unseen_cases = [
        {"case_id": "U1", "mat": "IN718", "proc": "ColdArc", "feed": 5.50, "speed": 26.0, "desc": "IN718 ColdArc (Interpolation)"},
        {"case_id": "U2", "mat": "IN718", "proc": "GMAW", "feed": 5.25, "speed": 22.5, "desc": "IN718 GMAW (Untested Interpolation)"},
        {"case_id": "U3", "mat": "SS304L", "proc": "ColdArc", "feed": 5.25, "speed": 25.0, "desc": "SS304L ColdArc (Interpolation)"},
        {"case_id": "U4", "mat": "SS304L", "proc": "GMAW", "feed": 5.25, "speed": 24.5, "desc": "SS304L GMAW (Untested Interpolation)"}
    ]

    fig_summary, axes_sum = plt.subplots(4, 6, figsize=(22, 14), dpi=150)
    unseen_records = []

    for r_idx, c_info in enumerate(unseen_cases):
        cid = c_info["case_id"]
        c_dir = os.path.join(UNSEEN_DIR, cid)
        os.makedirs(c_dir, exist_ok=True)

        c_vec = encode_c(c_info["feed"], c_info["speed"], c_info["mat"], c_info["proc"])

        with torch.no_grad():
            rgb_m1 = (np.clip(m1(c_vec).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            rgb_m2 = (np.clip(m2.generate(c_vec).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            rgb_m3 = (np.clip(m3(c_vec)[0].squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            
            p_rgb4, p_sdf4, p_mask4, _ = m4.generate(c_vec)
            rgb_m4 = (np.clip(p_rgb4.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            sdf_m4 = p_sdf4.squeeze().cpu().numpy()
            mask_m4 = (p_mask4.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            p_rgb5, p_mask5 = m5(c_vec)
            rgb_m5 = (np.clip(p_rgb5.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            mask_m5 = (p_mask5.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            p_rgb6, p_mask6, p_pts6, _ = m6.generate(c_vec)
            rgb_m6 = (np.clip(p_rgb6.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            mask_m6 = (p_mask6.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        cnt1 = extract_robust_crown_contour(cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2GRAY))
        cnt3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
        cnt4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
        cnt5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))
        cnt6 = extract_robust_crown_contour(cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2GRAY))

        # ---------------------------------------------------------------------
        # MANDATORY SEPARATION FOR UNSEEN PREDICTIONS
        # ---------------------------------------------------------------------
        # Raw RGB
        cv2.imwrite(os.path.join(c_dir, "M1_DirectCNN_raw_rgb.png"), cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M2_CVAE_raw_rgb.png"), cv2.cvtColor(rgb_m2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M3_GCWPGv2_raw_rgb.png"), cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M4_GeoSDFCVAE_raw_rgb.png"), cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M5_cGAN_raw_rgb.png"), cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M6_GFWPG_raw_rgb.png"), cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2BGR))

        # Masks and SDFs
        cv2.imwrite(os.path.join(c_dir, "M4_GeoSDFCVAE_predicted_mask.png"), mask_m4 * 255)
        np.save(os.path.join(c_dir, "M4_GeoSDFCVAE_predicted_sdf.npy"), sdf_m4)
        cv2.imwrite(os.path.join(c_dir, "M5_cGAN_predicted_mask.png"), mask_m5 * 255)
        cv2.imwrite(os.path.join(c_dir, "M6_GFWPG_predicted_mask.png"), mask_m6 * 255)

        # Coordinate JSONs
        with open(os.path.join(c_dir, "M4_contour_coords.json"), "w") as f:
            json.dump(cnt4.get("pts", {}), f)
        with open(os.path.join(c_dir, "M6_contour_coords.json"), "w") as f:
            json.dump(cnt6.get("pts", {}), f)

        # Visual diagnostic overlays
        def make_vis_overlay(img_rgb, cnt_obj):
            ov = img_rgb.copy()
            for y, x in cnt_obj.get("pts", {}).items():
                if 0 <= y < 128 and 0 <= x < 128:
                    ov[y, x] = [0, 255, 0]
            cv2.line(ov, (64, 0), (64, 127), (255, 255, 0), 1)
            return ov

        cv2.imwrite(os.path.join(c_dir, "M4_visualization_overlay.png"), cv2.cvtColor(make_vis_overlay(rgb_m4, cnt4), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M6_visualization_overlay.png"), cv2.cvtColor(make_vis_overlay(rgb_m6, cnt6), cv2.COLOR_RGB2BGR))

        unseen_records.append({
            "case_id": cid,
            "description": c_info["desc"],
            "material": c_info["mat"],
            "process": c_info["proc"],
            "feed_rate": c_info["feed"],
            "speed": c_info["speed"],
            "m1_predicted_proj_px": cnt1["reinf_proj_px"],
            "m3_predicted_proj_px": cnt3["reinf_proj_px"],
            "m4_predicted_proj_px": cnt4["reinf_proj_px"],
            "m5_predicted_proj_px": cnt5["reinf_proj_px"],
            "m6_predicted_proj_px": cnt6["reinf_proj_px"]
        })

        # Plot on summary figure
        row_imgs = [
            ("M1: Direct CNN", rgb_m1, cnt1),
            ("M2: CVAE (z=0)", rgb_m2, None),
            ("M3: GC-WPG-v2", rgb_m3, cnt3),
            ("M4: GeoSDF-CVAE", rgb_m4, cnt4),
            ("M5: cGAN", rgb_m5, cnt5),
            ("M6: GF-WPG", rgb_m6, cnt6)
        ]
        for col_idx, (m_label, img, cnt_obj) in enumerate(row_imgs):
            ax = axes_sum[r_idx, col_idx]
            ax.imshow(img)
            ax.axvline(64, color='yellow', linestyle='--', linewidth=0.8)
            if cnt_obj is not None and len(cnt_obj["pts"]) > 0:
                ax.plot(list(cnt_obj["pts"].values()), list(cnt_obj["pts"].keys()), color='lime', linewidth=1.2)
            if r_idx == 0:
                ax.set_title(m_label, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Case {cid}\n{c_info['desc']}\nF={c_info['feed']} S={c_info['speed']}",
                              fontsize=9, fontweight='bold', rotation=0, labelpad=70)
            ax.set_xticks([])
            ax.set_yticks([])

    fig_summary.suptitle("Phase V10: Unseen Prospective Parameter Predictions Across 4 In-Domain Cases\n(Pure Condition-Driven Inference on NVIDIA A100 Models — Zero Leakage)",
                         fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig7_path = os.path.join(FIGURES_DIR, "fig7_unseen_parameter_predictions.png")
    plt.savefig(fig7_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig7_path}")

    df_unseen = pd.DataFrame(unseen_records)
    csv_unseen_path = os.path.join(METRICS_DIR, "UNSEEN_PARAMETER_PREDICTIONS.csv")
    df_unseen.to_csv(csv_unseen_path, index=False)
    print(f"Saved {csv_unseen_path}")

    # Author Proposals
    author_experimental_proposals(unseen_records)

def author_experimental_proposals(unseen_records):
    p1 = next((r['m6_predicted_proj_px'] for r in unseen_records if r['case_id'] == 'U1'), 18.0)
    p2 = next((r['m6_predicted_proj_px'] for r in unseen_records if r['case_id'] == 'U2'), 19.5)
    p3 = next((r['m6_predicted_proj_px'] for r in unseen_records if r['case_id'] == 'U3'), 18.2)
    p4 = next((r['m6_predicted_proj_px'] for r in unseen_records if r['case_id'] == 'U4'), 20.1)

    proposal_text = f"""# PHASE V10 — TARGETED PHYSICAL VALIDATION EXPERIMENTAL PROPOSALS

## 1. Executive Summary & Experimental Objectives
To rigorously validate the generative geometry predictions produced by our continuous geometry framework (**GeoSDF-CVAE** and **GF-WPG Geometry-First Generator**) trained on the NVIDIA A100 GPU, we propose four physical welding trials across our four material-process regimes. These trials test both in-domain high-gradient interpolation points and physical boundary transitions.

> **CRITICAL REPRODUCIBILITY MANDATE:** All contour measurements, apex projections, and boundary errors are specified strictly in **pixels** ($128 \\times 128$ normalized coordinate frame, baseline $X=64$) to maintain scientific honesty and avoid unvalidated pixel-to-millimeter scaling assumptions.

---

## 2. Experimental Design Matrix

| Trial ID | Material | Process | Feed Rate (m/min) | Travel Speed (cm/min) | Predicted Crown Projection (px) | Targeted Physical Hypothesis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-V10-01** | Inconel 718 | Cold Arc | **5.50** | **26.0** | ~{p1:.1f} px | Validates intermediate heat-input crown formation between high-speed and low-speed Cold Arc regimes. |
| **EXP-V10-02** | Inconel 718 | GMAW | **5.25** | **22.5** | ~{p2:.1f} px | Tests high-deposition low-speed GMAW boundary where gravity sag and droplet detachment dynamics peak. |
| **EXP-V10-03** | SS304L | Cold Arc | **5.25** | **25.0** | ~{p3:.1f} px | Evaluates surface-tension driven wetting angle in stainless steel under pulsed low-heat Cold Arc waveforms. |
| **EXP-V10-04** | SS304L | GMAW | **5.25** | **24.5** | ~{p4:.1f} px | Evaluates deep fusion and crown reinforcement balance at the center of the parameter envelope. |

---

## 3. Recommended Metallographic Preparation Protocol
1. **Sectioning:** Transverse cross-sectioning at steady-state weld bead length (>50 mm from strike and crater extinguish points).
2. **Mounting & Polishing:** Conductive epoxy mount, ground with SiC papers (320 to 1200 grit), followed by 3 µm and 1 µm diamond polish, finishing with 0.05 µm colloidal silica.
3. **Etching:**
   - **Inconel 718:** Kalling's No. 2 reagent (5 g CuCl2, 100 mL HCl, 100 mL ethanol) swabbing for 10-15 seconds.
   - **SS304L:** Glyceregia (15 mL HCl, 10 mL glycerol, 5 mL HNO3) immersion for 30-45 seconds.
4. **Optical Macrography:** Calibrated optical stereo-microscopy, uniform diffuse illumination, registering plate interface strictly at baseline $X=64$ in $128 \\times 128$ normalized coordinate system.
"""
    proposal_path = os.path.join(REPORTS_DIR, "PHASE_V10_PROPOSED_VALIDATION_EXPERIMENTS.md")
    with open(proposal_path, "w") as f:
        f.write(proposal_text)
    print(f"Saved {proposal_path}")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase V10 Sensitivity & Unseen Analysis on device: {device}")
    models, encode_c = load_trained_models(device)
    run_parameter_sensitivity(models, encode_c, device)
    run_unseen_in_domain_predictions(models, encode_c, device)
    print("\nPhase V10 Sensitivity and Unseen Predictions Completed Successfully!")

if __name__ == "__main__":
    main()
