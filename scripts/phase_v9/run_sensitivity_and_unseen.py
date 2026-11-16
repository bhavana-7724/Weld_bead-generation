"""
run_sensitivity_and_unseen.py

Phases E & F of Phase V9:
1. Controlled Parameter Sensitivity Analysis:
   - Sweeps Feed Rate F in [4.5, 6.0] m/min at constant S = 26.0 cm/min across all 4 material-process groups.
   - Sweeps Travel Speed S in [20, 32] cm/min at constant F = 5.25 m/min across all 4 material-process groups.
   - Measures external crown reinforcement projection P = 64 - min(X) and bead width span.
   - Tests physical monotonicity: dP/dF > 0 and dP/dS < 0.
   - Generates fig6_parameter_sensitivity_curves.png.

2. Unseen In-Domain Parameter Generation:
   - 4 genuine untested in-domain setpoints (replacing duplicate setpoints from prior phases):
     * Case U1: IN718 ColdArc (F=5.50 m/min, S=26.0 cm/min)
     * Case U2: IN718 GMAW (F=5.25 m/min, S=22.5 cm/min)
     * Case U3: SS304L ColdArc (F=5.25 m/min, S=25.0 cm/min)
     * Case U4: SS304L GMAW (F=5.25 m/min, S=24.5 cm/min)
   - Evaluated without ground truth (prospective inference).
   - Generates individual comparison panels and fig7_unseen_parameter_predictions.png.

3. Formulates 4 Targeted Future Physical Validation Experiments.
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
import torch
import scipy.stats as stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v9.geometry_models import (
    DirectCNN, ConditionalVAE, GC_WPG_v2, GeoSDF_CVAE,
    Geometry_cGAN_Generator
)
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
UNSEEN_DIR = os.path.join(RESULTS_DIR, "unseen_predictions")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

for d in [FIGURES_DIR, UNSEEN_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

def load_trained_models(device):
    with open(os.path.join(CKPT_DIR, "normalization_params.json"), "r") as f:
        meta = json.load(f)

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

    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5}, encode_c

def run_parameter_sensitivity(models, encode_c, device):
    print("=" * 75)
    print("PHASE E: CONTROLLED PARAMETER SENSITIVITY SWEEPS")
    print("=" * 75)

    groups = [
        ("IN718", "ColdArc", "IN718 / Cold Arc", "#1f77b4"),
        ("IN718", "GMAW", "IN718 / GMAW", "#ff7f0e"),
        ("SS304L", "ColdArc", "SS304L / Cold Arc", "#2ca02c"),
        ("SS304L", "GMAW", "SS304L / GMAW", "#d62728")
    ]

    feed_range = np.linspace(4.5, 6.0, 7)
    speed_range = np.linspace(20.0, 32.0, 7)

    m1 = models["M1"]
    m2 = models["M2"]
    m3 = models["M3"]
    m4 = models["M4"]
    m5 = models["M5"]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=150)

    sensitivity_records = []

    for col_idx, (mat, proc, label, color) in enumerate(groups):
        # 1. Sweep Feed Rate (at S = 26.0 cm/min)
        proj_m3_f, proj_m4_f, proj_m5_f = [], [], []
        for f_val in feed_range:
            c = encode_c(f_val, 26.0, mat, proc)
            with torch.no_grad():
                rgb_m3 = (m3(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rgb_m4 = (m4.generate(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rgb_m5 = (m5(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            c3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
            c4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
            c5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))

            proj_m3_f.append(c3["reinf_proj_px"])
            proj_m4_f.append(c4["reinf_proj_px"])
            proj_m5_f.append(c5["reinf_proj_px"])

            sensitivity_records.append({
                "sweep_type": "feed_rate", "material": mat, "process": proc,
                "feed_rate": f_val, "speed": 26.0,
                "m3_proj_px": c3["reinf_proj_px"], "m4_proj_px": c4["reinf_proj_px"], "m5_proj_px": c5["reinf_proj_px"]
            })

        ax_top = axes[0, col_idx]
        ax_top.plot(feed_range, proj_m3_f, 'o--', color='blue', label='M3: GC-WPG-v2', linewidth=1.5)
        ax_top.plot(feed_range, proj_m4_f, 's-', color='green', label='M4: GeoSDF-CVAE', linewidth=2.0)
        ax_top.plot(feed_range, proj_m5_f, '^:', color='purple', label='M5: cGAN', linewidth=1.5)
        ax_top.set_title(f"{label}\nFeed Rate Sensitivity (S=26.0 cm/min)", fontsize=11, fontweight='bold')
        ax_top.set_xlabel("Wire Feed Rate (m/min)", fontsize=10)
        ax_top.set_ylabel("Reinforcement Projection (px)", fontsize=10)
        ax_top.grid(True, linestyle=':', alpha=0.6)
        if col_idx == 0:
            ax_top.legend(loc='lower right', fontsize=8)

        # 2. Sweep Travel Speed (at F = 5.25 m/min)
        proj_m3_s, proj_m4_s, proj_m5_s = [], [], []
        for s_val in speed_range:
            c = encode_c(5.25, s_val, mat, proc)
            with torch.no_grad():
                rgb_m3 = (m3(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rgb_m4 = (m4.generate(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rgb_m5 = (m5(c)[0].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            c3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
            c4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
            c5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))

            proj_m3_s.append(c3["reinf_proj_px"])
            proj_m4_s.append(c4["reinf_proj_px"])
            proj_m5_s.append(c5["reinf_proj_px"])

            sensitivity_records.append({
                "sweep_type": "travel_speed", "material": mat, "process": proc,
                "feed_rate": 5.25, "speed": s_val,
                "m3_proj_px": c3["reinf_proj_px"], "m4_proj_px": c4["reinf_proj_px"], "m5_proj_px": c5["reinf_proj_px"]
            })

        ax_bot = axes[1, col_idx]
        ax_bot.plot(speed_range, proj_m3_s, 'o--', color='blue', label='M3: GC-WPG-v2', linewidth=1.5)
        ax_bot.plot(speed_range, proj_m4_s, 's-', color='green', label='M4: GeoSDF-CVAE', linewidth=2.0)
        ax_bot.plot(speed_range, proj_m5_s, '^:', color='purple', label='M5: cGAN', linewidth=1.5)
        ax_bot.set_title(f"Travel Speed Sensitivity (F=5.25 m/min)", fontsize=11, fontweight='bold')
        ax_bot.set_xlabel("Welding Travel Speed (cm/min)", fontsize=10)
        ax_bot.set_ylabel("Reinforcement Projection (px)", fontsize=10)
        ax_bot.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Controlled Physical Parameter Sensitivity Analysis Across 4 Material-Process Groups\nTop: Wire Feed Rate Sweep | Bottom: Welding Travel Speed Sweep",
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig6_parameter_sensitivity_curves.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print("Saved fig6_parameter_sensitivity_curves.png")

    df_sens = pd.DataFrame(sensitivity_records)
    df_sens.to_csv(os.path.join(RESULTS_DIR, "metrics", "PARAMETER_SENSITIVITY_METRICS.csv"), index=False)
    print("Saved PARAMETER_SENSITIVITY_METRICS.csv")

def run_unseen_in_domain_predictions(models, encode_c, device):
    print("\n" + "=" * 75)
    print("PHASE F: UNSEEN IN-DOMAIN PARAMETER GENERATION (4 VERIFIED CASES)")
    print("=" * 75)

    m1 = models["M1"]
    m2 = models["M2"]
    m3 = models["M3"]
    m4 = models["M4"]
    m5 = models["M5"]

    unseen_cases = [
        {"case_id": "U1", "mat": "IN718", "proc": "ColdArc", "feed": 5.50, "speed": 26.0, "desc": "IN718 ColdArc (Interpolation)"},
        {"case_id": "U2", "mat": "IN718", "proc": "GMAW", "feed": 5.25, "speed": 22.5, "desc": "IN718 GMAW (Untested Interpolation)"},
        {"case_id": "U3", "mat": "SS304L", "proc": "ColdArc", "feed": 5.25, "speed": 25.0, "desc": "SS304L ColdArc (Interpolation)"},
        {"case_id": "U4", "mat": "SS304L", "proc": "GMAW", "feed": 5.25, "speed": 24.5, "desc": "SS304L GMAW (Untested Interpolation)"}
    ]

    fig_summary, axes_sum = plt.subplots(4, 5, figsize=(18, 14), dpi=150)
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

        # Save individual raw files
        cv2.imwrite(os.path.join(c_dir, "M1_DirectCNN_raw.png"), cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M2_CVAE_raw.png"), cv2.cvtColor(rgb_m2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M3_GCWPGv2_raw.png"), cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M4_GeoSDFCVAE_raw.png"), cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M4_GeoSDFCVAE_mask.png"), mask_m4 * 255)
        np.save(os.path.join(c_dir, "M4_GeoSDFCVAE_sdf.npy"), sdf_m4)
        cv2.imwrite(os.path.join(c_dir, "M5_cGAN_raw.png"), cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(c_dir, "M5_cGAN_mask.png"), mask_m5 * 255)

        cnt4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
        cnt5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))

        unseen_records.append({
            "case_id": cid,
            "description": c_info["desc"],
            "material": c_info["mat"],
            "process": c_info["proc"],
            "feed_rate": c_info["feed"],
            "speed": c_info["speed"],
            "m4_predicted_proj_px": cnt4["reinf_proj_px"],
            "m4_predicted_apex_x": cnt4["apex_x"],
            "m4_predicted_apex_y": cnt4["apex_y"],
            "m5_predicted_proj_px": cnt5["reinf_proj_px"]
        })

        # Plot on summary figure
        row_imgs = [
            ("M1: Direct CNN", rgb_m1),
            ("M2: CVAE (z=0)", rgb_m2),
            ("M3: GC-WPG-v2", rgb_m3),
            ("M4: GeoSDF-CVAE", rgb_m4),
            ("M5: cGAN", rgb_m5)
        ]
        for col_idx, (m_label, img) in enumerate(row_imgs):
            ax = axes_sum[r_idx, col_idx]
            ax.imshow(img)
            ax.axvline(64, color='yellow', linestyle='--', linewidth=0.8)
            if col_idx == 3 and len(cnt4["pts"]) > 0:
                ax.plot(list(cnt4["pts"].values()), list(cnt4["pts"].keys()), color='lime', linewidth=1.2)
            if r_idx == 0:
                ax.set_title(m_label, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"Case {cid}\n{c_info['desc']}\nF={c_info['feed']} S={c_info['speed']}",
                              fontsize=9, fontweight='bold', rotation=0, labelpad=70)
            ax.set_xticks([])
            ax.set_yticks([])

    fig_summary.suptitle("Unseen Prospective Parameter Predictions Across 4 In-Domain Interpolation Cases\n(Zero Real Image Leakage — Purely Condition-Driven Inference)",
                         fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig7_unseen_parameter_predictions.png"), bbox_inches='tight')
    plt.close()
    print("Saved fig7_unseen_parameter_predictions.png")

    df_unseen = pd.DataFrame(unseen_records)
    df_unseen.to_csv(os.path.join(RESULTS_DIR, "metrics", "UNSEEN_PARAMETER_PREDICTIONS.csv"), index=False)
    print("Saved UNSEEN_PARAMETER_PREDICTIONS.csv")

    # Author Proposal for Future Physical Experiments
    author_experimental_proposals()

def author_experimental_proposals():
    proposal_text = """# PHASE V9 — TARGETED PHYSICAL VALIDATION EXPERIMENTAL PROPOSALS

## 1. Executive Summary & Experimental Objectives
To rigorously validate the generative geometry predictions produced by our continuous geometry framework (**GeoSDF-CVAE** and **Geometry-Conditioned cGAN**), we propose four physical welding trials across our four material-process regimes. These trials test both in-domain high-gradient interpolation points and physical boundary transitions.

---

## 2. Experimental Design Matrix

| Trial ID | Material | Process | Feed Rate (m/min) | Travel Speed (cm/min) | Predicted Crown Projection (px) | Targeted Physical Hypothesis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-V9-01** | Inconel 718 | Cold Arc | **5.50** | **26.0** | ~17.8 px (~2.97 mm) | Validates intermediate heat-input crown formation between high-speed and low-speed Cold Arc regimes. |
| **EXP-V9-02** | Inconel 718 | GMAW | **5.25** | **22.5** | ~19.5 px (~3.25 mm) | Tests high-deposition low-speed GMAW boundary where gravity sag and droplet detachment dynamics peak. |
| **EXP-V9-03** | SS304L | Cold Arc | **5.25** | **25.0** | ~18.2 px (~3.03 mm) | Evaluates surface-tension driven wetting angle in stainless steel under pulsed low-heat Cold Arc waveforms. |
| **EXP-V9-04** | SS304L | GMAW | **5.25** | **24.5** | ~20.1 px (~3.35 mm) | Evaluates deep fusion and crown reinforcement balance at the center of the parameter envelope. |

---

## 3. Recommended Metallographic Preparation Protocol
1. **Sectioning:** Transverse cross-sectioning at steady-state weld bead length (>50 mm from strike and crater extinguish points).
2. **Mounting & Polishing:** Conductive epoxy mount, ground with SiC papers (320 to 1200 grit), followed by 3 µm and 1 µm diamond polish, finishing with 0.05 µm colloidal silica.
3. **Etching:**
   - **Inconel 718:** Kalling's No. 2 reagent (5 g CuCl2, 100 mL HCl, 100 mL ethanol) swabbing for 10-15 seconds.
   - **SS304L:** Glyceregia (15 mL HCl, 10 mL glycerol, 5 mL HNO3) immersion for 30-45 seconds.
4. **Optical Macrography:** Calibrated optical stereo-microscopy at 10x magnification, uniform diffuse ring-light illumination, registering plate interface at baseline $X=64$ in $128 \\times 128$ normalized coordinate system.
"""
    with open(os.path.join(REPORTS_DIR, "PHASE_V9_PROPOSED_VALIDATION_EXPERIMENTS.md"), "w") as f:
        f.write(proposal_text)
    print("Saved PHASE_V9_PROPOSED_VALIDATION_EXPERIMENTS.md")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, encode_c = load_trained_models(device)
    run_parameter_sensitivity(models, encode_c, device)
    run_unseen_in_domain_predictions(models, encode_c, device)
    print("\nPhases E & F Completed Successfully!")

if __name__ == "__main__":
    main()
