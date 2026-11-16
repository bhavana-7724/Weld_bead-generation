"""
train_and_generate_36.py

Phase D of Phase V9:
1. Trains canonical production checkpoints for all 5 architectures on all 36 physical experiments (72 views):
   - M1: Direct Conditional CNN
   - M2: Conditional VAE (CVAE)
   - M3: GC-WPG-v2 (Multi-Scale Spatial Prior CNN)
   - M4: GeoSDF-CVAE (Continuous Signed Distance Field VAE)
   - M5: Geometry-Conditioned cGAN (Spatial Geometry Conditioning + PatchGAN)
   Saves checkpoints to results/phase_v9_geometry_aware/checkpoints/

2. Generates canonical deterministic outputs for all 36 physical experiments (S01 to S36):
   - Individual sample directories in results/phase_v9_geometry_aware/generated_36/S{s:02d}/
   - Real View A and Real View B photographs
   - M1-M5 raw generated images
   - Predicted binary crown masks (explicitly generated for M4 and M5, extracted for M1-M3)
   - Predicted Continuous Signed Distance Field (SDF) array and visualization for M4
   - Contour overlays on baseline X=64
   - High-resolution 10-panel scientific comparison figure for each sample

3. Compiles presentation-quality 36-sample Contact Sheets:
   - fig1_36sample_contact_sheet_m1_direct_cnn.png
   - fig2_36sample_contact_sheet_m2_cvae.png
   - fig3_36sample_contact_sheet_m3_gc_wpg_v2.png
   - fig4_36sample_contact_sheet_m4_geosdf_cvae.png
   - fig5_36sample_contact_sheet_m5_geometry_cgan.png
   - fig5b_tri_view_comparison_summary.png
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v9.geometry_models import (
    DirectCNN, ConditionalVAE, GC_WPG_v2, GeoSDF_CVAE,
    Geometry_cGAN_Generator, PatchGAN_Discriminator
)
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware")
AUDIT_DIR = os.path.join(RESULTS_DIR, "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
SDF_DIR = os.path.join(AUDIT_DIR, "gt_sdf")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
GEN36_DIR = os.path.join(RESULTS_DIR, "generated_36")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

for d in [CKPT_DIR, GEN36_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

def train_production_models(device, df_clean, data_cache, epochs=35):
    print("=" * 75)
    print("TRAINING CANONICAL FULL-DATASET CHECKPOINTS (N=36 EXPERIMENTS, 72 VIEWS)")
    print(f"Device: {device} | Epochs: {epochs}")
    print("=" * 75)

    f_min, f_max = float(df_clean['feed_rate'].min()), float(df_clean['feed_rate'].max())
    s_min, s_max = float(df_clean['speed'].min()), float(df_clean['speed'].max())

    norm_meta = {
        "f_min": f_min, "f_max": f_max,
        "s_min": s_min, "s_max": s_max,
        "w_min": 6.82, "w_max": 12.02,
        "r_min": 1.20, "r_max": 4.85
    }
    with open(os.path.join(CKPT_DIR, "normalization_params.json"), "w") as f:
        json.dump(norm_meta, f, indent=2)

    def encode_c(row):
        f_norm = (float(row['feed_rate']) - f_min) / (f_max - f_min + 1e-8)
        s_norm = (float(row['speed']) - s_min) / (s_max - s_min + 1e-8)
        return [
            f_norm, s_norm,
            1.0 if row['material'] == 'IN718' else 0.0,
            1.0 if row['material'] == 'SS304L' else 0.0,
            1.0 if row['process'] == 'ColdArc' else 0.0,
            1.0 if row['process'] == 'GMAW' else 0.0
        ]

    c_list, img_list, mask_list, sdf_list, apex_list = [], [], [], [], []
    for _, row in df_clean.iterrows():
        s_id = int(row['sample'])
        c_vec = encode_c(row)
        for view in ["A", "B"]:
            d = data_cache[s_id][view]
            c_list.append(c_vec)
            img_list.append(d["rgb"].astype(np.float32) / 255.0)
            mask_list.append(d["mask"].astype(np.float32))
            sdf_list.append(d["sdf"].astype(np.float32))
            apex_list.append([float(d["c_info"]["apex_x"]), float(d["c_info"]["apex_y"])])

    t_c = torch.tensor(np.array(c_list), dtype=torch.float32, device=device)
    t_img = torch.tensor(np.array(img_list), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    t_mask = torch.tensor(np.array(mask_list), dtype=torch.float32, device=device).unsqueeze(1)
    t_sdf = torch.tensor(np.array(sdf_list), dtype=torch.float32, device=device).unsqueeze(1)
    t_apex = torch.tensor(np.array(apex_list), dtype=torch.float32, device=device)

    batch_size = 24
    n_batches = int(np.ceil(len(t_c) / batch_size))

    # --- M1: Direct CNN ---
    print("\n[1/5] Training M1: Direct Conditional ResUp CNN...")
    m1 = DirectCNN(6, 16).to(device)
    opt_m1 = torch.optim.AdamW(m1.parameters(), lr=3e-3, weight_decay=1e-4)
    m1.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m1.zero_grad()
            pred = m1(t_c[idx])
            loss = F.l1_loss(pred, t_img[idx]) + F.l1_loss(pred * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
            loss.backward()
            opt_m1.step()
    torch.save(m1.state_dict(), os.path.join(CKPT_DIR, "m1_direct_cnn_final.pt"))

    # --- M2: Conditional VAE ---
    print("[2/5] Training M2: Conditional VAE (CVAE)...")
    m2 = ConditionalVAE(6, 24).to(device)
    opt_m2 = torch.optim.AdamW(m2.parameters(), lr=2e-3, weight_decay=1e-4)
    m2.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m2.zero_grad()
            recon, mu, logvar = m2(t_img[idx], t_c[idx])
            l_recon = F.l1_loss(recon, t_img[idx]) + F.l1_loss(recon * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = l_recon + 0.005 * kl
            loss.backward()
            opt_m2.step()
    torch.save(m2.state_dict(), os.path.join(CKPT_DIR, "m2_cvae_final.pt"))

    # --- M3: GC-WPG-v2 ---
    print("[3/5] Training M3: GC-WPG-v2 Multi-Scale Spatial Prior...")
    m3 = GC_WPG_v2(6, 16, scaler_params=norm_meta).to(device)
    opt_m3 = torch.optim.AdamW(m3.parameters(), lr=3e-3, weight_decay=1e-4)
    m3.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m3.zero_grad()
            pred, pred_geom, _ = m3(t_c[idx])
            loss = F.l1_loss(pred, t_img[idx]) + F.l1_loss(pred * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
            loss.backward()
            opt_m3.step()
    torch.save(m3.state_dict(), os.path.join(CKPT_DIR, "m3_gc_wpg_v2_final.pt"))

    # --- M4: GeoSDF-CVAE ---
    print("[4/5] Training M4: GeoSDF-CVAE Continuous Geometry...")
    m4 = GeoSDF_CVAE(6, 16, beta=1.0).to(device)
    opt_m4 = torch.optim.AdamW(m4.parameters(), lr=2.5e-3, weight_decay=1e-4)
    m4.train()
    for ep in range(epochs + 5):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m4.zero_grad()
            p_rgb, p_sdf, p_mask, p_apex, mu, logvar = m4(t_sdf[idx], t_c[idx])
            l_sdf = F.l1_loss(p_sdf, t_sdf[idx])
            l_mask = F.binary_cross_entropy(p_mask, t_mask[idx])
            l_apex = F.l1_loss(p_apex, t_apex[idx])
            l_rgb = F.l1_loss(p_rgb, t_img[idx]) + F.l1_loss(p_rgb * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.5
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = 1.0 * l_sdf + 2.0 * l_mask + 0.1 * l_apex + 2.0 * l_rgb + 0.005 * kl
            loss.backward()
            opt_m4.step()
    torch.save(m4.state_dict(), os.path.join(CKPT_DIR, "m4_geosdf_cvae_final.pt"))

    # --- M5: Geometry-Conditioned cGAN ---
    print("[5/5] Training M5: Geometry-Conditioned cGAN...")
    m5_g = Geometry_cGAN_Generator(6).to(device)
    m5_d = PatchGAN_Discriminator(in_channels=4).to(device)
    opt_g = torch.optim.AdamW(m5_g.parameters(), lr=2e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(m5_d.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-3)
    m5_g.train()
    m5_d.train()
    for ep in range(epochs + 5):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            # Train D
            opt_d.zero_grad()
            p_rgb, p_mask = m5_g(t_c[idx])
            d_real = m5_d(t_mask[idx], t_img[idx])
            d_fake = m5_d(p_mask.detach(), p_rgb.detach())
            loss_d = 0.5 * (torch.mean((d_real - 1.0)**2) + torch.mean(d_fake**2))
            loss_d.backward()
            opt_d.step()
            # Train G
            opt_g.zero_grad()
            d_fake_g = m5_d(p_mask, p_rgb)
            loss_adv = torch.mean((d_fake_g - 1.0)**2)
            loss_g_recon = F.l1_loss(p_rgb, t_img[idx]) + F.l1_loss(p_rgb * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.5
            loss_g_mask = F.l1_loss(p_mask, t_mask[idx]) * 3.0
            loss_g = loss_adv * 0.1 + loss_g_recon + loss_g_mask
            loss_g.backward()
            opt_g.step()
    torch.save(m5_g.state_dict(), os.path.join(CKPT_DIR, "m5_geometry_cgan_final.pt"))
    print("All production model checkpoints successfully trained and saved!")

    return m1, m2, m3, m4, m5_g, encode_c

def generate_canonical_36_samples(m1, m2, m3, m4, m5_g, encode_c, device, df_clean, data_cache):
    print("\n" + "=" * 75)
    print("GENERATING CANONICAL DETERMINISTIC PREDICTIONS FOR ALL 36 SAMPLES")
    print("=" * 75)

    m1.eval()
    m2.eval()
    m3.eval()
    m4.eval()
    m5_g.eval()

    models = {
        "M1_DirectCNN": lambda c: (m1(c), None, None),
        "M2_CVAE": lambda c: (m2.generate(c), None, None),
        "M3_GCWPGv2": lambda c: (m3(c)[0], None, None),
        "M4_GeoSDFCVAE": lambda c: (m4.generate(c)[0], m4.generate(c)[1], m4.generate(c)[2]), # rgb, sdf, mask
        "M5_cGAN": lambda c: (m5_g(c)[0], None, m5_g(c)[1]) # rgb, None, mask
    }

    sample_outputs = {}

    for _, row in df_clean.iterrows():
        s_id = int(row['sample'])
        exp_code = f"S{s_id:02d}"
        s_dir = os.path.join(GEN36_DIR, exp_code)
        os.makedirs(s_dir, exist_ok=True)

        c_vec = torch.tensor([encode_c(row)], dtype=torch.float32, device=device)

        # Real A & B
        real_a = data_cache[s_id]["A"]["rgb"]
        real_b = data_cache[s_id]["B"]["rgb"]
        cv2.imwrite(os.path.join(s_dir, "Real_A.png"), cv2.cvtColor(real_a, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "Real_B.png"), cv2.cvtColor(real_b, cv2.COLOR_RGB2BGR))

        gt_pts_a = data_cache[s_id]["A"]["c_info"]["pts"]
        gt_pts_b = data_cache[s_id]["B"]["c_info"]["pts"]

        s_dict = {
            "Real_A": real_a,
            "Real_B": real_b,
            "gt_pts_a": gt_pts_a,
            "gt_pts_b": gt_pts_b,
            "preds": {}
        }

        for m_name, m_func in models.items():
            with torch.no_grad():
                rgb_t, sdf_t, mask_t = m_func(c_vec)

            rgb_np = (np.clip(rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(s_dir, f"{m_name}_raw.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

            c_info = extract_robust_crown_contour(cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY))
            if mask_t is not None:
                mask_bin = (mask_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            else:
                mask_bin = c_info["mask"]

            cv2.imwrite(os.path.join(s_dir, f"{m_name}_mask.png"), (mask_bin * 255).astype(np.uint8))

            # Overlay
            overlay = rgb_np.copy()
            cv2.line(overlay, (64, 0), (64, 127), (255, 255, 0), 1) # baseline
            for y, x in c_info["pts"].items():
                if 0 <= y < 128 and 0 <= x < 128:
                    cv2.circle(overlay, (x, y), 1, (0, 255, 0), -1) # green predicted contour
            cv2.imwrite(os.path.join(s_dir, f"{m_name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Special SDF save for M4
            if sdf_t is not None:
                sdf_np = sdf_t.squeeze().cpu().numpy()
                np.save(os.path.join(s_dir, f"{m_name}_sdf.npy"), sdf_np)
                # Visual SDF
                sdf_vis = ((np.clip(sdf_np, -15, 15) + 15) / 30.0 * 255).astype(np.uint8)
                sdf_vis = cv2.applyColorMap(sdf_vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(s_dir, f"{m_name}_sdf_vis.png"), sdf_vis)

            s_dict["preds"][m_name] = {
                "rgb": rgb_np,
                "mask": mask_bin,
                "pts": c_info["pts"],
                "reinf_proj_px": c_info["reinf_proj_px"]
            }

        # Create 10-Panel Scientific Comparison Figure
        fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150)
        axes[0, 0].imshow(real_a)
        axes[0, 0].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[0, 0].set_title(f"{exp_code} Real View A", fontsize=11, fontweight='bold')

        axes[0, 1].imshow(real_b)
        axes[0, 1].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[0, 1].set_title(f"{exp_code} Real View B", fontsize=11, fontweight='bold')

        axes[0, 2].imshow(s_dict["preds"]["M1_DirectCNN"]["rgb"])
        axes[0, 2].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[0, 2].set_title("M1: Direct CNN", fontsize=11)

        axes[0, 3].imshow(s_dict["preds"]["M2_CVAE"]["rgb"])
        axes[0, 3].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[0, 3].set_title("M2: CVAE (z=0)", fontsize=11)

        axes[0, 4].imshow(s_dict["preds"]["M3_GCWPGv2"]["rgb"])
        axes[0, 4].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[0, 4].set_title("M3: GC-WPG-v2", fontsize=11)

        axes[1, 0].imshow(s_dict["preds"]["M4_GeoSDFCVAE"]["rgb"])
        axes[1, 0].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[1, 0].set_title("M4: GeoSDF-CVAE", fontsize=11, fontweight='bold')

        axes[1, 1].imshow(s_dict["preds"]["M5_cGAN"]["rgb"])
        axes[1, 1].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[1, 1].set_title("M5: cGAN", fontsize=11)

        axes[1, 2].imshow(s_dict["preds"]["M4_GeoSDFCVAE"]["mask"], cmap='gray')
        axes[1, 2].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[1, 2].set_title("M4 Predicted Mask", fontsize=11)

        axes[1, 3].imshow(s_dict["preds"]["M5_cGAN"]["mask"], cmap='gray')
        axes[1, 3].axvline(64, color='yellow', linestyle='--', linewidth=1)
        axes[1, 3].set_title("M5 Predicted Mask", fontsize=11)

        # Panel 10: Multi-Contour Scientific Overlay
        ax_c = axes[1, 4]
        ax_c.imshow(real_a, alpha=0.6)
        ax_c.axvline(64, color='black', linestyle='--', linewidth=1, label="Base X=64")
        # GT A & B
        ax_c.plot(list(gt_pts_a.values()), list(gt_pts_a.keys()), color='cyan', linewidth=1.5, label="GT Real A")
        ax_c.plot(list(gt_pts_b.values()), list(gt_pts_b.keys()), color='blue', linewidth=1.5, linestyle=':', label="GT Real B")
        # M1-M5
        colors = {
            "M1_DirectCNN": "orange",
            "M2_CVAE": "purple",
            "M3_GCWPGv2": "red",
            "M4_GeoSDFCVAE": "green",
            "M5_cGAN": "magenta"
        }
        for m_k, col in colors.items():
            pts = s_dict["preds"][m_k]["pts"]
            if len(pts) > 0:
                ax_c.plot(list(pts.values()), list(pts.keys()), color=col, linewidth=1.2, label=m_k.split('_')[0])

        ax_c.set_xlim(0, 128)
        ax_c.set_ylim(128, 0)
        ax_c.set_title("Contour Comparison", fontsize=11, fontweight='bold')
        ax_c.legend(loc='lower left', fontsize=6, framealpha=0.7)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Sample {exp_code} | {row['material']} - {row['process']} | Feed={row['feed_rate']} m/min, Speed={row['speed']} cm/min",
                     fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(s_dir, f"sample_{s_id:02d}_10panel_comparison.png"), bbox_inches='tight')
        plt.close()

        sample_outputs[s_id] = s_dict

    print("Canonical 36 samples generated successfully!")
    return sample_outputs

def create_36sample_contact_sheets(sample_outputs, df_clean):
    print("\n" + "=" * 75)
    print("CREATING 36-SAMPLE CONTACT SHEETS FOR FIGURES 1 TO 5")
    print("=" * 75)

    models_info = [
        ("M1_DirectCNN", "fig1_36sample_contact_sheet_m1_direct_cnn.png", "Figure 1: 36-Sample Contact Sheet — M1 Direct Conditional CNN"),
        ("M2_CVAE", "fig2_36sample_contact_sheet_m2_cvae.png", "Figure 2: 36-Sample Contact Sheet — M2 Conditional VAE (z=0)"),
        ("M3_GCWPGv2", "fig3_36sample_contact_sheet_m3_gc_wpg_v2.png", "Figure 3: 36-Sample Contact Sheet — M3 GC-WPG-v2 Multi-Scale Prior"),
        ("M4_GeoSDFCVAE", "fig4_36sample_contact_sheet_m4_geosdf_cvae.png", "Figure 4: 36-Sample Contact Sheet — M4 GeoSDF-CVAE (Continuous Geometry)"),
        ("M5_cGAN", "fig5_36sample_contact_sheet_m5_geometry_cgan.png", "Figure 5: 36-Sample Contact Sheet — M5 Geometry-Conditioned cGAN")
    ]

    for m_key, fname, title_text in models_info:
        fig, axes = plt.subplots(6, 6, figsize=(18, 18), dpi=150)
        for idx in range(36):
            s_id = idx + 1
            row_idx = idx // 6
            col_idx = idx % 6
            ax = axes[row_idx, col_idx]

            row_data = df_clean[df_clean['sample'] == s_id].iloc[0]
            img = sample_outputs[s_id]["preds"][m_key]["rgb"]
            pts = sample_outputs[s_id]["preds"][m_key]["pts"]

            ax.imshow(img)
            ax.axvline(64, color='yellow', linestyle='--', linewidth=0.8, alpha=0.7)
            if len(pts) > 0:
                ax.plot(list(pts.values()), list(pts.keys()), color='lime', linewidth=1.0)

            ax.set_title(f"S{s_id:02d}: {row_data['material'][:2]}-{row_data['process'][:4]}\nF={row_data['feed_rate']} S={row_data['speed']}",
                         fontsize=8, pad=2)
            ax.axis('off')

        fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout()
        save_path = os.path.join(FIGURES_DIR, fname)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {fname}")

    # Tri-View Comparison Summary: Real A, Real B, M3, M4, M5 for 6 representative samples
    rep_samples = [4, 8, 14, 18, 26, 32] # representative of all 4 groups
    fig, axes = plt.subplots(len(rep_samples), 5, figsize=(16, 3.2 * len(rep_samples)), dpi=150)
    for r_idx, s_id in enumerate(rep_samples):
        row_data = df_clean[df_clean['sample'] == s_id].iloc[0]
        s_d = sample_outputs[s_id]

        # Col 0: Real A
        axes[r_idx, 0].imshow(s_d["Real_A"])
        axes[r_idx, 0].set_title(f"S{s_id:02d} Real A ({row_data['material']}-{row_data['process']})", fontsize=10)
        axes[r_idx, 0].axvline(64, color='yellow', linestyle='--', linewidth=0.8)

        # Col 1: Real B
        axes[r_idx, 1].imshow(s_d["Real_B"])
        axes[r_idx, 1].set_title(f"S{s_id:02d} Real B", fontsize=10)
        axes[r_idx, 1].axvline(64, color='yellow', linestyle='--', linewidth=0.8)

        # Col 2: M3
        axes[r_idx, 2].imshow(s_d["preds"]["M3_GCWPGv2"]["rgb"])
        axes[r_idx, 2].set_title("M3: GC-WPG-v2", fontsize=10)
        axes[r_idx, 2].axvline(64, color='yellow', linestyle='--', linewidth=0.8)

        # Col 3: M4 GeoSDF
        axes[r_idx, 3].imshow(s_d["preds"]["M4_GeoSDFCVAE"]["rgb"])
        axes[r_idx, 3].set_title("M4: GeoSDF-CVAE", fontsize=10, fontweight='bold')
        axes[r_idx, 3].axvline(64, color='yellow', linestyle='--', linewidth=0.8)

        # Col 4: M5 cGAN
        axes[r_idx, 4].imshow(s_d["preds"]["M5_cGAN"]["rgb"])
        axes[r_idx, 4].set_title("M5: cGAN", fontsize=10)
        axes[r_idx, 4].axvline(64, color='yellow', linestyle='--', linewidth=0.8)

        for c_idx in range(5):
            axes[r_idx, c_idx].axis('off')

    fig.suptitle("Tri-View Comparative Verification: Ground Truth A/B vs Advanced Geometry Models", fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    tri_path = os.path.join(FIGURES_DIR, "fig5b_tri_view_comparison_summary.png")
    plt.savefig(tri_path, bbox_inches='tight')
    plt.close()
    print("Saved fig5b_tri_view_comparison_summary.png")

def main():
    t0 = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    df_clean = pd.read_csv(CLEAN_CSV)

    # Preload data cache
    data_cache = {}
    for s in range(1, 37):
        data_cache[s] = {}
        for view in ["A", "B"]:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(PROCESSED_DIR, f"{s}-{view}.png")), cv2.COLOR_BGR2RGB)
            mask = (cv2.imread(os.path.join(MASKS_DIR, f"{s}-{view}.png"), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
            sdf = np.load(os.path.join(SDF_DIR, f"{s}-{view}.npy"))
            c_info = extract_robust_crown_contour(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            data_cache[s][view] = {
                "rgb": rgb, "mask": mask, "sdf": sdf, "c_info": c_info
            }

    m1, m2, m3, m4, m5_g, encode_c = train_production_models(device, df_clean, data_cache, epochs=25)
    sample_outputs = generate_canonical_36_samples(m1, m2, m3, m4, m5_g, encode_c, device, df_clean, data_cache)
    create_36sample_contact_sheets(sample_outputs, df_clean)
    print(f"\nPhase D Complete in {time.time() - t0:.1f}s!")

if __name__ == "__main__":
    main()
