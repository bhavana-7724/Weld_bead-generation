"""
train_production_a100.py

Phase V10 on NVIDIA A100 AI Server:
1. Trains production checkpoints for all 6 neural architectures on full dataset:
   - M1: Direct Conditional ResUp CNN
   - M2: Conditional VAE (CVAE)
   - M3: GC-WPG-v2 (Multi-Scale Spatial Prior CNN)
   - M4: GeoSDF-CVAE (Continuous Signed Distance Field VAE with FiLM & Dual-Path Prior)
   - M5: Geometry-Conditioned cGAN (Spatial Mask Prediction + PatchGAN)
   - M6: GF-WPG (GENUINE Geometry-First: Stage 1 Contour Net + Stage 2 Renderer)
   - B1: PCAContourRegressor (Statistical Shape Baseline)

2. Generates canonical outputs for all 36 physical experiments (S01 to S36) with STRICT SEPARATION:
   - M{k}_raw_rgb.png: Pure unaltered RGB output array
   - M{k}_predicted_mask.png: Pure binary/probability mask
   - M{k}_contour_coords.json: Exact numerical pixel coordinates
   - M{k}_visualization_overlay.png: Diagnostic visual overlay with green contour & baseline
   - 12-Panel Scientific Comparison Figure for each sample

3. Compiles presentation-quality 36-sample Contact Sheets:
   - fig1_36sample_contact_sheet_m1_direct_cnn.png
   - fig2_36sample_contact_sheet_m2_cvae.png
   - fig3_36sample_contact_sheet_m3_gc_wpg_v2.png
   - fig4_36sample_contact_sheet_m4_geosdf_cvae.png
   - fig5_36sample_contact_sheet_m5_geometry_cgan.png
   - fig5b_36sample_contact_sheet_m6_gf_wpg.png
   - fig5c_tri_view_comparison_summary.png
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

from scripts.phase_v10.geometry_models_cuda import (
    DirectCNN, ConditionalVAE, GC_WPG_v2, GeoSDF_CVAE,
    Geometry_cGAN_Generator, PatchGAN_Discriminator,
    GF_WPG, PCAContourRegressor
)
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")

AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
SDF_DIR = os.path.join(AUDIT_DIR, "gt_sdf")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v10_a100")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
GEN_DIR = os.path.join(RESULTS_DIR, "generated_36")

for d in [CKPT_DIR, FIGURES_DIR, GEN_DIR]:
    os.makedirs(d, exist_ok=True)

def train_production_models_a100(epochs=80):
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    print("=" * 80, flush=True)
    print("PHASE V10: PRODUCTION MODEL TRAINING (FULL DATASET, 36 SAMPLES, 72 VIEWS)", flush=True)
    print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"Hardware: {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)", flush=True)
    print(f"Production Epochs: {epochs}", flush=True)
    print("=" * 80, flush=True)

    df_clean = pd.read_csv(CLEAN_CSV)
    f_min, f_max = float(df_clean['feed_rate'].min()), float(df_clean['feed_rate'].max())
    s_min, s_max = float(df_clean['speed'].min()), float(df_clean['speed'].max())

    meta = {
        'f_min': f_min, 'f_max': f_max,
        's_min': s_min, 's_max': s_max,
        'w_min': 6.82, 'w_max': 12.02,
        'r_min': 1.20, 'r_max': 4.85
    }
    with open(os.path.join(CKPT_DIR, "normalization_params.json"), "w") as f:
        json.dump(meta, f, indent=2)

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

    # Preload full dataset (72 views)
    c_list, img_list, mask_list, sdf_list, apx_list, cnt_list = [], [], [], [], [], []
    for _, r in df_clean.iterrows():
        s_id = int(r['sample'])
        cvec = encode_c(r)
        for view in ["A", "B"]:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(PROCESSED_DIR, f"{s_id}-{view}.png")), cv2.COLOR_BGR2RGB)
            mask = (cv2.imread(os.path.join(MASKS_DIR, f"{s_id}-{view}.png"), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
            sdf = np.load(os.path.join(SDF_DIR, f"{s_id}-{view}.npy"))
            c_info = extract_robust_crown_contour(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))

            c_list.append(cvec)
            img_list.append(rgb.astype(np.float32) / 255.0)
            mask_list.append(mask.astype(np.float32))
            sdf_list.append(sdf.astype(np.float32))
            apx_list.append([float(c_info["apex_x"]), float(c_info["apex_y"])])

            pts_dict = c_info.get("pts", {})
            c_vec_95 = [float(pts_dict.get(y, 64)) for y in range(18, 113)]
            cnt_list.append(c_vec_95)

    t_c = torch.tensor(np.array(c_list), dtype=torch.float32, device=device)
    t_img = torch.tensor(np.array(img_list), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    t_mask = torch.tensor(np.array(mask_list), dtype=torch.float32, device=device).unsqueeze(1)
    t_sdf = torch.tensor(np.array(sdf_list), dtype=torch.float32, device=device).unsqueeze(1)
    t_apx = torch.tensor(np.array(apx_list), dtype=torch.float32, device=device)
    t_cnt = torch.tensor(np.array(cnt_list), dtype=torch.float32, device=device)

    batch_size = 16 if is_cuda else 24
    n_batches = int(np.ceil(len(t_c) / batch_size))

    # --- 1. M1 Direct CNN ---
    print("\nTraining Final M1: DirectCNN...", flush=True)
    m1 = DirectCNN(6, 16).to(device)
    opt_m1 = torch.optim.AdamW(m1.parameters(), lr=3e-3, weight_decay=1e-4)
    sched_m1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m1, T_max=epochs, eta_min=1e-4)
    scaler_m1 = torch.amp.GradScaler('cuda', enabled=is_cuda)
    m1.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m1.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                pred = m1(t_c[idx])
                loss = F.l1_loss(pred, t_img[idx]) + F.l1_loss(pred * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
            scaler_m1.scale(loss).backward()
            scaler_m1.step(opt_m1)
            scaler_m1.update()
        sched_m1.step()
    m1.eval()
    torch.save(m1.state_dict(), os.path.join(CKPT_DIR, "m1_direct_cnn_final.pt"))

    # --- 2. M2 Conditional VAE ---
    print("Training Final M2: Conditional VAE...", flush=True)
    m2 = ConditionalVAE(6, 24).to(device)
    opt_m2 = torch.optim.AdamW(m2.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched_m2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m2, T_max=epochs, eta_min=1e-4)
    scaler_m2 = torch.amp.GradScaler('cuda', enabled=is_cuda)
    m2.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m2.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                recon, mu, logvar = m2(t_img[idx], t_c[idx])
                l_rec = F.l1_loss(recon, t_img[idx]) + F.l1_loss(recon * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = l_rec + 0.005 * kl
            scaler_m2.scale(loss).backward()
            scaler_m2.step(opt_m2)
            scaler_m2.update()
        sched_m2.step()
    m2.eval()
    torch.save(m2.state_dict(), os.path.join(CKPT_DIR, "m2_cvae_final.pt"))

    # --- 3. M3 GC-WPG-v2 ---
    print("Training Final M3: GC-WPG-v2...", flush=True)
    m3 = GC_WPG_v2(6, 16, scaler_params=meta).to(device)
    opt_m3 = torch.optim.AdamW(m3.parameters(), lr=3e-3, weight_decay=1e-4)
    sched_m3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m3, T_max=epochs, eta_min=1e-4)
    scaler_m3 = torch.amp.GradScaler('cuda', enabled=is_cuda)
    m3.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m3.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                pred_rgb, _, _ = m3(t_c[idx])
                loss = F.l1_loss(pred_rgb, t_img[idx]) + F.l1_loss(pred_rgb * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.5
            scaler_m3.scale(loss).backward()
            scaler_m3.step(opt_m3)
            scaler_m3.update()
        sched_m3.step()
    m3.eval()
    torch.save(m3.state_dict(), os.path.join(CKPT_DIR, "m3_gc_wpg_v2_final.pt"))

    # --- 4. M4 GeoSDF-CVAE (Dual-Path Prior Mode) ---
    print("Training Final M4: GeoSDF-CVAE (Dual-Path Prior Training)...", flush=True)
    m4 = GeoSDF_CVAE(6, 16, beta=1.0).to(device)
    opt_m4 = torch.optim.AdamW(m4.parameters(), lr=2.5e-3, weight_decay=1e-4)
    sched_m4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m4, T_max=epochs, eta_min=1e-4)
    scaler_m4 = torch.amp.GradScaler('cuda', enabled=is_cuda)
    m4.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m4.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                # Posterior pass
                p_rgb, p_sdf, p_mask, p_apex, mu, logvar = m4(t_sdf[idx], t_c[idx])
                l_sdf = F.l1_loss(p_sdf, t_sdf[idx])
                p_mask_c = torch.clamp(p_mask, 1e-6, 1.0 - 1e-6)
                l_mask = F.binary_cross_entropy(p_mask_c, t_mask[idx])
                l_apex = F.l1_loss(p_apex, t_apx[idx])
                l_rgb = F.l1_loss(p_rgb, t_img[idx]) + F.l1_loss(p_rgb * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
                kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                l_post = l_sdf + 2.0 * l_mask + 0.1 * l_apex + l_rgb + 0.005 * kl

                # Prior pass (z=0)
                pr_rgb, pr_sdf, pr_mask, pr_apex = m4.forward_prior(t_c[idx])
                l_pr_sdf = F.l1_loss(pr_sdf, t_sdf[idx])
                pr_mask_c = torch.clamp(pr_mask, 1e-6, 1.0 - 1e-6)
                l_pr_mask = F.binary_cross_entropy(pr_mask_c, t_mask[idx])
                l_pr_rgb = F.l1_loss(pr_rgb, t_img[idx])
                l_prior = l_pr_sdf + 2.0 * l_pr_mask + l_pr_rgb

                loss = l_post + 1.0 * l_prior
            scaler_m4.scale(loss).backward()
            scaler_m4.step(opt_m4)
            scaler_m4.update()
        sched_m4.step()
    m4.eval()
    torch.save(m4.state_dict(), os.path.join(CKPT_DIR, "m4_geosdf_cvae_final.pt"))

    # --- 5. M5 Geometry-Conditioned cGAN ---
    print("Training Final M5: Geometry-Conditioned cGAN...", flush=True)
    m5_g = Geometry_cGAN_Generator(6).to(device)
    m5_d = PatchGAN_Discriminator(in_channels=4).to(device)
    opt_g = torch.optim.AdamW(m5_g.parameters(), lr=2e-3, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = torch.optim.AdamW(m5_d.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-3)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=epochs, eta_min=1e-4)
    sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=epochs, eta_min=1e-4)
    scaler_g = torch.amp.GradScaler('cuda', enabled=is_cuda)
    scaler_d = torch.amp.GradScaler('cuda', enabled=is_cuda)
    m5_g.train()
    m5_d.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_d.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                p_rgb, p_mask = m5_g(t_c[idx])
                d_real = m5_d(t_mask[idx], t_img[idx])
                d_fake = m5_d(p_mask.detach(), p_rgb.detach())
                loss_d = 0.5 * (torch.mean((d_real - 1.0)**2) + torch.mean(d_fake**2))
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d)
            scaler_d.update()

            opt_g.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                d_fake_g = m5_d(p_mask, p_rgb)
                loss_adv = torch.mean((d_fake_g - 1.0)**2)
                loss_g_recon = F.l1_loss(p_rgb, t_img[idx]) + F.l1_loss(p_rgb * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.5
                loss_g_mask = F.l1_loss(p_mask, t_mask[idx]) * 3.0
                loss_g = loss_adv * 0.1 + loss_g_recon + loss_g_mask
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
        sched_g.step()
        sched_d.step()
    m5_g.eval()
    torch.save(m5_g.state_dict(), os.path.join(CKPT_DIR, "m5_geometry_cgan_final.pt"))

    # --- 6. M6 GF-WPG (GENUINE Geometry-First) ---
    print("Training Final M6: GF-WPG (Geometry-First 2-Stage)...", flush=True)
    m6 = GF_WPG(cond_dim=6, num_contour_pts=95).to(device)
    opt_m6 = torch.optim.AdamW(m6.parameters(), lr=3e-3, weight_decay=1e-4)
    sched_m6 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m6, T_max=epochs, eta_min=1e-4)
    scaler_m6 = torch.amp.GradScaler('cuda', enabled=is_cuda)
    m6.train()
    for ep in range(epochs):
        perm = torch.randperm(len(t_c))
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            opt_m6.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                p_rgb, p_mask, p_pts, p_apx = m6(t_c[idx])
                l_geom = F.l1_loss(p_pts, t_cnt[idx]) + F.l1_loss(p_apx, t_apx[idx]) * 0.5
                l_mask = F.l1_loss(p_mask, t_mask[idx]) * 2.0
                l_render = F.l1_loss(p_rgb, t_img[idx]) + F.l1_loss(p_rgb * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
                loss = l_geom + l_mask + l_render
            scaler_m6.scale(loss).backward()
            scaler_m6.step(opt_m6)
            scaler_m6.update()
        sched_m6.step()
    m6.eval()
    torch.save(m6.state_dict(), os.path.join(CKPT_DIR, "m6_gf_wpg_final.pt"))

    # --- 7. Baseline B1: PCA Contour Regressor ---
    print("Fitting Final Baseline B1: PCA-Contour Regressor...", flush=True)
    b1_pca = PCAContourRegressor(n_components=4, alpha=1.0)
    b1_pca.fit(t_c.cpu().numpy(), t_cnt.cpu().numpy())

    # =========================================================================
    # CANONICAL GENERATION FOR ALL 36 SAMPLES (WITH EXPLICIT FILE SEPARATION)
    # =========================================================================
    print("\nGenerating canonical predictions for all 36 samples with explicit separation...", flush=True)
    all_sample_imgs = {"M1": [], "M2": [], "M3": [], "M4": [], "M5": [], "M6": [], "RealA": [], "RealB": []}

    for _, row in df_clean.iterrows():
        s_id = int(row['sample'])
        s_dir = os.path.join(GEN_DIR, f"S{s_id:02d}")
        os.makedirs(s_dir, exist_ok=True)

        c_vec = torch.tensor([encode_c(row)], dtype=torch.float32, device=device)
        c_np = np.array([encode_c(row)])

        real_a_rgb = cv2.cvtColor(cv2.imread(os.path.join(PROCESSED_DIR, f"{s_id}-A.png")), cv2.COLOR_BGR2RGB)
        real_b_rgb = cv2.cvtColor(cv2.imread(os.path.join(PROCESSED_DIR, f"{s_id}-B.png")), cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            rgb_m1 = (np.clip(m1(c_vec).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            rgb_m2 = (np.clip(m2.generate(c_vec).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            rgb_m3 = (np.clip(m3(c_vec)[0].squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)

            p_rgb4, p_sdf4, p_mask4, _ = m4.generate(c_vec)
            rgb_m4 = (np.clip(p_rgb4.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            sdf_m4 = p_sdf4.squeeze().cpu().numpy()
            mask_m4 = (p_mask4.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            p_rgb5, p_mask5 = m5_g(c_vec)
            rgb_m5 = (np.clip(p_rgb5.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            mask_m5 = (p_mask5.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            p_rgb6, p_mask6, p_pts6, _ = m6.generate(c_vec)
            rgb_m6 = (np.clip(p_rgb6.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            mask_m6 = (p_mask6.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # Contours extracted independently
        cnt_m1 = extract_robust_crown_contour(cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2GRAY))
        cnt_m2 = extract_robust_crown_contour(cv2.cvtColor(rgb_m2, cv2.COLOR_RGB2GRAY))
        cnt_m3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
        cnt_m4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
        cnt_m5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))
        cnt_m6 = extract_robust_crown_contour(cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2GRAY))

        # ---------------------------------------------------------------------
        # MANDATORY SEPARATION: Pure RAW RGB, Pure Mask, Numeric JSON, Overlays
        # ---------------------------------------------------------------------
        # Save Real Images
        cv2.imwrite(os.path.join(s_dir, "Real_ViewA.png"), cv2.cvtColor(real_a_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "Real_ViewB.png"), cv2.cvtColor(real_b_rgb, cv2.COLOR_RGB2BGR))

        # Save Pure RAW RGB files (Zero line drawings or annotations)
        cv2.imwrite(os.path.join(s_dir, "M1_raw_rgb.png"), cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M2_raw_rgb.png"), cv2.cvtColor(rgb_m2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M3_raw_rgb.png"), cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M4_raw_rgb.png"), cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M5_raw_rgb.png"), cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M6_raw_rgb.png"), cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2BGR))

        # Save Predicted Masks
        cv2.imwrite(os.path.join(s_dir, "M4_predicted_mask.png"), mask_m4 * 255)
        np.save(os.path.join(s_dir, "M4_predicted_sdf.npy"), sdf_m4)
        cv2.imwrite(os.path.join(s_dir, "M5_predicted_mask.png"), mask_m5 * 255)
        cv2.imwrite(os.path.join(s_dir, "M6_predicted_mask.png"), mask_m6 * 255)

        # Save Pure Contour Coordinate JSONs
        with open(os.path.join(s_dir, "M1_contour_coords.json"), "w") as f:
            json.dump(cnt_m1.get("pts", {}), f)
        with open(os.path.join(s_dir, "M4_contour_coords.json"), "w") as f:
            json.dump(cnt_m4.get("pts", {}), f)
        with open(os.path.join(s_dir, "M6_contour_coords.json"), "w") as f:
            json.dump(cnt_m6.get("pts", {}), f)

        # Save Diagnostic Visual Overlay (Green contour + Yellow baseline)
        def make_vis_overlay(img_rgb, cnt_obj):
            ov = img_rgb.copy()
            for y, x in cnt_obj.get("pts", {}).items():
                if 0 <= y < 128 and 0 <= x < 128:
                    ov[y, x] = [0, 255, 0] # Lime contour
            cv2.line(ov, (64, 0), (64, 127), (255, 255, 0), 1) # Yellow baseline
            return ov

        cv2.imwrite(os.path.join(s_dir, "M1_visualization_overlay.png"), cv2.cvtColor(make_vis_overlay(rgb_m1, cnt_m1), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M4_visualization_overlay.png"), cv2.cvtColor(make_vis_overlay(rgb_m4, cnt_m4), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(s_dir, "M6_visualization_overlay.png"), cv2.cvtColor(make_vis_overlay(rgb_m6, cnt_m6), cv2.COLOR_RGB2BGR))

        # Cache for contact sheets
        all_sample_imgs["M1"].append(rgb_m1)
        all_sample_imgs["M2"].append(rgb_m2)
        all_sample_imgs["M3"].append(rgb_m3)
        all_sample_imgs["M4"].append(rgb_m4)
        all_sample_imgs["M5"].append(rgb_m5)
        all_sample_imgs["M6"].append(rgb_m6)
        all_sample_imgs["RealA"].append(real_a_rgb)
        all_sample_imgs["RealB"].append(real_b_rgb)

    # -------------------------------------------------------------------------
    # COMPILE 36-SAMPLE CONTACT SHEETS (Figures 1 to 6)
    # -------------------------------------------------------------------------
    print("Compiling 36-Sample Presentation Contact Sheets...", flush=True)
    def compile_sheet(img_list, title, out_filename):
        fig, axes = plt.subplots(4, 9, figsize=(18, 9), dpi=150)
        axes = axes.flatten()
        for idx in range(36):
            ax = axes[idx]
            ax.imshow(img_list[idx])
            ax.axvline(64, color='yellow', linestyle='--', linewidth=0.7)
            r = df_clean.iloc[idx]
            ax.set_title(f"S{r['sample']:02d} ({r['material'][:2]})\nF={r['feed_rate']} S={r['speed']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, out_filename), bbox_inches='tight')
        plt.close()

    compile_sheet(all_sample_imgs["M1"], "Figure 1: Canonical 36-Sample Benchmark — M1: Direct CNN (ResUp)", "fig1_36sample_contact_sheet_m1_direct_cnn.png")
    compile_sheet(all_sample_imgs["M2"], "Figure 2: Canonical 36-Sample Benchmark — M2: Conditional VAE (z=0)", "fig2_36sample_contact_sheet_m2_cvae.png")
    compile_sheet(all_sample_imgs["M3"], "Figure 3: Canonical 36-Sample Benchmark — M3: GC-WPG-v2 (Multi-Scale Spatial Prior)", "fig3_36sample_contact_sheet_m3_gc_wpg_v2.png")
    compile_sheet(all_sample_imgs["M4"], "Figure 4: Canonical 36-Sample Benchmark — M4: GeoSDF-CVAE (Continuous Distance Field)", "fig4_36sample_contact_sheet_m4_geosdf_cvae.png")
    compile_sheet(all_sample_imgs["M5"], "Figure 5: Canonical 36-Sample Benchmark — M5: Geometry-Conditioned cGAN", "fig5_36sample_contact_sheet_m5_geometry_cgan.png")
    compile_sheet(all_sample_imgs["M6"], "Figure 5b: Canonical 36-Sample Benchmark — M6: GF-WPG (GENUINE Geometry-First)", "fig5b_36sample_contact_sheet_m6_gf_wpg.png")

    # Tri-View Summary Figure (Real View A vs Real View B vs Proposed M6 GF-WPG vs M4 GeoSDF-CVAE)
    fig_tri, ax_tri = plt.subplots(4, 4, figsize=(14, 14), dpi=150)
    rep_samples = [4, 13, 21, 31] # 0-indexed for S05, S14, S22, S32
    rep_labels = ["S05 (IN718 ColdArc)", "S14 (IN718 GMAW)", "S22 (SS304L ColdArc)", "S32 (SS304L GMAW)"]
    for row_i, (s_idx, s_lbl) in enumerate(zip(rep_samples, rep_labels)):
        ax_tri[row_i, 0].imshow(all_sample_imgs["RealA"][s_idx])
        ax_tri[row_i, 0].set_title(f"{s_lbl}\nReal View A", fontsize=10, fontweight='bold')
        ax_tri[row_i, 1].imshow(all_sample_imgs["RealB"][s_idx])
        ax_tri[row_i, 1].set_title("Real View B", fontsize=10, fontweight='bold')
        ax_tri[row_i, 2].imshow(all_sample_imgs["M4"][s_idx])
        ax_tri[row_i, 2].set_title("M4: GeoSDF-CVAE", fontsize=10, fontweight='bold')
        ax_tri[row_i, 3].imshow(all_sample_imgs["M6"][s_idx])
        ax_tri[row_i, 3].set_title("M6: GF-WPG (Geometry-First)", fontsize=10, fontweight='bold')
        for col_i in range(4):
            ax_tri[row_i, col_i].axvline(64, color='yellow', linestyle='--', linewidth=0.8)
            ax_tri[row_i, col_i].set_xticks([])
            ax_tri[row_i, col_i].set_yticks([])

    fig_tri.suptitle("Phase V10: Tri-View Physical Macrograph vs. Geometry-First Generative Validation", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "fig5c_tri_view_comparison_summary.png"), bbox_inches='tight')
    plt.close()

    t_total = time.time() - t_start
    print(f"\nProduction training and 36-sample generation completed in {t_total:.2f} s ({t_total/60:.2f} min)", flush=True)

if __name__ == "__main__":
    train_production_models_a100(epochs=80)
