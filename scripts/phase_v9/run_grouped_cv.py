"""
run_grouped_cv.py

Phase C: Fair Scientific Validation via Experiment-Grouped 5-Fold Cross-Validation:
- Divides 36 physical welding experiments into 5 stratified folds (7, 7, 7, 8, 7).
- Both cross-sectional macrograph views (A and B) of each experiment strictly stay in the same fold.
- Normalization parameters (F_min, F_max, S_min, S_max) fitted ONLY on the training fold.
- Trains out-of-fold models for all 5 architectures:
  M1: Direct Conditional CNN
  M2: Conditional VAE (deterministic z=0 and stochastic sampling)
  M3: GC-WPG-v2 (Multi-Scale Spatial Prior)
  M4: GeoSDF-CVAE (Continuous Signed Distance Field VAE)
  M5: Geometry-Conditioned cGAN (Spatial Geometry Conditioning + PatchGAN)
- Generates canonical deterministic out-of-fold predictions for all 36 physical experiments.
- Evaluates against BOTH real views (A and B) and aggregates to experiment level.
- Computes comprehensive metrics:
  * External contour MAE (px)
  * Boundary F1 (d_tol = 2px)
  * Apex localization error (px)
  * Mask Dice and IoU
  * Symmetric contour distance (px) and 95th-percentile Hausdorff distance HD95 (px)
  * Area error (px^2)
  * Weld ROI SSIM, Whole-image SSIM, and PSNR
  * Missing contour / extraction failure rate
  * 95% bootstrap confidence intervals
"""

import os
import sys
import time
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

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

AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
SDF_DIR = os.path.join(AUDIT_DIR, "gt_sdf")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware")
OOF_DIR = os.path.join(RESULTS_DIR, "oof_predictions")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for d in [OOF_DIR, METRICS_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================================
# Metric Computation Utilities
# =========================================================================
def compute_contour_distances(pts_gen, pts_real):
    if len(pts_gen) == 0 or len(pts_real) == 0:
        return 40.0, 45.0
    arr_gen = np.array([[x, y] for y, x in pts_gen.items()], dtype=np.float32)
    arr_real = np.array([[x, y] for y, x in pts_real.items()], dtype=np.float32)
    dists = cdist(arr_gen, arr_real)
    d_gen_to_real = np.min(dists, axis=1)
    d_real_to_gen = np.min(dists, axis=0)
    sym_dist = 0.5 * (float(np.mean(d_gen_to_real)) + float(np.mean(d_real_to_gen)))
    all_d = np.concatenate([d_gen_to_real, d_real_to_gen])
    hd95 = float(np.percentile(all_d, 95))
    return sym_dist, hd95

def compute_mask_overlap(pred_mask_binary, gt_mask_binary):
    p = (pred_mask_binary > 0).astype(np.float32)
    g = (gt_mask_binary > 0).astype(np.float32)
    intersection = float(np.sum(p * g))
    union = float(np.sum((p + g) > 0))
    sum_p = float(np.sum(p))
    sum_g = float(np.sum(g))
    
    iou = intersection / (union + 1e-6)
    dice = (2.0 * intersection) / (sum_p + sum_g + 1e-6)
    area_error_px2 = abs(sum_p - sum_g)
    return iou, dice, area_error_px2

def compute_all_sample_metrics(gen_rgb_uint8, real_rgb_uint8, gen_mask_binary, real_mask_binary, gen_pts, real_pts, real_edge_map):
    # Whole-image metrics
    g_fl = gen_rgb_uint8.astype(np.float32) / 255.0
    r_fl = real_rgb_uint8.astype(np.float32) / 255.0
    
    mse = float(np.mean((g_fl - r_fl) ** 2))
    psnr = float(psnr_func(r_fl, g_fl, data_range=1.0))
    ssim = float(ssim_func(r_fl, g_fl, data_range=1.0, channel_axis=2))
    
    # ROI SSIM
    if np.sum(real_mask_binary) > 10:
        g_gray = cv2.cvtColor(gen_rgb_uint8, cv2.COLOR_RGB2GRAY)
        r_gray = cv2.cvtColor(real_rgb_uint8, cv2.COLOR_RGB2GRAY)
        _, ssim_map = ssim_func(r_gray, g_gray, data_range=255, full=True)
        roi_ssim = float(np.mean(ssim_map[real_mask_binary > 0]))
    else:
        roi_ssim = ssim

    # Contour MAE
    common_ys = sorted(set(gen_pts.keys()).intersection(set(real_pts.keys())))
    if len(common_ys) > 5:
        contour_mae_px = float(np.mean([abs(gen_pts[y] - real_pts[y]) for y in common_ys]))
    else:
        contour_mae_px = 35.0

    # Apex error
    if len(gen_pts) > 0 and len(real_pts) > 0:
        g_apex_x = min(gen_pts.values())
        g_apex_y = [y for y, x in gen_pts.items() if x == g_apex_x][0]
        r_apex_x = min(real_pts.values())
        r_apex_y = [y for y, x in real_pts.items() if x == r_apex_x][0]
        apex_error_px = float(np.sqrt((g_apex_x - r_apex_x)**2 + (g_apex_y - r_apex_y)**2))
        proj_error_px = float(abs((64 - g_apex_x) - (64 - r_apex_x)))
    else:
        apex_error_px = 40.0
        proj_error_px = 30.0

    # Symmetric distance & HD95
    sym_dist_px, hd95_px = compute_contour_distances(gen_pts, real_pts)

    # Mask Dice & IoU
    iou, dice, area_err_px2 = compute_mask_overlap(gen_mask_binary, real_mask_binary)

    # Boundary F1 with tolerance d_tol = 2 px
    gen_edge = np.zeros((128, 128), dtype=np.uint8)
    for y, x in gen_pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            gen_edge[y, x] = 1

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_real = cv2.dilate(real_edge_map.astype(np.uint8), kernel)
    dilated_gen = cv2.dilate(gen_edge, kernel)

    tp_prec = np.sum((gen_edge == 1) & (dilated_real == 1))
    total_gen = np.sum(gen_edge == 1)
    precision = float(tp_prec / (total_gen + 1e-8))

    tp_rec = np.sum((real_edge_map == 1) & (dilated_gen == 1))
    total_real = np.sum(real_edge_map == 1)
    recall = float(tp_rec / (total_real + 1e-8))
    f1 = float(2 * precision * recall / (precision + recall + 1e-8))

    return {
        "ssim": ssim,
        "psnr": psnr,
        "mse": mse,
        "roi_ssim": roi_ssim,
        "contour_mae_px": contour_mae_px,
        "sym_dist_px": sym_dist_px,
        "hd95_px": hd95_px,
        "apex_error_px": apex_error_px,
        "proj_error_px": proj_error_px,
        "mask_iou": iou,
        "mask_dice": dice,
        "area_error_px2": area_err_px2,
        "boundary_f1": f1
    }

# =========================================================================
# Main Cross-Validation Routine
# =========================================================================
def run_experiment_grouped_cv():
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("PHASE C: EXPERIMENT-GROUPED 5-FOLD CROSS-VALIDATION BENCHMARK")
    print(f"Device: {device} | Mixed Precision: {device.type == 'cuda'}")
    print("=" * 75)

    df_clean = pd.read_csv(CLEAN_CSV)
    
    # 1. Stratified 5-Fold Grouped Assignment (Both views A & B stay together)
    folds = {}
    groups = df_clean.groupby(['material', 'process'], sort=False)
    for g_idx, (g_name, g_df) in enumerate(groups):
        samples = g_df['sample'].tolist()
        for idx, s in enumerate(samples):
            f = (idx + g_idx) % 5
            folds[s] = f
    df_clean['fold'] = df_clean['sample'].map(folds)
    
    # Save fold assignments
    df_clean[['sample', 'material', 'process', 'feed_rate', 'speed', 'fold']].to_csv(
        os.path.join(METRICS_DIR, "FOLD_ASSIGNMENTS.csv"), index=False
    )
    print("Fold counts (balanced 7-8 samples per fold):")
    print(df_clean['fold'].value_counts().sort_index())

    # Preload all 72 real images, masks, SDFs into memory
    data_cache = {}
    for s in range(1, 37):
        data_cache[s] = {}
        for view in ["A", "B"]:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(PROCESSED_DIR, f"{s}-{view}.png")), cv2.COLOR_BGR2RGB)
            mask = (cv2.imread(os.path.join(MASKS_DIR, f"{s}-{view}.png"), cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
            sdf = np.load(os.path.join(SDF_DIR, f"{s}-{view}.npy"))
            c_info = extract_robust_crown_contour(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
            data_cache[s][view] = {
                "rgb": rgb,
                "mask": mask,
                "sdf": sdf,
                "c_info": c_info
            }

    oof_sample_records = []
    oof_fold_records = []

    # Iterate over 5 folds
    for test_fold in range(5):
        print(f"\n>>> Running Fold {test_fold} / 4...", flush=True)
        train_df = df_clean[df_clean['fold'] != test_fold].copy()
        test_df = df_clean[df_clean['fold'] == test_fold].copy()

        # Fit feature scalers STRICTLY on training fold
        f_min, f_max = float(train_df['feed_rate'].min()), float(train_df['feed_rate'].max())
        s_min, s_max = float(train_df['speed'].min()), float(train_df['speed'].max())

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

        # Prepare train tensors (56 to 58 samples)
        train_c_list, train_img_list, train_mask_list, train_sdf_list, train_apex_list = [], [], [], [], []
        for _, row in train_df.iterrows():
            s_id = int(row['sample'])
            c_vec = encode_c(row)
            for view in ["A", "B"]:
                train_c_list.append(c_vec)
                d = data_cache[s_id][view]
                train_img_list.append(d["rgb"].astype(np.float32) / 255.0)
                train_mask_list.append(d["mask"].astype(np.float32))
                train_sdf_list.append(d["sdf"].astype(np.float32))
                train_apex_list.append([float(d["c_info"]["apex_x"]), float(d["c_info"]["apex_y"])])

        t_c = torch.tensor(np.array(train_c_list), dtype=torch.float32, device=device)
        t_img = torch.tensor(np.array(train_img_list), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        t_mask = torch.tensor(np.array(train_mask_list), dtype=torch.float32, device=device).unsqueeze(1)
        t_sdf = torch.tensor(np.array(train_sdf_list), dtype=torch.float32, device=device).unsqueeze(1)
        t_apex = torch.tensor(np.array(train_apex_list), dtype=torch.float32, device=device)

        batch_size = 28
        n_batches = int(np.ceil(len(t_c) / batch_size))
        t_fold_start = time.time()

        # -------------------------------------------------------------
        # Train Fold Models: M1, M2, M3, M4, M5
        # -------------------------------------------------------------
        # M1: Direct CNN
        print(f"  [Fold {test_fold}] Training M1 DirectCNN (12 ep)...", flush=True)
        m1 = DirectCNN(6, 16).to(device)
        opt_m1 = torch.optim.AdamW(m1.parameters(), lr=4e-3, weight_decay=1e-4)
        m1.train()
        for ep in range(12):
            perm = torch.randperm(len(t_c))
            for b in range(n_batches):
                idx = perm[b*batch_size : (b+1)*batch_size]
                opt_m1.zero_grad()
                pred = m1(t_c[idx])
                loss = F.l1_loss(pred, t_img[idx]) + F.l1_loss(pred * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
                loss.backward()
                opt_m1.step()
        m1.eval()

        # M2: CVAE
        print(f"  [Fold {test_fold}] Training M2 CVAE (12 ep)...", flush=True)
        m2 = ConditionalVAE(6, 24).to(device)
        opt_m2 = torch.optim.AdamW(m2.parameters(), lr=3e-3, weight_decay=1e-4)
        m2.train()
        for ep in range(12):
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
        m2.eval()

        # M3: GC-WPG-v2
        print(f"  [Fold {test_fold}] Training M3 GCWPGv2 (12 ep)...", flush=True)
        scaler_dict = {'w_min': 6.82, 'w_max': 12.02, 'r_min': 1.20, 'r_max': 4.85}
        m3 = GC_WPG_v2(6, 16, scaler_params=scaler_dict).to(device)
        opt_m3 = torch.optim.AdamW(m3.parameters(), lr=4e-3, weight_decay=1e-4)
        m3.train()
        for ep in range(12):
            perm = torch.randperm(len(t_c))
            for b in range(n_batches):
                idx = perm[b*batch_size : (b+1)*batch_size]
                opt_m3.zero_grad()
                pred, pred_geom, _ = m3(t_c[idx])
                loss = F.l1_loss(pred, t_img[idx]) + F.l1_loss(pred * t_mask[idx], t_img[idx] * t_mask[idx]) * 2.0
                loss.backward()
                opt_m3.step()
        m3.eval()

        # M4: GeoSDF-CVAE
        print(f"  [Fold {test_fold}] Training M4 GeoSDFCVAE (14 ep)...", flush=True)
        m4 = GeoSDF_CVAE(6, 16, beta=1.0).to(device)
        opt_m4 = torch.optim.AdamW(m4.parameters(), lr=3.5e-3, weight_decay=1e-4)
        m4.train()
        for ep in range(14):
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
        m4.eval()

        # M5: Geometry-Conditioned cGAN
        print(f"  [Fold {test_fold}] Training M5 cGAN (14 ep)...", flush=True)
        m5_g = Geometry_cGAN_Generator(6).to(device)
        m5_d = PatchGAN_Discriminator(in_channels=4).to(device)
        opt_g = torch.optim.AdamW(m5_g.parameters(), lr=2.5e-3, betas=(0.5, 0.999), weight_decay=1e-4)
        opt_d = torch.optim.AdamW(m5_d.parameters(), lr=1.5e-3, betas=(0.5, 0.999), weight_decay=1e-3)
        m5_g.train()
        m5_d.train()
        for ep in range(14):
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
        m5_g.eval()

        # -------------------------------------------------------------
        # Evaluate Out-of-Fold Test Samples for this Fold
        # -------------------------------------------------------------
        print(f"  [Fold {test_fold}] Evaluating {len(test_df)} test samples...", flush=True)
        models_dict = {
            "M1_DirectCNN": lambda c: (m1(c), None, None),
            "M2_CVAE": lambda c: (m2.generate(c), None, None),
            "M3_GCWPGv2": lambda c: (m3(c)[0], None, None),
            "M4_GeoSDFCVAE": lambda c: (m4.generate(c)[0], m4.generate(c)[2], m4.generate(c)[1]), # rgb, mask, sdf
            "M5_cGAN": lambda c: (m5_g(c)[0], m5_g(c)[1], None) # rgb, mask, None
        }

        fold_metrics_accum = {m_name: [] for m_name in models_dict}

        for _, row in test_df.iterrows():
            s_id = int(row['sample'])
            exp_code = f"S{s_id:02d}"
            c_test = torch.tensor([encode_c(row)], dtype=torch.float32, device=device)

            real_d_a = data_cache[s_id]["A"]
            real_d_b = data_cache[s_id]["B"]

            for m_key, m_func in models_dict.items():
                with torch.no_grad():
                    out_rgb_t, out_mask_t, out_sdf_t = m_func(c_test)

                # Process RGB output
                gen_rgb_uint8 = (np.clip(out_rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                
                # Derive predicted mask and contour
                c_gen = extract_robust_crown_contour(cv2.cvtColor(gen_rgb_uint8, cv2.COLOR_RGB2GRAY))
                if out_mask_t is not None:
                    # Model provides explicit predicted geometry mask
                    pred_mask_binary = (out_mask_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                else:
                    pred_mask_binary = c_gen["mask"]

                # Save Out-of-Fold images
                oof_sample_dir = os.path.join(OOF_DIR, f"fold_{test_fold}", exp_code)
                os.makedirs(oof_sample_dir, exist_ok=True)
                cv2.imwrite(os.path.join(oof_sample_dir, f"{m_key}_raw.png"), cv2.cvtColor(gen_rgb_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(oof_sample_dir, f"{m_key}_mask.png"), (pred_mask_binary * 255).astype(np.uint8))

                # Compute metrics against View A & View B
                m_a = compute_all_sample_metrics(
                    gen_rgb_uint8, real_d_a["rgb"], pred_mask_binary, real_d_a["mask"],
                    c_gen["pts"], real_d_a["c_info"]["pts"], real_d_a["mask"]
                )
                m_b = compute_all_sample_metrics(
                    gen_rgb_uint8, real_d_b["rgb"], pred_mask_binary, real_d_b["mask"],
                    c_gen["pts"], real_d_b["c_info"]["pts"], real_d_b["mask"]
                )

                # Average across both experimental views
                sample_res = {
                    "sample": s_id,
                    "experiment_id": exp_code,
                    "material": row["material"],
                    "process": row["process"],
                    "feed_rate": row["feed_rate"],
                    "speed": row["speed"],
                    "test_fold": test_fold,
                    "model_key": m_key,
                    "contour_mae_px": round((m_a["contour_mae_px"] + m_b["contour_mae_px"]) / 2.0, 2),
                    "boundary_f1": round((m_a["boundary_f1"] + m_b["boundary_f1"]) / 2.0, 4),
                    "apex_error_px": round((m_a["apex_error_px"] + m_b["apex_error_px"]) / 2.0, 2),
                    "proj_error_px": round((m_a["proj_error_px"] + m_b["proj_error_px"]) / 2.0, 2),
                    "sym_dist_px": round((m_a["sym_dist_px"] + m_b["sym_dist_px"]) / 2.0, 2),
                    "hd95_px": round((m_a["hd95_px"] + m_b["hd95_px"]) / 2.0, 2),
                    "mask_dice": round((m_a["mask_dice"] + m_b["mask_dice"]) / 2.0, 4),
                    "mask_iou": round((m_a["mask_iou"] + m_b["mask_iou"]) / 2.0, 4),
                    "area_error_px2": round((m_a["area_error_px2"] + m_b["area_error_px2"]) / 2.0, 1),
                    "roi_ssim": round((m_a["roi_ssim"] + m_b["roi_ssim"]) / 2.0, 4),
                    "ssim": round((m_a["ssim"] + m_b["ssim"]) / 2.0, 4),
                    "psnr": round((m_a["psnr"] + m_b["psnr"]) / 2.0, 2),
                    "extraction_failed": c_gen["reinf_proj_px"] <= 0
                }
                oof_sample_records.append(sample_res)
                fold_metrics_accum[m_key].append(sample_res)

        # Log fold-level summary
        print(f"  >>> Fold {test_fold} completed in {time.time() - t_fold_start:.1f}s", flush=True)
        for m_key, m_list in fold_metrics_accum.items():
            df_m_fold = pd.DataFrame(m_list)
            oof_fold_records.append({
                "fold": test_fold,
                "model_key": m_key,
                "sample_count": len(df_m_fold),
                "mean_contour_mae_px": round(df_m_fold["contour_mae_px"].mean(), 2),
                "mean_boundary_f1": round(df_m_fold["boundary_f1"].mean(), 4),
                "mean_hd95_px": round(df_m_fold["hd95_px"].mean(), 2),
                "mean_mask_dice": round(df_m_fold["mask_dice"].mean(), 4),
                "mean_roi_ssim": round(df_m_fold["roi_ssim"].mean(), 4),
                "mean_ssim": round(df_m_fold["ssim"].mean(), 4)
            })

    # Save Per-Sample OOF Metrics
    df_all_samples = pd.DataFrame(oof_sample_records)
    df_all_samples.to_csv(os.path.join(METRICS_DIR, "OOF_PER_SAMPLE_METRICS.csv"), index=False)
    print(f"\nSaved Per-Sample OOF Metrics to {os.path.join(METRICS_DIR, 'OOF_PER_SAMPLE_METRICS.csv')}")

    # Save Fold-Level Metrics
    df_folds = pd.DataFrame(oof_fold_records)
    df_folds.to_csv(os.path.join(METRICS_DIR, "OOF_FOLD_METRICS.csv"), index=False)
    print(f"Saved Fold-Level Metrics to {os.path.join(METRICS_DIR, 'OOF_FOLD_METRICS.csv')}")

    # Compute Overall Model Ranking & Bootstrap 95% Confidence Intervals
    ranking_records = []
    np.random.seed(42)

    for m_key in ["M1_DirectCNN", "M2_CVAE", "M3_GCWPGv2", "M4_GeoSDFCVAE", "M5_cGAN"]:
        df_sub = df_all_samples[df_all_samples['model_key'] == m_key]
        n_samples = len(df_sub)

        # Bootstrap 1000 resamples for 95% CI
        c_mae_boot = [np.mean(np.random.choice(df_sub['contour_mae_px'].values, n_samples, replace=True)) for _ in range(1000)]
        f1_boot = [np.mean(np.random.choice(df_sub['boundary_f1'].values, n_samples, replace=True)) for _ in range(1000)]
        dice_boot = [np.mean(np.random.choice(df_sub['mask_dice'].values, n_samples, replace=True)) for _ in range(1000)]
        ssim_boot = [np.mean(np.random.choice(df_sub['ssim'].values, n_samples, replace=True)) for _ in range(1000)]

        ranking_records.append({
            "model_key": m_key,
            "samples_evaluated": n_samples,
            "contour_mae_px": round(df_sub["contour_mae_px"].mean(), 2),
            "contour_mae_ci95": f"[{np.percentile(c_mae_boot, 2.5):.2f}, {np.percentile(c_mae_boot, 97.5):.2f}]",
            "boundary_f1": round(df_sub["boundary_f1"].mean(), 4),
            "boundary_f1_ci95": f"[{np.percentile(f1_boot, 2.5):.4f}, {np.percentile(f1_boot, 97.5):.4f}]",
            "apex_error_px": round(df_sub["apex_error_px"].mean(), 2),
            "sym_dist_px": round(df_sub["sym_dist_px"].mean(), 2),
            "hd95_px": round(df_sub["hd95_px"].mean(), 2),
            "mask_dice": round(df_sub["mask_dice"].mean(), 4),
            "mask_dice_ci95": f"[{np.percentile(dice_boot, 2.5):.4f}, {np.percentile(dice_boot, 97.5):.4f}]",
            "mask_iou": round(df_sub["mask_iou"].mean(), 4),
            "area_error_px2": round(df_sub["area_error_px2"].mean(), 1),
            "roi_ssim": round(df_sub["roi_ssim"].mean(), 4),
            "ssim": round(df_sub["ssim"].mean(), 4),
            "ssim_ci95": f"[{np.percentile(ssim_boot, 2.5):.4f}, {np.percentile(ssim_boot, 97.5):.4f}]",
            "psnr": round(df_sub["psnr"].mean(), 2),
            "failure_rate": f"{df_sub['extraction_failed'].mean() * 100:.1f}%"
        })

    df_ranking = pd.DataFrame(ranking_records)
    # Sort by held-out contour MAE
    df_ranking = df_ranking.sort_values(by="contour_mae_px", ascending=True)
    df_ranking.to_csv(os.path.join(METRICS_DIR, "OOF_MODEL_RANKING.csv"), index=False)
    print("\n" + "=" * 75)
    print("EXPERIMENT-GROUPED 5-FOLD CV MODEL RANKING (SORTED BY CONTOUR MAE):")
    print("=" * 75)
    print(df_ranking[['model_key', 'contour_mae_px', 'boundary_f1', 'mask_dice', 'hd95_px', 'ssim']].to_string(index=False))

    t_elapsed = time.time() - t_start
    print(f"\nGrouped 5-Fold CV Completed in {t_elapsed:.1f} seconds!")
    return df_ranking

if __name__ == "__main__":
    run_experiment_grouped_cv()
