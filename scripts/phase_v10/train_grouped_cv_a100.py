"""
train_grouped_cv_a100.py

Phase V10 on NVIDIA A100 AI Server:
High-Performance CUDA Experiment-Grouped 5-Fold Cross-Validation Benchmark

Key Methodological Corrections:
1. Training-Fold-Only Inner Validation:
   - Each outer training fold is partitioned into inner_train (80%) and inner_val (20%), grouped by sample.
   - The outer 5-fold test set is strictly sequestered and NEVER touched during model training,
     validation loss tracking, early stopping, or checkpoint selection.
2. Best-Checkpoint Selection & Early Stopping:
   - At each epoch, computes inner_val loss.
   - Tracks best checkpoint and implements early stopping with patience=20 epochs.
   - Detailed learning curves logged to metrics/learning_curves/fold_{k}_{model}_curve.csv.
3. Complete Model Inventory (6 Neural Models + 1 Statistical Shape Baseline):
   - M1: Direct Conditional ResUp CNN
   - M2: Conditional VAE (deterministic z=0 prior mode)
   - M3: GC-WPG-v2 (Multi-Scale Spatial Prior CNN)
   - M4: GeoSDF-CVAE (Continuous Signed Distance Field with FiLM + Prior Loss)
   - M5: Geometry-Conditioned cGAN (Spatial Mask Prediction + PatchGAN)
   - M6: GF-WPG (GENUINE Geometry-First: Stage 1 Contour Net + Stage 2 Renderer)
   - B1: PCAContourRegressor (Compact PCA Contour + Ridge Regression Baseline)
4. Comprehensive Out-of-Fold Metrics & 1,000 Bootstrap 95% CIs.
   - All contour metrics reported strictly in PIXELS.
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

from scripts.phase_v10.geometry_models_cuda import (
    DirectCNN, ConditionalVAE, GC_WPG_v2, GeoSDF_CVAE,
    Geometry_cGAN_Generator, PatchGAN_Discriminator,
    GF_WPG, PCAContourRegressor
)
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v10_a100")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FOLDS_CKPT_DIR = os.path.join(CKPT_DIR, "folds")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CURVES_DIR = os.path.join(METRICS_DIR, "learning_curves")

AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
SDF_DIR = os.path.join(AUDIT_DIR, "gt_sdf")

for d in [FOLDS_CKPT_DIR, METRICS_DIR, CURVES_DIR]:
    os.makedirs(d, exist_ok=True)

# =========================================================================
# Metric Computation Functions
# =========================================================================
def compute_comprehensive_metrics(real_rgb, gen_rgb, real_mask, gen_mask, real_cnt, gen_cnt):
    # 1. Whole Image SSIM & PSNR
    ssim = float(ssim_func(real_rgb, gen_rgb, channel_axis=2, data_range=255))
    psnr = float(psnr_func(real_rgb, gen_rgb, data_range=255))
    mse = float(np.mean((real_rgb.astype(np.float32) - gen_rgb.astype(np.float32)) ** 2))

    # 2. ROI SSIM (Crown deposition region)
    roi_mask = np.zeros((128, 128), dtype=bool)
    roi_mask[:, :64] = True
    real_roi = real_rgb.copy()
    gen_roi = gen_rgb.copy()
    real_roi[~roi_mask] = 0
    gen_roi[~roi_mask] = 0
    roi_ssim = float(ssim_func(real_roi, gen_roi, channel_axis=2, data_range=255))

    # 3. Contour Overlap Metrics
    r_pts = real_cnt.get("pts", {})
    g_pts = gen_cnt.get("pts", {})

    y_common = sorted(list(set(r_pts.keys()).intersection(set(g_pts.keys()))))
    if len(y_common) >= 5:
        diffs = [abs(r_pts[y] - g_pts[y]) for y in y_common]
        contour_mae_px = float(np.mean(diffs))
    else:
        contour_mae_px = 35.0 # Standard fallback penalty for severe mismatch

    # 4. Apex and Projection Error
    apex_error_px = float(np.sqrt(
        (real_cnt.get("apex_x", 64) - gen_cnt.get("apex_x", 64))**2 +
        (real_cnt.get("apex_y", 64) - gen_cnt.get("apex_y", 64))**2
    ))
    proj_error_px = float(abs(real_cnt.get("reinf_proj_px", 0) - gen_cnt.get("reinf_proj_px", 0)))

    # 5. Mask Dice & IoU
    inter = np.sum((real_mask > 0) & (gen_mask > 0))
    union = np.sum((real_mask > 0) | (gen_mask > 0))
    iou = float(inter / (union + 1e-8))
    dice = float(2.0 * inter / (np.sum(real_mask > 0) + np.sum(gen_mask > 0) + 1e-8))
    area_err_px2 = float(abs(np.sum(real_mask > 0) - np.sum(gen_mask > 0)))

    # 6. Symmetric Hausdorff / Distance
    r_coords = np.array([[x, y] for y, x in r_pts.items()])
    g_coords = np.array([[x, y] for y, x in g_pts.items()])

    if len(r_coords) > 0 and len(g_coords) > 0:
        dists_rg = cdist(r_coords, g_coords)
        d_r_to_g = dists_rg.min(axis=1)
        d_g_to_r = dists_rg.min(axis=0)
        sym_dist_px = float((d_r_to_g.mean() + d_g_to_r.mean()) / 2.0)
        all_dists = np.concatenate([d_r_to_g, d_g_to_r])
        hd95_px = float(np.percentile(all_dists, 95))
    else:
        sym_dist_px = 40.0
        hd95_px = 50.0

    # 7. Boundary F1 Score at d_tol = 2 px
    real_edge_map = np.zeros((128, 128), dtype=np.uint8)
    for y, x in r_pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            real_edge_map[y, x] = 1

    gen_edge = np.zeros((128, 128), dtype=np.uint8)
    for y, x in g_pts.items():
        if 0 <= y < 128 and 0 <= x < 128:
            gen_edge[y, x] = 1

    kernel_2px = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated_real_2px = cv2.dilate(real_edge_map, kernel_2px)
    dilated_gen_2px = cv2.dilate(gen_edge, kernel_2px)

    tp_prec_2px = np.sum((gen_edge == 1) & (dilated_real_2px == 1))
    prec_2px = float(tp_prec_2px / (np.sum(gen_edge == 1) + 1e-8))
    tp_rec_2px = np.sum((real_edge_map == 1) & (dilated_gen_2px == 1))
    rec_2px = float(tp_rec_2px / (np.sum(real_edge_map == 1) + 1e-8))
    f1_2px = float(2 * prec_2px * rec_2px / (prec_2px + rec_2px + 1e-8))

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
        "boundary_f1": f1_2px
    }

# =========================================================================
# Main Cross-Validation Routine
# =========================================================================
def run_a100_grouped_cv(max_epochs=80, patience=20):
    t_start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    print("=" * 80, flush=True)
    print("PHASE V10: NVIDIA A100 EXPERIMENT-GROUPED 5-FOLD CROSS-VALIDATION", flush=True)
    print(f"Target Device: {device} | CUDA Available: {torch.cuda.is_available()}", flush=True)
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Hardware:  {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)", flush=True)
        print("Mixed Precision: torch.amp.autocast('cuda') enabled", flush=True)
    print(f"Max Epochs: {max_epochs} | Early Stopping Patience: {patience} epochs", flush=True)
    print("=" * 80, flush=True)

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
    
    df_clean[['sample', 'material', 'process', 'feed_rate', 'speed', 'fold']].to_csv(
        os.path.join(METRICS_DIR, "FOLD_ASSIGNMENTS.csv"), index=False
    )
    print("Stratified Outer Fold Assignment Counts:")
    print(df_clean['fold'].value_counts().sort_index(), flush=True)

    # Preload all 72 real images, masks, SDFs into memory
    print("\nPre-loading 72 macrograph cross-sections and ground-truth geometry arrays...", flush=True)
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

    oof_sample_records = []

    # Iterate over 5 folds
    for test_fold in range(5):
        t_fold_start = time.time()
        print(f"\n" + "-" * 75, flush=True)
        print(f">>> RUNNING OUTER FOLD {test_fold} / 4 (Grouped Holdout Validation)", flush=True)
        print("-" * 75, flush=True)

        outer_train_df = df_clean[df_clean['fold'] != test_fold].copy()
        outer_test_df = df_clean[df_clean['fold'] == test_fold].copy()

        # Fit feature scalers STRICTLY on outer training fold
        f_min, f_max = float(outer_train_df['feed_rate'].min()), float(outer_train_df['feed_rate'].max())
        s_min, s_max = float(outer_train_df['speed'].min()), float(outer_train_df['speed'].max())

        scaler_meta = {
            'f_min': f_min, 'f_max': f_max,
            's_min': s_min, 's_max': s_max,
            'w_min': 6.82, 'w_max': 12.02,
            'r_min': 1.20, 'r_max': 4.85
        }

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

        # ---------------------------------------------------------------------
        # TRAINING-FOLD-ONLY INNER VALIDATION SPLIT (Task 2)
        # Keeps outer test data 100% excluded from model selection / early stopping!
        # ---------------------------------------------------------------------
        # Assign ~20% of training samples (~5-6 samples) to inner_val, stratified by group
        inner_val_samples = []
        for _, g_sub in outer_train_df.groupby(['material', 'process']):
            sub_samples = g_sub['sample'].tolist()
            # Pick 1 sample per group for inner validation
            val_pick = sub_samples[test_fold % len(sub_samples)]
            inner_val_samples.append(val_pick)

        inner_train_df = outer_train_df[~outer_train_df['sample'].isin(inner_val_samples)].copy()
        inner_val_df = outer_train_df[outer_train_df['sample'].isin(inner_val_samples)].copy()

        print(f"  Inner Train Samples: {len(inner_train_df)} ({len(inner_train_df)*2} views) | "
              f"Inner Val Samples: {len(inner_val_df)} ({len(inner_val_df)*2} views) | "
              f"Outer Test: {len(outer_test_df)} ({len(outer_test_df)*2} views)", flush=True)

        def make_tensors(df_subset):
            c_l, img_l, mask_l, sdf_l, apex_l, contour_l = [], [], [], [], [], []
            for _, r in df_subset.iterrows():
                s_id = int(r['sample'])
                cvec = encode_c(r)
                for view in ["A", "B"]:
                    d = data_cache[s_id][view]
                    c_l.append(cvec)
                    img_l.append(d["rgb"].astype(np.float32) / 255.0)
                    mask_l.append(d["mask"].astype(np.float32))
                    sdf_l.append(d["sdf"].astype(np.float32))
                    apex_l.append([float(d["c_info"]["apex_x"]), float(d["c_info"]["apex_y"])])

                    # 95 contour x-coordinates for y in [18, 112]
                    pts_dict = d["c_info"].get("pts", {})
                    c_vec_95 = [float(pts_dict.get(y, 64)) for y in range(18, 113)]
                    contour_l.append(c_vec_95)

            t_c = torch.tensor(np.array(c_l), dtype=torch.float32, device=device)
            t_img = torch.tensor(np.array(img_l), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            t_mask = torch.tensor(np.array(mask_l), dtype=torch.float32, device=device).unsqueeze(1)
            t_sdf = torch.tensor(np.array(sdf_l), dtype=torch.float32, device=device).unsqueeze(1)
            t_apx = torch.tensor(np.array(apex_l), dtype=torch.float32, device=device)
            t_cnt = torch.tensor(np.array(contour_l), dtype=torch.float32, device=device)
            return t_c, t_img, t_mask, t_sdf, t_apx, t_cnt

        tr_c, tr_img, tr_mask, tr_sdf, tr_apx, tr_cnt = make_tensors(inner_train_df)
        val_c, val_img, val_mask, val_sdf, val_apx, val_cnt = make_tensors(inner_val_df)

        batch_size = 16 if is_cuda else 24
        n_batches = int(np.ceil(len(tr_c) / batch_size))

        # Helper function for training-fold-only validation and early stopping
        def train_and_select_best(model_name, model, opt, sched, scaler, forward_train_fn, forward_val_fn):
            best_val = float('inf')
            best_state = None
            patience_cnt = 0
            curve_records = []

            for ep in range(1, max_epochs + 1):
                model.train()
                perm = torch.randperm(len(tr_c))
                train_losses = []
                for b in range(n_batches):
                    idx = perm[b*batch_size : (b+1)*batch_size]
                    opt.zero_grad()
                    with torch.amp.autocast('cuda', enabled=is_cuda):
                        loss = forward_train_fn(idx)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    train_losses.append(loss.item())

                sched.step()
                curr_lr = opt.param_groups[0]['lr']
                train_loss_ep = float(np.mean(train_losses))

                # Validation strictly on inner_val (no gradients)
                model.eval()
                with torch.no_grad():
                    with torch.amp.autocast('cuda', enabled=is_cuda):
                        val_loss_ep = float(forward_val_fn().item())

                curve_records.append({
                    "epoch": ep, "train_loss": train_loss_ep, "val_loss": val_loss_ep, "lr": curr_lr
                })

                # Best checkpoint tracking
                if val_loss_ep < best_val:
                    best_val = val_loss_ep
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1

                if patience_cnt >= patience:
                    print(f"    [{model_name}] Early stopping at epoch {ep} (Best Val Loss: {best_val:.4f})")
                    break

            # Save learning curve CSV
            df_curve = pd.DataFrame(curve_records)
            curve_path = os.path.join(CURVES_DIR, f"fold_{test_fold}_{model_name}_curve.csv")
            df_curve.to_csv(curve_path, index=False)

            # Load best weights
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            ckpt_path = os.path.join(FOLDS_CKPT_DIR, f"fold_{test_fold}_{model_name}.pt")
            torch.save(model.state_dict(), ckpt_path)
            return model

        # =====================================================================
        # MODEL 1: Direct CNN
        # =====================================================================
        print(f"  [Fold {test_fold}] Training M1 DirectCNN...", flush=True)
        m1 = DirectCNN(6, 16).to(device)
        opt_m1 = torch.optim.AdamW(m1.parameters(), lr=3e-3, weight_decay=1e-4)
        sched_m1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m1, T_max=max_epochs, eta_min=1e-4)
        scaler_m1 = torch.amp.GradScaler('cuda', enabled=is_cuda)

        def fwd_train_m1(idx):
            pred = m1(tr_c[idx])
            return F.l1_loss(pred, tr_img[idx]) + F.l1_loss(pred * tr_mask[idx], tr_img[idx] * tr_mask[idx]) * 2.0

        def fwd_val_m1():
            pred = m1(val_c)
            return F.l1_loss(pred, val_img) + F.l1_loss(pred * val_mask, val_img * val_mask) * 2.0

        m1 = train_and_select_best("m1_direct_cnn", m1, opt_m1, sched_m1, scaler_m1, fwd_train_m1, fwd_val_m1)

        # =====================================================================
        # MODEL 2: Conditional VAE
        # =====================================================================
        print(f"  [Fold {test_fold}] Training M2 CVAE...", flush=True)
        m2 = ConditionalVAE(6, 24).to(device)
        opt_m2 = torch.optim.AdamW(m2.parameters(), lr=2.5e-3, weight_decay=1e-4)
        sched_m2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m2, T_max=max_epochs, eta_min=1e-4)
        scaler_m2 = torch.amp.GradScaler('cuda', enabled=is_cuda)

        def fwd_train_m2(idx):
            recon, mu, logvar = m2(tr_img[idx], tr_c[idx])
            l_rec = F.l1_loss(recon, tr_img[idx]) + F.l1_loss(recon * tr_mask[idx], tr_img[idx] * tr_mask[idx]) * 2.0
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            return l_rec + 0.005 * kl

        def fwd_val_m2():
            # Validate in prospective prior mode (z=0)
            pred_prior = m2.generate(val_c)
            return F.l1_loss(pred_prior, val_img) + F.l1_loss(pred_prior * val_mask, val_img * val_mask) * 2.0

        m2 = train_and_select_best("m2_cvae", m2, opt_m2, sched_m2, scaler_m2, fwd_train_m2, fwd_val_m2)

        # =====================================================================
        # MODEL 3: GC-WPG-v2 (Multi-Scale Spatial Prior)
        # =====================================================================
        print(f"  [Fold {test_fold}] Training M3 GC-WPG-v2...", flush=True)
        m3 = GC_WPG_v2(6, 16, scaler_params=scaler_meta).to(device)
        opt_m3 = torch.optim.AdamW(m3.parameters(), lr=3e-3, weight_decay=1e-4)
        sched_m3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m3, T_max=max_epochs, eta_min=1e-4)
        scaler_m3 = torch.amp.GradScaler('cuda', enabled=is_cuda)

        def fwd_train_m3(idx):
            pred_rgb, pred_geom, _ = m3(tr_c[idx])
            l_rgb = F.l1_loss(pred_rgb, tr_img[idx]) + F.l1_loss(pred_rgb * tr_mask[idx], tr_img[idx] * tr_mask[idx]) * 2.5
            return l_rgb

        def fwd_val_m3():
            pred_rgb, _, _ = m3(val_c)
            return F.l1_loss(pred_rgb, val_img) + F.l1_loss(pred_rgb * val_mask, val_img * val_mask) * 2.5

        m3 = train_and_select_best("m3_gc_wpg_v2", m3, opt_m3, sched_m3, scaler_m3, fwd_train_m3, fwd_val_m3)

        # =====================================================================
        # MODEL 4: GeoSDF-CVAE (With Dual-Path Prior Training & FiLM)
        # =====================================================================
        print(f"  [Fold {test_fold}] Training M4 GeoSDF-CVAE (Dual-Path Prior Mode)...", flush=True)
        m4 = GeoSDF_CVAE(6, 16, beta=1.0).to(device)
        opt_m4 = torch.optim.AdamW(m4.parameters(), lr=2.5e-3, weight_decay=1e-4)
        sched_m4 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m4, T_max=max_epochs, eta_min=1e-4)
        scaler_m4 = torch.amp.GradScaler('cuda', enabled=is_cuda)

        def fwd_train_m4(idx):
            # 1. Posterior pass
            p_rgb, p_sdf, p_mask, p_apex, mu, logvar = m4(tr_sdf[idx], tr_c[idx])
            l_sdf = F.l1_loss(p_sdf, tr_sdf[idx])
            p_mask_clamped = torch.clamp(p_mask, 1e-6, 1.0 - 1e-6)
            l_mask = F.binary_cross_entropy(p_mask_clamped, tr_mask[idx])
            l_apex = F.l1_loss(p_apex, tr_apx[idx])
            l_rgb = F.l1_loss(p_rgb, tr_img[idx]) + F.l1_loss(p_rgb * tr_mask[idx], tr_img[idx] * tr_mask[idx]) * 2.0
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            l_post = l_sdf + 2.0 * l_mask + 0.1 * l_apex + l_rgb + 0.005 * kl

            # 2. Prior pass (z=0) - Directly trains decoder to respond to c alone!
            pr_rgb, pr_sdf, pr_mask, pr_apex = m4.forward_prior(tr_c[idx])
            l_pr_sdf = F.l1_loss(pr_sdf, tr_sdf[idx])
            pr_mask_c = torch.clamp(pr_mask, 1e-6, 1.0 - 1e-6)
            l_pr_mask = F.binary_cross_entropy(pr_mask_c, tr_mask[idx])
            l_pr_rgb = F.l1_loss(pr_rgb, tr_img[idx])
            l_prior = l_pr_sdf + 2.0 * l_pr_mask + l_pr_rgb

            return l_post + 1.0 * l_prior

        def fwd_val_m4():
            pr_rgb, pr_sdf, pr_mask, _ = m4.forward_prior(val_c)
            pr_mask_c = torch.clamp(pr_mask, 1e-6, 1.0 - 1e-6)
            return F.l1_loss(pr_sdf, val_sdf) + 2.0 * F.binary_cross_entropy(pr_mask_c, val_mask) + F.l1_loss(pr_rgb, val_img)

        m4 = train_and_select_best("m4_geosdf_cvae", m4, opt_m4, sched_m4, scaler_m4, fwd_train_m4, fwd_val_m4)

        # =====================================================================
        # MODEL 5: Geometry-Conditioned cGAN
        # =====================================================================
        print(f"  [Fold {test_fold}] Training M5 cGAN...", flush=True)
        m5_g = Geometry_cGAN_Generator(6).to(device)
        m5_d = PatchGAN_Discriminator(in_channels=4).to(device)
        opt_g = torch.optim.AdamW(m5_g.parameters(), lr=2e-3, betas=(0.5, 0.999), weight_decay=1e-4)
        opt_d = torch.optim.AdamW(m5_d.parameters(), lr=1e-3, betas=(0.5, 0.999), weight_decay=1e-3)
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=max_epochs, eta_min=1e-4)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=max_epochs, eta_min=1e-4)
        scaler_g = torch.amp.GradScaler('cuda', enabled=is_cuda)
        scaler_d = torch.amp.GradScaler('cuda', enabled=is_cuda)

        best_g_val = float('inf')
        best_g_state = None
        patience_g = 0

        for ep in range(1, max_epochs + 1):
            m5_g.train()
            m5_d.train()
            perm = torch.randperm(len(tr_c))
            for b in range(n_batches):
                idx = perm[b*batch_size : (b+1)*batch_size]
                # Train D
                opt_d.zero_grad()
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    p_rgb, p_mask = m5_g(tr_c[idx])
                    d_real = m5_d(tr_mask[idx], tr_img[idx])
                    d_fake = m5_d(p_mask.detach(), p_rgb.detach())
                    loss_d = 0.5 * (torch.mean((d_real - 1.0)**2) + torch.mean(d_fake**2))
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

                # Train G
                opt_g.zero_grad()
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    d_fake_g = m5_d(p_mask, p_rgb)
                    loss_adv = torch.mean((d_fake_g - 1.0)**2)
                    loss_g_recon = F.l1_loss(p_rgb, tr_img[idx]) + F.l1_loss(p_rgb * tr_mask[idx], tr_img[idx] * tr_mask[idx]) * 2.5
                    loss_g_mask = F.l1_loss(p_mask, tr_mask[idx]) * 3.0
                    loss_g = loss_adv * 0.1 + loss_g_recon + loss_g_mask
                scaler_g.scale(loss_g).backward()
                scaler_g.step(opt_g)
                scaler_g.update()

            sched_g.step()
            sched_d.step()

            # Inner Validation
            m5_g.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    pv_rgb, pv_mask = m5_g(val_c)
                    val_g_loss = F.l1_loss(pv_rgb, val_img) + F.l1_loss(pv_mask, val_mask) * 2.0
                    v_val = val_g_loss.item()

            if v_val < best_g_val:
                best_g_val = v_val
                best_g_state = {k: v.cpu().clone() for k, v in m5_g.state_dict().items()}
                patience_g = 0
            else:
                patience_g += 1

            if patience_g >= patience:
                print(f"    [m5_cgan] Early stopping at epoch {ep} (Best Val: {best_g_val:.4f})")
                break

        m5_g.load_state_dict({k: v.to(device) for k, v in best_g_state.items()})
        m5_g.eval()
        torch.save(m5_g.state_dict(), os.path.join(FOLDS_CKPT_DIR, f"fold_{test_fold}_m5_geometry_cgan.pt"))

        # =====================================================================
        # MODEL 6: GF-WPG (GENUINE Geometry-First Weld Profile Generator)
        # =====================================================================
        print(f"  [Fold {test_fold}] Training M6 GF-WPG (Geometry-First 2-Stage)...", flush=True)
        m6 = GF_WPG(cond_dim=6, num_contour_pts=95).to(device)
        opt_m6 = torch.optim.AdamW(m6.parameters(), lr=3e-3, weight_decay=1e-4)
        sched_m6 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_m6, T_max=max_epochs, eta_min=1e-4)
        scaler_m6 = torch.amp.GradScaler('cuda', enabled=is_cuda)

        def fwd_train_m6(idx):
            p_rgb, p_mask, p_pts, p_apx = m6(tr_c[idx])
            # Stage 1: Explicit Geometry Loss
            l_geom = F.l1_loss(p_pts, tr_cnt[idx]) + F.l1_loss(p_apx, tr_apx[idx]) * 0.5
            l_mask = F.l1_loss(p_mask, tr_mask[idx]) * 2.0
            # Stage 2: Profile Rendering Loss
            l_render = F.l1_loss(p_rgb, tr_img[idx]) + F.l1_loss(p_rgb * tr_mask[idx], tr_img[idx] * tr_mask[idx]) * 2.0
            return l_geom + l_mask + l_render

        def fwd_val_m6():
            p_rgb, p_mask, p_pts, p_apx = m6(val_c)
            l_geom = F.l1_loss(p_pts, val_cnt) + F.l1_loss(p_apx, val_apx) * 0.5
            l_render = F.l1_loss(p_rgb, val_img)
            return l_geom + l_render

        m6 = train_and_select_best("m6_gf_wpg", m6, opt_m6, sched_m6, scaler_m6, fwd_train_m6, fwd_val_m6)

        # =====================================================================
        # BASELINE B1: PCAContourRegressor (Statistical Shape Model)
        # =====================================================================
        print(f"  [Fold {test_fold}] Fitting Baseline B1 (PCA-Contour + Ridge)...", flush=True)
        b1_pca = PCAContourRegressor(n_components=4, alpha=1.0)
        C_train_np = tr_c.cpu().numpy()
        Cnt_train_np = tr_cnt.cpu().numpy()
        b1_pca.fit(C_train_np, Cnt_train_np)

        # ---------------------------------------------------------------------
        # OUT-OF-FOLD TEST BENCHMARK (Strictly Held-Out Outer Test Fold)
        # ---------------------------------------------------------------------
        print(f"  [Fold {test_fold}] Evaluating on Sequestered Outer Test Set ({len(outer_test_df)} experiments)...", flush=True)

        for _, row in outer_test_df.iterrows():
            s_id = int(row['sample'])
            c_test_vec = torch.tensor([encode_c(row)], dtype=torch.float32, device=device)
            c_test_np = np.array([encode_c(row)])

            with torch.no_grad():
                # Inference on M1 - M6
                rgb_m1 = (np.clip(m1(c_test_vec).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                rgb_m2 = (np.clip(m2.generate(c_test_vec).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                rgb_m3 = (np.clip(m3(c_test_vec)[0].squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                
                pr_rgb4, pr_sdf4, pr_mask4, _ = m4.generate(c_test_vec)
                rgb_m4 = (np.clip(pr_rgb4.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                mask_m4 = (pr_mask4.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                pr_rgb5, pr_mask5 = m5_g(c_test_vec)
                rgb_m5 = (np.clip(pr_rgb5.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                mask_m5 = (pr_mask5.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                pr_rgb6, pr_mask6, pr_pts6, _ = m6.generate(c_test_vec)
                rgb_m6 = (np.clip(pr_rgb6.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
                mask_m6 = (pr_mask6.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                # B1 PCA Baseline
                mask_b1 = b1_pca.rasterize(c_test_np)[0]
                # Render simple baseline RGB from PCA mask
                rgb_b1 = np.full((128, 128, 3), 160, dtype=np.uint8)
                rgb_b1[mask_b1 > 0] = [80, 80, 80]
                rgb_b1[:, 64:] = [120, 120, 120]

            # Extract contours from generated images independently
            cnt_m1 = extract_robust_crown_contour(cv2.cvtColor(rgb_m1, cv2.COLOR_RGB2GRAY))
            cnt_m2 = extract_robust_crown_contour(cv2.cvtColor(rgb_m2, cv2.COLOR_RGB2GRAY))
            cnt_m3 = extract_robust_crown_contour(cv2.cvtColor(rgb_m3, cv2.COLOR_RGB2GRAY))
            cnt_m4 = extract_robust_crown_contour(cv2.cvtColor(rgb_m4, cv2.COLOR_RGB2GRAY))
            cnt_m5 = extract_robust_crown_contour(cv2.cvtColor(rgb_m5, cv2.COLOR_RGB2GRAY))
            cnt_m6 = extract_robust_crown_contour(cv2.cvtColor(rgb_m6, cv2.COLOR_RGB2GRAY))

            # B1 contour directly from PCA prediction
            pts_b1_arr = b1_pca.predict_contour(c_test_np)[0]
            cnt_b1 = {
                "pts": {18 + i: int(round(x)) for i, x in enumerate(pts_b1_arr)},
                "apex_x": float(np.min(pts_b1_arr)),
                "apex_y": float(18 + np.argmin(pts_b1_arr)),
                "reinf_proj_px": float(64.0 - np.min(pts_b1_arr))
            }

            model_eval_dict = {
                "M1_DirectCNN": (rgb_m1, cnt_m1.get("mask", np.zeros((128,128))), cnt_m1),
                "M2_CVAE": (rgb_m2, cnt_m2.get("mask", np.zeros((128,128))), cnt_m2),
                "M3_GCWPGv2": (rgb_m3, cnt_m3.get("mask", np.zeros((128,128))), cnt_m3),
                "M4_GeoSDF_CVAE": (rgb_m4, mask_m4, cnt_m4),
                "M5_cGAN": (rgb_m5, mask_m5, cnt_m5),
                "M6_GF_WPG": (rgb_m6, mask_m6, cnt_m6),
                "B1_PCA_Baseline": (rgb_b1, mask_b1, cnt_b1)
            }

            # Evaluate against BOTH View A and View B
            for view in ["A", "B"]:
                real_d = data_cache[s_id][view]
                real_rgb = real_d["rgb"]
                real_mask = real_d["mask"]
                real_cnt = real_d["c_info"]

                for m_name, (g_rgb, g_mask, g_cnt) in model_eval_dict.items():
                    metrics = compute_comprehensive_metrics(real_rgb, g_rgb, real_mask, g_mask, real_cnt, g_cnt)
                    oof_sample_records.append({
                        "test_fold": test_fold,
                        "sample": s_id,
                        "view": view,
                        "model": m_name,
                        "material": row['material'],
                        "process": row['process'],
                        "feed_rate": row['feed_rate'],
                        "speed": row['speed'],
                        **metrics
                    })

        t_fold_end = time.time()
        print(f"  [Fold {test_fold} Complete] Duration: {t_fold_end - t_fold_start:.2f} s", flush=True)

    # -------------------------------------------------------------------------
    # COMPILE OUT-OF-FOLD METRICS TABLE WITH 1000 BOOTSTRAP RESAMPLES
    # -------------------------------------------------------------------------
    df_oof = pd.DataFrame(oof_sample_records)
    all_sample_csv = os.path.join(METRICS_DIR, "ALL_SAMPLE_OOF_PREDICTIONS.csv")
    df_oof.to_csv(all_sample_csv, index=False)
    print(f"\nSaved all out-of-fold predictions to: {all_sample_csv}")

    print("\nComputing 1,000 Bootstrap 95% Confidence Intervals...", flush=True)
    summary_rows = []
    models_list = ["M1_DirectCNN", "M2_CVAE", "M3_GCWPGv2", "M4_GeoSDF_CVAE", "M5_cGAN", "M6_GF_WPG", "B1_PCA_Baseline"]

    for m_name in models_list:
        sub = df_oof[df_oof['model'] == m_name]
        
        def bootstrap_ci(arr_values, n_boot=1000, ci=95):
            vals = np.array(arr_values)
            boots = []
            for _ in range(n_boot):
                sample_idx = np.random.choice(len(vals), size=len(vals), replace=True)
                boots.append(np.mean(vals[sample_idx]))
            low = np.percentile(boots, (100 - ci) / 2)
            high = np.percentile(boots, 100 - (100 - ci) / 2)
            return float(np.mean(vals)), float(low), float(high)

        c_mean, c_low, c_high = bootstrap_ci(sub['contour_mae_px'])
        s_mean, s_low, s_high = bootstrap_ci(sub['ssim'])
        d_mean, d_low, d_high = bootstrap_ci(sub['mask_dice'])
        p_mean, _, _ = bootstrap_ci(sub['psnr'])
        f1_mean, _, _ = bootstrap_ci(sub['boundary_f1'])
        apx_mean, _, _ = bootstrap_ci(sub['apex_error_px'])
        iou_mean, _, _ = bootstrap_ci(sub['mask_iou'])
        area_mean, _, _ = bootstrap_ci(sub['area_error_px2'])

        fail_rate = float(np.mean(sub['contour_mae_px'] >= 30.0) * 100.0)

        summary_rows.append({
            "Architecture": m_name,
            "Contour_MAE_px": round(c_mean, 2),
            "Contour_MAE_95CI": f"[{c_low:.2f}, {c_high:.2f}]",
            "Boundary_F1_2px": round(f1_mean, 4),
            "Apex_Error_px": round(apx_mean, 2),
            "Mask_Dice": round(d_mean, 4),
            "Mask_Dice_95CI": f"[{d_low:.4f}, {d_high:.4f}]",
            "Mask_IoU": round(iou_mean, 4),
            "Area_Error_px2": round(area_mean, 1),
            "Whole_SSIM": round(s_mean, 4),
            "Whole_SSIM_95CI": f"[{s_low:.4f}, {s_high:.4f}]",
            "PSNR_dB": round(p_mean, 2),
            "Failure_Rate_pct": round(fail_rate, 1)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.sort_values(by="Contour_MAE_px", inplace=True)
    ranking_csv = os.path.join(METRICS_DIR, "OOF_MODEL_RANKING.csv")
    df_summary.to_csv(ranking_csv, index=False)
    print(f"Saved definitive out-of-fold model ranking to: {ranking_csv}")
    print("\n" + "=" * 80)
    print("PHASE V10 A100 OUT-OF-FOLD BENCHMARK RESULTS")
    print("=" * 80)
    print(df_summary.to_string(index=False), flush=True)

    t_total = time.time() - t_start
    print(f"\nTotal 5-Fold Grouped Cross-Validation completed in {t_total:.2f} s ({t_total/60:.2f} min)", flush=True)

if __name__ == "__main__":
    run_a100_grouped_cv(max_epochs=80, patience=20)
