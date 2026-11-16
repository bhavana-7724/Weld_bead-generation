"""
train_grouped_cv_v13.py

Phase V13 Stage 1: Experiment-Grouped 5-Fold Cross-Validation Benchmark
Target System: Single NVIDIA A100 GPU (CUDA_VISIBLE_DEVICES=0) / Fallback CPU
Model: GF-WPG v2 (Enhanced Geometry-First Weld Profile Generator v2)

Methodological Standards:
1. Zero Data Leakage:
   - Outer test fold (7-8 samples) strictly sequestered.
   - Dual views (A and B) of each specimen always assigned to the same fold.
   - Feature scalers fit strictly on the outer training fold.
2. Inner-Validation Early Stopping:
   - Outer train split into inner_train (80%) and inner_val (20%), grouped by sample.
   - Validation loss and contour MAE tracked every epoch.
   - Patience = 20 epochs; best checkpoint saved per fold.
3. Detailed Metrics & Confidence Intervals:
   - Out-of-fold Contour MAE (px), Boundary F1 (@2px), Mask Dice & IoU.
   - Apex error (px), Toe top & bottom errors (px), SSIM, and PSNR.
   - 1,000-iteration bootstrap 95% confidence intervals.
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
from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import torch
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v13.models_v13 import GF_WPG_v2, GF_WPG_Loss
from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")

AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
GEN36_FALLBACK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v13_a100")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FOLDS_CKPT_DIR = os.path.join(CKPT_DIR, "folds")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CURVES_DIR = os.path.join(METRICS_DIR, "learning_curves")

for d in [FOLDS_CKPT_DIR, METRICS_DIR, CURVES_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================================
# Evaluation Metric Functions
# =========================================================================
def evaluate_sample_predictions(real_rgb, gen_rgb, real_mask, gen_mask, real_cnt, pred_pts_dict, pred_apex, pred_toes):
    """
    Computes rigorous physical and image metrics between real and predicted weld cross-sections.
    All contour and landmark distances are strictly in PIXELS.
    """
    # 1. Photometric
    ssim = float(ssim_func(real_rgb, gen_rgb, channel_axis=2, data_range=255))
    psnr = float(psnr_func(real_rgb, gen_rgb, data_range=255))

    # ROI SSIM (deposition half, x < 64)
    roi_real = real_rgb[:, :64, :].copy()
    roi_gen = gen_rgb[:, :64, :].copy()
    roi_ssim = float(ssim_func(roi_real, roi_gen, channel_axis=2, data_range=255))

    # 2. Contour Overlap MAE
    r_pts = real_cnt.get("pts", {})
    y_common = sorted(list(set(r_pts.keys()).intersection(set(pred_pts_dict.keys()))))
    if len(y_common) >= 5:
        diffs = [abs(r_pts[y] - pred_pts_dict[y]) for y in y_common]
        contour_mae_px = float(np.mean(diffs))
    else:
        contour_mae_px = 35.0

    # 3. Apex Error
    real_apex_x = real_cnt.get("apex_x", 64.0)
    real_apex_y = real_cnt.get("apex_y", 64.0)
    apex_error_px = float(np.sqrt((real_apex_x - pred_apex[0]) ** 2 + (real_apex_y - pred_apex[1]) ** 2))

    # 4. Toe Errors
    if len(r_pts) > 0:
        real_toe_top = min(r_pts.keys())
        real_toe_bot = max(r_pts.keys())
    else:
        real_toe_top = 38.0
        real_toe_bot = 98.0
    toe_top_err_px = float(abs(real_toe_top - pred_toes[0]))
    toe_bot_err_px = float(abs(real_toe_bot - pred_toes[1]))
    mean_toe_err_px = float((toe_top_err_px + toe_bot_err_px) / 2.0)

    # 5. Mask Dice and IoU
    inter = np.sum((real_mask > 0) & (gen_mask > 0))
    union = np.sum((real_mask > 0) | (gen_mask > 0))
    iou = float(inter / (union + 1e-8))
    dice = float(2.0 * inter / (np.sum(real_mask > 0) + np.sum(gen_mask > 0) + 1e-8))

    # 6. Boundary F1 Score at d_tol = 2 px
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


# =========================================================================
# Main Cross-Validation Routine
# =========================================================================
def run_grouped_cv(max_epochs=80, patience=20, smoke_test=False, target_device=None):
    t_start = time.time()
    if target_device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(target_device)
    is_cuda = device.type == "cuda"

    print("=" * 80)
    print("PHASE V13 [STAGE 1]: EXPERIMENT-GROUPED 5-FOLD CROSS-VALIDATION")
    print(f"Device:           {device}")
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Hardware:     {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
        print(f"CUDA Version:     {torch.version.cuda}")
    print(f"Max Epochs:       {max_epochs} | Early Stopping Patience: {patience}")
    print(f"Smoke Test Mode:  {smoke_test}")
    print("=" * 80)

    # Load master dataset
    df_clean = pd.read_csv(CLEAN_CSV)

    # Stratified 5-Fold Grouped Assignment (Views A & B stay together)
    folds = {}
    groups = df_clean.groupby(['material', 'process'], sort=False)
    for g_idx, (g_name, g_df) in enumerate(groups):
        samples = g_df['sample'].tolist()
        for idx, s in enumerate(samples):
            f = (idx + g_idx) % 5
            folds[s] = f
    df_clean['fold'] = df_clean['sample'].map(folds)

    fold_csv_path = os.path.join(METRICS_DIR, "FOLD_ASSIGNMENTS.csv")
    df_clean[['sample', 'material', 'process', 'feed_rate', 'speed', 'fold']].to_csv(fold_csv_path, index=False)
    print(f"Fold assignments saved to: {fold_csv_path}")

    # Preload all 72 experimental images, masks, and ground-truth geometry
    print("\nPreloading 72 macrograph cross-sections and GT geometry arrays into memory...")
    data_cache = {}
    for s in range(1, 37):
        data_cache[s] = {}
        for view in ["A", "B"]:
            p1 = os.path.join(PROCESSED_DIR, f"{s}-{view}.png")
            if not os.path.exists(p1):
                p1 = os.path.join(GEN36_FALLBACK_DIR, f"S{s:02d}", f"Real_{view}.png")
            bgr = cv2.imread(p1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            mask_path = os.path.join(MASKS_DIR, f"{s}-{view}.png")
            if os.path.exists(mask_path):
                mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
            else:
                mask = np.zeros((128, 128), dtype=np.uint8)

            c_info = extract_robust_crown_contour(gray, x_baseline=64)
            pts_dict = c_info.get("pts", {})

            # 95 scanline array for y in [18, 112]
            pts_arr = np.zeros(95, dtype=np.float32)
            for idx in range(95):
                y = 18 + idx
                pts_arr[idx] = float(pts_dict.get(y, 64.0))

            apex_coord = np.array([float(c_info.get("apex_x", 64.0)), float(c_info.get("apex_y", 64.0))], dtype=np.float32)
            
            if len(pts_dict) > 0:
                toe_coord = np.array([float(min(pts_dict.keys())), float(max(pts_dict.keys()))], dtype=np.float32)
            else:
                toe_coord = np.array([38.0, 98.0], dtype=np.float32)

            data_cache[s][view] = {
                "rgb": rgb,
                "mask": mask,
                "c_info": c_info,
                "pts_arr": pts_arr,
                "apex_coord": apex_coord,
                "toe_coord": toe_coord
            }

    oof_records = []
    num_folds_to_run = 1 if smoke_test else 5

    # Cross-Validation Loop
    for test_fold in range(num_folds_to_run):
        t_fold_start = time.time()
        print(f"\n" + "-" * 75)
        print(f">>> RUNNING OUTER FOLD {test_fold} / 4 (Grouped Outer Validation)")
        print("-" * 75)

        outer_train_df = df_clean[df_clean['fold'] != test_fold].copy()
        outer_test_df = df_clean[df_clean['fold'] == test_fold].copy()

        # Strict feature scalers on outer training fold
        f_min, f_max = float(outer_train_df['feed_rate'].min()), float(outer_train_df['feed_rate'].max())
        s_min, s_max = float(outer_train_df['speed'].min()), float(outer_train_df['speed'].max())

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

        # Inner-Validation Split (80% inner_train, 20% inner_val grouped by sample)
        train_samples = sorted(outer_train_df['sample'].unique().tolist())
        np.random.seed(42 + test_fold)
        shuffled = np.random.permutation(train_samples)
        n_val = max(1, int(len(train_samples) * 0.20))
        inner_val_samples = sorted(shuffled[:n_val].tolist())
        inner_train_samples = sorted(shuffled[n_val:].tolist())

        print(f"  Outer Train: {len(train_samples)} samples | Outer Test: {len(outer_test_df)} samples")
        print(f"  Inner Train: {len(inner_train_samples)} samples | Inner Val: {len(inner_val_samples)} samples")

        # Build Inner Train Tensors
        it_c, it_rgb, it_mask, it_pts, it_apex, it_toes = [], [], [], [], [], []
        for s in inner_train_samples:
            row = df_clean[df_clean['sample'] == s].iloc[0]
            c_vec = encode_condition(row)
            for v in ["A", "B"]:
                d = data_cache[s][v]
                it_c.append(c_vec)
                it_rgb.append(d["rgb"].transpose(2, 0, 1) / 255.0)
                it_mask.append(d["mask"][np.newaxis, :, :].astype(np.float32))
                it_pts.append(d["pts_arr"])
                it_apex.append(d["apex_coord"])
                it_toes.append(d["toe_coord"])

        it_c_t = torch.tensor(np.array(it_c), dtype=torch.float32, device=device)
        it_rgb_t = torch.tensor(np.array(it_rgb), dtype=torch.float32, device=device)
        it_mask_t = torch.tensor(np.array(it_mask), dtype=torch.float32, device=device)
        it_pts_t = torch.tensor(np.array(it_pts), dtype=torch.float32, device=device)
        it_apex_t = torch.tensor(np.array(it_apex), dtype=torch.float32, device=device)
        it_toes_t = torch.tensor(np.array(it_toes), dtype=torch.float32, device=device)

        # Build Inner Val Tensors
        iv_c, iv_rgb, iv_mask, iv_pts, iv_apex, iv_toes = [], [], [], [], [], []
        for s in inner_val_samples:
            row = df_clean[df_clean['sample'] == s].iloc[0]
            c_vec = encode_condition(row)
            for v in ["A", "B"]:
                d = data_cache[s][v]
                iv_c.append(c_vec)
                iv_rgb.append(d["rgb"].transpose(2, 0, 1) / 255.0)
                iv_mask.append(d["mask"][np.newaxis, :, :].astype(np.float32))
                iv_pts.append(d["pts_arr"])
                iv_apex.append(d["apex_coord"])
                iv_toes.append(d["toe_coord"])

        iv_c_t = torch.tensor(np.array(iv_c), dtype=torch.float32, device=device)
        iv_rgb_t = torch.tensor(np.array(iv_rgb), dtype=torch.float32, device=device)
        iv_mask_t = torch.tensor(np.array(iv_mask), dtype=torch.float32, device=device)
        iv_pts_t = torch.tensor(np.array(iv_pts), dtype=torch.float32, device=device)
        iv_apex_t = torch.tensor(np.array(iv_apex), dtype=torch.float32, device=device)
        iv_toes_t = torch.tensor(np.array(iv_toes), dtype=torch.float32, device=device)

        # Initialize GF-WPG v2
        model = GF_WPG_v2(cond_dim=6, num_contour_pts=95).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4)
        criterion = GF_WPG_Loss().to(device)
        scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

        best_val_mae = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_state_dict = None
        epoch_history = []

        batch_size = 8
        n_batches = int(np.ceil(len(it_c_t) / batch_size))

        # Training Loop
        epochs_to_run = 1 if smoke_test else max_epochs
        for epoch in range(1, epochs_to_run + 1):
            model.train()
            perm = torch.randperm(len(it_c_t))
            train_loss_epoch = 0.0

            for b in range(n_batches):
                idx = perm[b * batch_size : (b + 1) * batch_size]
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    p_rgb, p_mask, p_pts, p_apex, p_toes = model(it_c_t[idx])
                    loss, l_dict = criterion(
                        p_rgb, p_mask, p_pts, p_apex, p_toes,
                        it_rgb_t[idx], it_mask_t[idx], it_pts_t[idx], it_apex_t[idx], it_toes_t[idx]
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_epoch += loss.item()

            train_loss_epoch /= n_batches
            scheduler.step()

            # Validation Step on Inner Val Set
            model.eval()
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=is_cuda):
                    v_rgb, v_mask, v_pts, v_apex, v_toes = model(iv_c_t)
                    v_loss, v_ldict = criterion(
                        v_rgb, v_mask, v_pts, v_apex, v_toes,
                        iv_rgb_t, iv_mask_t, iv_pts_t, iv_apex_t, iv_toes_t
                    )
                # Compute inner val contour MAE
                val_mae = float(F.l1_loss(v_pts, iv_pts_t).item())

            epoch_rec = {
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "val_loss": v_loss.item(),
                "val_mae_px": val_mae,
                "pts_loss": v_ldict["pts"],
                "apex_loss": v_ldict["apex"],
                "toe_loss": v_ldict["toe"],
                "mask_loss": v_ldict["mask"],
                "rgb_loss": v_ldict["rgb"],
                "ssim_loss": v_ldict["ssim"]
            }
            epoch_history.append(epoch_rec)

            if epoch % 10 == 0 or epoch == 1 or epoch == epochs_to_run:
                print(f"  Ep {epoch:02d}/{epochs_to_run:02d} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {v_loss.item():.4f} | Val MAE: {val_mae:.2f} px")

            # Early Stopping Check
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
                patience_counter = 0
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience and not smoke_test:
                    print(f"  [EARLY STOPPING] Patience {patience} reached at epoch {epoch}. Best epoch was {best_epoch} (Val MAE: {best_val_mae:.2f} px).")
                    break

        # Save Best Checkpoint for this Fold
        fold_ckpt_path = os.path.join(FOLDS_CKPT_DIR, f"gfwpg_v2_fold_{test_fold}.pt")
        if best_state_dict is not None:
            torch.save(best_state_dict, fold_ckpt_path)
            model.load_state_dict(best_state_dict)
        else:
            torch.save(model.state_dict(), fold_ckpt_path)

        # Save Learning Curve CSV
        df_curve = pd.DataFrame(epoch_history)
        curve_csv_path = os.path.join(CURVES_DIR, f"fold_{test_fold}_gfwpg_v2_curve.csv")
        df_curve.to_csv(curve_csv_path, index=False)

        # Plot Learning Curve
        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax1.plot(df_curve['epoch'], df_curve['train_loss'], label='Train Total Loss', color='#2563EB', linewidth=1.8)
        ax1.plot(df_curve['epoch'], df_curve['val_loss'], label='Val Total Loss', color='#D97706', linewidth=1.8, linestyle='--')
        ax1.set_xlabel('Epoch', fontweight='bold')
        ax1.set_ylabel('Loss', fontweight='bold', color='#1E293B')
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(df_curve['epoch'], df_curve['val_mae_px'], label='Val Contour MAE (px)', color='#DC2626', linewidth=1.8)
        ax2.set_ylabel('Contour MAE (px)', fontweight='bold', color='#DC2626')
        ax2.axvline(best_epoch, color='#059669', linestyle=':', label=f'Best Ep {best_epoch}')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)
        plt.title(f"GF-WPG v2 Training Curves — Fold {test_fold} (Best MAE: {best_val_mae:.2f} px)", fontweight='bold')
        plt.tight_layout()
        curve_png_path = os.path.join(CURVES_DIR, f"fold_{test_fold}_loss_curve.png")
        plt.savefig(curve_png_path)
        plt.close()

        # =====================================================================
        # Out-of-Fold Evaluation on Strictly Sequestered Outer Test Fold
        # =====================================================================
        print(f"  Evaluating Fold {test_fold} on sequestered outer test set ({len(outer_test_df)} samples)...")
        model.eval()

        for _, row in outer_test_df.iterrows():
            s_id = int(row['sample'])
            c_vec = encode_condition(row)
            c_t = torch.tensor([c_vec], dtype=torch.float32, device=device)

            with torch.no_grad():
                pred_rgb_t, pred_mask_t, pred_pts_t, pred_apex_t, pred_toes_t = model.generate(c_t)

            pred_rgb_np = (np.clip(pred_rgb_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype(np.uint8)
            pred_mask_np = (pred_mask_t.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            pred_pts_np = pred_pts_t.squeeze().cpu().numpy()
            pred_apex_np = pred_apex_t.squeeze().cpu().numpy()
            pred_toes_np = pred_toes_t.squeeze().cpu().numpy()

            # Format pts dict for evaluation
            pred_pts_dict = {18 + idx: pred_pts_np[idx] for idx in range(95)}

            # Evaluate against View A and View B
            for view in ["A", "B"]:
                real_d = data_cache[s_id][view]
                metrics = evaluate_sample_predictions(
                    real_d["rgb"], pred_rgb_np,
                    real_d["mask"], pred_mask_np,
                    real_d["c_info"],
                    pred_pts_dict, pred_apex_np, pred_toes_np
                )

                rec = {
                    "fold": test_fold,
                    "sample": s_id,
                    "sample_code": f"S{s_id:02d}",
                    "view": view,
                    "material": row['material'],
                    "process": row['process'],
                    "feed_rate": row['feed_rate'],
                    "speed": row['speed'],
                    **metrics
                }
                oof_records.append(rec)

        t_fold_end = time.time()
        print(f"[FOLD {test_fold} COMPLETE] Runtime: {t_fold_end - t_fold_start:.1f} s")

    # =========================================================================
    # Compile and Aggregate Out-of-Fold Results
    # =========================================================================
    df_oof = pd.DataFrame(oof_records)
    oof_csv_path = os.path.join(METRICS_DIR, "cv_metrics.csv")
    df_oof.to_csv(oof_csv_path, index=False)
    print(f"\nDetailed 72-record CV metrics saved to: {oof_csv_path}")

    # Dual-View Sample Averaging
    numeric_cols = [
        "contour_mae_px", "boundary_f1", "mask_dice", "mask_iou",
        "apex_error_px", "toe_top_err_px", "toe_bot_err_px", "mean_toe_err_px",
        "ssim", "psnr", "roi_ssim"
    ]
    df_sample_avg = df_oof.groupby(
        ["sample", "sample_code", "material", "process", "feed_rate", "speed", "fold"]
    )[numeric_cols].mean().reset_index()

    sample_avg_csv = os.path.join(METRICS_DIR, "cv_sample_metrics.csv")
    df_sample_avg.to_csv(sample_avg_csv, index=False)
    print(f"Sample-averaged CV metrics saved to: {sample_avg_csv}")

    # Summary Statistics with 1,000-Iteration Bootstrap 95% CIs
    print("\nComputing 1,000-iteration bootstrap 95% confidence intervals...")
    summary_rows = []
    np.random.seed(42)

    for col in numeric_cols:
        vals = df_sample_avg[col].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))

        # Bootstrap
        boot_means = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(1000)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        summary_rows.append({
            "metric": col,
            "mean": mean_val,
            "std": std_val,
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "formatted": f"{mean_val:.2f} +/- {std_val:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]"
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(METRICS_DIR, "cv_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Summary metrics table saved to: {summary_csv_path}")

    # Console Display
    print("\n" + "=" * 80)
    print("PHASE V13: GF-WPG v2 OUT-OF-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Metric':<20} | {'Mean +/- Std':<18} | {'95% CI':<20}")
    print("-" * 65)
    for _, r in df_summary.iterrows():
        print(f"{r['metric']:<20} | {r['mean']:>6.2f} +/- {r['std']:<6.2f}    | [{r['ci_lower_95']:>6.2f}, {r['ci_upper_95']:>6.2f}]")
    print("=" * 80)

    t_total = time.time() - t_start
    print(f"Total Stage 1 execution time: {t_total:.1f} s ({t_total/60:.2f} min)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase V13 Grouped 5-Fold Cross-Validation")
    parser.add_argument("--max-epochs", type=int, default=80, help="Maximum epochs per fold (default: 80)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 fold, 1 epoch test")
    parser.add_argument("--device", type=str, default=None, help="Target device (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    sys.exit(run_grouped_cv(
        max_epochs=args.max_epochs,
        patience=args.patience,
        smoke_test=args.smoke_test,
        target_device=args.device
    ))
