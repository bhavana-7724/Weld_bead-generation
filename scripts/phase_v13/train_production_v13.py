"""
train_production_v13.py

Phase V13 Stage 2: Full Dataset Production Training of GF-WPG v2
Target System: Single NVIDIA A100 GPU (CUDA_VISIBLE_DEVICES=0) / Fallback CPU
Model: Enhanced Geometry-First Weld Profile Generator v2 (GF-WPG v2)

Trains on all 36 physical welding experiments (all 72 views) to produce the
canonical production model for 36-sample synthesis and unseen parameter inference.

Saves:
- Production checkpoint: results/phase_v13_a100/checkpoints/gfwpg_v2_final.pt
- Normalization metadata: results/phase_v13_a100/checkpoints/normalization_params.json
- Training curve CSV: results/phase_v13_a100/metrics/learning_curves/production_gfwpg_v2_curve.csv
- Loss curve plot: results/phase_v13_a100/metrics/learning_curves/production_loss_curve.png
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
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CURVES_DIR = os.path.join(METRICS_DIR, "learning_curves")

for d in [CKPT_DIR, METRICS_DIR, CURVES_DIR]:
    os.makedirs(d, exist_ok=True)


def run_production_training(epochs=80, smoke_test=False, target_device=None):
    t_start = time.time()
    if target_device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(target_device)
    is_cuda = device.type == "cuda"

    print("=" * 80)
    print("PHASE V13 [STAGE 2]: FULL PRODUCTION TRAINING OF GF-WPG v2")
    print(f"Device:          {device}")
    if is_cuda:
        props = torch.cuda.get_device_properties(0)
        print(f"GPU Hardware:    {props.name} ({props.total_memory / (1024**3):.2f} GB VRAM)")
    print(f"Target Epochs:   {epochs}")
    print(f"Smoke Test Mode: {smoke_test}")
    print("=" * 80)

    df_clean = pd.read_csv(CLEAN_CSV)
    f_min = float(df_clean['feed_rate'].min())
    f_max = float(df_clean['feed_rate'].max())
    s_min = float(df_clean['speed'].min())
    s_max = float(df_clean['speed'].max())

    norm_params = {
        "f_min": f_min,
        "f_max": f_max,
        "s_min": s_min,
        "s_max": s_max,
        "materials": ["IN718", "SS304L"],
        "processes": ["ColdArc", "GMAW"],
        "num_samples": len(df_clean),
        "target_image_size": [128, 128],
        "substrate_baseline_x": 64,
        "scanline_y_range": [18, 112]
    }
    norm_path = os.path.join(CKPT_DIR, "normalization_params.json")
    with open(norm_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    print(f"Normalization parameters saved to: {norm_path}")

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

    # Preload and build training tensors
    print("Building full 72-view training dataset...")
    all_c, all_rgb, all_mask, all_pts, all_apex, all_toes = [], [], [], [], [], []

    for _, row in df_clean.iterrows():
        s_id = int(row['sample'])
        c_vec = encode_condition(row)

        for view in ["A", "B"]:
            p1 = os.path.join(PROCESSED_DIR, f"{s_id}-{view}.png")
            if not os.path.exists(p1):
                p1 = os.path.join(GEN36_FALLBACK_DIR, f"S{s_id:02d}", f"Real_{view}.png")

            bgr = cv2.imread(p1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            mask_path = os.path.join(MASKS_DIR, f"{s_id}-{view}.png")
            if os.path.exists(mask_path):
                mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128).astype(np.uint8)
            else:
                mask = np.zeros((128, 128), dtype=np.uint8)

            c_info = extract_robust_crown_contour(gray, x_baseline=64)
            pts_dict = c_info.get("pts", {})

            pts_arr = np.zeros(95, dtype=np.float32)
            for idx in range(95):
                y = 18 + idx
                pts_arr[idx] = float(pts_dict.get(y, 64.0))

            apex_coord = np.array([float(c_info.get("apex_x", 64.0)), float(c_info.get("apex_y", 64.0))], dtype=np.float32)

            if len(pts_dict) > 0:
                toe_coord = np.array([float(min(pts_dict.keys())), float(max(pts_dict.keys()))], dtype=np.float32)
            else:
                toe_coord = np.array([38.0, 98.0], dtype=np.float32)

            all_c.append(c_vec)
            all_rgb.append(rgb.transpose(2, 0, 1) / 255.0)
            all_mask.append(mask[np.newaxis, :, :].astype(np.float32))
            all_pts.append(pts_arr)
            all_apex.append(apex_coord)
            all_toes.append(toe_coord)

    c_tensor = torch.tensor(np.array(all_c), dtype=torch.float32, device=device)
    rgb_tensor = torch.tensor(np.array(all_rgb), dtype=torch.float32, device=device)
    mask_tensor = torch.tensor(np.array(all_mask), dtype=torch.float32, device=device)
    pts_tensor = torch.tensor(np.array(all_pts), dtype=torch.float32, device=device)
    apex_tensor = torch.tensor(np.array(all_apex), dtype=torch.float32, device=device)
    toes_tensor = torch.tensor(np.array(all_toes), dtype=torch.float32, device=device)

    print(f"Loaded {len(c_tensor)} training views (36 experiments x 2 views).")

    # Initialize model
    model = GF_WPG_v2(cond_dim=6, num_contour_pts=95).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    epochs_to_run = 1 if smoke_test else epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_to_run, eta_min=1e-4)
    criterion = GF_WPG_Loss().to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    batch_size = 8
    n_batches = int(np.ceil(len(c_tensor) / batch_size))
    training_history = []

    print("\nStarting production model optimization...")
    model.train()
    for ep in range(1, epochs_to_run + 1):
        perm = torch.randperm(len(c_tensor))
        ep_loss = 0.0
        ep_dict = {"pts": 0.0, "apex": 0.0, "toe": 0.0, "mask": 0.0, "rgb": 0.0, "crown": 0.0, "ssim": 0.0, "grad": 0.0}

        for b in range(n_batches):
            idx = perm[b * batch_size : (b + 1) * batch_size]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=is_cuda):
                p_rgb, p_mask, p_pts, p_apex, p_toes = model(c_tensor[idx])
                loss, l_dict = criterion(
                    p_rgb, p_mask, p_pts, p_apex, p_toes,
                    rgb_tensor[idx], mask_tensor[idx], pts_tensor[idx], apex_tensor[idx], toes_tensor[idx]
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ep_loss += loss.item()
            for k in ep_dict:
                ep_dict[k] += l_dict[k]

        scheduler.step()
        ep_loss /= n_batches
        for k in ep_dict:
            ep_dict[k] /= n_batches

        rec = {
            "epoch": ep,
            "total_loss": ep_loss,
            **ep_dict,
            "lr": float(scheduler.get_last_lr()[0])
        }
        training_history.append(rec)

        if ep % 10 == 0 or ep == 1 or ep == epochs_to_run:
            print(f"  Ep {ep:02d}/{epochs_to_run:02d} | Loss: {ep_loss:.4f} | Pts L1: {ep_dict['pts']:.2f} px | Mask: {ep_dict['mask']:.3f} | RGB L1: {ep_dict['rgb']:.3f} | SSIM: {1.0 - ep_dict['ssim']:.3f}")

    # Save final model
    model.eval()
    final_ckpt_path = os.path.join(CKPT_DIR, "gfwpg_v2_final.pt")
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"\nFinal production checkpoint saved to: {final_ckpt_path}")

    # Save Training Curve CSV
    df_curve = pd.DataFrame(training_history)
    curve_csv_path = os.path.join(CURVES_DIR, "production_gfwpg_v2_curve.csv")
    df_curve.to_csv(curve_csv_path, index=False)
    print(f"Production training curves saved to: {curve_csv_path}")

    # Plot Loss Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Left: Total Loss & Sub-losses
    ax1.plot(df_curve['epoch'], df_curve['total_loss'], label='Total Multi-Objective Loss', color='#1E293B', linewidth=2.0)
    ax1.plot(df_curve['epoch'], df_curve['pts'], label='Geometric Contour Loss (px)', color='#2563EB', linewidth=1.5)
    ax1.plot(df_curve['epoch'], df_curve['mask'], label='Mask BCE+Dice Loss', color='#059669', linewidth=1.5)
    ax1.plot(df_curve['epoch'], df_curve['rgb'], label='RGB Photometric L1', color='#D97706', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss Value', fontweight='bold')
    ax1.set_title('Production Training Losses', fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)

    # Right: Geometric Landmark Convergence
    ax2.plot(df_curve['epoch'], df_curve['pts'], label='Contour Scanline L1 (px)', color='#2563EB', linewidth=1.8)
    ax2.plot(df_curve['epoch'], df_curve['apex'], label='Apex L1 (px)', color='#7C3AED', linewidth=1.5)
    ax2.plot(df_curve['epoch'], df_curve['toe'], label='Toe Attachment L1 (px)', color='#DC2626', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Error (pixels)', fontweight='bold')
    ax2.set_title('Geometric Landmark Convergence', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('GF-WPG v2 Production Optimization on Full 36-Experiment Dataset', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    curve_png_path = os.path.join(CURVES_DIR, "production_loss_curve.png")
    plt.savefig(curve_png_path)
    plt.close()
    print(f"Production loss figure saved to: {curve_png_path}")

    t_total = time.time() - t_start
    print(f"Production training completed in {t_total:.1f} s ({t_total/60:.2f} min)")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase V13 Production Training")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (default: 80)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch test")
    parser.add_argument("--device", type=str, default=None, help="Target device (e.g. cuda:0, cpu)")
    args = parser.parse_args()

    sys.exit(run_production_training(
        epochs=args.epochs,
        smoke_test=args.smoke_test,
        target_device=args.device
    ))
