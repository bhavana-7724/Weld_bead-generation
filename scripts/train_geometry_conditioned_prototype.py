"""
train_geometry_conditioned_prototype.py

Implements Task 4: Lightweight Geometry-Conditioned Image Decoder Prototype (GC-WPG).
Architecture Flow:
  Welding Setpoints c in R^6
      │
      ├─► Geometry Estimator MLP ──► Predicted [W_pred, R_pred]
      │                                   │
      │                                   ▼
      │                             Spatial Geometry Prior Map G in R^(2 x 128 x 128)
      │                             (Channel 0: Smooth Crown Prior x < 64)
      │                             (Channel 1: Plate Substrate Prior x >= 64)
      │                                   │
      └─► Latent Condition MLP ──► ResUp Decoder (8->128) ──► Concatenate [Feat; G]
                                                                  │
                                                                  ▼
                                                      Output RGB Image I_hat

CRITICAL PROTOCOL:
- Strictly Prospective: During holdout inference, only pre-weld setpoints c are provided.
  NO ground-truth test dimensions or test contours are ever fed!
- Evaluated against EC-WPG and Conditional VAE on the exact same 4 holdouts (Samples 5, 14, 22, 32).

Outputs:
- results/final_integration/guide_demo_v6_verified/gc_wpg_prototype_checkpoint.pt
- results/final_integration/guide_demo_v6_verified/geometry_conditioned_prototype_metrics.csv
- results/final_integration/guide_demo_v6_verified/fig11_geometry_conditioned_decoder.png
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v6_verified")
os.makedirs(OUT_DIR, exist_ok=True)

PX_PER_MM = 6.0

print("=" * 70)
print("  TRAINING GEOMETRY-CONDITIONED IMAGE DECODER PROTOTYPE (GC-WPG)  ")
print("=" * 70)

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample")

train_df = full_df[~full_df['is_holdout']].copy().reset_index(drop=True)
holdout_df = full_df[full_df['is_holdout']].copy().reset_index(drop=True)

f_min, f_max = train_df['feed_rate'].min(), train_df['feed_rate'].max()
s_min, s_max = train_df['speed'].min(), train_df['speed'].max()
w_min, w_max = train_df['width'].min(), train_df['width'].max()
r_min, r_max = train_df['reinforcement'].min(), train_df['reinforcement'].max()

def build_prospective_condition(row):
    f_norm = (row['feed_rate'] - f_min) / (f_max - f_min + 1e-8)
    s_norm = (row['speed'] - s_min) / (s_max - s_min + 1e-8)
    c_cat = [
        1.0 if row['material'] == 'IN718' else 0.0,
        1.0 if row['material'] == 'SS304L' else 0.0,
        1.0 if row['process'] == 'ColdArc' else 0.0,
        1.0 if row['process'] == 'GMAW' else 0.0
    ]
    return np.array([f_norm, s_norm] + c_cat, dtype=np.float32)

def load_dataset(df_subset):
    imgs, rois, conds, geoms = [], [], [], []
    roi_dir = os.path.join(DATA_DIR, "processed", "aligned_128_rois")
    for _, r in df_subset.iterrows():
        sid, view = r['sample'], r['view']
        im = cv2.imread(r['path'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(im.transpose(2, 0, 1))
        
        roi = cv2.imread(os.path.join(roi_dir, f"{sid}-{view}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        rois.append(roi[np.newaxis, :, :])
        
        conds.append(build_prospective_condition(r))
        
        # Normalized geometry target: [Width_norm, Reinf_norm]
        w_norm = (r['width'] - w_min) / (w_max - w_min + 1e-8)
        reinf_norm = (r['reinforcement'] - r_min) / (r_max - r_min + 1e-8)
        geoms.append(np.array([w_norm, reinf_norm], dtype=np.float32))
        
    return (torch.tensor(np.array(imgs), dtype=torch.float32),
            torch.tensor(np.array(rois), dtype=torch.float32),
            torch.tensor(np.array(conds), dtype=torch.float32),
            torch.tensor(np.array(geoms), dtype=torch.float32))

train_imgs, train_rois, train_conds, train_geoms = load_dataset(train_df)
val_imgs, val_rois, val_conds, val_geoms = load_dataset(holdout_df)
print(f"Loaded {len(train_imgs)} train images and {len(val_imgs)} holdout images.")

# -------------------------------------------------------------------------
# 2. Geometry-Conditioned Architecture (GC-WPG)
# -------------------------------------------------------------------------
class ResUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(self.act(self.bn_up(self.up(x))))

class GeometryConditionedGenerator(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        
        # 1. Geometry Estimator Branch: Predicts [Width_norm, Reinf_norm]
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        
        # 2. Latent Condition Feature Encoder
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 3. Convolutional Decoder Trunk
        self.up1 = ResUpBlock(128, 64) # 8 -> 16
        self.up2 = ResUpBlock(64, 32)  # 16 -> 32
        self.up3 = ResUpBlock(32, 16)  # 32 -> 64
        self.up4 = ResUpBlock(16, 16)  # 64 -> 128
        
        # 4. Spatially Conditioned Fusion Block
        # Concatenates 16 visual feature channels + 2 spatial geometry prior channels = 18 channels
        self.fusion_head = nn.Sequential(
            nn.Conv2d(18, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Precompute coordinate grids for spatial prior generation
        y_grid, x_grid = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                        torch.arange(128, dtype=torch.float32), indexing='ij')
        self.register_buffer("y_coords", y_grid) # (128, 128)
        self.register_buffer("x_coords", x_grid) # (128, 128)

    def generate_spatial_geometry_prior(self, w_norm, reinf_norm):
        """
        Differentiable spatial geometry prior map:
        - Un-normalizes [W, R] to physical pixels.
        - Computes smooth signed distance function of the external reinforcement crown (x < 64).
        """
        batch_size = w_norm.size(0)
        
        # Scale to physical pixel dimensions
        w_px = (w_norm * (w_max - w_min) + w_min) * PX_PER_MM # (B, 1)
        r_px = (reinf_norm * (r_max - r_min) + r_min) * PX_PER_MM # (B, 1)
        
        w_px = w_px.view(batch_size, 1, 1)
        r_px = r_px.view(batch_size, 1, 1)
        
        # Parabolic crown boundary for rows: x_bnd(y) = 64 - r_px * (1 - ((y - 64)/(w_px/2))^2)
        y_dist = (self.y_coords.unsqueeze(0) - 64.0) / (w_px / 2.0 + 1e-6)
        profile_factor = torch.clamp(1.0 - y_dist**2, min=0.0)
        x_crown_bnd = 64.0 - r_px * profile_factor
        
        # Channel 0: Smooth Reinforcement Crown Spatial Prior
        # High inside metal crown (x > x_crown_bnd and x < 64), 0 in background (x < x_crown_bnd)
        crown_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - x_crown_bnd)) * torch.sigmoid(1.5 * (64.0 - self.x_coords.unsqueeze(0)))
        
        # Channel 1: Plate Substrate Prior (x >= 64)
        plate_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - 64.0)).expand(batch_size, -1, -1)
        
        # Concatenate along channel dimension -> (B, 2, 128, 128)
        spatial_prior = torch.stack([crown_prior, plate_prior], dim=1)
        return spatial_prior

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
            
        # 1. Predict geometry
        pred_geom = self.geom_head(c) # (B, 2)
        w_pred = pred_geom[:, 0:1]
        r_pred = pred_geom[:, 1:2]
        
        # 2. Generate 2D spatial geometry prior map
        spatial_prior = self.generate_spatial_geometry_prior(w_pred, r_pred) # (B, 2, 128, 128)
        
        # 3. Decode visual features
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h)))) # (B, 16, 128, 128)
        
        # 4. Spatially conditioned fusion
        fused = torch.cat([feat, spatial_prior], dim=1) # (B, 18, 128, 128)
        out_img = self.fusion_head(fused)
        
        return out_img, pred_geom, spatial_prior

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            if seed is not None:
                g = torch.Generator(device=c.device)
                g.manual_seed(seed)
                z = torch.randn((c.size(0), self.noise_dim), generator=g, dtype=torch.float32, device=c.device) * 0.05
            else:
                z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
            return self.forward(c, z)

# -------------------------------------------------------------------------
# 3. Training Loop
# -------------------------------------------------------------------------
model = GeometryConditionedGenerator(cond_dim=6, noise_dim=16).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-4)

epochs = 80
batch_size = 8
n_samples = len(train_imgs)
n_batches = int(np.ceil(n_samples / batch_size))

t0 = time.time()
print(f"Training GC-WPG Prototype for {epochs} epochs...", flush=True)

for ep in range(1, epochs + 1):
    model.train()
    perm = torch.randperm(n_samples)
    ep_loss, ep_img, ep_roi, ep_geom = 0.0, 0.0, 0.0, 0.0
    
    for b in range(n_batches):
        idx = perm[b*batch_size : (b+1)*batch_size]
        b_imgs = train_imgs[idx].to(device)
        b_rois = train_rois[idx].to(device)
        b_conds = train_conds[idx].to(device)
        b_geoms = train_geoms[idx].to(device)
        b_z = torch.randn((len(idx), 16), dtype=torch.float32, device=device) * 0.05
        
        optimizer.zero_grad()
        pred_img, pred_geom, _ = model(b_conds, b_z)
        
        # 1. Global image reconstruction
        l_img = F.l1_loss(pred_img, b_imgs) + 0.5 * F.mse_loss(pred_img, b_imgs)
        
        # 2. ROI weighted loss
        diff = torch.abs(pred_img - b_imgs)
        roi_exp = b_rois.repeat(1, 3, 1, 1)
        l_roi = (diff * roi_exp).sum() / (roi_exp.sum() + 1e-6)
        
        # 3. Geometry prediction loss (supervising the geometry estimator head on training samples)
        l_geom = F.mse_loss(pred_geom, b_geoms)
        
        total_loss = l_img + 2.0 * l_roi + 5.0 * l_geom
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        ep_loss += total_loss.item()
        ep_img += l_img.item()
        ep_roi += l_roi.item()
        ep_geom += l_geom.item()
        
    scheduler.step()
    if ep % 20 == 0 or ep == epochs:
        print(f"Epoch {ep:2d}/{epochs} | Loss: {ep_loss/n_batches:.4f} | Img: {ep_img/n_batches:.4f} | ROI: {ep_roi/n_batches:.4f} | Geom MSE: {ep_geom/n_batches:.4f}", flush=True)

elapsed = time.time() - t0
print(f"Training completed in {elapsed:.1f} seconds.")

# Save Checkpoint
ckpt_save_path = os.path.join(OUT_DIR, "gc_wpg_prototype_checkpoint.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_params": {
        "feed_rate_min": float(f_min), "feed_rate_max": float(f_max),
        "speed_min": float(s_min), "speed_max": float(s_max),
        "width_min": float(w_min), "width_max": float(w_max),
        "reinforcement_min": float(r_min), "reinforcement_max": float(r_max)
    },
    "cond_dim": 6, "noise_dim": 16, "seed": SEED
}, ckpt_save_path)
print(f"Saved GC-WPG checkpoint to: {ckpt_save_path}")

# -------------------------------------------------------------------------
# 4. Rigorous Evaluation on Holdouts (Actual Generated Image Boundary)
# -------------------------------------------------------------------------
print("\nEvaluating GC-WPG on 8 Unseen Holdout Views...", flush=True)
model.eval()

# Helper for boundary extraction
from scripts.audit_actual_image_boundaries import extract_external_crown_contour

gc_records = []
vis_outputs = []

for i in range(len(val_imgs)):
    r = holdout_df.iloc[i]
    sid = r['sample']
    view = r['view']
    
    gt_img = val_imgs[i].permute(1, 2, 0).numpy()
    gt_gray = cv2.cvtColor((gt_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gt_pts, gt_edge_map = extract_external_crown_contour(gt_gray, x_baseline=64, threshold=165)
    gt_apex_x = min(gt_pts.values()) if gt_pts else 64
    gt_apex_y = [y for y, x in gt_pts.items() if x == gt_apex_x][0] if gt_pts else 64
    
    # Prospective Inference (ZERO ground truth fed!)
    c = val_conds[i:i+1].to(device)
    with torch.no_grad():
        pr_img_t, pred_geom, spatial_prior = model.generate(c, seed=42)
        pr_img = pr_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pr_gray = cv2.cvtColor((pr_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        prior_map = spatial_prior.squeeze(0)[0].cpu().numpy() # crown prior
        
    vis_outputs.append((gt_img, pr_img, prior_map, gt_pts))
    
    # Extract actual generated crown boundary
    bg_val = np.median(pr_gray[20:110, :20])
    thresh = max(145, min(170, bg_val - 12))
    pr_pts, pr_edge_map = extract_external_crown_contour(pr_gray, x_baseline=64, threshold=thresh)
    pr_apex_x = min(pr_pts.values()) if pr_pts else 64
    pr_apex_y = [y for y, x in pr_pts.items() if x == pr_apex_x][0] if pr_pts else 64
    
    # Image metrics
    p_val = psnr(gt_img, pr_img, data_range=1.0)
    s_val = ssim(gt_img, pr_img, channel_axis=2, data_range=1.0)
    
    # ROI SSIM
    r_bool = val_rois[i].squeeze(0).numpy() > 0.5
    r_min_y, r_max_y = np.where(r_bool)[0].min(), np.where(r_bool)[0].max()
    r_min_x, r_max_x = np.where(r_bool)[1].min(), np.where(r_bool)[1].max()
    roi_ssim = ssim(gt_img[r_min_y:r_max_y+1, r_min_x:r_max_x+1],
                    pr_img[r_min_y:r_max_y+1, r_min_x:r_max_x+1],
                    channel_axis=2, data_range=1.0)
    
    # Apex Error
    apex_err_mm = abs(gt_apex_x - pr_apex_x) / PX_PER_MM
    
    # Contour MAE
    y_common = set(gt_pts.keys()).intersection(set(pr_pts.keys()))
    if len(y_common) > 10:
        diffs = [abs(gt_pts[y] - pr_pts[y]) for y in y_common]
        mae_contour_mm = np.mean(diffs) / PX_PER_MM
        hausdorff_mm = np.max(diffs) / PX_PER_MM
        coverage = len(y_common) / max(len(gt_pts), 1)
    else:
        mae_contour_mm = abs(gt_apex_x - 64) / PX_PER_MM
        hausdorff_mm = mae_contour_mm
        coverage = 0.0
        
    # Boundary F1 strictly on X < 64
    k1 = np.ones((3, 3), np.uint8)
    gt_dil1 = cv2.dilate(gt_edge_map[:, :64], k1)
    tp1 = np.sum((pr_edge_map[:, :64] == 1) & (gt_dil1 == 1))
    f1_1px = 2.0 * tp1 / (np.sum(pr_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)
    
    k2 = np.ones((5, 5), np.uint8)
    gt_dil2 = cv2.dilate(gt_edge_map[:, :64], k2)
    tp2 = np.sum((pr_edge_map[:, :64] == 1) & (gt_dil2 == 1))
    f1_2px = 2.0 * tp2 / (np.sum(pr_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)
    
    gc_records.append({
        "model": "Geometry-Conditioned Prototype (GC-WPG)",
        "sample": sid, "view": view, "material": r['material'], "process": r['process'],
        "psnr_db": round(float(p_val), 2),
        "ssim": round(float(s_val), 4),
        "roi_ssim": round(float(roi_ssim), 4),
        "apex_err_x_mm": round(float(apex_err_mm), 3),
        "contour_mae_mm": round(float(mae_contour_mm), 3),
        "hausdorff_mm": round(float(hausdorff_mm), 3),
        "reinf_boundary_f1_1px": round(float(f1_1px), 4),
        "reinf_boundary_f1_2px": round(float(f1_2px), 4),
        "contour_coverage": round(float(coverage), 3)
    })

df_gc = pd.DataFrame(gc_records)
gc_csv_path = os.path.join(OUT_DIR, "geometry_conditioned_prototype_metrics.csv")
df_gc.to_csv(gc_csv_path, index=False)
print("\nGC-WPG Prototype Holdout Evaluation Results:")
print(df_gc[['sample', 'view', 'psnr_db', 'ssim', 'roi_ssim', 'apex_err_x_mm', 'contour_mae_mm', 'reinf_boundary_f1_1px']].to_string())

print(f"\nMean PSNR:         {df_gc['psnr_db'].mean():.2f} dB")
print(f"Mean SSIM:         {df_gc['ssim'].mean():.4f}")
print(f"Mean ROI SSIM:     {df_gc['roi_ssim'].mean():.4f}")
print(f"Mean Contour MAE:  {df_gc['contour_mae_mm'].mean():.3f} mm")
print(f"Mean Apex Error:   {df_gc['apex_err_x_mm'].mean():.3f} mm")
print(f"Mean Boundary F1:  {df_gc['reinf_boundary_f1_1px'].mean():.4f}")

# -------------------------------------------------------------------------
# 5. Generate Figure 11: Geometry-Conditioned Decoder Architecture & Results
# -------------------------------------------------------------------------
print("\nGenerating Figure 11: Geometry-Conditioned Decoder Demonstration...", flush=True)
fig = plt.figure(figsize=(16, 9), dpi=300)
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.25)

# Row 1: 4 Holdout Views: Ground Truth vs Spatial Prior vs Generated Photograph
indices_demo = [0, 2, 4, 6] # Samples 5, 14, 22, 32 View A
titles_demo = ["Sample 05 (IN718 ColdArc)", "Sample 14 (IN718 GMAW)", "Sample 22 (SS304L ColdArc)", "Sample 32 (SS304L GMAW)"]

for idx, h_idx in enumerate(indices_demo):
    gt_im, pr_im, prior_m, g_pts = vis_outputs[h_idx]
    
    # Top row: Spatial Prior Map overlaid on Ground Truth
    ax_top = fig.add_subplot(gs[0, idx])
    ax_top.imshow(prior_m, cmap="magma")
    ax_top.axvline(64, color='white', linestyle=':', alpha=0.7)
    # Overlay GT contour
    ys = sorted(g_pts.keys())
    xs = [g_pts[y] for y in ys]
    ax_top.plot(xs, ys, color='cyan', linestyle='--', linewidth=1.5, label="GT Contour")
    ax_top.set_title(f"{titles_demo[idx]}\nPredicted Spatial Prior G", fontsize=9, fontweight="bold")
    ax_top.axis("off")
    if idx == 0:
        ax_top.legend(loc="lower left", fontsize=7.5)
        
    # Bottom row: Synthesized Cross-Section with Extracted Boundary
    ax_bot = fig.add_subplot(gs[1, idx])
    ax_bot.imshow(pr_im)
    ax_bot.axvline(64, color='red', linestyle=':', alpha=0.6)
    
    # Extract predicted boundary
    pr_gray_sub = cv2.cvtColor((pr_im * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    p_pts, _ = extract_external_crown_contour(pr_gray_sub, threshold=155)
    
    # Plot GT in cyan dashed
    ax_bot.plot(xs, ys, color='cyan', linestyle='--', linewidth=1.2, label="GT Contour")
    # Plot Generated in green
    if p_pts:
        p_ys = sorted(p_pts.keys())
        p_xs = [p_pts[y] for y in p_ys]
        ax_bot.plot(p_xs, p_ys, color='#2ecc71', linewidth=1.5, label="Generated Boundary")
        
    ax_bot.set_title(f"GC-WPG Synthesized Image\nSSIM: {df_gc.iloc[h_idx]['ssim']:.3f} | Cont Err: {df_gc.iloc[h_idx]['contour_mae_mm']:.2f} mm",
                     fontsize=9, fontweight="bold")
    ax_bot.axis("off")
    if idx == 0:
        ax_bot.legend(loc="lower left", fontsize=7.5)

plt.suptitle("Figure 11: Geometry-Conditioned Image Decoder Prototype (GC-WPG)\nTop: Predicted Spatial Prior G(W_pred, R_pred) Guides Convolutional Layers | Bottom: Synthesized RGB Macrograph with Sharp Physical Crown",
             fontsize=12, fontweight="bold", y=0.98)

fig11_path = os.path.join(OUT_DIR, "fig11_geometry_conditioned_decoder.png")
plt.savefig(fig11_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Figure 11 to: {fig11_path}")
print("=" * 70)
