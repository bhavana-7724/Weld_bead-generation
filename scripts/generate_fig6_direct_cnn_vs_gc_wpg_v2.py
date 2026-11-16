"""
generate_fig6_direct_cnn_vs_gc_wpg_v2.py

Generates Figure 6:
fig6_direct_cnn_vs_gc_wpg_v2.png
in results/final_integration/guide_demo_v7_refined/

Compares:
1. Ground Truth Experimental Macrograph
2. Direct Conditional CNN (Baseline)
3. GC-WPG Prototype (v1 - Late Fusion)
4. GC-WPG-v2 (Multi-Scale Spatial Prior Injection)
5. Zoomed Boundary Contour Overlay (GT vs Direct CNN vs GC-WPG-v1 vs GC-WPG-v2)

Across representative development holdout samples:
- Sample 5 (IN718, ColdArc)
- Sample 14 (IN718, GMAW)
- Sample 22 (SS304L, ColdArc)
- Sample 32 (SS304L, GMAW - Deep Penetration)

Plus quantitative cross-validation performance comparison bars.
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7_refined")
os.makedirs(OUT_DIR, exist_ok=True)

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

class MultiScaleSpatialFusion(nn.Module):
    def __init__(self, channels, prior_channels=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels + prior_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x, prior):
        return x + self.block(torch.cat([x, prior], dim=1))

class GC_WPG_v2(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.scaler = scaler_params
        
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.up1 = ResUpBlock(128, 64)
        self.fuse1 = MultiScaleSpatialFusion(64, prior_channels=2)
        self.up2 = ResUpBlock(64, 32)
        self.fuse2 = MultiScaleSpatialFusion(32, prior_channels=2)
        self.up3 = ResUpBlock(32, 16)
        self.fuse3 = MultiScaleSpatialFusion(16, prior_channels=2)
        self.up4 = ResUpBlock(16, 16)
        self.fuse4 = MultiScaleSpatialFusion(16, prior_channels=2)
        
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        y_grid, x_grid = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                        torch.arange(128, dtype=torch.float32), indexing='ij')
        self.register_buffer("y_coords", y_grid.contiguous().clone(), persistent=False)
        self.register_buffer("x_coords", x_grid.contiguous().clone(), persistent=False)

    def generate_spatial_geometry_prior(self, w_norm, reinf_norm):
        batch_size = w_norm.size(0)
        px_scale = 6.0
        w_px = (w_norm * (self.scaler['w_max'] - self.scaler['w_min']) + self.scaler['w_min']) * px_scale
        r_px = (reinf_norm * (self.scaler['r_max'] - self.scaler['r_min']) + self.scaler['r_min']) * px_scale
        w_px = w_px.view(batch_size, 1, 1)
        r_px = r_px.view(batch_size, 1, 1)
        y_dist = (self.y_coords.unsqueeze(0) - 64.0) / (w_px / 2.0 + 1e-6)
        profile_factor = torch.clamp(1.0 - y_dist**2, min=0.0)
        x_crown_bnd = 64.0 - r_px * profile_factor
        crown_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - x_crown_bnd)) * torch.sigmoid(1.5 * (64.0 - self.x_coords.unsqueeze(0)))
        plate_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - 64.0)).expand(batch_size, -1, -1)
        return torch.stack([crown_prior, plate_prior], dim=1)

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        pred_geom = self.geom_head(c)
        prior_128 = self.generate_spatial_geometry_prior(pred_geom[:, 0:1], pred_geom[:, 1:2])
        prior_64 = F.interpolate(prior_128, size=(64, 64), mode='bilinear', align_corners=False)
        prior_32 = F.interpolate(prior_128, size=(32, 32), mode='bilinear', align_corners=False)
        prior_16 = F.interpolate(prior_128, size=(16, 16), mode='bilinear', align_corners=False)
        
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        h1 = self.fuse1(self.up1(h), prior_16)
        h2 = self.fuse2(self.up2(h1), prior_32)
        h3 = self.fuse3(self.up3(h2), prior_64)
        h4 = self.fuse4(self.up4(h3), prior_128)
        return self.out_conv(h4), pred_geom, prior_128

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            out, geom, prior = self.forward(c)
            return out, geom, prior

class DirectCNN(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.out_conv(feat)

class GC_WPG_v1(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.scaler = scaler_params
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 2),
            nn.Sigmoid()
        )
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)
        self.fusion_head = nn.Sequential(
            nn.Conv2d(16 + 2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        y_grid, x_grid = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                        torch.arange(128, dtype=torch.float32), indexing='ij')
        self.register_buffer("y_coords", y_grid.contiguous().clone(), persistent=False)
        self.register_buffer("x_coords", x_grid.contiguous().clone(), persistent=False)

    def generate_spatial_geometry_prior(self, w_norm, reinf_norm):
        batch_size = w_norm.size(0)
        px_scale = 6.0
        w_px = (w_norm * (self.scaler['w_max'] - self.scaler['w_min']) + self.scaler['w_min']) * px_scale
        r_px = (reinf_norm * (self.scaler['r_max'] - self.scaler['r_min']) + self.scaler['r_min']) * px_scale
        w_px = w_px.view(batch_size, 1, 1)
        r_px = r_px.view(batch_size, 1, 1)
        y_dist = (self.y_coords.unsqueeze(0) - 64.0) / (w_px / 2.0 + 1e-6)
        profile_factor = torch.clamp(1.0 - y_dist**2, min=0.0)
        x_crown_bnd = 64.0 - r_px * profile_factor
        crown_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - x_crown_bnd)) * torch.sigmoid(1.5 * (64.0 - self.x_coords.unsqueeze(0)))
        plate_prior = torch.sigmoid(1.5 * (self.x_coords.unsqueeze(0) - 64.0)).expand(batch_size, -1, -1)
        return torch.stack([crown_prior, plate_prior], dim=1)

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        pred_geom = self.geom_head(c)
        spatial_prior = self.generate_spatial_geometry_prior(pred_geom[:, 0:1], pred_geom[:, 1:2])
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        fused = torch.cat([feat, spatial_prior], dim=1)
        return self.fusion_head(fused), pred_geom, spatial_prior

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            out, geom, prior = self.forward(c)
            return out, geom, prior

def extract_crown_contour(gray_img, x_baseline=64, y_min=20, y_max=110, threshold=155):
    pts = {}
    edge_map = np.zeros_like(gray_img, dtype=np.uint8)
    for y in range(y_min, y_max + 1):
        row_vals = gray_img[y, :x_baseline]
        metal_cols = np.where(row_vals < threshold)[0]
        if len(metal_cols) > 0 and metal_cols[0] < x_baseline:
            x_min = int(metal_cols[0])
            pts[y] = x_min
            edge_map[y, x_min] = 1
    return pts, edge_map

def main():
    print("Loading datasets and model checkpoints for Figure 6 generation...", flush=True)
    meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
    clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
    full_df = pd.merge(meta_df, clean_df, on="sample").sort_values(by=["sample", "view"]).reset_index(drop=True)
    
    DEV_HOLDOUTS = [5, 14, 22, 32]
    train_pool = full_df[~full_df['sample'].isin(DEV_HOLDOUTS)]
    val_samples_dev = [1, 10, 19, 28]
    dev_train = train_pool[~train_pool['sample'].isin(val_samples_dev)]
    
    dev_scaler = {
        "f_min": float(dev_train['feed_rate'].min()), "f_max": float(dev_train['feed_rate'].max()),
        "s_min": float(dev_train['speed'].min()), "s_max": float(dev_train['speed'].max()),
        "w_min": float(dev_train['width'].min()), "w_max": float(dev_train['width'].max()),
        "r_min": float(dev_train['reinforcement'].min()), "r_max": float(dev_train['reinforcement'].max()),
        "d_min": float(dev_train['depth'].min()), "d_max": float(dev_train['depth'].max())
    }
    
    # Load GC-WPG-v2 checkpoint
    v2_ckpt_path = os.path.join(OUT_DIR, "gc_wpg_v2_checkpoint.pt")
    v2_ckpt = torch.load(v2_ckpt_path, map_location="cpu")
    model_v2 = GC_WPG_v2(6, 16, scaler_params=v2_ckpt["scaler"])
    model_v2.load_state_dict(v2_ckpt["model_state_dict"])
    model_v2.eval()
    
    # Fast train Direct CNN and GC-WPG-v1 on dev_train (batch size 8 takes ~8s)
    print("Training Direct CNN and GC-WPG-v1 baselines on dev_train...", flush=True)
    t_imgs = torch.tensor(np.array([cv2.cvtColor(cv2.imread(r['path']), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) / 255.0 for _, r in dev_train.iterrows()]), dtype=torch.float32)
    t_conds = torch.tensor(np.array([
        [
            (r['feed_rate'] - dev_scaler['f_min']) / (dev_scaler['f_max'] - dev_scaler['f_min'] + 1e-8),
            (r['speed'] - dev_scaler['s_min']) / (dev_scaler['s_max'] - dev_scaler['s_min'] + 1e-8),
            1.0 if r['material'] == 'IN718' else 0.0,
            1.0 if r['material'] == 'SS304L' else 0.0,
            1.0 if r['process'] == 'ColdArc' else 0.0,
            1.0 if r['process'] == 'GMAW' else 0.0
        ] for _, r in dev_train.iterrows()
    ]), dtype=torch.float32)
    t_geoms = torch.tensor(np.array([
        [
            (r['width'] - dev_scaler['w_min']) / (dev_scaler['w_max'] - dev_scaler['w_min'] + 1e-8),
            (r['reinforcement'] - dev_scaler['r_min']) / (dev_scaler['r_max'] - dev_scaler['r_min'] + 1e-8)
        ] for _, r in dev_train.iterrows()
    ]), dtype=torch.float32)
    
    bs = 8
    n_batches = int(np.ceil(len(t_imgs) / bs))
    
    model_direct = DirectCNN(6, 16)
    opt_d = torch.optim.AdamW(model_direct.parameters(), lr=1e-3, weight_decay=1e-4)
    model_direct.train()
    for ep in range(60):
        perm = torch.randperm(len(t_imgs))
        for b in range(n_batches):
            idx = perm[b*bs : (b+1)*bs]
            opt_d.zero_grad()
            p = model_direct(t_conds[idx])
            loss = F.l1_loss(p, t_imgs[idx]) + F.l1_loss(p[:, :, 20:110, 10:64], t_imgs[idx][:, :, 20:110, 10:64]) * 2.0
            loss.backward()
            opt_d.step()
    model_direct.eval()
    
    model_v1 = GC_WPG_v1(6, 16, scaler_params=dev_scaler)
    opt_v1 = torch.optim.AdamW(model_v1.parameters(), lr=1e-3, weight_decay=1e-4)
    model_v1.train()
    for ep in range(60):
        perm = torch.randperm(len(t_imgs))
        for b in range(n_batches):
            idx = perm[b*bs : (b+1)*bs]
            opt_v1.zero_grad()
            p, pg, _ = model_v1(t_conds[idx])
            loss = F.l1_loss(p, t_imgs[idx]) + F.l1_loss(p[:, :, 20:110, 10:64], t_imgs[idx][:, :, 20:110, 10:64]) * 2.0 + F.mse_loss(pg, t_geoms[idx]) * 0.2
            loss.backward()
            opt_v1.step()
    model_v1.eval()
    print("Baseline models ready.", flush=True)

    # Plot Figure 6
    fig = plt.figure(figsize=(20, 18), dpi=300)
    gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 1, 0.75], hspace=0.30, wspace=0.15)
    
    holdout_samples = [5, 14, 22, 32]
    
    for row_idx, sid in enumerate(holdout_samples):
        row_data = full_df[(full_df['sample'] == sid) & (full_df['view'] == 'A')].iloc[0]
        gt_bgr = cv2.imread(row_data['path'])
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
        
        gt_pts, _ = extract_crown_contour(gt_gray, threshold=165)
        gt_ax = min(gt_pts.values()) if gt_pts else 64
        gt_ay = [y for y, x in gt_pts.items() if x == gt_ax][0] if gt_pts else 64
        gt_reinf = 64 - gt_ax
        
        c = torch.tensor([[
            (row_data['feed_rate'] - dev_scaler['f_min']) / (dev_scaler['f_max'] - dev_scaler['f_min'] + 1e-8),
            (row_data['speed'] - dev_scaler['s_min']) / (dev_scaler['s_max'] - dev_scaler['s_min'] + 1e-8),
            1.0 if row_data['material'] == 'IN718' else 0.0,
            1.0 if row_data['material'] == 'SS304L' else 0.0,
            1.0 if row_data['process'] == 'ColdArc' else 0.0,
            1.0 if row_data['process'] == 'GMAW' else 0.0
        ]], dtype=torch.float32)
        
        with torch.no_grad():
            img_dir = np.clip(model_direct(c).squeeze(0).permute(1, 2, 0).numpy(), 0, 1)
            img_v1, _, _ = model_v1.generate(c, seed=42)
            img_v1 = np.clip(img_v1.squeeze(0).permute(1, 2, 0).numpy(), 0, 1)
            img_v2, _, _ = model_v2.generate(c, seed=42)
            img_v2 = np.clip(img_v2.squeeze(0).permute(1, 2, 0).numpy(), 0, 1)
            
        def get_pred_pts(img):
            p_gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            bg_v = np.median(p_gray[20:110, :20])
            pts, _ = extract_crown_contour(p_gray, threshold=max(145, min(170, bg_v - 12)))
            ax = min(pts.values()) if pts else 64
            ay = [y for y, x in pts.items() if x == ax][0] if pts else 64
            y_common = set(gt_pts.keys()).intersection(set(pts.keys()))
            mae = np.mean([abs(gt_pts[y] - pts[y]) for y in y_common]) if len(y_common) >= 5 else abs(gt_ax - 64)
            return pts, ax, ay, mae
            
        pts_dir, ax_dir, ay_dir, mae_dir = get_pred_pts(img_dir)
        pts_v1, ax_v1, ay_v1, mae_v1 = get_pred_pts(img_v1)
        pts_v2, ax_v2, ay_v2, mae_v2 = get_pred_pts(img_v2)
        
        mat_proc = f"{row_data['material']} {row_data['process']}"
        cond_str = f"F={row_data['feed_rate']:.1f} m/min, S={row_data['speed']:.1f} cm/min"
        
        # Col 0: GT
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(gt_rgb)
        ax0.axvline(64, color='cyan', linestyle=':', alpha=0.7)
        if gt_pts:
            ax0.plot([gt_pts[y] for y in sorted(gt_pts.keys())], sorted(gt_pts.keys()), color='#2ecc71', linewidth=1.8)
            ax0.plot(gt_ax, gt_ay, 'ro', markersize=5)
        ax0.set_title(f"Sample {sid}: {mat_proc}\n{cond_str}\nGT Reinf: {gt_reinf} px (Apex X={gt_ax})", fontsize=9.5, fontweight='bold')
        ax0.axis('off')
        
        # Col 1: Direct CNN
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(img_dir)
        ax1.axvline(64, color='red', linestyle=':', alpha=0.5)
        if pts_dir:
            ax1.plot([pts_dir[y] for y in sorted(pts_dir.keys())], sorted(pts_dir.keys()), color='#3498db', linewidth=1.8)
            ax1.plot(ax_dir, ay_dir, 'bo', markersize=5)
        ax1.set_title(f"Direct CNN\nContour MAE: {mae_dir:.2f} px\nApex Err: {abs(gt_ax - ax_dir):.1f} px", fontsize=9.5, fontweight='bold')
        ax1.axis('off')
        
        # Col 2: GC-WPG-v1
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.imshow(img_v1)
        ax2.axvline(64, color='red', linestyle=':', alpha=0.5)
        if pts_v1:
            ax2.plot([pts_v1[y] for y in sorted(pts_v1.keys())], sorted(pts_v1.keys()), color='#e67e22', linewidth=1.8)
            ax2.plot(ax_v1, ay_v1, 'o', color='#e67e22', markersize=5)
        ax2.set_title(f"GC-WPG-v1 (Late Fusion)\nContour MAE: {mae_v1:.2f} px\nApex Err: {abs(gt_ax - ax_v1):.1f} px", fontsize=9.5, fontweight='bold')
        ax2.axis('off')
        
        # Col 3: GC-WPG-v2
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.imshow(img_v2)
        ax3.axvline(64, color='red', linestyle=':', alpha=0.5)
        if pts_v2:
            ax3.plot([pts_v2[y] for y in sorted(pts_v2.keys())], sorted(pts_v2.keys()), color='#e74c3c', linewidth=1.8)
            ax3.plot(ax_v2, ay_v2, 'ro', markersize=5)
        ax3.set_title(f"GC-WPG-v2 (Multi-Scale Prior)\nContour MAE: {mae_v2:.2f} px\nApex Err: {abs(gt_ax - ax_v2):.1f} px", fontsize=9.5, fontweight='bold')
        ax3.axis('off')
        
        # Col 4: Zoomed Overlay
        ax4 = fig.add_subplot(gs[row_idx, 4])
        ax4.imshow(gt_rgb[20:110, 10:64, :])
        ax4.axvline(54, color='cyan', linestyle=':', alpha=0.8, label='Interface')
        if gt_pts:
            ax4.plot([gt_pts[y] - 10 for y in sorted(gt_pts.keys()) if 20 <= y <= 110],
                     [y - 20 for y in sorted(gt_pts.keys()) if 20 <= y <= 110],
                     color='#2ecc71', linewidth=2.2, label='Ground Truth')
        if pts_dir:
            ax4.plot([pts_dir[y] - 10 for y in sorted(pts_dir.keys()) if 20 <= y <= 110],
                     [y - 20 for y in sorted(pts_dir.keys()) if 20 <= y <= 110],
                     color='#3498db', linewidth=1.8, linestyle='--', label='Direct CNN')
        if pts_v1:
            ax4.plot([pts_v1[y] - 10 for y in sorted(pts_v1.keys()) if 20 <= y <= 110],
                     [y - 20 for y in sorted(pts_v1.keys()) if 20 <= y <= 110],
                     color='#e67e22', linewidth=1.8, linestyle=':', label='GC-WPG-v1')
        if pts_v2:
            ax4.plot([pts_v2[y] - 10 for y in sorted(pts_v2.keys()) if 20 <= y <= 110],
                     [y - 20 for y in sorted(pts_v2.keys()) if 20 <= y <= 110],
                     color='#e74c3c', linewidth=2.0, label='GC-WPG-v2')
        ax4.set_title(f"Zoomed Crown Overlay\nSample {sid} ({mat_proc})", fontsize=9.5, fontweight='bold')
        ax4.axis('off')
        if row_idx == 0:
            ax4.legend(loc='lower left', fontsize=7.5, framealpha=0.9)
            
    # Bottom Row: Summary Performance Charts from REFINED_MODEL_COMPARISON.csv
    refined_csv_path = os.path.join(OUT_DIR, "REFINED_MODEL_COMPARISON.csv")
    if os.path.exists(refined_csv_path):
        df_comp = pd.read_csv(refined_csv_path)
        sub_cv = df_comp[df_comp['evaluation_suite'].str.contains("5-Fold")].copy()
        
        ax_m1 = fig.add_subplot(gs[4, :2])
        models = sub_cv['model'].tolist()
        short_names = [m.replace("Direct Conditional CNN", "Direct CNN").replace("Prototype", "v1").replace(" (Multi-Scale Prior)", "") for m in models]
        x_indices = np.arange(len(models))
        width = 0.35
        
        ax_m1.bar(x_indices - width/2, sub_cv['contour_mae_px_mean'], width, yerr=sub_cv['contour_mae_px_std'],
                  label='Contour MAE (px)', color='#e74c3c', capsize=3, alpha=0.85)
        ax_m1.bar(x_indices + width/2, sub_cv['apex_err_x_px_mean'], width, yerr=sub_cv['apex_err_x_px_std'],
                  label='Apex Error X (px)', color='#3498db', capsize=3, alpha=0.85)
        ax_m1.set_xticks(x_indices)
        ax_m1.set_xticklabels(short_names, fontsize=8.5, rotation=10)
        ax_m1.set_ylabel("Error (Pixels, lower is better)", fontsize=9, fontweight='bold')
        ax_m1.set_title("5-Fold CV Boundary Geometry Accuracy", fontsize=10, fontweight='bold')
        ax_m1.legend(fontsize=8, loc='upper left')
        ax_m1.grid(axis='y', linestyle='--', alpha=0.5)
        
        ax_m2 = fig.add_subplot(gs[4, 2:4])
        ax_m2.bar(x_indices - width/2, sub_cv['ssim_mean'], width, yerr=sub_cv['ssim_std'],
                  label='Whole SSIM', color='#2ecc71', capsize=3, alpha=0.85)
        ax_m2.bar(x_indices + width/2, sub_cv['roi_ssim_mean'], width, yerr=sub_cv['roi_ssim_std'],
                  label='Weld-ROI SSIM', color='#f39c12', capsize=3, alpha=0.85)
        ax_m2.set_xticks(x_indices)
        ax_m2.set_xticklabels(short_names, fontsize=8.5, rotation=10)
        ax_m2.set_ylabel("SSIM (higher is better)", fontsize=9, fontweight='bold')
        ax_m2.set_ylim(0.65, 0.85)
        ax_m2.set_title("5-Fold CV Image & ROI Perceptual Similarity", fontsize=10, fontweight='bold')
        ax_m2.legend(fontsize=8, loc='lower right')
        ax_m2.grid(axis='y', linestyle='--', alpha=0.5)
        
        ax_m3 = fig.add_subplot(gs[4, 4])
        f1_vals = sub_cv['reinf_boundary_f1_1px_mean'] * 100
        bars = ax_m3.bar(x_indices, f1_vals, width=0.5, color='#9b59b6', alpha=0.85)
        ax_m3.set_xticks(x_indices)
        ax_m3.set_xticklabels(short_names, fontsize=8, rotation=15)
        ax_m3.set_ylabel("1-px Boundary F1 (%)", fontsize=9, fontweight='bold')
        ax_m3.set_title("1-Pixel Boundary F1 Score", fontsize=10, fontweight='bold')
        ax_m3.grid(axis='y', linestyle='--', alpha=0.5)
        for b in bars:
            h = b.get_height()
            ax_m3.text(b.get_x() + b.get_width()/2., h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=7.5)

    fig.suptitle(
        "Figure 6: Comprehensive Model Comparison — Direct CNN vs. GC-WPG-v1 vs. GC-WPG-v2 (Multi-Scale Prior)\n"
        "Evaluating Boundary Fidelity, External Crown Localization, and Photometric Realism on Unseen Welding Conditions",
        fontsize=13, fontweight='bold', y=0.985
    )
    
    out_fig = os.path.join(OUT_DIR, "fig6_direct_cnn_vs_gc_wpg_v2.png")
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Figure 6 to: {out_fig}", flush=True)

if __name__ == "__main__":
    main()
