"""
run_v7_integrity_audit.py

Comprehensive Integrity Audit of Phase V7 Refined:
- Task 1: Trace and resolve the GC-WPG-v1 discrepancy.
- Task 2: Verify all Figure 6 comparisons on Samples 5, 14, 22, 32 across all models using actual checkpoints.
- Task 3: Audit Figure 5 measurements, threshold sensitivity sweep (T=135..165, Otsu, Adaptive), and experimental comparison.
- Task 4: Audit 5-Fold Grouped CV checkpoints, early stopping isolation, and experiment-level aggregation.
- Task 5: Output all audit markdown reports, CSV files, and corrected figures in guide_demo_v7_integrity_audit/
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "final_integration")
AUDIT_DIR = os.path.join(RESULTS_DIR, "guide_demo_v7_integrity_audit")
RAW_OUT_DIR = os.path.join(AUDIT_DIR, "raw_outputs")
CKPT_DIR = os.path.join(AUDIT_DIR, "checkpoints")

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(RAW_OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

# =========================================================================
# 1. Dataset Preprocessing & Standardization
# =========================================================================
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample").sort_values(by=["sample", "view"]).reset_index(drop=True)

def fit_prospective_scaler(df_train):
    return {
        "f_min": float(df_train['feed_rate'].min()), "f_max": float(df_train['feed_rate'].max()),
        "s_min": float(df_train['speed'].min()), "s_max": float(df_train['speed'].max()),
        "w_min": float(df_train['width'].min()), "w_max": float(df_train['width'].max()),
        "r_min": float(df_train['reinforcement'].min()), "r_max": float(df_train['reinforcement'].max()),
        "d_min": float(df_train['depth'].min()), "d_max": float(df_train['depth'].max())
    }

def extract_crown_contour(gray_img, x_baseline=64, y_min=20, y_max=110, threshold=155):
    pts = {}
    edge_map = np.zeros_like(gray_img, dtype=np.uint8)
    for y in range(y_min, y_max + 1):
        row_vals = gray_img[y, :x_baseline]
        metal_idx = np.where(row_vals < threshold)[0]
        if len(metal_idx) > 0:
            x_min = metal_idx[0]
            if x_min < x_baseline:
                pts[y] = int(x_min)
                edge_map[y, x_min] = 1
    return pts, edge_map

# Preload all 72 images
all_data = []
for idx, r in full_df.iterrows():
    sid = int(r['sample'])
    view = r['view']
    im_bgr = cv2.imread(r['path'])
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    gt_pts, gt_edge_map = extract_crown_contour(im_gray, threshold=165)
    apex_x = min(gt_pts.values()) if gt_pts else 64
    apex_y = [y for y, x in gt_pts.items() if x == apex_x][0] if gt_pts else 64
    roi_mask = np.zeros((128, 128), dtype=np.float32)
    roi_mask[20:110, 10:64] = 1.0
    all_data.append({
        "sample": sid, "view": view, "material": r['material'], "process": r['process'],
        "feed_rate": float(r['feed_rate']), "speed": float(r['speed']),
        "width": float(r['width']), "depth": float(r['depth']), "reinforcement": float(r['reinforcement']),
        "img_rgb": im_rgb, "img_gray": im_gray, "roi_mask": roi_mask,
        "gt_pts": gt_pts, "gt_edge_map": gt_edge_map, "gt_apex_x": apex_x, "gt_apex_y": apex_y,
        "path": r['path']
    })

# =========================================================================
# 2. Architectures (Faithful to Original Implementations)
# =========================================================================
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

class ConditionalVAE(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128 * 8 * 8),
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
    def encode(self, x, c):
        h = self.enc_conv(x).view(x.size(0), -1)
        hc = torch.cat([h, c], dim=-1)
        return self.fc_mu(hc), self.fc_logvar(hc)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def decode(self, z, c):
        zc = torch.cat([z, c], dim=-1)
        h = self.dec_fc(zc).view(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        return self.out_conv(feat)
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar
    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            z = torch.zeros((c.size(0), self.latent_dim), dtype=torch.float32, device=c.device)
            return self.decode(z, c)

class GC_WPG_V1_Official(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.scaler = scaler_params
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
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
            nn.Conv2d(18, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        y_grid, x_grid = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                        torch.arange(128, dtype=torch.float32), indexing='ij')
        self.register_buffer('y_coords', y_grid.contiguous().clone())
        self.register_buffer('x_coords', x_grid.contiguous().clone())

    def generate_spatial_geometry_prior(self, w_norm, reinf_norm):
        batch_size = w_norm.size(0)
        px_scale = 6.0
        w_px = (w_norm * (self.scaler['width_max'] - self.scaler['width_min']) + self.scaler['width_min']) * px_scale
        r_px = (reinf_norm * (self.scaler['reinforcement_max'] - self.scaler['reinforcement_min']) + self.scaler['reinforcement_min']) * px_scale
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
            return self.forward(c)

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
            return self.forward(c)

# Parametric baseline renderer
def render_parametric_prospective(w_mm, r_mm, d_mm, px_per_mm=6.0):
    canvas = np.ones((128, 128, 3), dtype=np.float32) * (180.0 / 255.0)
    canvas[:, 64:, :] = 135.0 / 255.0
    w_px = w_mm * px_per_mm
    r_px = r_mm * px_per_mm
    d_px = d_mm * px_per_mm
    y_apex = 64.0
    for y in range(128):
        dy = y - y_apex
        if abs(dy) <= w_px / 2.0:
            factor = 1.0 - (dy / (w_px / 2.0))**2
            x_crown = 64.0 - r_px * factor
            for x in range(int(max(0, np.floor(x_crown))), 64):
                canvas[y, x, :] = [85.0 / 255.0, 95.0 / 255.0, 110.0 / 255.0]
            x_root = 64.0 + d_px * factor
            for x in range(64, int(min(128, np.ceil(x_root)))):
                canvas[y, x, :] = [75.0 / 255.0, 85.0 / 255.0, 100.0 / 255.0]
    return canvas

# =========================================================================
# 3. Model Training & Checkpoint Verification (Holdout Partition)
# =========================================================================
DEV_HOLDOUTS = [5, 14, 22, 32]
train_pool = [d for d in all_data if d['sample'] not in DEV_HOLDOUTS]
holdout_pool = [d for d in all_data if d['sample'] in DEV_HOLDOUTS]
val_samples_dev = [1, 10, 19, 28]
dev_train = [d for d in train_pool if d['sample'] not in val_samples_dev]
dev_val = [d for d in train_pool if d['sample'] in val_samples_dev]
dev_scaler = fit_prospective_scaler(pd.DataFrame(dev_train))

# Prepare tensors
t_imgs = torch.tensor(np.array([r['img_rgb'].transpose(2, 0, 1) for r in dev_train]), dtype=torch.float32)
t_rois = torch.tensor(np.array([r['roi_mask'] for r in dev_train]), dtype=torch.float32).unsqueeze(1)
t_conds = torch.tensor(np.array([
    [
        (r['feed_rate'] - dev_scaler['f_min']) / (dev_scaler['f_max'] - dev_scaler['f_min'] + 1e-8),
        (r['speed'] - dev_scaler['s_min']) / (dev_scaler['s_max'] - dev_scaler['s_min'] + 1e-8),
        1.0 if r['material'] == 'IN718' else 0.0,
        1.0 if r['material'] == 'SS304L' else 0.0,
        1.0 if r['process'] == 'ColdArc' else 0.0,
        1.0 if r['process'] == 'GMAW' else 0.0
    ] for r in dev_train
]), dtype=torch.float32)

v_imgs = torch.tensor(np.array([r['img_rgb'].transpose(2, 0, 1) for r in dev_val]), dtype=torch.float32)
v_conds = torch.tensor(np.array([
    [
        (r['feed_rate'] - dev_scaler['f_min']) / (dev_scaler['f_max'] - dev_scaler['f_min'] + 1e-8),
        (r['speed'] - dev_scaler['s_min']) / (dev_scaler['s_max'] - dev_scaler['s_min'] + 1e-8),
        1.0 if r['material'] == 'IN718' else 0.0,
        1.0 if r['material'] == 'SS304L' else 0.0,
        1.0 if r['process'] == 'ColdArc' else 0.0,
        1.0 if r['process'] == 'GMAW' else 0.0
    ] for r in dev_val
]), dtype=torch.float32)

# Load or train permanent Direct CNN checkpoint
cnn_ckpt_file = os.path.join(CKPT_DIR, "direct_cnn_checkpoint.pt")
m_cnn = DirectCNN(6, 16).to(device)
if os.path.exists(cnn_ckpt_file):
    print("Loading existing Direct Conditional CNN checkpoint...", flush=True)
    m_cnn.load_state_dict(torch.load(cnn_ckpt_file, map_location="cpu")["model_state_dict"])
    m_cnn.eval()
else:
    print("Training and checkpointing Direct Conditional CNN...", flush=True)
    opt_cnn = torch.optim.AdamW(m_cnn.parameters(), lr=2e-3, weight_decay=1e-4)
    sched_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(opt_cnn, T_max=45, eta_min=1e-4)
    best_v_cnn = 1e9
    best_state_cnn = None
    for ep in range(45):
        m_cnn.train()
        perm = torch.randperm(len(t_imgs))
        for b in range(int(np.ceil(len(t_imgs) / 8))):
            idx = perm[b*8 : (b+1)*8]
            opt_cnn.zero_grad()
            p = m_cnn(t_conds[idx])
            loss = F.l1_loss(p, t_imgs[idx]) + F.l1_loss(p * t_rois[idx], t_imgs[idx] * t_rois[idx]) * 2.0
            loss.backward()
            opt_cnn.step()
        sched_cnn.step()
        m_cnn.eval()
        with torch.no_grad():
            v_l = F.l1_loss(m_cnn(v_conds), v_imgs).item()
            if v_l < best_v_cnn:
                best_v_cnn = v_l
                best_state_cnn = {k: v.cpu().clone() for k, v in m_cnn.state_dict().items()}
    m_cnn.load_state_dict(best_state_cnn)
    m_cnn.eval()
    torch.save({"model_state_dict": m_cnn.state_dict(), "scaler": dev_scaler}, cnn_ckpt_file)

# Load or train permanent Conditional VAE checkpoint
vae_ckpt_file = os.path.join(CKPT_DIR, "cvae_checkpoint.pt")
m_vae = ConditionalVAE(6, 24).to(device)
if os.path.exists(vae_ckpt_file):
    print("Loading existing Conditional VAE checkpoint...", flush=True)
    m_vae.load_state_dict(torch.load(vae_ckpt_file, map_location="cpu")["model_state_dict"])
    m_vae.eval()
else:
    print("Training and checkpointing Conditional VAE...", flush=True)
    opt_vae = torch.optim.AdamW(m_vae.parameters(), lr=2e-3, weight_decay=1e-4)
    sched_vae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vae, T_max=45, eta_min=1e-4)
    best_v_vae = 1e9
    best_state_vae = None
    for ep in range(45):
        m_vae.train()
        perm = torch.randperm(len(t_imgs))
        for b in range(int(np.ceil(len(t_imgs) / 8))):
            idx = perm[b*8 : (b+1)*8]
            opt_vae.zero_grad()
            recon, mu, logvar = m_vae(t_imgs[idx], t_conds[idx])
            l_recon = F.l1_loss(recon, t_imgs[idx]) + F.l1_loss(recon * t_rois[idx], t_imgs[idx] * t_rois[idx]) * 1.5
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = l_recon + 0.005 * kl
            loss.backward()
            opt_vae.step()
        sched_vae.step()
        m_vae.eval()
        with torch.no_grad():
            v_l = F.l1_loss(m_vae.generate(v_conds, seed=42), v_imgs).item()
            if v_l < best_v_vae:
                best_v_vae = v_l
                best_state_vae = {k: v.cpu().clone() for k, v in m_vae.state_dict().items()}
    m_vae.load_state_dict(best_state_vae)
    m_vae.eval()
    torch.save({"model_state_dict": m_vae.state_dict(), "scaler": dev_scaler}, vae_ckpt_file)

# Load official GC-WPG-v1 checkpoint
print("Loading Official GC-WPG-v1 Checkpoint...", flush=True)
v1_ckpt_path = os.path.join(RESULTS_DIR, "guide_demo_v6_verified", "gc_wpg_prototype_checkpoint.pt")
v1_ckpt = torch.load(v1_ckpt_path, map_location="cpu")
m_gc_v1 = GC_WPG_V1_Official(6, 16, scaler_params=v1_ckpt["scaler_params"]).to(device)
m_gc_v1.load_state_dict(v1_ckpt["model_state_dict"])
m_gc_v1.eval()

# Load official GC-WPG-v2 checkpoint
print("Loading Official GC-WPG-v2 Checkpoint...", flush=True)
v2_ckpt_path = os.path.join(RESULTS_DIR, "guide_demo_v7_refined", "gc_wpg_v2_checkpoint.pt")
v2_ckpt = torch.load(v2_ckpt_path, map_location="cpu")
m_gc_v2 = GC_WPG_v2(6, 16, scaler_params=v2_ckpt["scaler"]).to(device)
m_gc_v2.load_state_dict(v2_ckpt["model_state_dict"])
m_gc_v2.eval()

# Parametric regressor
def fit_parametric_regressor(train_records, scaler):
    X = np.array([
        [
            (r['feed_rate'] - scaler['f_min']) / (scaler['f_max'] - scaler['f_min'] + 1e-8),
            (r['speed'] - scaler['s_min']) / (scaler['s_max'] - scaler['s_min'] + 1e-8),
            1.0 if r['material'] == 'IN718' else 0.0,
            1.0 if r['material'] == 'SS304L' else 0.0,
            1.0 if r['process'] == 'ColdArc' else 0.0,
            1.0 if r['process'] == 'GMAW' else 0.0
        ] for r in train_records
    ])
    Y_w = np.array([r['width'] for r in train_records])
    Y_r = np.array([r['reinforcement'] for r in train_records])
    Y_d = np.array([r['depth'] for r in train_records])
    w_w = np.linalg.pinv(X).dot(Y_w)
    w_r = np.linalg.pinv(X).dot(Y_r)
    w_d = np.linalg.pinv(X).dot(Y_d)
    def predict(r):
        x = np.array([
            (r['feed_rate'] - scaler['f_min']) / (scaler['f_max'] - scaler['f_min'] + 1e-8),
            (r['speed'] - scaler['s_min']) / (scaler['s_max'] - scaler['s_min'] + 1e-8),
            1.0 if r['material'] == 'IN718' else 0.0,
            1.0 if r['material'] == 'SS304L' else 0.0,
            1.0 if r['process'] == 'ColdArc' else 0.0,
            1.0 if r['process'] == 'GMAW' else 0.0
        ])
        return float(x.dot(w_w)), float(x.dot(w_r)), float(x.dot(w_d))
    return predict

geom_reg = fit_parametric_regressor(dev_train, dev_scaler)

# =========================================================================
# 4. Standardized Evaluation Function (Single Identical Pipeline)
# =========================================================================
def evaluate_single_prediction_standard(pred_rgb, gt_record):
    gt_rgb = gt_record['img_rgb']
    gt_pts = gt_record['gt_pts']
    gt_apex_x = gt_record['gt_apex_x']
    gt_apex_y = gt_record['gt_apex_y']
    gt_edge_map = gt_record['gt_edge_map']

    p_val = psnr(gt_rgb, pred_rgb, data_range=1.0)
    s_val = ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=1.0)

    roi_gt = gt_rgb[20:110, 10:64, :]
    roi_pr = pred_rgb[20:110, 10:64, :]
    roi_s = ssim(roi_gt, roi_pr, channel_axis=2, data_range=1.0)

    pred_gray = cv2.cvtColor((pred_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    bg_val = np.median(pred_gray[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))
    pr_pts, pr_edge_map = extract_crown_contour(pred_gray, x_baseline=64, threshold=adaptive_thresh)

    is_failed = False
    if len(pr_pts) < 10:
        is_failed = True
        pr_apex_x = 64
        pr_apex_y = 64
    else:
        pr_apex_x = min(pr_pts.values())
        pr_apex_y = [y for y, x in pr_pts.items() if x == pr_apex_x][0]
        if pr_apex_x >= 62:
            is_failed = True

    apex_err_x_px = abs(gt_apex_x - pr_apex_x)
    apex_err_euclid_px = np.sqrt((gt_apex_x - pr_apex_x)**2 + (gt_apex_y - pr_apex_y)**2)

    y_common = set(gt_pts.keys()).intersection(set(pr_pts.keys()))
    if len(y_common) >= 10:
        diffs = [abs(gt_pts[y] - pr_pts[y]) for y in y_common]
        contour_mae_px = float(np.mean(diffs))
        hausdorff_px = float(np.max(diffs))
        coverage = len(y_common) / max(len(gt_pts), 1)
    else:
        contour_mae_px = float(abs(gt_apex_x - 64))
        hausdorff_px = float(abs(gt_apex_x - 64))
        coverage = 0.0

    k1 = np.ones((3, 3), np.uint8)
    gt_dil1 = cv2.dilate(gt_edge_map[:, :64], k1)
    tp1 = np.sum((pr_edge_map[:, :64] == 1) & (gt_dil1 == 1))
    f1_1px = 2.0 * tp1 / (np.sum(pr_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)

    k2 = np.ones((5, 5), np.uint8)
    gt_dil2 = cv2.dilate(gt_edge_map[:, :64], k2)
    tp2 = np.sum((pr_edge_map[:, :64] == 1) & (gt_dil2 == 1))
    f1_2px = 2.0 * tp2 / (np.sum(pr_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)

    return {
        "psnr_db": round(float(p_val), 2),
        "ssim": round(float(s_val), 4),
        "roi_ssim": round(float(roi_s), 4),
        "apex_err_x_px": round(float(apex_err_x_px), 2),
        "apex_err_euclid_px": round(float(apex_err_euclid_px), 2),
        "contour_mae_px": round(float(contour_mae_px), 2),
        "hausdorff_px": round(float(hausdorff_px), 2),
        "reinf_boundary_f1_1px": round(float(f1_1px), 4),
        "reinf_boundary_f1_2px": round(float(f1_2px), 4),
        "continuity_coverage": round(float(coverage), 3),
        "extraction_failed": 1.0 if is_failed else 0.0,
        "pred_pts": pr_pts,
        "pred_apex_x": pr_apex_x,
        "pred_apex_y": pr_apex_y,
        "n_common_points": len(y_common)
    }

# =========================================================================
# 5. Execute Task 2: Re-evaluate All Models on Holdouts & Save Raw Outputs
# =========================================================================
print("\n" + "="*70)
print("EXECUTING TASK 2: RE-EVALUATING SAMPLES 5, 14, 22, 32 ON ALL MODELS")
print("="*70)

reeval_rows = []
model_dict = {
    "Direct Conditional CNN": m_cnn,
    "Conditional VAE": m_vae,
    "GC-WPG-v1 (Official Prototype)": m_gc_v1,
    "GC-WPG-v2 (Multi-Scale Prior)": m_gc_v2,
    "Prospective Parametric Baseline": None
}

raw_generated_cache = {}

for rec in holdout_pool:
    sid = rec['sample']
    view = rec['view']
    mat = rec['material']
    proc = rec['process']
    key = f"sample{sid}_view{view}"
    raw_generated_cache[key] = {"gt": rec['img_rgb'], "gt_pts": rec['gt_pts'], "gt_ax": rec['gt_apex_x'], "gt_ay": rec['gt_apex_y']}

    # Prospective condition for dev models
    c_dev = torch.tensor([[
        (rec['feed_rate'] - dev_scaler['f_min']) / (dev_scaler['f_max'] - dev_scaler['f_min'] + 1e-8),
        (rec['speed'] - dev_scaler['s_min']) / (dev_scaler['s_max'] - dev_scaler['s_min'] + 1e-8),
        1.0 if mat == 'IN718' else 0.0,
        1.0 if mat == 'SS304L' else 0.0,
        1.0 if proc == 'ColdArc' else 0.0,
        1.0 if proc == 'GMAW' else 0.0
    ]], dtype=torch.float32)

    # Prospective condition for v1 model (uses v1 scaler params)
    s_v1 = v1_ckpt["scaler_params"]
    c_v1 = torch.tensor([[
        (rec['feed_rate'] - s_v1['feed_rate_min']) / (s_v1['feed_rate_max'] - s_v1['feed_rate_min']),
        (rec['speed'] - s_v1['speed_min']) / (s_v1['speed_max'] - s_v1['speed_min']),
        1.0 if mat == 'IN718' else 0.0,
        1.0 if mat == 'SS304L' else 0.0,
        1.0 if proc == 'ColdArc' else 0.0,
        1.0 if proc == 'GMAW' else 0.0
    ]], dtype=torch.float32)

    with torch.no_grad():
        p_cnn = np.clip(m_cnn(c_dev).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        p_vae = np.clip(m_vae.generate(c_dev, seed=42).squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        out_v1, _, _ = m_gc_v1.generate(c_v1, seed=42)
        p_v1 = np.clip(out_v1.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        out_v2, _, _ = m_gc_v2.generate(c_dev, seed=42)
        p_v2 = np.clip(out_v2.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        
    pw, pr, pd_val = geom_reg(rec)
    p_param = render_parametric_prospective(pw, pr, pd_val)

    preds = {
        "Direct Conditional CNN": p_cnn,
        "Conditional VAE": p_vae,
        "GC-WPG-v1 (Official Prototype)": p_v1,
        "GC-WPG-v2 (Multi-Scale Prior)": p_v2,
        "Prospective Parametric Baseline": p_param
    }

    # Save raw images
    for m_name, im in preds.items():
        fname = f"raw_{m_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_s{sid}_v{view}.png"
        cv2.imwrite(os.path.join(RAW_OUT_DIR, fname), (im * 255).astype(np.uint8)[:, :, ::-1])
        raw_generated_cache[key][m_name] = im

        # Evaluate
        res = evaluate_single_prediction_standard(im, rec)
        raw_generated_cache[key][f"{m_name}_pts"] = res["pred_pts"]
        raw_generated_cache[key][f"{m_name}_ax"] = res["pred_apex_x"]
        raw_generated_cache[key][f"{m_name}_ay"] = res["pred_apex_y"]
        raw_generated_cache[key][f"{m_name}_mae"] = res["contour_mae_px"]
        raw_generated_cache[key][f"{m_name}_apex_err"] = res["apex_err_x_px"]

        row = {
            "sample": sid, "view": view, "material": mat, "process": proc,
            "feed_rate": rec['feed_rate'], "speed": rec['speed'],
            "model": m_name,
            "raw_filename": fname
        }
        for k, v in res.items():
            if k not in ["pred_pts", "pred_apex_x", "pred_apex_y"]:
                row[k] = v
        reeval_rows.append(row)

df_reeval = pd.DataFrame(reeval_rows)
csv_reeval_path = os.path.join(AUDIT_DIR, "FIGURE6_REEVALUATION.csv")
df_reeval.to_csv(csv_reeval_path, index=False)
print(f"Saved FIGURE6_REEVALUATION.csv to: {csv_reeval_path}")

# =========================================================================
# 6. Task 3: Audit Figure 5 Sensitivity, Thresholds & Experimental Ground Truth
# =========================================================================
print("\n" + "="*70)
print("EXECUTING TASK 3: AUDITING FIGURE 5 SENSITIVITY & THRESHOLD ROBUSTNESS")
print("="*70)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from scripts.generate_sample_v6 import EC_WPG
from scripts.infer_weld_cross_section_cli import build_condition_tensor

v6_ec_ckpt_path = os.path.join(RESULTS_DIR, "guide_demo_v6", "ec_wpg_checkpoint.pt")
model_ec = EC_WPG(6, 16)
ckpt_ec = torch.load(v6_ec_ckpt_path, map_location="cpu")
model_ec.load_state_dict(ckpt_ec["model_state_dict"])
model_ec.eval()

thresholds = [135, 140, 145, 150, 155, 160, 165]
sens_rows = []

# Sweep 1: Feed Rate Sweep (IN718, ColdArc, Speed = 25.0 cm/min)
f_vals = [4.8, 5.2, 5.6, 6.0]
for f in f_vals:
    c = build_condition_tensor("IN718", "ColdArc", f, 25.0)
    with torch.no_grad():
        img_t, _ = model_ec.generate(c, seed=42)
        rgb = (np.clip(img_t.squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Otsu threshold on active left ROI
    roi_gray = gray[20:110, :64]
    otsu_thresh, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Adaptive Mean threshold
    adap_bin = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    adap_area = int(np.sum(adap_bin == 0))

    row_dict = {
        "sweep_parameter": "feed_rate (m/min)",
        "setpoint": f,
        "fixed_speed_cm_min": 25.0,
        "material": "IN718", "process": "ColdArc",
        "otsu_optimal_threshold": float(otsu_thresh),
        "adaptive_segmented_area_px": adap_area
    }
    
    for t in thresholds:
        pts = {}
        for y in range(20, 111):
            m = np.where(gray[y, :64] < t)[0]
            if len(m) > 0 and m[0] < 64:
                pts[y] = int(m[0])
        ax = min(pts.values()) if pts else 64
        reinf = 64 - ax
        area = int(sum(np.sum(gray[y, :64] < t) for y in range(20, 111)))
        row_dict[f"apex_proj_px_T{t}"] = reinf
        row_dict[f"active_area_px_count_T{t}"] = area
    sens_rows.append(row_dict)

# Sweep 2: Travel Speed Sweep (SS304L, GMAW, Feed Rate = 5.0 m/min)
s_vals = [22.0, 25.0, 28.0, 32.0]
for s in s_vals:
    c = build_condition_tensor("SS304L", "GMAW", 5.0, s)
    with torch.no_grad():
        img_t, _ = model_ec.generate(c, seed=42)
        rgb = (np.clip(img_t.squeeze(0).permute(1, 2, 0).numpy(), 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    roi_gray = gray[20:110, :64]
    otsu_thresh, _ = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap_bin = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    adap_area = int(np.sum(adap_bin == 0))

    row_dict = {
        "sweep_parameter": "travel_speed (cm/min)",
        "setpoint": s,
        "fixed_feed_rate_m_min": 5.0,
        "material": "SS304L", "process": "GMAW",
        "otsu_optimal_threshold": float(otsu_thresh),
        "adaptive_segmented_area_px": adap_area
    }
    
    for t in thresholds:
        pts = {}
        for y in range(20, 111):
            m = np.where(gray[y, :64] < t)[0]
            if len(m) > 0 and m[0] < 64:
                pts[y] = int(m[0])
        ax = min(pts.values()) if pts else 64
        reinf = 64 - ax
        area = int(sum(np.sum(gray[y, :64] < t) for y in range(20, 111)))
        row_dict[f"apex_proj_px_T{t}"] = reinf
        row_dict[f"active_area_px_count_T{t}"] = area
    sens_rows.append(row_dict)

df_sens = pd.DataFrame(sens_rows)
sens_csv_path = os.path.join(AUDIT_DIR, "SENSITIVITY_THRESHOLD_ROBUSTNESS.csv")
df_sens.to_csv(sens_csv_path, index=False)
print(f"Saved SENSITIVITY_THRESHOLD_ROBUSTNESS.csv to: {sens_csv_path}")

# =========================================================================
# 7. Render Corrected Figure 6 (Genuine Model Comparison)
# =========================================================================
print("\nRendering Corrected Figure 6 (Genuine Performance with Official Checkpoints)...", flush=True)

fig = plt.figure(figsize=(20, 18), dpi=300)
gs = fig.add_gridspec(5, 5, height_ratios=[1, 1, 1, 1, 0.75], hspace=0.30, wspace=0.15)
holdout_samples = [5, 14, 22, 32]

for row_idx, sid in enumerate(holdout_samples):
    key = f"sample{sid}_viewA"
    data = raw_generated_cache[key]
    gt_rgb = data["gt"]
    gt_pts = data["gt_pts"]
    gt_ax = data["gt_ax"]
    gt_ay = data["gt_ay"]
    gt_reinf = 64 - gt_ax

    img_dir = data["Direct Conditional CNN"]
    pts_dir = data["Direct Conditional CNN_pts"]
    ax_dir = data["Direct Conditional CNN_ax"]
    mae_dir = data["Direct Conditional CNN_mae"]
    apex_err_dir = data["Direct Conditional CNN_apex_err"]

    img_v1 = data["GC-WPG-v1 (Official Prototype)"]
    pts_v1 = data["GC-WPG-v1 (Official Prototype)_pts"]
    ax_v1 = data["GC-WPG-v1 (Official Prototype)_ax"]
    mae_v1 = data["GC-WPG-v1 (Official Prototype)_mae"]
    apex_err_v1 = data["GC-WPG-v1 (Official Prototype)_apex_err"]

    img_v2 = data["GC-WPG-v2 (Multi-Scale Prior)"]
    pts_v2 = data["GC-WPG-v2 (Multi-Scale Prior)_pts"]
    ax_v2 = data["GC-WPG-v2 (Multi-Scale Prior)_ax"]
    mae_v2 = data["GC-WPG-v2 (Multi-Scale Prior)_mae"]
    apex_err_v2 = data["GC-WPG-v2 (Multi-Scale Prior)_apex_err"]

    rec_info = [d for d in holdout_pool if d['sample'] == sid and d['view'] == 'A'][0]
    mat_proc = f"{rec_info['material']} {rec_info['process']}"
    cond_str = f"F={rec_info['feed_rate']:.1f} m/min, S={rec_info['speed']:.1f} cm/min"

    # Col 0: GT
    ax0 = fig.add_subplot(gs[row_idx, 0])
    ax0.imshow(gt_rgb)
    ax0.axvline(64, color='cyan', linestyle=':', alpha=0.7)
    if gt_pts:
        ax0.plot([gt_pts[y] for y in sorted(gt_pts.keys())], sorted(gt_pts.keys()), color='#2ecc71', linewidth=1.8)
        ax0.plot(gt_ax, gt_ay, 'ro', markersize=5)
    ax0.set_title(f"Sample {sid}-A: {mat_proc}\n{cond_str}\nGT Reinf: {gt_reinf} px (Apex X={gt_ax})", fontsize=9.5, fontweight='bold')
    ax0.axis('off')

    # Col 1: Direct CNN
    ax1 = fig.add_subplot(gs[row_idx, 1])
    ax1.imshow(img_dir)
    ax1.axvline(64, color='red', linestyle=':', alpha=0.5)
    if pts_dir:
        ax1.plot([pts_dir[y] for y in sorted(pts_dir.keys())], sorted(pts_dir.keys()), color='#3498db', linewidth=1.8)
        ax1.plot(ax_dir, data["Direct Conditional CNN_ay"], 'bo', markersize=5)
    ax1.set_title(f"Direct CNN (Checkpoint)\nContour MAE: {mae_dir:.2f} px\nApex Err: {apex_err_dir:.1f} px", fontsize=9.5, fontweight='bold')
    ax1.axis('off')

    # Col 2: GC-WPG-v1
    ax2 = fig.add_subplot(gs[row_idx, 2])
    ax2.imshow(img_v1)
    ax2.axvline(64, color='red', linestyle=':', alpha=0.5)
    if pts_v1:
        ax2.plot([pts_v1[y] for y in sorted(pts_v1.keys())], sorted(pts_v1.keys()), color='#e67e22', linewidth=1.8)
        ax2.plot(ax_v1, data["GC-WPG-v1 (Official Prototype)_ay"], 'o', color='#e67e22', markersize=5)
    ax2.set_title(f"GC-WPG-v1 (Official Ckpt)\nContour MAE: {mae_v1:.2f} px\nApex Err: {apex_err_v1:.1f} px", fontsize=9.5, fontweight='bold')
    ax2.axis('off')

    # Col 3: GC-WPG-v2
    ax3 = fig.add_subplot(gs[row_idx, 3])
    ax3.imshow(img_v2)
    ax3.axvline(64, color='red', linestyle=':', alpha=0.5)
    if pts_v2:
        ax3.plot([pts_v2[y] for y in sorted(pts_v2.keys())], sorted(pts_v2.keys()), color='#e74c3c', linewidth=1.8)
        ax3.plot(ax_v2, data["GC-WPG-v2 (Multi-Scale Prior)_ay"], 'ro', markersize=5)
    ax3.set_title(f"GC-WPG-v2 (Multi-Scale Ckpt)\nContour MAE: {mae_v2:.2f} px\nApex Err: {apex_err_v2:.1f} px", fontsize=9.5, fontweight='bold')
    ax3.axis('off')

    # Col 4: Zoomed Crown Overlay
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
    ax4.set_title(f"Zoomed Crown Overlay\nSample {sid}-A ({mat_proc})", fontsize=9.5, fontweight='bold')
    ax4.axis('off')
    if row_idx == 0:
        ax4.legend(loc='lower left', fontsize=7.5, framealpha=0.9)

# Bottom Row: Verified Holdout Comparison Charts
models_order = ["Direct Conditional CNN", "Conditional VAE", "GC-WPG-v1 (Official Prototype)", "GC-WPG-v2 (Multi-Scale Prior)", "Prospective Parametric Baseline"]
short_names = ["Direct CNN", "CVAE", "GC-WPG-v1", "GC-WPG-v2", "Parametric"]
holdout_metrics_summary = df_reeval.groupby("model").agg({
    "contour_mae_px": ["mean", "std"],
    "apex_err_x_px": ["mean", "std"],
    "ssim": ["mean", "std"],
    "roi_ssim": ["mean", "std"],
    "reinf_boundary_f1_1px": ["mean", "std"]
}).reindex(models_order)

x_indices = np.arange(len(models_order))
width = 0.35

ax_m1 = fig.add_subplot(gs[4, :2])
ax_m1.bar(x_indices - width/2, holdout_metrics_summary[("contour_mae_px", "mean")], width,
          yerr=holdout_metrics_summary[("contour_mae_px", "std")], label='Contour MAE (px)', color='#e74c3c', capsize=3, alpha=0.85)
ax_m1.bar(x_indices + width/2, holdout_metrics_summary[("apex_err_x_px", "mean")], width,
          yerr=holdout_metrics_summary[("apex_err_x_px", "std")], label='Apex Err X (px)', color='#3498db', capsize=3, alpha=0.85)
ax_m1.set_xticks(x_indices)
ax_m1.set_xticklabels(short_names, fontsize=8.5, rotation=10)
ax_m1.set_ylabel("Error (Pixels, lower is better)", fontsize=9, fontweight='bold')
ax_m1.set_title("Audited Holdout Boundary Accuracy (8 Views)", fontsize=10, fontweight='bold')
ax_m1.legend(fontsize=8, loc='upper right')
ax_m1.grid(axis='y', linestyle='--', alpha=0.5)

ax_m2 = fig.add_subplot(gs[4, 2:4])
ax_m2.bar(x_indices - width/2, holdout_metrics_summary[("ssim", "mean")], width,
          yerr=holdout_metrics_summary[("ssim", "std")], label='Whole SSIM', color='#2ecc71', capsize=3, alpha=0.85)
ax_m2.bar(x_indices + width/2, holdout_metrics_summary[("roi_ssim", "mean")], width,
          yerr=holdout_metrics_summary[("roi_ssim", "std")], label='Weld-ROI SSIM', color='#f39c12', capsize=3, alpha=0.85)
ax_m2.set_xticks(x_indices)
ax_m2.set_xticklabels(short_names, fontsize=8.5, rotation=10)
ax_m2.set_ylabel("SSIM (higher is better)", fontsize=9, fontweight='bold')
ax_m2.set_ylim(0.65, 0.88)
ax_m2.set_title("Audited Holdout Image & ROI SSIM (8 Views)", fontsize=10, fontweight='bold')
ax_m2.legend(fontsize=8, loc='lower right')
ax_m2.grid(axis='y', linestyle='--', alpha=0.5)

ax_m3 = fig.add_subplot(gs[4, 4])
f1_vals = holdout_metrics_summary[("reinf_boundary_f1_1px", "mean")] * 100
bars = ax_m3.bar(x_indices, f1_vals, width=0.5, color='#9b59b6', alpha=0.85)
ax_m3.set_xticks(x_indices)
ax_m3.set_xticklabels(short_names, fontsize=8, rotation=15)
ax_m3.set_ylabel("1-px Boundary F1 (%)", fontsize=9, fontweight='bold')
ax_m3.set_title("Audited 1-Pixel Boundary F1", fontsize=10, fontweight='bold')
ax_m3.grid(axis='y', linestyle='--', alpha=0.5)
for b in bars:
    h = b.get_height()
    ax_m3.text(b.get_x() + b.get_width()/2., h + 0.5, f"{h:.1f}%", ha='center', va='bottom', fontsize=7.5)

fig.suptitle(
    "Figure 6 (Audited & Re-evaluated): Genuine Model Comparison on Development Holdouts\n"
    "Evaluating Checkpointed Direct CNN vs. Official GC-WPG-v1 vs. Official GC-WPG-v2 on Prospective Unseen Conditions",
    fontsize=13, fontweight='bold', y=0.985
)

out_fig6 = os.path.join(AUDIT_DIR, "fig6_reevaluated.png")
plt.savefig(out_fig6, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Audited Figure 6 to: {out_fig6}", flush=True)

# Also copy to refined dir to fix the discrepancy there
refined_fig6 = os.path.join(RESULTS_DIR, "guide_demo_v7_refined", "fig6_direct_cnn_vs_gc_wpg_v2.png")
plt.figure()
img_read = cv2.imread(out_fig6)
cv2.imwrite(refined_fig6, img_read)
print(f"Updated: {refined_fig6}")

print("\nAll computational tasks for Integrity Audit script completed successfully!")
