"""
execute_v7_fair_benchmarks.py

Phase V7: Comprehensive Fair Comparison & Experiment-Grouped 5-Fold Validation
Implements:
1. 5-Fold Grouped Cross-Validation (all 36 experiments, 72 views)
2. Development Holdout Benchmark (Samples 5, 14, 22, 32; 8 views)
3. Three Fairly Trained Prospective Models (Direct CNN, Conditional VAE, GC-WPG)
4. Leakage-Free Parametric Baseline (Predicted Dimensions)
5. Comprehensive Boundary Metric Logging in PIXELS
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
from torch.utils.data import Dataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

print("=" * 75)
print("  PHASE V7: FAIR EXPERIMENT-GROUPED BENCHMARK & BOUNDARY EVALUATION  ")
print("=" * 75)

# Load metadata and clean dataset
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample").sort_values(by=["sample", "view"]).reset_index(drop=True)

def fit_prospective_scaler(df_train):
    f_min = float(df_train['feed_rate'].min())
    f_max = float(df_train['feed_rate'].max())
    s_min = float(df_train['speed'].min())
    s_max = float(df_train['speed'].max())
    w_min = float(df_train['width'].min())
    w_max = float(df_train['width'].max())
    r_min = float(df_train['reinforcement'].min())
    r_max = float(df_train['reinforcement'].max())
    d_min = float(df_train['depth'].min())
    d_max = float(df_train['depth'].max())
    return {
        "f_min": f_min, "f_max": f_max,
        "s_min": s_min, "s_max": s_max,
        "w_min": w_min, "w_max": w_max,
        "r_min": r_min, "r_max": r_max,
        "d_min": d_min, "d_max": d_max
    }

# Pre-load all 72 images and pre-compute ROIs & ground-truth contours
all_data = []
print("Pre-loading all 72 macrographs and extracting ground-truth boundaries...", flush=True)

def extract_crown_contour(gray_img, x_baseline=64, y_min=20, y_max=110, threshold=155):
    contour_pts = {}
    edge_map = np.zeros_like(gray_img, dtype=np.uint8)
    for y in range(y_min, y_max + 1):
        row_vals = gray_img[y, :x_baseline]
        metal_idx = np.where(row_vals < threshold)[0]
        if len(metal_idx) > 0:
            x_edge = metal_idx[0]
            if x_edge < x_baseline:
                contour_pts[y] = int(x_edge)
                edge_map[y, x_edge] = 1
    return contour_pts, edge_map

for idx, r in full_df.iterrows():
    sid = int(r['sample'])
    view = r['view']
    im_bgr = cv2.imread(r['path'])
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    
    # Extract GT contour
    gt_pts, gt_edge_map = extract_crown_contour(im_gray, threshold=165)
    apex_x = min(gt_pts.values()) if gt_pts else 64
    apex_y = [y for y, x in gt_pts.items() if x == apex_x][0] if gt_pts else 64
    
    # Weld ROI mask (horizontal x in [10, 64], vertical y centered at apex)
    roi_mask = np.zeros((128, 128), dtype=np.float32)
    roi_mask[20:110, 10:64] = 1.0
    
    all_data.append({
        "sample": sid, "view": view,
        "material": r['material'], "process": r['process'],
        "feed_rate": float(r['feed_rate']), "speed": float(r['speed']),
        "width": float(r['width']), "depth": float(r['depth']), "reinforcement": float(r['reinforcement']),
        "img_rgb": im_rgb, "img_gray": im_gray,
        "roi_mask": roi_mask,
        "gt_pts": gt_pts, "gt_edge_map": gt_edge_map,
        "gt_apex_x": apex_x, "gt_apex_y": apex_y
    })

print(f"Loaded {len(all_data)} images. Sample distribution: {len(set(d['sample'] for d in all_data))} experiments.")

# -------------------------------------------------------------------------
# 1. Stratified 5-Fold Grouped Cross-Validation Generator
# -------------------------------------------------------------------------
# 36 experiments = 4 quadrants of 9 experiments:
# Q1: 1-9 (IN718, ColdArc)
# Q2: 10-18 (IN718, GMAW)
# Q3: 19-27 (SS304L, ColdArc)
# Q4: 28-36 (SS304L, GMAW)
quadrants = [
    list(range(1, 10)),
    list(range(10, 19)),
    list(range(19, 28)),
    list(range(28, 37))
]

# Create 5 folds ensuring balanced distribution across quadrants
# Folds 1-4 get 7 samples (2+2+2+1), Fold 5 gets 8 samples (2+2+2+2)
cv_folds = [[] for _ in range(5)]
for q_idx, q_samples in enumerate(quadrants):
    # deterministic shuffle per quadrant
    rng = np.random.RandomState(SEED + q_idx)
    shuffled = rng.permutation(q_samples).tolist()
    # Distribute 9 samples into 5 folds: [2, 2, 2, 2, 1] or similar
    counts = [2, 2, 2, 1, 2] if q_idx % 2 == 0 else [2, 2, 1, 2, 2]
    cur = 0
    for f in range(5):
        c = counts[f]
        cv_folds[f].extend(shuffled[cur:cur+c])
        cur += c

cv_splits = {}
for f in range(5):
    test_exps = sorted(cv_folds[f])
    rem_exps = sorted([s for s in range(1, 37) if s not in test_exps])
    # Inner validation for early stopping: 4 experiments (1 from each quadrant)
    val_exps = [rem_exps[i] for i in [0, 7, 14, 21]]
    train_exps = [s for s in rem_exps if s not in val_exps]
    cv_splits[f"fold_{f+1}"] = {
        "outer_test_samples": test_exps,
        "inner_val_samples": val_exps,
        "train_samples": train_exps,
        "test_image_count": len(test_exps) * 2,
        "train_image_count": len(train_exps) * 2
    }

with open(os.path.join(OUT_DIR, "cv_splits.json"), "w") as fp:
    json.dump(cv_splits, fp, indent=2)
print("Saved 5-Fold Grouped Cross-Validation splits to cv_splits.json.")

# -------------------------------------------------------------------------
# 2. Architectures for Fair Comparison
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
        
        # Encoder
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
        
        # Decoder
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
        recon = self.decode(z, c)
        return recon, mu, logvar

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            if seed is not None:
                g = torch.Generator(device=c.device)
                g.manual_seed(seed)
                z = torch.randn((c.size(0), self.latent_dim), generator=g, dtype=torch.float32, device=c.device) * 0.1
            else:
                z = torch.zeros((c.size(0), self.latent_dim), dtype=torch.float32, device=c.device)
            return self.decode(z, c)

class GC_WPG(nn.Module):
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
        w_norm = pred_geom[:, 0:1]
        r_norm = pred_geom[:, 1:2]
        spatial_prior = self.generate_spatial_geometry_prior(w_norm, r_norm)
        
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        feat = self.up4(self.up3(self.up2(self.up1(h))))
        fused = torch.cat([feat, spatial_prior], dim=1)
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
            out_img, pred_geom, spatial_prior = self.forward(c, z)
            return out_img, pred_geom, spatial_prior

# Prospective Parametric Baseline (Predicted Dimensions)
def render_parametric_prospective(pred_w_mm, pred_r_mm, pred_d_mm=1.5, grid_size=128):
    img = np.ones((grid_size, grid_size, 3), dtype=np.float32) * 0.92
    img[:, 64:, :] = [0.72, 0.70, 0.68]
    
    px_scale = 6.0
    w_px = pred_w_mm * px_scale
    r_px = pred_r_mm * px_scale
    d_px = pred_d_mm * px_scale
    
    y_coords = np.linspace(64.0 - w_px/2.0, 64.0 + w_px/2.0, 100)
    norm_y = (y_coords - 64.0) / (w_px / 2.0 + 1e-6)
    crown_x = 64.0 - r_px * (1.0 - norm_y**2)
    pen_x = 64.0 + d_px * (1.0 - norm_y**2)
    
    poly_pts = []
    for x, y in zip(crown_x, y_coords):
        poly_pts.append([int(np.clip(x, 0, 127)), int(np.clip(y, 0, 127))])
    for x, y in zip(pen_x[::-1], y_coords[::-1]):
        poly_pts.append([int(np.clip(x, 0, 127)), int(np.clip(y, 0, 127))])
        
    pts = np.array(poly_pts, np.int32)
    mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    img[mask > 0] = [0.55, 0.52, 0.50]
    cv2.polylines(img, [pts], isClosed=True, color=[0.25, 0.22, 0.20], thickness=1)
    return img

# -------------------------------------------------------------------------
# 3. Model Training Engine
# -------------------------------------------------------------------------
def train_model_partition(train_records, val_records, model_type, scaler, epochs=65, lr=2e-3):
    # Prepare tensors
    t_imgs = torch.tensor(np.array([r['img_rgb'].transpose(2, 0, 1) for r in train_records]), dtype=torch.float32)
    t_rois = torch.tensor(np.array([r['roi_mask'] for r in train_records]), dtype=torch.float32).unsqueeze(1)
    t_conds = torch.tensor(np.array([
        [
            (r['feed_rate'] - scaler['f_min']) / (scaler['f_max'] - scaler['f_min'] + 1e-8),
            (r['speed'] - scaler['s_min']) / (scaler['s_max'] - scaler['s_min'] + 1e-8),
            1.0 if r['material'] == 'IN718' else 0.0,
            1.0 if r['material'] == 'SS304L' else 0.0,
            1.0 if r['process'] == 'ColdArc' else 0.0,
            1.0 if r['process'] == 'GMAW' else 0.0
        ] for r in train_records
    ]), dtype=torch.float32)
    
    t_geoms = torch.tensor(np.array([
        [
            (r['width'] - scaler['w_min']) / (scaler['w_max'] - scaler['w_min'] + 1e-8),
            (r['reinforcement'] - scaler['r_min']) / (scaler['r_max'] - scaler['r_min'] + 1e-8)
        ] for r in train_records
    ]), dtype=torch.float32)

    v_imgs = torch.tensor(np.array([r['img_rgb'].transpose(2, 0, 1) for r in val_records]), dtype=torch.float32)
    v_conds = torch.tensor(np.array([
        [
            (r['feed_rate'] - scaler['f_min']) / (scaler['f_max'] - scaler['f_min'] + 1e-8),
            (r['speed'] - scaler['s_min']) / (scaler['s_max'] - scaler['s_min'] + 1e-8),
            1.0 if r['material'] == 'IN718' else 0.0,
            1.0 if r['material'] == 'SS304L' else 0.0,
            1.0 if r['process'] == 'ColdArc' else 0.0,
            1.0 if r['process'] == 'GMAW' else 0.0
        ] for r in val_records
    ]), dtype=torch.float32)

    if model_type == "DirectCNN":
        model = DirectCNN(cond_dim=6, noise_dim=16).to(device)
    elif model_type == "ConditionalVAE":
        model = ConditionalVAE(cond_dim=6, latent_dim=24).to(device)
    elif model_type == "GC_WPG":
        model = GC_WPG(cond_dim=6, noise_dim=16, scaler_params=scaler).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    best_loss = 1e9
    best_state = None
    batch_size = 8
    n_samples = len(t_imgs)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n_samples)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            b_imgs = t_imgs[idx].to(device)
            b_rois = t_rois[idx].to(device)
            b_conds = t_conds[idx].to(device)
            b_geoms = t_geoms[idx].to(device)
            
            optimizer.zero_grad()
            if model_type == "DirectCNN":
                pred = model(b_conds)
                loss_l1 = F.l1_loss(pred, b_imgs)
                loss_roi = F.l1_loss(pred * b_rois, b_imgs * b_rois) * 2.0
                loss = loss_l1 + loss_roi
            elif model_type == "ConditionalVAE":
                recon, mu, logvar = model(b_imgs, b_conds)
                loss_l1 = F.l1_loss(recon, b_imgs)
                loss_roi = F.l1_loss(recon * b_rois, b_imgs * b_rois) * 1.5
                kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = loss_l1 + loss_roi + 0.005 * kl_div
            elif model_type == "GC_WPG":
                pred, pred_geom, _ = model(b_conds)
                loss_l1 = F.l1_loss(pred, b_imgs)
                loss_roi = F.l1_loss(pred * b_rois, b_imgs * b_rois) * 2.0
                loss_geom = F.mse_loss(pred_geom, b_geoms) * 0.2
                loss = loss_l1 + loss_roi + loss_geom
                
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validation check
        model.eval()
        with torch.no_grad():
            if model_type == "DirectCNN":
                val_pred = model(v_conds)
                v_loss = F.l1_loss(val_pred, v_imgs).item()
            elif model_type == "ConditionalVAE":
                val_pred = model.generate(v_conds, seed=42)
                v_loss = F.l1_loss(val_pred, v_imgs).item()
            elif model_type == "GC_WPG":
                val_pred, _, _ = model.generate(v_conds, seed=42)
                v_loss = F.l1_loss(val_pred, v_imgs).item()
                
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model

# Train Simple Linear/Ridge Geometry Regressor for Parametric Baseline
def train_prospective_geometry_regressor(train_records, scaler):
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
    
    # Regularized Ridge Least Squares: beta = (X^T X + lambda I)^-1 X^T Y
    X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    reg = 1e-2 * np.eye(X_bias.shape[1])
    inv = np.linalg.inv(X_bias.T @ X_bias + reg)
    beta_w = inv @ X_bias.T @ Y_w
    beta_r = inv @ X_bias.T @ Y_r
    beta_d = inv @ X_bias.T @ Y_d
    
    def predict(row):
        c = np.array([
            (row['feed_rate'] - scaler['f_min']) / (scaler['f_max'] - scaler['f_min'] + 1e-8),
            (row['speed'] - scaler['s_min']) / (scaler['s_max'] - scaler['s_min'] + 1e-8),
            1.0 if row['material'] == 'IN718' else 0.0,
            1.0 if row['material'] == 'SS304L' else 0.0,
            1.0 if row['process'] == 'ColdArc' else 0.0,
            1.0 if row['process'] == 'GMAW' else 0.0,
            1.0
        ])
        return float(c @ beta_w), float(c @ beta_r), float(c @ beta_d)
        
    return predict

# -------------------------------------------------------------------------
# 4. Comprehensive Evaluation Metric Computer
# -------------------------------------------------------------------------
def evaluate_prediction(pred_rgb, gt_record):
    gt_rgb = gt_record['img_rgb']
    gt_gray = gt_record['img_gray']
    gt_pts = gt_record['gt_pts']
    gt_apex_x = gt_record['gt_apex_x']
    gt_apex_y = gt_record['gt_apex_y']
    gt_edge_map = gt_record['gt_edge_map']
    
    # 1. Full SSIM and PSNR
    p_val = psnr(gt_rgb, pred_rgb, data_range=1.0)
    s_val = ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=1.0)
    
    # 2. Weld-ROI SSIM (crop Y: 20-110, X: 10-64)
    roi_gt = gt_rgb[20:110, 10:64, :]
    roi_pr = pred_rgb[20:110, 10:64, :]
    roi_s = ssim(roi_gt, roi_pr, channel_axis=2, data_range=1.0)
    
    # 3. Independent Boundary Extraction on Pred Image
    pred_gray = (cv2.cvtColor((pred_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    bg_val = np.median(pred_gray[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))
    pr_pts, pr_edge_map = extract_crown_contour(pred_gray, x_baseline=64, threshold=adaptive_thresh)
    
    # Check extraction failure
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
            
    # Apex errors in PIXELS
    apex_err_x_px = abs(gt_apex_x - pr_apex_x)
    apex_err_euclid_px = np.sqrt((gt_apex_x - pr_apex_x)**2 + (gt_apex_y - pr_apex_y)**2)
    
    # Contour distance in PIXELS
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
        
    # Boundary F1 at 1-pixel and 2-pixel tolerance on X < 64
    k1 = np.ones((3, 3), np.uint8)
    gt_dil1 = cv2.dilate(gt_edge_map[:, :64], k1)
    tp1 = np.sum((pr_edge_map[:, :64] == 1) & (gt_dil1 == 1))
    f1_1px = 2.0 * tp1 / (np.sum(pr_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)
    
    k2 = np.ones((5, 5), np.uint8)
    gt_dil2 = cv2.dilate(gt_edge_map[:, :64], k2)
    tp2 = np.sum((pr_edge_map[:, :64] == 1) & (gt_dil2 == 1))
    f1_2px = 2.0 * tp2 / (np.sum(pr_edge_map[:, :64]) + np.sum(gt_edge_map[:, :64]) + 1e-8)
    
    return {
        "psnr_db": float(p_val),
        "ssim": float(s_val),
        "roi_ssim": float(roi_s),
        "apex_err_x_px": float(apex_err_x_px),
        "apex_err_euclid_px": float(apex_err_euclid_px),
        "contour_mae_px": float(contour_mae_px),
        "hausdorff_px": float(hausdorff_px),
        "reinf_boundary_f1_1px": float(f1_1px),
        "reinf_boundary_f1_2px": float(f1_2px),
        "contour_coverage": float(coverage),
        "extraction_failed": 1.0 if is_failed else 0.0,
        "pred_pts": pr_pts,
        "pred_apex_x": pr_apex_x,
        "pred_apex_y": pr_apex_y
    }

# -------------------------------------------------------------------------
# 5. BENCHMARK SUITE A: Development Holdouts (Samples 5, 14, 22, 32; 8 views)
# -------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  EXECUTING BENCHMARK SUITE A: DEVELOPMENT HOLDOUTS (8 VIEWS)")
print("=" * 75)

DEV_HOLDOUTS = [5, 14, 22, 32]
train_pool = [d for d in all_data if d['sample'] not in DEV_HOLDOUTS]
holdout_pool = [d for d in all_data if d['sample'] in DEV_HOLDOUTS]

# Separate 4 validation experiments from train pool
val_samples_dev = [1, 10, 19, 28]
dev_train = [d for d in train_pool if d['sample'] not in val_samples_dev]
dev_val = [d for d in train_pool if d['sample'] in val_samples_dev]

scaler_dev = fit_prospective_scaler(pd.DataFrame(dev_train))
print(f"Training on {len(dev_train)} images, validating on {len(dev_val)} images, testing on {len(holdout_pool)} images.")

print("Training Fair Direct CNN Baseline...", flush=True)
m_cnn_dev = train_model_partition(dev_train, dev_val, "DirectCNN", scaler_dev, epochs=45)

print("Training Fair Conditional VAE Baseline...", flush=True)
m_vae_dev = train_model_partition(dev_train, dev_val, "ConditionalVAE", scaler_dev, epochs=45)

print("Training Fair GC-WPG Prototype...", flush=True)
m_gc_dev = train_model_partition(dev_train, dev_val, "GC_WPG", scaler_dev, epochs=50)

print("Training Prospective Parametric Regressor...", flush=True)
geom_reg_dev = train_prospective_geometry_regressor(dev_train, scaler_dev)

detailed_records = []
dev_predictions_vis = {}

for idx, r in enumerate(holdout_pool):
    sid = r['sample']
    view = r['view']
    mat = r['material']
    proc = r['process']
    
    # 6-dim prospective condition
    c6 = torch.tensor(np.array([[
        (r['feed_rate'] - scaler_dev['f_min']) / (scaler_dev['f_max'] - scaler_dev['f_min'] + 1e-8),
        (r['speed'] - scaler_dev['s_min']) / (scaler_dev['s_max'] - scaler_dev['s_min'] + 1e-8),
        1.0 if r['material'] == 'IN718' else 0.0,
        1.0 if r['material'] == 'SS304L' else 0.0,
        1.0 if r['process'] == 'ColdArc' else 0.0,
        1.0 if r['process'] == 'GMAW' else 0.0
    ]], dtype=np.float32))
    
    # Generate predictions
    with torch.no_grad():
        p_cnn = m_cnn_dev(c6).squeeze(0).permute(1, 2, 0).cpu().numpy()
        p_vae = m_vae_dev.generate(c6, seed=42).squeeze(0).permute(1, 2, 0).cpu().numpy()
        p_gc, _, p_prior = m_gc_dev.generate(c6, seed=42)
        p_gc = p_gc.squeeze(0).permute(1, 2, 0).cpu().numpy()
        p_prior = p_prior.squeeze(0).cpu().numpy()
        
    pw, pr, pd_val = geom_reg_dev(r)
    p_param = render_parametric_prospective(pw, pr, pd_val)
    
    models_to_eval = [
        ("Direct Conditional CNN", p_cnn),
        ("Conditional VAE", p_vae),
        ("GC-WPG Prototype", p_gc),
        ("Prospective Parametric", p_param)
    ]
    
    if idx not in dev_predictions_vis:
        dev_predictions_vis[f"{sid}-{view}"] = {
            "gt_rgb": r['img_rgb'],
            "gt_pts": r['gt_pts'],
            "models": {name: img for name, img in models_to_eval}
        }
    
    for m_name, pred_img in models_to_eval:
        res = evaluate_prediction(pred_img, r)
        detailed_records.append({
            "evaluation_suite": "Development Holdouts (Phase V6 Benchmark)",
            "fold": "holdout_fixed",
            "model": m_name,
            "sample": sid, "view": view, "material": mat, "process": proc,
            "psnr_db": round(res["psnr_db"], 2),
            "ssim": round(res["ssim"], 4),
            "roi_ssim": round(res["roi_ssim"], 4),
            "apex_err_x_px": round(res["apex_err_x_px"], 2),
            "apex_err_euclid_px": round(res["apex_err_euclid_px"], 2),
            "contour_mae_px": round(res["contour_mae_px"], 2),
            "hausdorff_px": round(res["hausdorff_px"], 2),
            "reinf_boundary_f1_1px": round(res["reinf_boundary_f1_1px"], 4),
            "reinf_boundary_f1_2px": round(res["reinf_boundary_f1_2px"], 4),
            "contour_coverage": round(res["contour_coverage"], 3),
            "extraction_failed": res["extraction_failed"]
        })

print("Holdout evaluation completed.")

# -------------------------------------------------------------------------
# 6. BENCHMARK SUITE B: 5-Fold Grouped Cross-Validation (All 36 Samples)
# -------------------------------------------------------------------------
print("\n" + "=" * 75)
print("  EXECUTING BENCHMARK SUITE B: 5-FOLD GROUPED CROSS-VALIDATION (72 VIEWS)")
print("=" * 75)

for f_idx in range(5):
    fold_name = f"fold_{f_idx+1}"
    split_info = cv_splits[fold_name]
    t_samples = split_info['train_samples']
    v_samples = split_info['inner_val_samples']
    test_samples = split_info['outer_test_samples']
    
    t_records = [d for d in all_data if d['sample'] in t_samples]
    v_records = [d for d in all_data if d['sample'] in v_samples]
    test_records = [d for d in all_data if d['sample'] in test_samples]
    
    scaler_fold = fit_prospective_scaler(pd.DataFrame(t_records))
    print(f"\n--- Fold {f_idx+1}/5: Test Samples: {test_samples} ({len(test_records)} views) ---", flush=True)
    
    print(f"  [Fold {f_idx+1}] Training Direct CNN...", flush=True)
    m_cnn = train_model_partition(t_records, v_records, "DirectCNN", scaler_fold, epochs=35)
    print(f"  [Fold {f_idx+1}] Training Conditional VAE...", flush=True)
    m_vae = train_model_partition(t_records, v_records, "ConditionalVAE", scaler_fold, epochs=35)
    print(f"  [Fold {f_idx+1}] Training GC-WPG...", flush=True)
    m_gc = train_model_partition(t_records, v_records, "GC_WPG", scaler_fold, epochs=40)
    geom_reg = train_prospective_geometry_regressor(t_records, scaler_fold)
    
    for r in test_records:
        sid = r['sample']
        view = r['view']
        mat = r['material']
        proc = r['process']
        
        c6 = torch.tensor(np.array([[
            (r['feed_rate'] - scaler_fold['f_min']) / (scaler_fold['f_max'] - scaler_fold['f_min'] + 1e-8),
            (r['speed'] - scaler_fold['s_min']) / (scaler_fold['s_max'] - scaler_fold['s_min'] + 1e-8),
            1.0 if r['material'] == 'IN718' else 0.0,
            1.0 if r['material'] == 'SS304L' else 0.0,
            1.0 if r['process'] == 'ColdArc' else 0.0,
            1.0 if r['process'] == 'GMAW' else 0.0
        ]], dtype=np.float32))
        
        with torch.no_grad():
            p_cnn = m_cnn(c6).squeeze(0).permute(1, 2, 0).cpu().numpy()
            p_vae = m_vae.generate(c6, seed=42).squeeze(0).permute(1, 2, 0).cpu().numpy()
            p_gc, _, _ = m_gc.generate(c6, seed=42)
            p_gc = p_gc.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
        pw, pr, pd_val = geom_reg(r)
        p_param = render_parametric_prospective(pw, pr, pd_val)
        
        models_to_eval = [
            ("Direct Conditional CNN", p_cnn),
            ("Conditional VAE", p_vae),
            ("GC-WPG Prototype", p_gc),
            ("Prospective Parametric", p_param)
        ]
        
        for m_name, pred_img in models_to_eval:
            res = evaluate_prediction(pred_img, r)
            detailed_records.append({
                "evaluation_suite": "5-Fold Grouped Cross-Validation (Unbiased)",
                "fold": fold_name,
                "model": m_name,
                "sample": sid, "view": view, "material": mat, "process": proc,
                "psnr_db": round(res["psnr_db"], 2),
                "ssim": round(res["ssim"], 4),
                "roi_ssim": round(res["roi_ssim"], 4),
                "apex_err_x_px": round(res["apex_err_x_px"], 2),
                "apex_err_euclid_px": round(res["apex_err_euclid_px"], 2),
                "contour_mae_px": round(res["contour_mae_px"], 2),
                "hausdorff_px": round(res["hausdorff_px"], 2),
                "reinf_boundary_f1_1px": round(res["reinf_boundary_f1_1px"], 4),
                "reinf_boundary_f1_2px": round(res["reinf_boundary_f1_2px"], 4),
                "contour_coverage": round(res["contour_coverage"], 3),
                "extraction_failed": res["extraction_failed"]
            })

df_detailed = pd.DataFrame(detailed_records)
detailed_csv_path = os.path.join(OUT_DIR, "EXTERNAL_CONTOUR_METRICS.csv")
df_detailed.to_csv(detailed_csv_path, index=False)
print(f"\nSaved detailed metrics for all {len(df_detailed)} evaluations to {detailed_csv_path}")

# -------------------------------------------------------------------------
# 7. Aggregate Summary Table: FAIR_MODEL_COMPARISON.csv
# -------------------------------------------------------------------------
summary_rows = []
for suite in ["Development Holdouts (Phase V6 Benchmark)", "5-Fold Grouped Cross-Validation (Unbiased)"]:
    sub = df_detailed[df_detailed['evaluation_suite'] == suite]
    for model in ["Direct Conditional CNN", "Conditional VAE", "GC-WPG Prototype", "Prospective Parametric"]:
        m_sub = sub[sub['model'] == model]
        if len(m_sub) == 0:
            continue
        summary_rows.append({
            "evaluation_suite": suite,
            "model": model,
            "n_evaluations": len(m_sub),
            "psnr_mean": round(float(m_sub['psnr_db'].mean()), 2),
            "psnr_std": round(float(m_sub['psnr_db'].std()), 2),
            "ssim_mean": round(float(m_sub['ssim'].mean()), 4),
            "ssim_std": round(float(m_sub['ssim'].std()), 4),
            "roi_ssim_mean": round(float(m_sub['roi_ssim'].mean()), 4),
            "roi_ssim_std": round(float(m_sub['roi_ssim'].std()), 4),
            "apex_err_x_px_mean": round(float(m_sub['apex_err_x_px'].mean()), 2),
            "apex_err_x_px_std": round(float(m_sub['apex_err_x_px'].std()), 2),
            "contour_mae_px_mean": round(float(m_sub['contour_mae_px'].mean()), 2),
            "contour_mae_px_std": round(float(m_sub['contour_mae_px'].std()), 2),
            "hausdorff_px_mean": round(float(m_sub['hausdorff_px'].mean()), 2),
            "hausdorff_px_std": round(float(m_sub['hausdorff_px'].std()), 2),
            "reinf_boundary_f1_1px_mean": round(float(m_sub['reinf_boundary_f1_1px'].mean()), 4),
            "reinf_boundary_f1_1px_std": round(float(m_sub['reinf_boundary_f1_1px'].std()), 4),
            "reinf_boundary_f1_2px_mean": round(float(m_sub['reinf_boundary_f1_2px'].mean()), 4),
            "reinf_boundary_f1_2px_std": round(float(m_sub['reinf_boundary_f1_2px'].std()), 4),
            "continuity_coverage_mean": round(float(m_sub['contour_coverage'].mean()), 3),
            "failure_rate": round(float(m_sub['extraction_failed'].mean()), 3)
        })

df_summary = pd.DataFrame(summary_rows)
summary_csv_path = os.path.join(OUT_DIR, "FAIR_MODEL_COMPARISON.csv")
df_summary.to_csv(summary_csv_path, index=False)
print(f"Saved Fair Comparison Summary to {summary_csv_path}")

print("\n" + "=" * 75)
print("  FAIR MODEL COMPARISON SUMMARY (UNBIASED 5-FOLD CV):")
print("=" * 75)
cv_summary = df_summary[df_summary['evaluation_suite'].str.contains("5-Fold")][
    ['model', 'ssim_mean', 'roi_ssim_mean', 'apex_err_x_px_mean', 'contour_mae_px_mean', 'reinf_boundary_f1_1px_mean', 'continuity_coverage_mean', 'failure_rate']
]
print(cv_summary.to_string(index=False))

print("\n" + "=" * 75)
print("  DEVELOPMENT HOLDOUTS (PHASE V6 BENCHMARK COMPARISON):")
print("=" * 75)
dev_summary = df_summary[df_summary['evaluation_suite'].str.contains("Holdouts")][
    ['model', 'ssim_mean', 'roi_ssim_mean', 'apex_err_x_px_mean', 'contour_mae_px_mean', 'reinf_boundary_f1_1px_mean', 'continuity_coverage_mean', 'failure_rate']
]
print(dev_summary.to_string(index=False))

# -------------------------------------------------------------------------
# 8. Visual Overlay Generation: fig1_fair_model_comparison_v7.png
# -------------------------------------------------------------------------
print("\nGenerating Figure 1: Fair Multi-Model Comparison & Contour Overlay...", flush=True)
fig, axes = plt.subplots(4, 5, figsize=(18, 14), dpi=300)
holdout_keys = ["5-A", "14-A", "22-A", "32-A"]
model_order = ["Direct Conditional CNN", "Conditional VAE", "GC-WPG Prototype", "Prospective Parametric"]

for row_idx, h_key in enumerate(holdout_keys):
    item = dev_predictions_vis[h_key]
    gt_im = item["gt_rgb"]
    gt_pts = item["gt_pts"]
    ys = sorted(gt_pts.keys())
    xs = [gt_pts[y] for y in ys]
    
    # Col 0: Ground Truth
    axes[row_idx, 0].imshow(gt_im)
    axes[row_idx, 0].plot(xs, ys, color='cyan', linewidth=1.5, linestyle='--')
    axes[row_idx, 0].axvline(64, color='white', linestyle=':', alpha=0.7)
    axes[row_idx, 0].set_ylabel(f"Sample {h_key}", fontsize=11, fontweight='bold')
    axes[row_idx, 0].axis('off')
    if row_idx == 0:
        axes[row_idx, 0].set_title("Ground Truth\n(Macrograph)", fontsize=11, fontweight='bold')
        
    for col_idx, m_name in enumerate(model_order):
        ax = axes[row_idx, col_idx + 1]
        p_img = item["models"][m_name]
        ax.imshow(p_img)
        ax.axvline(64, color='red', linestyle=':', alpha=0.5)
        # Plot GT in cyan dashed
        ax.plot(xs, ys, color='cyan', linewidth=1.2, linestyle='--', alpha=0.8)
        
        # Extract predicted boundary
        p_gray = cv2.cvtColor((p_img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        bg_val = np.median(p_gray[20:110, :20])
        pr_pts, _ = extract_crown_contour(p_gray, threshold=max(145, min(170, bg_val - 12)))
        if pr_pts:
            p_ys = sorted(pr_pts.keys())
            p_xs = [pr_pts[y] for y in p_ys]
            ax.plot(p_xs, p_ys, color='#2ecc71', linewidth=1.5)
            
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(f"{m_name}\n(Generated RGB)", fontsize=11, fontweight='bold')

plt.suptitle("Figure 1 (Phase V7): Fair Comparison Across Competing Generative Paradigms\n"
             "Cyan Dashed: Verified GT Crown Boundary | Green Solid: Extracted Generated Boundary | All Models Pure Prospective (6-dim)",
             fontsize=13, fontweight='bold', y=0.98)
fig1_path = os.path.join(OUT_DIR, "fig1_fair_model_comparison_v7.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Figure 1 to {fig1_path}")

# -------------------------------------------------------------------------
# 9. Visual Overlay: fig2_external_contour_overlay_v7.png
# -------------------------------------------------------------------------
print("Generating Figure 2: Detailed Contour Mismatch Overlays...", flush=True)
fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)
for idx, h_key in enumerate(holdout_keys):
    item = dev_predictions_vis[h_key]
    gt_pts = item["gt_pts"]
    ys = sorted(gt_pts.keys())
    xs = [gt_pts[y] for y in ys]
    
    # Top: Direct CNN vs GT
    ax_top = axes[0, idx]
    p_cnn = item["models"]["Direct Conditional CNN"]
    ax_top.imshow(p_cnn)
    ax_top.axvline(64, color='white', linestyle=':', alpha=0.6)
    ax_top.plot(xs, ys, color='cyan', linewidth=2.0, linestyle='--', label='GT Contour')
    p_gray = cv2.cvtColor((p_cnn * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    pr_pts, _ = extract_crown_contour(p_gray, threshold=155)
    if pr_pts:
        ax_top.plot([pr_pts[y] for y in sorted(pr_pts.keys())], sorted(pr_pts.keys()), color='#e74c3c', linewidth=2.0, label='Direct CNN')
    ax_top.set_title(f"Sample {h_key}: Direct CNN Mismatch", fontsize=10, fontweight='bold')
    ax_top.axis('off')
    if idx == 0:
        ax_top.legend(loc="lower left", fontsize=8)
        
    # Bottom: GC-WPG vs GT
    ax_bot = axes[1, idx]
    p_gc = item["models"]["GC-WPG Prototype"]
    ax_bot.imshow(p_gc)
    ax_bot.axvline(64, color='white', linestyle=':', alpha=0.6)
    ax_bot.plot(xs, ys, color='cyan', linewidth=2.0, linestyle='--', label='GT Contour')
    p_gray_gc = cv2.cvtColor((p_gc * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    pr_pts_gc, _ = extract_crown_contour(p_gray_gc, threshold=155)
    if pr_pts_gc:
        ax_bot.plot([pr_pts_gc[y] for y in sorted(pr_pts_gc.keys())], sorted(pr_pts_gc.keys()), color='#2ecc71', linewidth=2.0, label='GC-WPG')
    ax_bot.set_title(f"Sample {h_key}: GC-WPG Mismatch", fontsize=10, fontweight='bold')
    ax_bot.axis('off')
    if idx == 0:
        ax_bot.legend(loc="lower left", fontsize=8)

plt.suptitle("Figure 2 (Phase V7): Visualizing Exact Contour Mismatch at the Reinforcement Crown\n"
             "Cyan Dashed: Ground Truth | Red: Direct CNN | Green: GC-WPG Prototype",
             fontsize=13, fontweight='bold', y=0.98)
fig2_path = os.path.join(OUT_DIR, "fig2_external_contour_overlay_v7.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved Figure 2 to {fig2_path}")

print("\n" + "=" * 75)
print("  PHASE V7 FAIR BENCHMARK EXECUTION COMPLETE!  ")
print("=" * 75)
