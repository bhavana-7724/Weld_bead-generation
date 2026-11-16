"""
run_v7_fair_comparison.py

Phase V7: Verified Weld Boundary Prediction and Generalization
Fair, leakage-free comparative evaluation of:
1. Direct Conditional CNN (Deterministic Prospective Baseline)
2. True Conditional VAE (Stochastic Latent Baseline)
3. Geometry-Conditioned Generator (GC-WPG Prototype)
4. Prospective Parametric Baseline (Predicted Dimensions, Zero Test Leakage)
5. Historical EC-WPG & DDPM Baselines (Documented context)

All models operate under identical conditions:
- Pure prospective 6-dimensional input vector (Feed rate, Speed, Material [one-hot], Process [one-hot])
- Identical 128x128 resolution, aligned coordinate frame
- Grouped cross-validation (Views A and B never separated)
- Scalers fitted strictly on training partitions
- Primary boundary metrics evaluated strictly in PIXELS on the external reinforcement crown (X < 64)
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

# Load dataset metadata
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample").sort_values(by=["sample", "view"]).reset_index(drop=True)

# -------------------------------------------------------------------------
# 1. Standardized Feature Processing (Pure Prospective: 6 Features)
# -------------------------------------------------------------------------
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

def transform_prospective(row, scaler):
    f_norm = (row['feed_rate'] - scaler['f_min']) / (scaler['f_max'] - scaler['f_min'] + 1e-8)
    s_norm = (row['speed'] - scaler['s_min']) / (scaler['s_max'] - scaler['s_min'] + 1e-8)
    c_cat = [
        1.0 if row['material'] == 'IN718' else 0.0,
        1.0 if row['material'] == 'SS304L' else 0.0,
        1.0 if row['process'] == 'ColdArc' else 0.0,
        1.0 if row['process'] == 'GMAW' else 0.0
    ]
    return np.array([f_norm, s_norm] + c_cat, dtype=np.float32)

# -------------------------------------------------------------------------
# 2. Strict Boundary Extraction Algorithm (Independent of GT)
# -------------------------------------------------------------------------
def extract_external_crown_contour(gray_img, x_baseline=64, y_min=20, y_max=110, threshold=155):
    """
    Extracts 1D boundary profile x(y) for rows y in [y_min, y_max] where the specimen crown
    projects laterally into the background (x < x_baseline).
    Returns dict {y: x} of leftmost boundary points and a binary edge image (128x128).
    Strictly independent of ground truth reference.
    """
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

# -------------------------------------------------------------------------
# 3. Model Architectures
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

# Model 1: Direct Conditional CNN
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

# Model 2: Genuine Conditional VAE
class ConditionalVAE(nn.Module):
    def __init__(self, cond_dim=6, latent_dim=24):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        # Encoder: 3x128x128 -> latent
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1), # 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc_mu = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8 + cond_dim, latent_dim)
        
        # Decoder: (latent + cond) -> 3x128x128
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
                z = torch.randn((c.size(0), self.latent_dim), generator=g, dtype=torch.float32, device=c.device)
            else:
                z = torch.zeros((c.size(0), self.latent_dim), dtype=torch.float32, device=c.device)
            return self.decode(z, c)

# Model 3: Geometry-Conditioned Generator (GC-WPG)
class GC_WPG(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, scaler_params=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.scaler = scaler_params
        
        # Upstream geometry regression head: predicts [W, R] in normalized space
        self.geom_head = nn.Sequential(
            nn.Linear(cond_dim, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Linear(32, 2), # [W_norm, R_norm]
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
        
        # Final fusion head: 16 visual channels + 2 spatial prior channels -> 3 RGB
        self.fusion_head = nn.Sequential(
            nn.Conv2d(16 + 2, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        y_grid, x_grid = torch.meshgrid(torch.arange(128, dtype=torch.float32),
                                        torch.arange(128, dtype=torch.float32), indexing='ij')
        self.register_buffer("y_coords", y_grid)
        self.register_buffer("x_coords", x_grid)

    def generate_spatial_geometry_prior(self, w_norm, reinf_norm):
        batch_size = w_norm.size(0)
        # Scale to pixel coordinates
        # Assumed nominal scale 6.0 px/mm only inside the generator coordinate layer
        px_scale = 6.0
        w_px = (w_norm * (self.scaler['w_max'] - self.scaler['w_min']) + self.scaler['w_min']) * px_scale
        r_px = (reinf_norm * (self.scaler['r_max'] - self.scaler['r_min']) + self.scaler['r_min']) * px_scale
        w_px = w_px.view(batch_size, 1, 1)
        r_px = r_px.view(batch_size, 1, 1)
        
        y_dist = (self.y_coords.unsqueeze(0) - 64.0) / (w_px / 2.0 + 1e-6)
        profile_factor = torch.clamp(1.0 - y_dist**2, min=0.0)
        x_crown_bnd = 64.0 - r_px * profile_factor
        
        # Smooth spatial mask
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

# Model 4: Prospective Parametric Renderer (Zero Test Leakage)
def render_parametric_prospective(pred_w_mm, pred_r_mm, pred_d_mm=1.5, grid_size=128):
    """
    Renders analytical parametric profile using PREDICTED dimensions (zero ground truth leakage).
    """
    img = np.ones((grid_size, grid_size, 3), dtype=np.float32) * 0.92
    img[:, 64:, :] = [0.72, 0.70, 0.68] # base metal
    
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

print("V7 Architecture and Boundary modules defined successfully.", flush=True)
