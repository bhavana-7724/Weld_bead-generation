"""
train_and_evaluate_gc_wpg_v2.py

Trains and evaluates GC-WPG-v2 with Multi-Scale Spatial Prior Injection:
- Injects spatial geometry priors at 16x16, 32x32, 64x64, and 128x128
- Uses gradient alignment loss to prevent boundary dissipation
- Evaluated on:
  1. Development Holdouts (Samples 5, 14, 22, 32; 8 views)
  2. 5-Fold Grouped Cross-Validation (all 36 samples, 72 views)
- Reports all spatial metrics in PIXELS
- Saves GC_WPG_V2_METRICS.csv and REFINED_MODEL_COMPARISON.csv
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
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7_refined")
V7_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cpu")

# Load existing cv_splits.json
with open(os.path.join(V7_DIR, "cv_splits.json"), "r") as fp:
    cv_splits = json.load(fp)

# Load metadata and clean dataset exactly as in Phase V7
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample").sort_values(by=["sample", "view"]).reset_index(drop=True)

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

all_data = []
print("Pre-loading all 72 macrographs and extracting ground-truth boundaries...", flush=True)
for idx, r in full_df.iterrows():
    sid = int(r['sample'])
    view = r['view']
    im_bgr = cv2.imread(r['path'])
    if im_bgr is None:
        raise RuntimeError(f"Missing image: {r['path']}")
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    
    gt_pts, gt_edge_map = extract_crown_contour(im_gray, threshold=165)
    if len(gt_pts) > 0:
        apex_x = min(gt_pts.values())
        apex_y = [y for y, x in gt_pts.items() if x == apex_x][0]
    else:
        apex_x, apex_y = 64, 64
        
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

print(f"Loaded {len(all_data)} images across {len(set(d['sample'] for d in all_data))} experiments.")

def fit_prospective_scaler(records):
    f_vals = [r['feed_rate'] for r in records]
    s_vals = [r['speed'] for r in records]
    w_vals = [r['width'] for r in records]
    r_vals = [r['reinforcement'] for r in records]
    d_vals = [r['depth'] for r in records]
    return {
        "f_min": float(min(f_vals)), "f_max": float(max(f_vals)),
        "s_min": float(min(s_vals)), "s_max": float(max(s_vals)),
        "w_min": float(min(w_vals)), "w_max": float(max(w_vals)),
        "r_min": float(min(r_vals)), "r_max": float(max(r_vals)),
        "d_min": float(min(d_vals)), "d_max": float(max(d_vals))
    }

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
        
        # 1. Upstream Geometry Prediction Head
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
        
        # 2. Main Decoder Backbone
        in_dim = cond_dim + noise_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Upsampling stages
        self.up1 = ResUpBlock(128, 64)  # 8 -> 16
        self.fuse1 = MultiScaleSpatialFusion(64, prior_channels=2)
        
        self.up2 = ResUpBlock(64, 32)   # 16 -> 32
        self.fuse2 = MultiScaleSpatialFusion(32, prior_channels=2)
        
        self.up3 = ResUpBlock(32, 16)   # 32 -> 64
        self.fuse3 = MultiScaleSpatialFusion(16, prior_channels=2)
        
        self.up4 = ResUpBlock(16, 16)   # 64 -> 128
        self.fuse4 = MultiScaleSpatialFusion(16, prior_channels=2)
        
        # Final output convolution
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
        w_norm = pred_geom[:, 0:1]
        r_norm = pred_geom[:, 1:2]
        
        # 128x128 full prior
        prior_128 = self.generate_spatial_geometry_prior(w_norm, r_norm)
        
        # Downsampled priors for intermediate stages
        prior_64 = F.interpolate(prior_128, size=(64, 64), mode='bilinear', align_corners=False)
        prior_32 = F.interpolate(prior_128, size=(32, 32), mode='bilinear', align_corners=False)
        prior_16 = F.interpolate(prior_128, size=(16, 16), mode='bilinear', align_corners=False)
        
        # Decoder forward pass with multi-scale injection
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        
        h1 = self.fuse1(self.up1(h), prior_16)
        h2 = self.fuse2(self.up2(h1), prior_32)
        h3 = self.fuse3(self.up3(h2), prior_64)
        h4 = self.fuse4(self.up4(h3), prior_128)
        
        out_img = self.out_conv(h4)
        return out_img, pred_geom, prior_128

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            if seed is not None:
                g = torch.Generator(device=c.device)
                g.manual_seed(seed)
                z = torch.randn((c.size(0), self.noise_dim), generator=g, dtype=torch.float32, device=c.device) * 0.05
            else:
                z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
            out_img, pred_geom, prior = self.forward(c, z)
            return out_img, pred_geom, prior

def train_gc_wpg_v2(train_records, val_records, scaler, epochs=80, lr=1e-3):
    t_imgs = torch.tensor(np.array([r['img_rgb'].transpose(2, 0, 1) for r in train_records]), dtype=torch.float32)
    t_rois = torch.tensor(np.array([np.expand_dims(r['roi_mask'], 0) for r in train_records]), dtype=torch.float32)
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

    model = GC_WPG_v2(cond_dim=6, noise_dim=16, scaler_params=scaler).to(device)
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
            pred, pred_geom, _ = model(b_conds)
            
            loss_l1 = F.l1_loss(pred, b_imgs)
            loss_roi = F.l1_loss(pred * b_rois, b_imgs * b_rois) * 2.0
            loss_geom = F.mse_loss(pred_geom, b_geoms) * 0.2
            
            # Boundary gradient alignment loss
            pred_grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
            gt_grad_x = torch.abs(b_imgs[:, :, :, 1:] - b_imgs[:, :, :, :-1])
            loss_grad = F.l1_loss(pred_grad_x * b_rois[:, :, :, 1:], gt_grad_x * b_rois[:, :, :, 1:]) * 0.5
            
            loss = loss_l1 + loss_roi + loss_geom + loss_grad
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validation check
        model.eval()
        with torch.no_grad():
            val_pred, _, _ = model.generate(v_conds, seed=42)
            v_loss = F.l1_loss(val_pred, v_imgs).item()
            if v_loss < best_loss:
                best_loss = v_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model

def evaluate_prediction(pred_rgb, gt_record):
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
    
    pred_gray = (cv2.cvtColor((pred_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    bg_val = np.median(pred_gray[20:110, :20])
    adaptive_thresh = max(145, min(170, bg_val - 12))
    pr_pts, pr_edge_map = extract_crown_contour(pred_gray, x_baseline=64, threshold=adaptive_thresh)
    
    is_failed = False
    if len(pr_pts) < 10:
        is_failed = True
        pr_apex_x, pr_apex_y = 64, 64
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
        "pred_apex_y": pr_apex_y,
        "pred_rgb": pred_rgb
    }

# -------------------------------------------------------------------------
# Execution Pipeline
# -------------------------------------------------------------------------
all_eval_metrics = []

# Suite A: Development Holdouts
print("\n" + "="*70)
print("TRAINING & EVALUATING GC-WPG-v2 ON DEVELOPMENT HOLDOUTS (8 VIEWS)")
print("="*70)
DEV_HOLDOUTS = [5, 14, 22, 32]
train_pool = [d for d in all_data if d['sample'] not in DEV_HOLDOUTS]
holdout_pool = [d for d in all_data if d['sample'] in DEV_HOLDOUTS]
val_samples_dev = [1, 10, 19, 28]
dev_train = [d for d in train_pool if d['sample'] not in val_samples_dev]
dev_val = [d for d in train_pool if d['sample'] in val_samples_dev]
dev_scaler = fit_prospective_scaler(dev_train)

print(f"Training GC-WPG-v2 on {len(dev_train)} images, validating on {len(dev_val)} images...")
t0 = time.time()
model_dev = train_gc_wpg_v2(dev_train, dev_val, dev_scaler, epochs=80)
print(f"Completed training in {time.time() - t0:.1f}s.")

dev_holdout_preds = {}
for rec in holdout_pool:
    c = torch.tensor([[
        (rec['feed_rate'] - dev_scaler['f_min']) / (dev_scaler['f_max'] - dev_scaler['f_min'] + 1e-8),
        (rec['speed'] - dev_scaler['s_min']) / (dev_scaler['s_max'] - dev_scaler['s_min'] + 1e-8),
        1.0 if rec['material'] == 'IN718' else 0.0,
        1.0 if rec['material'] == 'SS304L' else 0.0,
        1.0 if rec['process'] == 'ColdArc' else 0.0,
        1.0 if rec['process'] == 'GMAW' else 0.0
    ]], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred_t, _, _ = model_dev.generate(c, seed=42)
        pred_rgb = np.clip(pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        
    metrics = evaluate_prediction(pred_rgb, rec)
    dev_holdout_preds[f"s{rec['sample']}_v{rec['view']}"] = metrics
    
    metrics_row = {
        "evaluation_suite": "Development Holdouts (Phase V6 Benchmark)",
        "fold": "holdout_dev",
        "sample": rec['sample'], "view": rec['view'],
        "material": rec['material'], "process": rec['process'],
        "feed_rate": rec['feed_rate'], "speed": rec['speed'],
        "model": "GC-WPG-v2 (Multi-Scale Prior)"
    }
    for k, v in metrics.items():
        if k not in ["pred_pts", "pred_rgb"]:
            metrics_row[k] = v
    all_eval_metrics.append(metrics_row)

# Suite B: 5-Fold Grouped Cross-Validation
print("\n" + "="*70)
print("TRAINING & EVALUATING GC-WPG-v2 ON 5-FOLD GROUPED CV (72 VIEWS)")
print("="*70)

for f_idx in range(5):
    fold_name = f"fold_{f_idx+1}"
    split = cv_splits[fold_name]
    test_exps = split["outer_test_samples"]
    val_exps = split["inner_val_samples"]
    train_exps = split["train_samples"]
    
    fold_train = [d for d in all_data if d['sample'] in train_exps]
    fold_val = [d for d in all_data if d['sample'] in val_exps]
    fold_test = [d for d in all_data if d['sample'] in test_exps]
    
    scaler_cv = fit_prospective_scaler(fold_train)
    print(f"\nFold {f_idx+1}/5: Train {len(fold_train)} imgs, Val {len(fold_val)} imgs, Test {len(fold_test)} imgs...")
    t0 = time.time()
    model_cv = train_gc_wpg_v2(fold_train, fold_val, scaler_cv, epochs=80)
    print(f"Fold {f_idx+1} trained in {time.time() - t0:.1f}s.")
    
    for rec in fold_test:
        c = torch.tensor([[
            (rec['feed_rate'] - scaler_cv['f_min']) / (scaler_cv['f_max'] - scaler_cv['f_min'] + 1e-8),
            (rec['speed'] - scaler_cv['s_min']) / (scaler_cv['s_max'] - scaler_cv['s_min'] + 1e-8),
            1.0 if rec['material'] == 'IN718' else 0.0,
            1.0 if rec['material'] == 'SS304L' else 0.0,
            1.0 if rec['process'] == 'ColdArc' else 0.0,
            1.0 if rec['process'] == 'GMAW' else 0.0
        ]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            pred_t, _, _ = model_cv.generate(c, seed=42)
            pred_rgb = np.clip(pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
            
        metrics = evaluate_prediction(pred_rgb, rec)
        metrics_row = {
            "evaluation_suite": "5-Fold Grouped Cross-Validation (Unbiased)",
            "fold": fold_name,
            "sample": rec['sample'], "view": rec['view'],
            "material": rec['material'], "process": rec['process'],
            "feed_rate": rec['feed_rate'], "speed": rec['speed'],
            "model": "GC-WPG-v2 (Multi-Scale Prior)"
        }
        for k, v in metrics.items():
            if k not in ["pred_pts", "pred_rgb"]:
                metrics_row[k] = v
        all_eval_metrics.append(metrics_row)

df_all = pd.DataFrame(all_eval_metrics)
metrics_csv_path = os.path.join(OUT_DIR, "GC_WPG_V2_METRICS.csv")
df_all.to_csv(metrics_csv_path, index=False)
print(f"\nSaved GC_WPG_V2_METRICS.csv to: {metrics_csv_path}")

# Load previous FAIR_MODEL_COMPARISON.csv
prev_summary_path = os.path.join(V7_DIR, "FAIR_MODEL_COMPARISON.csv")
df_prev = pd.read_csv(prev_summary_path)

# Compute GC-WPG-v2 summary statistics
summary_rows = []
for suite in ["Development Holdouts (Phase V6 Benchmark)", "5-Fold Grouped Cross-Validation (Unbiased)"]:
    sub = df_all[df_all["evaluation_suite"] == suite]
    summary_rows.append({
        "evaluation_suite": suite,
        "model": "GC-WPG-v2 (Multi-Scale Prior)",
        "n_evaluations": len(sub),
        "psnr_mean": round(sub["psnr_db"].mean(), 2),
        "psnr_std": round(sub["psnr_db"].std(), 2),
        "ssim_mean": round(sub["ssim"].mean(), 4),
        "ssim_std": round(sub["ssim"].std(), 4),
        "roi_ssim_mean": round(sub["roi_ssim"].mean(), 4),
        "roi_ssim_std": round(sub["roi_ssim"].std(), 4),
        "apex_err_x_px_mean": round(sub["apex_err_x_px"].mean(), 2),
        "apex_err_x_px_std": round(sub["apex_err_x_px"].std(), 2),
        "contour_mae_px_mean": round(sub["contour_mae_px"].mean(), 2),
        "contour_mae_px_std": round(sub["contour_mae_px"].std(), 2),
        "hausdorff_px_mean": round(sub["hausdorff_px"].mean(), 2),
        "hausdorff_px_std": round(sub["hausdorff_px"].std(), 2),
        "reinf_boundary_f1_1px_mean": round(sub["reinf_boundary_f1_1px"].mean(), 4),
        "reinf_boundary_f1_1px_std": round(sub["reinf_boundary_f1_1px"].std(), 4),
        "reinf_boundary_f1_2px_mean": round(sub["reinf_boundary_f1_2px"].mean(), 4),
        "reinf_boundary_f1_2px_std": round(sub["reinf_boundary_f1_2px"].std(), 4),
        "continuity_coverage_mean": round(sub["contour_coverage"].mean(), 3),
        "failure_rate": round(sub["extraction_failed"].mean(), 3)
    })

df_v2_summary = pd.DataFrame(summary_rows)
df_combined = pd.concat([df_prev, df_v2_summary], ignore_index=True)
combined_csv_path = os.path.join(OUT_DIR, "REFINED_MODEL_COMPARISON.csv")
df_combined.to_csv(combined_csv_path, index=False)
print(f"Saved REFINED_MODEL_COMPARISON.csv to: {combined_csv_path}")

# Print summary
print("\n" + "="*80)
print("REFINED MODEL COMPARISON SUMMARY (5-Fold CV):")
print("="*80)
sub_cv = df_combined[df_combined["evaluation_suite"] == "5-Fold Grouped Cross-Validation (Unbiased)"]
cols_show = ["model", "contour_mae_px_mean", "apex_err_x_px_mean", "hausdorff_px_mean", "ssim_mean", "roi_ssim_mean", "reinf_boundary_f1_1px_mean", "failure_rate"]
print(sub_cv[cols_show].to_string(index=False))

# Save dev model checkpoint for Figure 6 and CLI
torch.save({
    "model_state_dict": model_dev.state_dict(),
    "scaler": dev_scaler
}, os.path.join(OUT_DIR, "gc_wpg_v2_checkpoint.pt"))
print(f"Saved GC-WPG-v2 checkpoint to: {os.path.join(OUT_DIR, 'gc_wpg_v2_checkpoint.pt')}")
