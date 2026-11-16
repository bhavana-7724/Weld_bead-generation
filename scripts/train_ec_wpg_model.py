"""
train_ec_wpg_model.py

Trains the Edge-Aware Conditional Weld Profile Generator (EC-WPG) on the
physically aligned 128x128 weld macrograph dataset (64 training images, excluding holdouts 5, 14, 22, 32).

Key Features:
1. Prospective Conditioning: c = [feed_rate, speed, material_IN718, material_SS304L, process_ColdArc, process_GMAW]
2. Dual-Head Architecture:
   - Head 1: High-fidelity RGB weld macrograph generator
   - Head 2: Auxiliary boundary/edge prediction head
3. Multi-Component Loss Formulation:
   - L_img: Global L1 + MSE loss
   - L_roi: Weld ROI masked loss focusing capacity on bead reinforcement and fusion zone
   - L_grad: Differentiable 2D Sobel spatial gradient loss
   - L_edge: Auxiliary boundary loss (BCE + Dice) supervising edge localization against Canny pseudo-labels
4. Produces:
   - ec_wpg_checkpoint.pt
   - ec_wpg_training_metadata.json
   - Quantitative evaluation on 4 unseen holdouts
"""

import os
import sys
import json
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

# Set random seed for scientific reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v6")
os.makedirs(OUT_DIR, exist_ok=True)

HOLDOUTS = [5, 14, 22, 32]

print("=" * 70, flush=True)
print("  TRAINING EDGE-AWARE CONDITIONAL WELD PROFILE GENERATOR (EC-WPG)  ", flush=True)
print("=" * 70, flush=True)

# -------------------------------------------------------------------------
# 1. Load Metadata and Pre-load Dataset into Memory
# -------------------------------------------------------------------------
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample")

train_df = full_df[~full_df['is_holdout']].copy().reset_index(drop=True)
holdout_df = full_df[full_df['is_holdout']].copy().reset_index(drop=True)

# Compute min/max scaling strictly on training set for prospective parameters
f_min = train_df['feed_rate'].min()
f_max = train_df['feed_rate'].max()
s_min = train_df['speed'].min()
s_max = train_df['speed'].max()

scaler_params = {
    "feed_rate_min": float(f_min),
    "feed_rate_max": float(f_max),
    "speed_min": float(s_min),
    "speed_max": float(s_max)
}

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

def load_data_subset(df_subset):
    imgs = []
    edges = []
    rois = []
    conds = []
    edge_dir = os.path.join(DATA_DIR, "processed", "aligned_128_edges")
    roi_dir = os.path.join(DATA_DIR, "processed", "aligned_128_rois")
    
    for _, r in df_subset.iterrows():
        sid = r['sample']
        view = r['view']
        
        # Load aligned RGB image
        im = cv2.imread(r['path'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(im.transpose(2, 0, 1))
        
        # Load Canny edge map
        edge_path = os.path.join(edge_dir, f"{sid}-{view}.png")
        edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        edges.append(edge[np.newaxis, :, :])
        
        # Load ROI mask
        roi_path = os.path.join(roi_dir, f"{sid}-{view}.png")
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        rois.append(roi[np.newaxis, :, :])
        
        conds.append(build_prospective_condition(r))
        
    return (torch.tensor(np.array(imgs), dtype=torch.float32),
            torch.tensor(np.array(edges), dtype=torch.float32),
            torch.tensor(np.array(rois), dtype=torch.float32),
            torch.tensor(np.array(conds), dtype=torch.float32))

print("Loading training dataset (64 images, edges, ROIs)...", flush=True)
train_imgs, train_edges, train_rois, train_conds = load_data_subset(train_df)
print("Loading holdout dataset (8 images, edges, ROIs)...", flush=True)
val_imgs, val_edges, val_rois, val_conds = load_data_subset(holdout_df)
print(f"Train tensors: Img {train_imgs.shape}, Edge {train_edges.shape}, ROI {train_rois.shape}, Cond {train_conds.shape}", flush=True)

# -------------------------------------------------------------------------
# 2. Model Architecture: Edge-Aware Conditional Weld Profile Generator
# -------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

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
        x = self.act(self.bn_up(self.up(x)))
        return self.conv(x)

class EC_WPG(nn.Module):
    """
    Edge-Aware Conditional Weld Profile Generator.
    Condition -> Deep Shared Feature Maps -> Dual Heads (RGB Image + Edge Boundary Map)
    """
    def __init__(self, cond_dim=6, noise_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        in_dim = cond_dim + noise_dim
        
        # Condition projection to 8x8 feature map
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Shared trunk: 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.up1 = ResUpBlock(128, 64)  # 8 -> 16
        self.up2 = ResUpBlock(64, 32)   # 16 -> 32
        self.up3 = ResUpBlock(32, 16)   # 32 -> 64
        self.up4 = ResUpBlock(16, 16)   # 64 -> 128
        
        # Head 1: Primary RGB Macrograph Image Synthesis Head
        self.head_img = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Head 2: Auxiliary Boundary/Edge Prediction Head
        self.head_edge = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        inp = torch.cat([c, z], dim=-1)
        
        h = self.fc(inp).view(-1, 128, 8, 8)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        feat = self.up4(h)
        
        out_img = self.head_img(feat)
        out_edge = self.head_edge(feat)
        return out_img, out_edge

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
# 3. Differentiable Losses
# -------------------------------------------------------------------------
class SobelGradientLoss(nn.Module):
    """
    Computes spatial gradient magnitude using fixed 3x3 Sobel convolution kernels.
    """
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32) / 4.0
        
        # 3 channels (RGB) grouped
        self.weight_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)
        self.weight_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.weight_x, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, self.weight_y, padding=1, groups=3)
        pred_mag = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)
        
        target_grad_x = F.conv2d(target, self.weight_x, padding=1, groups=3)
        target_grad_y = F.conv2d(target, self.weight_y, padding=1, groups=3)
        target_mag = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)
        
        return F.l1_loss(pred_mag, target_mag)

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        intersection = 2.0 * (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (intersection + self.eps) / (union + self.eps)
        return (1.0 - dice).mean()

# -------------------------------------------------------------------------
# 4. Training Procedure
# -------------------------------------------------------------------------
def train_model(epochs=300, batch_size=8, lr=2e-3,
                lambda_roi=2.0, lambda_grad=1.0, lambda_edge=1.0):
    
    model = EC_WPG(cond_dim=6, noise_dim=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    sobel_loss_fn = SobelGradientLoss(device)
    dice_loss_fn = DiceLoss()
    
    n_samples = len(train_imgs)
    n_batches = int(np.ceil(n_samples / batch_size))
    
    history = {
        "epoch": [], "total_loss": [], "l_img": [], "l_roi": [], "l_grad": [], "l_edge": [],
        "val_psnr": [], "val_ssim": [], "val_roi_ssim": []
    }
    
    start_time = time.time()
    print(f"\nStarting training: {epochs} epochs | Batch size {batch_size} | Weights: ROI={lambda_roi}, Grad={lambda_grad}, Edge={lambda_edge}", flush=True)
    
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        
        ep_total = 0.0
        ep_img = 0.0
        ep_roi = 0.0
        ep_grad = 0.0
        ep_edge = 0.0
        
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            b_imgs = train_imgs[idx].to(device)
            b_edges = train_edges[idx].to(device)
            b_rois = train_rois[idx].to(device)
            b_conds = train_conds[idx].to(device)
            
            # Latent noise regularization
            b_z = torch.randn((len(idx), 16), dtype=torch.float32, device=device) * 0.05
            
            optimizer.zero_grad()
            pred_img, pred_edge = model(b_conds, b_z)
            
            # 1. Image Loss (L1 + MSE)
            l_img = F.l1_loss(pred_img, b_imgs) + 0.5 * F.mse_loss(pred_img, b_imgs)
            
            # 2. ROI Weighted Loss (focus on reinforcement & fusion zone)
            diff = torch.abs(pred_img - b_imgs)
            roi_expanded = b_rois.repeat(1, 3, 1, 1)
            l_roi = (diff * roi_expanded).sum() / (roi_expanded.sum() + 1e-6)
            
            # 3. Spatial Gradient Loss (differentiable Sobel)
            l_grad = sobel_loss_fn(pred_img, b_imgs)
            
            # 4. Auxiliary Edge Loss (BCE + Dice)
            l_edge_bce = F.binary_cross_entropy(pred_edge, b_edges)
            l_edge_dice = dice_loss_fn(pred_edge, b_edges)
            l_edge = l_edge_bce + l_edge_dice
            
            # Total loss
            loss = l_img + lambda_roi * l_roi + lambda_grad * l_grad + lambda_edge * l_edge
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            ep_total += loss.item()
            ep_img += l_img.item()
            ep_roi += l_roi.item()
            ep_grad += l_grad.item()
            ep_edge += l_edge.item()
            
        scheduler.step()
        
        # Periodic evaluation on validation holdouts
        if ep % 25 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val_pred_img, _ = model.generate(val_conds.to(device), seed=42)
                
                psnrs, ssims, roi_ssims = [], [], []
                for i in range(len(val_imgs)):
                    gt = val_imgs[i].permute(1, 2, 0).numpy()
                    pr = val_pred_img[i].permute(1, 2, 0).numpy()
                    roi = val_rois[i].squeeze(0).numpy()
                    
                    p_val = psnr(gt, pr, data_range=1.0)
                    s_val = ssim(gt, pr, channel_axis=2, data_range=1.0)
                    
                    # Compute ROI SSIM
                    r_bool = roi > 0.5
                    if np.sum(r_bool) > 50:
                        r_min_y, r_max_y = np.where(r_bool)[0].min(), np.where(r_bool)[0].max()
                        r_min_x, r_max_x = np.where(r_bool)[1].min(), np.where(r_bool)[1].max()
                        gt_crop = gt[r_min_y:r_max_y+1, r_min_x:r_max_x+1]
                        pr_crop = pr[r_min_y:r_max_y+1, r_min_x:r_max_x+1]
                        rs_val = ssim(gt_crop, pr_crop, channel_axis=2, data_range=1.0)
                    else:
                        rs_val = s_val
                    
                    psnrs.append(p_val)
                    ssims.append(s_val)
                    roi_ssims.append(rs_val)
                    
            mean_psnr = np.mean(psnrs)
            mean_ssim = np.mean(ssims)
            mean_roi_ssim = np.mean(roi_ssims)
            
            history["epoch"].append(ep)
            history["total_loss"].append(ep_total / n_batches)
            history["l_img"].append(ep_img / n_batches)
            history["l_roi"].append(ep_roi / n_batches)
            history["l_grad"].append(ep_grad / n_batches)
            history["l_edge"].append(ep_edge / n_batches)
            history["val_psnr"].append(float(mean_psnr))
            history["val_ssim"].append(float(mean_ssim))
            history["val_roi_ssim"].append(float(mean_roi_ssim))
            
            print(f"Epoch {ep:3d}/{epochs} | Total: {ep_total/n_batches:.4f} | Img: {ep_img/n_batches:.4f} | "
                  f"ROI: {ep_roi/n_batches:.4f} | Grad: {ep_grad/n_batches:.4f} | Edge: {ep_edge/n_batches:.4f} || "
                  f"Val SSIM: {mean_ssim:.4f} | Val ROI SSIM: {mean_roi_ssim:.4f} | Val PSNR: {mean_psnr:.2f} dB", flush=True)

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds ({elapsed/epochs:.4f} s/epoch)", flush=True)
    return model, history

# Run primary training
model, history = train_model(epochs=300, batch_size=8, lr=2e-3,
                             lambda_roi=2.0, lambda_grad=1.0, lambda_edge=1.0)

# Save checkpoint and metadata
ckpt_path = os.path.join(OUT_DIR, "ec_wpg_checkpoint.pt")
torch.save({
    "model_state_dict": model.state_dict(),
    "scaler_params": scaler_params,
    "cond_dim": 6,
    "noise_dim": 16,
    "seed": SEED,
    "holdouts": HOLDOUTS
}, ckpt_path)
print(f"\nSaved EC-WPG checkpoint to: {ckpt_path}", flush=True)

# Save training history and metadata JSON
meta_out = {
    "model_name": "Edge-Aware Conditional Weld Profile Generator (EC-WPG)",
    "training_samples": 32,
    "training_images": 64,
    "holdout_samples": HOLDOUTS,
    "epochs": 300,
    "condition_dimension": 6,
    "condition_features": ["feed_rate", "speed", "IN718", "SS304L", "ColdArc", "GMAW"],
    "scaler_params": scaler_params,
    "loss_weights": {
        "lambda_img": 1.0,
        "lambda_roi": 2.0,
        "lambda_grad": 1.0,
        "lambda_edge": 1.0
    },
    "final_holdout_metrics": {
        "mean_ssim": float(history["val_ssim"][-1]),
        "mean_roi_ssim": float(history["val_roi_ssim"][-1]),
        "mean_psnr_db": float(history["val_psnr"][-1])
    }
}

json_path = os.path.join(OUT_DIR, "ec_wpg_training_metadata.json")
with open(json_path, "w") as f:
    json.dump(meta_out, f, indent=4)
print(f"Saved metadata to: {json_path}", flush=True)

# -------------------------------------------------------------------------
# 5. Holdout Evaluation Table (Samples 5, 14, 22, 32)
# -------------------------------------------------------------------------
print("\n--- Detailed Holdout Evaluation (Unseen Welding Conditions) ---", flush=True)
model.eval()
holdout_records = []

for idx, r in holdout_df.iterrows():
    sid = int(r['sample'])
    view = r['view']
    mat = r['material']
    proc = r['process']
    f_val = r['feed_rate']
    s_val = r['speed']
    
    gt_img = val_imgs[idx].permute(1, 2, 0).numpy()
    gt_edge = val_edges[idx].squeeze(0).numpy()
    gt_roi = val_rois[idx].squeeze(0).numpy()
    
    cond_vec = val_conds[idx:idx+1].to(device)
    with torch.no_grad():
        pr_img_t, pr_edge_t = model.generate(cond_vec, seed=42)
        pr_img = pr_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        pr_edge = pr_edge_t.squeeze(0).squeeze(0).cpu().numpy()
        
    p_val = psnr(gt_img, pr_img, data_range=1.0)
    s_val = ssim(gt_img, pr_img, channel_axis=2, data_range=1.0)
    
    # ROI metrics
    r_bool = gt_roi > 0.5
    roi_mae = np.mean(np.abs(gt_img[r_bool] - pr_img[r_bool]))
    
    r_min_y, r_max_y = np.where(r_bool)[0].min(), np.where(r_bool)[0].max()
    r_min_x, r_max_x = np.where(r_bool)[1].min(), np.where(r_bool)[1].max()
    gt_crop = gt_img[r_min_y:r_max_y+1, r_min_x:r_max_x+1]
    pr_crop = pr_img[r_min_y:r_max_y+1, r_min_x:r_max_x+1]
    roi_ssim = ssim(gt_crop, pr_crop, channel_axis=2, data_range=1.0)
    
    # Boundary / Edge F1 score (threshold at 0.35)
    pr_edge_bin = (pr_edge > 0.35).astype(np.uint8)
    gt_edge_bin = (gt_edge > 0.5).astype(np.uint8)
    
    # Dilate GT edge by 1 pixel tolerance for fair physical boundary matching
    kernel = np.ones((3, 3), np.uint8)
    gt_dilated = cv2.dilate(gt_edge_bin, kernel)
    pr_dilated = cv2.dilate(pr_edge_bin, kernel)
    
    tp = np.sum((pr_edge_bin == 1) & (gt_dilated == 1))
    fp = np.sum((pr_edge_bin == 1) & (gt_dilated == 0))
    fn = np.sum((gt_edge_bin == 1) & (pr_dilated == 0))
    
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
    
    holdout_records.append({
        "sample": sid,
        "view": view,
        "material": mat,
        "process": proc,
        "feed_rate": f_val,
        "speed": s_val,
        "full_psnr_db": round(float(p_val), 2),
        "full_ssim": round(float(s_val), 4),
        "roi_ssim": round(float(roi_ssim), 4),
        "roi_mae": round(float(roi_mae), 4),
        "edge_f1_score": round(float(f1), 4)
    })
    
eval_df = pd.DataFrame(holdout_records)
eval_csv_path = os.path.join(OUT_DIR, "ec_wpg_holdouts_evaluation.csv")
eval_df.to_csv(eval_csv_path, index=False)
print(eval_df.to_string(), flush=True)
print(f"\nSaved holdouts evaluation to: {eval_csv_path}", flush=True)
print("=" * 70, flush=True)
