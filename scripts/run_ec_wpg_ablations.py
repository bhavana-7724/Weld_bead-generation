"""
run_ec_wpg_ablations.py

Executes the Controlled Ablation Study for Phase V6-H:
Evaluates the contribution of each architectural and loss component under identical
prospective conditions and fixed random seed (42).

5 Ablation Configurations:
- A0 (Baseline Direct CNN): L_img only (L1 + MSE)
- A1 (+ Sobel Gradient Loss): L_img + lambda_grad * L_grad
- A2 (+ Weld ROI Masked Loss): L_img + lambda_roi * L_roi
- A3 (+ Auxiliary Canny Edge Head): L_img + lambda_edge * L_edge
- A4 (Full Proposed EC-WPG): L_img + lambda_roi * L_roi + lambda_grad * L_grad + lambda_edge * L_edge

Produces:
1. ablation_study_metrics.csv
2. fig8_ablation_study.png (Visual comparison + quantitative bar chart)
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v6")
os.makedirs(OUT_DIR, exist_ok=True)

HOLDOUTS = [5, 14, 22, 32]

# Load Data
meta_df = pd.read_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"))
clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
full_df = pd.merge(meta_df, clean_df, on="sample")

train_df = full_df[~full_df['is_holdout']].copy().reset_index(drop=True)
holdout_df = full_df[full_df['is_holdout']].copy().reset_index(drop=True)

f_min = train_df['feed_rate'].min()
f_max = train_df['feed_rate'].max()
s_min = train_df['speed'].min()
s_max = train_df['speed'].max()

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
    imgs, edges, rois, conds = [], [], [], []
    edge_dir = os.path.join(DATA_DIR, "processed", "aligned_128_edges")
    roi_dir = os.path.join(DATA_DIR, "processed", "aligned_128_rois")
    
    for _, r in df_subset.iterrows():
        sid = r['sample']
        view = r['view']
        im = cv2.imread(r['path'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        imgs.append(im.transpose(2, 0, 1))
        
        edge = cv2.imread(os.path.join(edge_dir, f"{sid}-{view}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        edges.append(edge[np.newaxis, :, :])
        
        roi = cv2.imread(os.path.join(roi_dir, f"{sid}-{view}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        rois.append(roi[np.newaxis, :, :])
        
        conds.append(build_prospective_condition(r))
        
    return (torch.tensor(np.array(imgs), dtype=torch.float32),
            torch.tensor(np.array(edges), dtype=torch.float32),
            torch.tensor(np.array(rois), dtype=torch.float32),
            torch.tensor(np.array(conds), dtype=torch.float32))

train_imgs, train_edges, train_rois, train_conds = load_data_subset(train_df)
val_imgs, val_edges, val_rois, val_conds = load_data_subset(holdout_df)

# Architecture Components
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

class GenericAblationModel(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16, has_edge_head=True):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        self.has_edge_head = has_edge_head
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
        
        self.head_img = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        if has_edge_head:
            self.head_edge = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            self.head_edge = None

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
        out_edge = self.head_edge(feat) if self.has_edge_head else None
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

class SobelGradientLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0],
                                [-2.0, 0.0, 2.0],
                                [-1.0, 0.0, 1.0]], dtype=torch.float32) / 4.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0],
                                [ 0.0,  0.0,  0.0],
                                [ 1.0,  2.0,  1.0]], dtype=torch.float32) / 4.0
        self.weight_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)
        self.weight_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(device)

    def compute_mag(self, x):
        gx = F.conv2d(x, self.weight_x, padding=1, groups=3)
        gy = F.conv2d(x, self.weight_y, padding=1, groups=3)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, pred, target_mag):
        pred_mag = self.compute_mag(pred)
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

sobel_fn_global = SobelGradientLoss(device)
with torch.no_grad():
    train_sobel_mag = sobel_fn_global.compute_mag(train_imgs)

def run_single_ablation(name, lambda_roi=0.0, lambda_grad=0.0, lambda_edge=0.0, has_edge_head=False, epochs=50):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    model = GenericAblationModel(cond_dim=6, noise_dim=16, has_edge_head=has_edge_head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    
    dice_fn = DiceLoss()
    
    n_samples = len(train_imgs)
    batch_size = 8
    n_batches = int(np.ceil(n_samples / batch_size))
    
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        for b in range(n_batches):
            idx = perm[b*batch_size : (b+1)*batch_size]
            b_imgs = train_imgs[idx].to(device)
            b_edges = train_edges[idx].to(device)
            b_rois = train_rois[idx].to(device)
            b_conds = train_conds[idx].to(device)
            b_smag = train_sobel_mag[idx].to(device)
            b_z = torch.randn((len(idx), 16), dtype=torch.float32, device=device) * 0.05
            
            optimizer.zero_grad()
            pred_img, pred_edge = model(b_conds, b_z)
            
            # Base loss
            l_img = F.l1_loss(pred_img, b_imgs) + 0.5 * F.mse_loss(pred_img, b_imgs)
            loss = l_img
            
            if lambda_roi > 0:
                diff = torch.abs(pred_img - b_imgs)
                roi_exp = b_rois.repeat(1, 3, 1, 1)
                l_roi = (diff * roi_exp).sum() / (roi_exp.sum() + 1e-6)
                loss += lambda_roi * l_roi
                
            if lambda_grad > 0:
                l_grad = sobel_fn_global(pred_img, b_smag)
                loss += lambda_grad * l_grad
                
            if lambda_edge > 0 and pred_edge is not None:
                l_edge = F.binary_cross_entropy(pred_edge, b_edges) + dice_fn(pred_edge, b_edges)
                loss += lambda_edge * l_edge
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if ep % 25 == 0 or ep == epochs:
            print(f"  [{name}] Epoch {ep}/{epochs} loss={loss.item():.4f}", flush=True)
        
    # Evaluate on Holdouts
    model.eval()
    holdout_preds = []
    with torch.no_grad():
        val_pred_imgs, val_pred_edges = model.generate(val_conds.to(device), seed=42)
        
    psnrs, ssims, roi_ssims, roi_maes, f1s, apex_errs = [], [], [], [], [], []
    for i in range(len(val_imgs)):
        gt = val_imgs[i].permute(1, 2, 0).numpy()
        pr = val_pred_imgs[i].permute(1, 2, 0).numpy()
        roi = val_rois[i].squeeze(0).numpy()
        gt_e = val_edges[i].squeeze(0).numpy()
        
        p_val = psnr(gt, pr, data_range=1.0)
        s_val = ssim(gt, pr, channel_axis=2, data_range=1.0)
        
        r_bool = roi > 0.5
        r_mae = np.mean(np.abs(gt[r_bool] - pr[r_bool]))
        
        r_min_y, r_max_y = np.where(r_bool)[0].min(), np.where(r_bool)[0].max()
        r_min_x, r_max_x = np.where(r_bool)[1].min(), np.where(r_bool)[1].max()
        gt_crop = gt[r_min_y:r_max_y+1, r_min_x:r_max_x+1]
        pr_crop = pr[r_min_y:r_max_y+1, r_min_x:r_max_x+1]
        r_ssim = ssim(gt_crop, pr_crop, channel_axis=2, data_range=1.0)
        
        # Boundary F1
        if has_edge_head and val_pred_edges is not None:
            pr_e_map = val_pred_edges[i].squeeze(0).numpy()
            pr_e_bin = (pr_e_map > 0.35).astype(np.uint8)
        else:
            # Extract Sobel gradient as boundary representation
            pr_gray = cv2.cvtColor((pr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            gx = cv2.Sobel(pr_gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(pr_gray, cv2.CV_32F, 0, 1, ksize=3)
            gmag = np.sqrt(gx**2 + gy**2)
            pr_e_bin = (gmag > 50).astype(np.uint8)
            
        gt_e_bin = (gt_e > 0.5).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        gt_dil = cv2.dilate(gt_e_bin, kernel)
        pr_dil = cv2.dilate(pr_e_bin, kernel)
        
        tp = np.sum((pr_e_bin == 1) & (gt_dil == 1))
        fp = np.sum((pr_e_bin == 1) & (gt_dil == 0))
        fn = np.sum((gt_e_bin == 1) & (pr_dil == 0))
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2.0 * prec * rec / (prec + rec + 1e-8)
        
        # Apex Error (outermost crown projection along horizontal axis X < 64)
        # Find leftmost pixel with specimen intensity
        gt_crown = (cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) < 165)
        pr_crown = (cv2.cvtColor((pr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) < 165)
        
        gt_apex_x = np.where(gt_crown[:, :64])[1].min() if np.any(gt_crown[:, :64]) else 64
        pr_apex_x = np.where(pr_crown[:, :64])[1].min() if np.any(pr_crown[:, :64]) else 64
        apex_err = abs(gt_apex_x - pr_apex_x)
        
        psnrs.append(p_val)
        ssims.append(s_val)
        roi_ssims.append(r_ssim)
        roi_maes.append(r_mae)
        f1s.append(f1)
        apex_errs.append(apex_err)
        
    return {
        "variant": name,
        "mean_ssim": np.mean(ssims),
        "mean_psnr_db": np.mean(psnrs),
        "mean_roi_ssim": np.mean(roi_ssims),
        "mean_roi_mae": np.mean(roi_maes),
        "mean_edge_f1": np.mean(f1s),
        "mean_apex_err_px": np.mean(apex_errs)
    }, val_pred_imgs[1].permute(1, 2, 0).numpy() # Sample 14 view B (representative)

print("Running Controlled Ablation Study (5 configurations)...", flush=True)

configs = [
    ("A0_Baseline_CNN", 0.0, 0.0, 0.0, False),
    ("A1_Sobel_Gradient", 0.0, 1.0, 0.0, False),
    ("A2_Weld_ROI", 2.0, 0.0, 0.0, False),
    ("A3_Edge_Head", 0.0, 0.0, 1.0, True),
    ("A4_Full_EC_WPG", 2.0, 1.0, 1.0, True),
]

results = []
vis_images = {}

for name, l_roi, l_grad, l_edge, has_head in configs:
    t0 = time.time()
    res, vis = run_single_ablation(name, lambda_roi=l_roi, lambda_grad=l_grad,
                                   lambda_edge=l_edge, has_edge_head=has_head, epochs=50)
    dur = time.time() - t0
    print(f"[{name}] Done in {dur:.1f}s | SSIM: {res['mean_ssim']:.4f} | ROI SSIM: {res['mean_roi_ssim']:.4f} | F1: {res['mean_edge_f1']:.4f} | Apex Err: {res['mean_apex_err_px']:.1f}px", flush=True)
    results.append(res)
    vis_images[name] = vis

ablation_df = pd.DataFrame(results)
csv_out = os.path.join(OUT_DIR, "ablation_study_metrics.csv")
ablation_df.to_csv(csv_out, index=False)
print(f"\nSaved ablation metrics to: {csv_out}", flush=True)

# -------------------------------------------------------------------------
# Generate Figure 8: Ablation Study Comparison
# -------------------------------------------------------------------------
print("Generating Figure 8: Ablation Study Publication Figure...", flush=True)
fig = plt.figure(figsize=(14, 8), dpi=300)
gs = fig.add_gridspec(2, 6, height_ratios=[1.2, 1.0], hspace=0.35, wspace=0.25)

# Row 1: Visual comparison of generated weld profile for Sample 14 (Holdout, IN718 GMAW)
gt_sample14 = val_imgs[1].permute(1, 2, 0).numpy()

ax_gt = fig.add_subplot(gs[0, 0])
ax_gt.imshow(gt_sample14)
ax_gt.set_title("Ground Truth\n(Sample 14 Holdout)", fontsize=10, fontweight="bold")
ax_gt.axvline(64, color='red', linestyle='--', alpha=0.6, linewidth=1.0)
ax_gt.axis("off")

for idx, (name, _, _, _, _) in enumerate(configs):
    ax = fig.add_subplot(gs[0, idx + 1])
    ax.imshow(vis_images[name])
    labels = ["A0: Baseline CNN\n(Img Loss only)",
              "A1: + Sobel Grad\n(Edge Gradient)",
              "A2: + Weld ROI\n(Focused Area)",
              "A3: + Edge Head\n(Boundary Head)",
              "A4: Full EC-WPG\n(Combined Framework)"]
    ax.set_title(labels[idx], fontsize=9, fontweight="bold")
    ax.axvline(64, color='red', linestyle='--', alpha=0.6, linewidth=1.0)
    ax.axis("off")

# Row 2: Quantitative bar charts
ax_b1 = fig.add_subplot(gs[1, 0:2])
names = ["A0", "A1", "A2", "A3", "A4"]
roi_ssims = ablation_df['mean_roi_ssim']
bars1 = ax_b1.bar(names, roi_ssims, color=['#95a5a6', '#3498db', '#e67e22', '#9b59b6', '#2ecc71'])
ax_b1.set_ylabel("Weld ROI SSIM", fontsize=10, fontweight="bold")
ax_b1.set_title("Weld Zone Structural Similarity", fontsize=11, fontweight="bold")
ax_b1.set_ylim([0.5, 0.9])
ax_b1.grid(axis='y', linestyle=':', alpha=0.6)
for bar in bars1:
    h = bar.get_height()
    ax_b1.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax_b2 = fig.add_subplot(gs[1, 2:4])
f1s = ablation_df['mean_edge_f1']
bars2 = ax_b2.bar(names, f1s, color=['#95a5a6', '#3498db', '#e67e22', '#9b59b6', '#2ecc71'])
ax_b2.set_ylabel("Boundary Edge F1 Score", fontsize=10, fontweight="bold")
ax_b2.set_title("Boundary Localization Accuracy", fontsize=11, fontweight="bold")
ax_b2.set_ylim([0.2, 0.75])
ax_b2.grid(axis='y', linestyle=':', alpha=0.6)
for bar in bars2:
    h = bar.get_height()
    ax_b2.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

ax_b3 = fig.add_subplot(gs[1, 4:6])
apex = ablation_df['mean_apex_err_px']
bars3 = ax_b3.bar(names, apex, color=['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71'])
ax_b3.set_ylabel("Apex Position Error (pixels)", fontsize=10, fontweight="bold")
ax_b3.set_title("Weld Crown Apex Error (Lower is Better)", fontsize=11, fontweight="bold")
ax_b3.set_ylim([0, 8.0])
ax_b3.grid(axis='y', linestyle=':', alpha=0.6)
for bar in bars3:
    h = bar.get_height()
    ax_b3.annotate(f"{h:.1f} px", xy=(bar.get_x() + bar.get_width()/2, h),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

plt.suptitle("Phase V6 Ablation Study: Progressive Architectural and Loss Contributions\nEvaluated on Unseen Development Holdouts (Samples 5, 14, 22, 32)",
             fontsize=13, fontweight="bold", y=0.98)

fig_path = os.path.join(OUT_DIR, "fig8_ablation_study.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Figure 8 to: {fig_path}", flush=True)
