import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.dataset import get_interpolation_dataset
from ddpm.sample import generate_samples

out_dir = os.path.join(Config.OUTPUT_DIR, "conditioning_diagnostic")
os.makedirs(out_dir, exist_ok=True)

def get_metrics(img1, img2):
    mse = np.mean((img1 - img2)**2)
    s = ssim(img1, img2, data_range=255)
    return mse, s

def get_foreground_metrics(img1, img2):
    # Mask background using Otsu threshold on the combined images or simple fixed threshold
    # The weld bead is typically darker. We'll find a threshold on img1 and apply to both.
    thresh = threshold_otsu(img1)
    mask1 = img1 < thresh
    mask2 = img2 < thresh
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    
    # MSE on the union foreground
    if np.sum(union) > 0:
        mse_fg = np.mean((img1[union] - img2[union])**2)
    else:
        mse_fg = 0.0
    return mse_fg, iou

def main():
    device = Config.DEVICE
    
    # Task 1: Check scaler loading and condition differences
    ds = get_interpolation_dataset()
    
    # We will pick 3 specific experiments
    exps = [5, 14, 22]
    
    cond_vectors = []
    
    cond_dict = {}
    for i in range(len(ds)):
        img, cond, exp_id, _ = ds[i]
        if exp_id in exps and exp_id not in cond_dict:
            cond_dict[exp_id] = cond
            cond_vectors.append({
                "experiment_id": exp_id,
                "feed_rate": float(cond[0]),
                "speed": float(cond[1]),
                "HIP": float(cond[2]),
                "efficiency": float(cond[3]),
                "mat_in718": float(cond[4]),
                "mat_ss304l": float(cond[5]),
                "proc_coldarc": float(cond[6]),
                "proc_gmaw": float(cond[7])
            })
            
    df_cond = pd.DataFrame(cond_vectors)
    df_cond.to_csv(os.path.join(out_dir, "condition_vectors.csv"), index=False)
    
    # Load model
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    chk_path = os.path.join(Config.MODEL_DIR, "checkpoint_best.pt")
    model.load_state_dict(torch.load(chk_path, map_location=device)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    
    # Task 3: Controlled same-seed intervention
    # Generate one image per condition with seed=42
    imgs_same_seed = {}
    for exp_id in exps:
        c = cond_dict[exp_id].unsqueeze(0).to(device)
        out = generate_samples(model, scheduler, c, num_samples_per_cond=1, seed=42)
        imgs_same_seed[exp_id] = out[0].squeeze().cpu().numpy()
        
    # Task 4: Controlled same-condition intervention
    # Generate 5 images per condition with seeds 100, 101, 102, 103, 104
    seeds = [100, 101, 102, 103, 104]
    imgs_multiple_seeds = {exp_id: {} for exp_id in exps}
    for exp_id in exps:
        c = cond_dict[exp_id].unsqueeze(0).to(device)
        for s in seeds:
            out = generate_samples(model, scheduler, c, num_samples_per_cond=1, seed=s)
            imgs_multiple_seeds[exp_id][s] = out[0].squeeze().cpu().numpy()
            
    # Compile metrics
    pairwise_metrics = []
    
    # Same seed, different conditions
    for i, e1 in enumerate(exps):
        for e2 in exps[i+1:]:
            mse_val, ssim_val = get_metrics(imgs_same_seed[e1], imgs_same_seed[e2])
            mse_fg, iou = get_foreground_metrics(imgs_same_seed[e1], imgs_same_seed[e2])
            pairwise_metrics.append({
                "type": "inter_condition_same_seed",
                "seed": 42,
                "exp_1": e1,
                "exp_2": e2,
                "MSE_global": mse_val,
                "SSIM": ssim_val,
                "MSE_foreground": mse_fg,
                "IoU_foreground": iou
            })
            
    # Same condition, different seeds
    for exp_id in exps:
        for i, s1 in enumerate(seeds):
            for s2 in seeds[i+1:]:
                mse_val, ssim_val = get_metrics(imgs_multiple_seeds[exp_id][s1], imgs_multiple_seeds[exp_id][s2])
                mse_fg, iou = get_foreground_metrics(imgs_multiple_seeds[exp_id][s1], imgs_multiple_seeds[exp_id][s2])
                pairwise_metrics.append({
                    "type": "intra_condition_diff_seed",
                    "seed": f"{s1}_vs_{s2}",
                    "exp_1": exp_id,
                    "exp_2": exp_id,
                    "MSE_global": mse_val,
                    "SSIM": ssim_val,
                    "MSE_foreground": mse_fg,
                    "IoU_foreground": iou
                })
                
    df_metrics = pd.DataFrame(pairwise_metrics)
    df_metrics.to_csv(os.path.join(out_dir, "pairwise_metrics.csv"), index=False)
    
    # Visual grid for same-seed (Task 3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, exp_id in enumerate(exps):
        axes[i].imshow(imgs_same_seed[exp_id], cmap='gray')
        axes[i].set_title(f"Exp {exp_id}\nSeed 42")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "controlled_generation_grid.png"), bbox_inches='tight')
    plt.close()
    
    print("Diagnostics completed.")

if __name__ == "__main__":
    main()
