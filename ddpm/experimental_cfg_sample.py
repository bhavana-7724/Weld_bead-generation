import os
import torch
import numpy as np
import pandas as pd
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

from ddpm.config import Config
from ddpm.model_v2 import ConditionalUNetV2
from ddpm.diffusion import DiffusionScheduler
from ddpm.dataset import WeldDataset

OUT_DIR = "results/ddpm/v2_cfg_experiment"
os.makedirs(OUT_DIR, exist_ok=True)

def cfg_p_sample(scheduler, img, t, t_index, pred):
    betas_t = scheduler._extract(scheduler.betas, t, img.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, img.shape)
    sqrt_recip_alphas_t = scheduler._extract(scheduler.sqrt_recip_alphas, t, img.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        img - betas_t * pred / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = scheduler._extract(scheduler.posterior_variance, t, img.shape)
        noise = torch.randn_like(img)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def cfg_generate_samples(model, scheduler, conds, scale, seed=None):
    device = conds.device
    if seed is not None:
        torch.manual_seed(seed)
        
    b = conds.shape[0]
    shape = (b, Config.IMG_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
    img = torch.randn(shape, device=device)
    uncond = torch.zeros_like(conds)
    
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc=f'CFG Sampling (scale={scale})', leave=False):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        
        cond_pred = model(img, t, conds)
        if scale == 1.0:
            pred = cond_pred
        else:
            uncond_pred = model(img, t, uncond)
            pred = uncond_pred + scale * (cond_pred - uncond_pred)
            
        img = cfg_p_sample(scheduler, img, t, i, pred)
        
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = (img * 255).type(torch.uint8)
    return img

def main():
    device = Config.DEVICE
    model = ConditionalUNetV2(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    chk = torch.load("results/ddpm/v2_diagnostic/v2_diagnostic_chk.pt", map_location=device)
    model.load_state_dict(chk['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    dataset = WeldDataset(Config.TRAIN_CSV, 'data/processed/images_36_v2', is_train=False)
    
    holdout_exps = [5, 14, 22, 32]
    master_df = pd.read_csv(Config.MASTER_CSV)
    eval_df = master_df[master_df['experiment_id'].isin(holdout_exps)].drop_duplicates(subset=['experiment_id']).copy()
    
    # 1. Prepare Conditions
    cond_tensors = []
    for _, row in eval_df.iterrows():
        cont_vals = row[['feed_rate', 'speed', 'HIP', 'efficiency']].values.astype(np.float64).reshape(1, -1)
        cont_scaled = dataset.scaler.transform(cont_vals).flatten().astype(np.float32)
        cat_encoded = np.array([
            1.0 if row['material'] == 'IN718' else 0.0,
            1.0 if row['material'] == 'SS304L' else 0.0,
            1.0 if row['process'] == 'ColdArc' else 0.0,
            1.0 if row['process'] == 'GMAW' else 0.0
        ], dtype=np.float32)
        cond = torch.tensor(np.concatenate([cont_scaled, cat_encoded]), dtype=torch.float32)
        cond_tensors.append(cond)
        
    all_conds = torch.stack(cond_tensors).to(device) # Shape: (4, COND_DIM)
    
    results_log = []
    generated_dict = {}
    
    # Test 1: Identical Noise, Varying Scales
    scales = [1.0, 2.0, 3.0]
    fixed_seed = 42
    
    for scale in scales:
        print(f"Generating for scale {scale}...")
        imgs = cfg_generate_samples(model, scheduler, all_conds, scale, seed=fixed_seed)
        generated_dict[f"scale_{scale}"] = imgs.cpu().numpy()
        
        for i, exp_id in enumerate(holdout_exps):
            img_np = imgs[i].squeeze().cpu().numpy()
            img_hash = hashlib.md5(img_np.tobytes()).hexdigest()
            Image.fromarray(img_np, mode='L').save(os.path.join(OUT_DIR, f"exp_{exp_id}_scale_{scale}.png"))
            results_log.append({"experiment_id": exp_id, "scale": scale, "seed": fixed_seed, "hash": img_hash})
            
    # Test 2: Independent Seeds, Fixed Scale (2.0)
    indep_seeds = [101, 102, 103]
    for seed in indep_seeds:
        print(f"Generating for seed {seed}...")
        imgs = cfg_generate_samples(model, scheduler, all_conds, 2.0, seed=seed)
        generated_dict[f"seed_{seed}"] = imgs.cpu().numpy()
        
        for i, exp_id in enumerate(holdout_exps):
            img_np = imgs[i].squeeze().cpu().numpy()
            img_hash = hashlib.md5(img_np.tobytes()).hexdigest()
            Image.fromarray(img_np, mode='L').save(os.path.join(OUT_DIR, f"exp_{exp_id}_seed_{seed}.png"))
            results_log.append({"experiment_id": exp_id, "scale": 2.0, "seed": seed, "hash": img_hash})
            
    pd.DataFrame(results_log).to_csv(os.path.join(OUT_DIR, "cfg_experiment_log.csv"), index=False)
    
    # Generate pairwise differences
    print("Calculating Pairwise Differences...")
    diff_log = []
    
    # Across conditions, same scale/seed
    for scale in scales:
        imgs = generated_dict[f"scale_{scale}"]
        for i in range(len(holdout_exps)):
            for j in range(i+1, len(holdout_exps)):
                mae = np.mean(np.abs(imgs[i].astype(float) - imgs[j].astype(float)))
                corr = np.corrcoef(imgs[i].flatten(), imgs[j].flatten())[0,1]
                diff_log.append({"Comparison": f"Exp {holdout_exps[i]} vs {holdout_exps[j]}", "Scale": scale, "Seed": fixed_seed, "MAE": mae, "Corr": corr})
                
    # Across scales, same condition
    for i, exp_id in enumerate(holdout_exps):
        s1 = generated_dict["scale_1.0"][i]
        s2 = generated_dict["scale_2.0"][i]
        s3 = generated_dict["scale_3.0"][i]
        mae_12 = np.mean(np.abs(s1.astype(float) - s2.astype(float)))
        mae_23 = np.mean(np.abs(s2.astype(float) - s3.astype(float)))
        diff_log.append({"Comparison": f"Exp {exp_id} Scale 1.0 vs 2.0", "Scale": "N/A", "Seed": fixed_seed, "MAE": mae_12, "Corr": np.corrcoef(s1.flatten(), s2.flatten())[0,1]})
        diff_log.append({"Comparison": f"Exp {exp_id} Scale 2.0 vs 3.0", "Scale": "N/A", "Seed": fixed_seed, "MAE": mae_23, "Corr": np.corrcoef(s2.flatten(), s3.flatten())[0,1]})

    # Across seeds, same condition
    for i, exp_id in enumerate(holdout_exps):
        sd1 = generated_dict["seed_101"][i]
        sd2 = generated_dict["seed_102"][i]
        mae = np.mean(np.abs(sd1.astype(float) - sd2.astype(float)))
        diff_log.append({"Comparison": f"Exp {exp_id} Seed 101 vs 102", "Scale": 2.0, "Seed": "N/A", "MAE": mae, "Corr": np.corrcoef(sd1.flatten(), sd2.flatten())[0,1]})

    pd.DataFrame(diff_log).to_csv(os.path.join(OUT_DIR, "pairwise_diffs.csv"), index=False)
    
    # Plot Contact Sheet
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    cols = ['Scale 1.0', 'Scale 2.0', 'Scale 3.0', 'Seed 101 (Sc 2.0)', 'Seed 102 (Sc 2.0)']
    
    for col, title in enumerate(cols):
        axes[0, col].set_title(title)
        
    for i, exp_id in enumerate(holdout_exps):
        axes[i, 0].imshow(generated_dict["scale_1.0"][i].squeeze(), cmap='gray')
        axes[i, 0].set_ylabel(f"Exp {exp_id}")
        axes[i, 1].imshow(generated_dict["scale_2.0"][i].squeeze(), cmap='gray')
        axes[i, 2].imshow(generated_dict["scale_3.0"][i].squeeze(), cmap='gray')
        axes[i, 3].imshow(generated_dict["seed_101"][i].squeeze(), cmap='gray')
        axes[i, 4].imshow(generated_dict["seed_102"][i].squeeze(), cmap='gray')
        
        for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cfg_contact_sheet.png"))
    print("Inference Diagnostic Complete.")

if __name__ == '__main__':
    main()
