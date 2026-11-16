import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from tqdm import tqdm

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.dataset import WeldDataset
from ddpm.utils import set_seed

OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "ddpm", "cfg_ablation")
os.makedirs(OUT_DIR, exist_ok=True)
CHK_PATH = os.path.join(OUT_DIR, "cfg_checkpoint.pt")
HIST_PATH = os.path.join(OUT_DIR, "training_history.csv")
METRICS_PATH = os.path.join(OUT_DIR, "evaluation_metrics.csv")

NULL_CONDITION = torch.zeros(Config.COND_DIM)

@torch.no_grad()
def cfg_p_sample(model, scheduler, x, t, t_index, c, guidance_scale):
    b = x.shape[0]
    device = x.device
    c_null = NULL_CONDITION.unsqueeze(0).repeat(b, 1).to(device)
    
    uncond_pred = model(x, t, c_null)
    cond_pred = model(x, t, c)
    noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    
    betas_t = scheduler._extract(scheduler.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = scheduler._extract(scheduler.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = scheduler._extract(scheduler.sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = scheduler._extract(scheduler.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def cfg_p_sample_loop(model, scheduler, shape, c, guidance_scale, seed=None):
    device = next(model.parameters()).device
    b = shape[0]
    if seed is not None:
        torch.manual_seed(seed)
    img = torch.randn(shape, device=device)
    
    for i in tqdm(reversed(range(0, scheduler.timesteps)), desc=f'CFG s={guidance_scale}', total=scheduler.timesteps, leave=False):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = cfg_p_sample(model, scheduler, img, t, i, c, guidance_scale)
        
    img = (img + 1.0) / 2.0
    img = img.clamp(0, 1)
    img = (img * 255).type(torch.uint8)
    return img

def apply_condition_dropout(conds, dropout_prob=0.15):
    b = conds.shape[0]
    device = conds.device
    mask = torch.rand(b, device=device) < dropout_prob
    c_null = NULL_CONDITION.to(device).unsqueeze(0).repeat(b, 1)
    return torch.where(mask.unsqueeze(1), c_null, conds)

def get_metrics(img1, img2):
    mse = np.mean((img1 - img2)**2)
    s = ssim(img1, img2, data_range=255)
    
    thresh = threshold_otsu(img1)
    mask1 = img1 < thresh
    mask2 = img2 < thresh
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
    
    return mse, s, iou

def main():
    set_seed(42)
    device = Config.DEVICE
    
    train_dataset = WeldDataset(Config.TRAIN_CSV, Config.PROCESSED_IMG_DIR, is_train=True)
    dataloader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    
    start_epoch = 1
    epochs = 1000
    
    if os.path.exists(CHK_PATH):
        print(f"Resuming from checkpoint {CHK_PATH}")
        checkpoint = torch.load(CHK_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    
    if start_epoch <= epochs:
        print(f"Starting CFG training from epoch {start_epoch} to {epochs}...")
        model.train()
        for epoch in range(start_epoch, epochs + 1):
            epoch_loss = 0
            for imgs, conds, exp_ids, _ in dataloader:
                imgs = imgs.to(device)
                conds = conds.to(device)
                
                # Apply Condition Dropout (15%)
                conds = apply_condition_dropout(conds, dropout_prob=0.15)
                
                b = imgs.shape[0]
                t = torch.randint(0, Config.TIMESTEPS, (b,), device=device).long()
                noise = torch.randn_like(imgs)
                x_noisy = scheduler.q_sample(imgs, t, noise=noise)
                
                noise_pred = model(x_noisy, t, conds)
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
                optimizer.step()
                
                epoch_loss += loss.item() * b
            
            epoch_loss /= len(train_dataset)
            
            # Incremental logging
            with open(HIST_PATH, "a") as f:
                if epoch == 1 and not os.path.exists(CHK_PATH):
                    f.write("epoch,loss\n")
                f.write(f"{epoch},{epoch_loss:.5f}\n")
                
            if epoch % 50 == 0 or epoch == epochs:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
                }, CHK_PATH)
                print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss:.5f}")
                
    # --- EVALUATION PHASE ---
    print("\n--- Starting Evaluation Phase ---")
    model.eval()
    holdout_exps = [5, 14, 22, 32]
    seeds = [100, 101, 102, 103, 104]
    scales = [1.0, 3.0, 5.0]
    
    # Load original master metadata to get condition vectors for holdouts
    master_df = pd.read_csv(Config.MASTER_CSV)
    eval_df = master_df[master_df['experiment_id'].isin(holdout_exps)].drop_duplicates(subset=['experiment_id']).copy()
    
    # Prepare cond vectors using the TRAIN scaler
    continuous_cols = ['feed_rate', 'speed', 'HIP', 'efficiency']
    scaler = train_dataset.scaler
    
    holdout_conds = {}
    for _, row in eval_df.iterrows():
        exp_id = row['experiment_id']
        cont_vals = row[continuous_cols].values.astype(np.float64).reshape(1, -1)
        cont_scaled = scaler.transform(cont_vals).flatten().astype(np.float32)
        
        mat_in718 = 1.0 if row['material'] == 'IN718' else 0.0
        mat_ss304l = 1.0 if row['material'] == 'SS304L' else 0.0
        proc_coldarc = 1.0 if row['process'] == 'ColdArc' else 0.0
        proc_gmaw = 1.0 if row['process'] == 'GMAW' else 0.0
        cat_encoded = np.array([mat_in718, mat_ss304l, proc_coldarc, proc_gmaw], dtype=np.float32)
        
        cond = torch.tensor(np.concatenate([cont_scaled, cat_encoded]), dtype=torch.float32)
        holdout_conds[exp_id] = cond.unsqueeze(0).to(device)
        
    # Generate Images
    gen_dir = os.path.join(OUT_DIR, "generated")
    os.makedirs(gen_dir, exist_ok=True)
    
    # Check completed
    metrics = []
    
    for s in scales:
        for exp_id in holdout_exps:
            c = holdout_conds[exp_id]
            for seed in seeds:
                img_path = os.path.join(gen_dir, f"exp{exp_id}_s{seed}_cfg{s}.png")
                if not os.path.exists(img_path):
                    # generate
                    out = cfg_p_sample_loop(model, scheduler, (1, Config.IMG_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE), c, guidance_scale=s, seed=seed)
                    gen_numpy = out[0].squeeze().cpu().numpy()
                    Image.fromarray(gen_numpy, mode='L').save(img_path)
                else:
                    gen_numpy = np.array(Image.open(img_path))
                    
    print("Generation complete. Calculating pairwise metrics...")
    # Calculate pairwise differences (comparing CFG scales against each other, and baseline behavior)
    for exp_id in holdout_exps:
        for seed in seeds:
            img_s1 = np.array(Image.open(os.path.join(gen_dir, f"exp{exp_id}_s{seed}_cfg1.0.png")))
            img_s3 = np.array(Image.open(os.path.join(gen_dir, f"exp{exp_id}_s{seed}_cfg3.0.png")))
            img_s5 = np.array(Image.open(os.path.join(gen_dir, f"exp{exp_id}_s{seed}_cfg5.0.png")))
            
            # Diff between cfg1 and cfg3
            mse_1_3, ssim_1_3, iou_1_3 = get_metrics(img_s1, img_s3)
            # Diff between cfg1 and cfg5
            mse_1_5, ssim_1_5, iou_1_5 = get_metrics(img_s1, img_s5)
            
            metrics.append({
                "exp_id": exp_id,
                "seed": seed,
                "MSE_1vs3": mse_1_3,
                "SSIM_1vs3": ssim_1_3,
                "IoU_1vs3": iou_1_3,
                "MSE_1vs5": mse_1_5,
                "SSIM_1vs5": ssim_1_5,
                "IoU_1vs5": iou_1_5
            })
            
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_PATH, index=False)
    
    # Create comparison grid for one seed (e.g. 100)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for col, exp_id in enumerate(holdout_exps):
        img1 = Image.open(os.path.join(gen_dir, f"exp{exp_id}_s100_cfg1.0.png"))
        img3 = Image.open(os.path.join(gen_dir, f"exp{exp_id}_s100_cfg3.0.png"))
        img5 = Image.open(os.path.join(gen_dir, f"exp{exp_id}_s100_cfg5.0.png"))
        
        axes[0, col].imshow(img1, cmap='gray')
        axes[0, col].set_title(f"Exp {exp_id} | CFG 1.0")
        axes[1, col].imshow(img3, cmap='gray')
        axes[1, col].set_title(f"Exp {exp_id} | CFG 3.0")
        axes[2, col].imshow(img5, cmap='gray')
        axes[2, col].set_title(f"Exp {exp_id} | CFG 5.0")
        
    for ax in axes.flatten():
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison_grid.png"))
    plt.close()
    
    print("Ablation experiment successfully completed.")

if __name__ == "__main__":
    main()
