import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler
from skimage.metrics import structural_similarity as ssim
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import joblib

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.sample import generate_samples
from ddpm.utils import set_seed

OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "ddpm", "two_condition_overfit")
os.makedirs(OUT_DIR, exist_ok=True)

class OverfitDataset(Dataset):
    def __init__(self, df, img_dir, scaler=None):
        self.metadata = df.copy()
        self.img_dir = img_dir
        self.continuous_cols = ['feed_rate', 'speed', 'HIP', 'efficiency']
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.metadata[self.continuous_cols].values)
            joblib.dump(self.scaler, os.path.join(OUT_DIR, "overfit_scaler.pkl"))
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.metadata)
        
    def _encode_categorical(self, material, process):
        mat_in718 = 1.0 if material == 'IN718' else 0.0
        mat_ss304l = 1.0 if material == 'SS304L' else 0.0
        proc_coldarc = 1.0 if process == 'ColdArc' else 0.0
        proc_gmaw = 1.0 if process == 'GMAW' else 0.0
        return [mat_in718, mat_ss304l, proc_coldarc, proc_gmaw]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        exp_id = row['experiment_id']
        img_id = row['image_id']
        
        img_filename = f"{exp_id}-{row['A/B']}.png"
        img_path = os.path.join(self.img_dir, img_filename)
        
        img = Image.open(img_path)
        if img.mode != 'L':
            img = img.convert('L')
            
        img_arr = np.array(img, dtype=np.float32)
        img_arr = (img_arr / 127.5) - 1.0
        img_tensor = torch.tensor(img_arr).unsqueeze(0)
        
        cont_vals = row[self.continuous_cols].values.astype(np.float64).reshape(1, -1)
        cont_scaled = self.scaler.transform(cont_vals).flatten().astype(np.float32)
        
        cat_encoded = np.array(self._encode_categorical(row['material'], row['process']), dtype=np.float32)
        cond_tensor = torch.tensor(np.concatenate([cont_scaled, cat_encoded]))
        
        return img_tensor, cond_tensor, exp_id, img_id, img_arr

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
    
    # 1. Setup Data
    master_df = pd.read_csv(Config.MASTER_CSV)
    # Target Exps: 1 and 3 (from clean_dataset_36.csv sample column)
    target_exps = [1, 3]
    df_overfit = master_df[master_df['experiment_id'].isin(target_exps)].copy()
    
    dataset = OverfitDataset(df_overfit, Config.PROCESSED_IMG_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) # 4 images total
    
    # 2. Setup Model
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    
    # 3. Train Loop
    epochs = 1000
    history = []
    
    print("Starting Training on 4 images...")
    model.train()
    for epoch in range(1, epochs + 1):
        for imgs, conds, exp_ids, _, _ in dataloader:
            imgs = imgs.to(device)
            conds = conds.to(device)
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
            
        if epoch % 100 == 0 or epoch == 1:
            history.append({"epoch": epoch, "loss": loss.item()})
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.4f}")
            
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(OUT_DIR, "overfit_checkpoint.pt"))
    pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "training_history.csv"), index=False)
    
    # 4. Controlled Generation
    model.eval()
    
    cond_vectors = {}
    real_imgs = {}
    for i in range(len(dataset)):
        _, cond, exp_id, _, img_raw = dataset[i]
        if exp_id not in cond_vectors:
            cond_vectors[exp_id] = cond
            real_imgs[exp_id] = ((img_raw + 1.0)/2.0 * 255).astype(np.uint8) # denormalized true image
            
    seeds = [200, 201, 202, 203, 204]
    
    gen_imgs = {exp_id: {} for exp_id in target_exps}
    
    print("Generating samples...")
    for exp_id in target_exps:
        c = cond_vectors[exp_id].unsqueeze(0).to(device)
        for s in seeds:
            out = generate_samples(model, scheduler, c, num_samples_per_cond=1, seed=s)
            gen_imgs[exp_id][s] = out[0].squeeze().cpu().numpy()
            
    # 5. Quantitative Evaluation
    metrics = []
    
    # Matched-seed inter-condition comparisons (Exp 1 vs Exp 3 on same seed)
    for s in seeds:
        img1 = gen_imgs[1][s]
        img3 = gen_imgs[3][s]
        mse_val, ssim_val, iou_val = get_metrics(img1, img3)
        metrics.append({
            "comparison_type": "matched_seed_different_condition",
            "seed": s,
            "exp_A": 1,
            "exp_B": 3,
            "MSE": mse_val,
            "SSIM": ssim_val,
            "Foreground_IoU": iou_val
        })
        
    pd.DataFrame(metrics).to_csv(os.path.join(OUT_DIR, "generation_metrics.csv"), index=False)
    
    # 6. Grids
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    
    # Row 0: Real images (just repeating A view to pad the 5 columns)
    for j in range(5):
        # alternate between real 1 and real 3
        exp_disp = target_exps[j % 2]
        axes[0, j].imshow(real_imgs[exp_disp], cmap='gray')
        axes[0, j].set_title(f"Real Exp {exp_disp}")
        axes[0, j].axis('off')
        
    # Row 1: Generated Exp 1
    for j, s in enumerate(seeds):
        axes[1, j].imshow(gen_imgs[1][s], cmap='gray')
        axes[1, j].set_title(f"Gen Exp 1 (Seed {s})")
        axes[1, j].axis('off')
        
    # Row 2: Generated Exp 3
    for j, s in enumerate(seeds):
        axes[2, j].imshow(gen_imgs[3][s], cmap='gray')
        axes[2, j].set_title(f"Gen Exp 3 (Seed {s})")
        axes[2, j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "condition_comparison_grid.png"), bbox_inches='tight')
    plt.close()
    print("Overfit experiment completed.")

if __name__ == "__main__":
    main()
