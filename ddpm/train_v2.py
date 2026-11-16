import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ddpm.config import Config
from ddpm.dataset import WeldDataset
from ddpm.diffusion import DiffusionScheduler
from ddpm.model_v2 import ConditionalUNetV2
from ddpm.sample import generate_samples

V2_IMG_DIR = "data/processed/images_36_v2"
V2_OUT_DIR = "results/ddpm/v2_diagnostic"

def train_v2_diagnostic():
    device = Config.DEVICE
    torch.manual_seed(42)
    
    train_dataset = WeldDataset(Config.TRAIN_CSV, V2_IMG_DIR, is_train=True)
    val_dataset = WeldDataset(Config.VAL_CSV, V2_IMG_DIR, scaler=train_dataset.scaler, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = ConditionalUNetV2(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    history = []
    max_epochs = 100 # Diagnostic short run
    
    print("Starting V2 Diagnostic Training...")
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for imgs, conds, _, _ in train_loader:
            imgs, conds = imgs.to(device), conds.to(device)
            optimizer.zero_grad()
            
            # Condition Dropout (15%)
            if np.random.rand() < 0.15:
                conds = torch.zeros_like(conds)
                
            t = torch.randint(0, Config.TIMESTEPS, (imgs.shape[0],), device=device).long()
            noise = torch.randn_like(imgs)
            noisy_imgs = scheduler.q_sample(imgs, t, noise)
            
            predicted_noise = model(noisy_imgs, t, conds)
            loss = F.mse_loss(predicted_noise, noise)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()
            
            train_loss += loss.item() * imgs.shape[0]
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, conds, _, _ in val_loader:
                imgs, conds = imgs.to(device), conds.to(device)
                t = torch.randint(0, Config.TIMESTEPS, (imgs.shape[0],), device=device).long()
                noise = torch.randn_like(imgs)
                noisy_imgs = scheduler.q_sample(imgs, t, noise)
                
                predicted_noise = model(noisy_imgs, t, conds)
                loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item() * imgs.shape[0]
                
        val_loss /= len(val_dataset)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
    
    # Save diagnostic model
    torch.save({'model_state_dict': model.state_dict()}, os.path.join(V2_OUT_DIR, "v2_diagnostic_chk.pt"))
    pd.DataFrame(history).to_csv(os.path.join(V2_OUT_DIR, "v2_diagnostic_history.csv"), index=False)
    
    # Generate Samples for Visual Inspection
    print("Generating validation samples from experimental V2 model...")
    model.eval()
    holdout_exps = [5, 14, 22, 32]
    master_df = pd.read_csv(Config.MASTER_CSV)
    eval_df = master_df[master_df['experiment_id'].isin(holdout_exps)].drop_duplicates(subset=['experiment_id']).copy()
    
    fig, axes = plt.subplots(4, 2, figsize=(8, 16))
    
    for i, (_, row) in enumerate(eval_df.iterrows()):
        exp_id = row['experiment_id']
        cont_vals = row[['feed_rate', 'speed', 'HIP', 'efficiency']].values.astype(np.float64).reshape(1, -1)
        cont_scaled = train_dataset.scaler.transform(cont_vals).flatten().astype(np.float32)
        
        mat_in718 = 1.0 if row['material'] == 'IN718' else 0.0
        mat_ss304l = 1.0 if row['material'] == 'SS304L' else 0.0
        proc_coldarc = 1.0 if row['process'] == 'ColdArc' else 0.0
        proc_gmaw = 1.0 if row['process'] == 'GMAW' else 0.0
        cat_encoded = np.array([mat_in718, mat_ss304l, proc_coldarc, proc_gmaw], dtype=np.float32)
        
        cond = torch.tensor(np.concatenate([cont_scaled, cat_encoded]), dtype=torch.float32)
        cond_tensor = cond.unsqueeze(0).to(device)
        
        # Real V2 image
        real_img_path = os.path.join(V2_IMG_DIR, f"{exp_id}-A.png")
        real_img = Image.open(real_img_path).convert('L')
        real_img_numpy = np.array(real_img, dtype=np.uint8)
        
        # Generate with seed=100
        gen_tensors = generate_samples(model, scheduler, cond_tensor, 1, 100)
        gen_img_numpy = gen_tensors[0].squeeze().cpu().numpy()
        
        axes[i, 0].imshow(real_img_numpy, cmap='gray')
        axes[i, 0].set_title(f'Real Exp {exp_id} (V2 Padded)')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gen_img_numpy, cmap='gray')
        axes[i, 1].set_title(f'Generated Exp {exp_id} (V2 Diagnostic)')
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(V2_OUT_DIR, "v2_diagnostic_generated_samples.png"))
    print("Done!")

if __name__ == "__main__":
    train_v2_diagnostic()
