import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
torch.set_num_threads(1)
import argparse
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from ddpm.config import Config
from ddpm.dataset import WeldDataset
from ddpm.diffusion import DiffusionScheduler
from ddpm.model_v3 import ConditionalUNetV3A, ConditionalUNetV3B, ConditionalUNetV3C

def train_v3(variant, epochs=15):
    device = Config.DEVICE
    torch.manual_seed(42)
    np.random.seed(42)
    
    out_dir = f"results/ddpm/v3_ablation_{variant}"
    os.makedirs(out_dir, exist_ok=True)
    
    # Select Model
    if variant == "A":
        model = ConditionalUNetV3A(cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM).to(device)
    elif variant == "B":
        model = ConditionalUNetV3B(cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM).to(device)
    elif variant == "C":
        model = ConditionalUNetV3C(cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM).to(device)
    else:
        raise ValueError("Invalid variant")
        
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Datasets
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    
    holdouts = [5, 14, 22, 32]
    train_df = train_df[~train_df['experiment_id'].isin(holdouts)].copy()
    val_df = val_df[~val_df['experiment_id'].isin(holdouts)].copy()
    
    train_csv = os.path.join(out_dir, "train_split.csv")
    val_csv = os.path.join(out_dir, "val_split.csv")
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    dataset_dir = 'data/processed/images_36_v3' if os.path.exists('data/processed/images_36_v3') else 'data/processed/images_36_v2'
    
    train_dataset = WeldDataset(train_csv, dataset_dir, is_train=True)
    val_dataset = WeldDataset(val_csv, dataset_dir, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Resume Logic
    start_epoch = 1
    best_val_loss = float('inf')
    history_file = os.path.join(out_dir, "history.csv")
    chk_path = os.path.join(out_dir, "checkpoint.pt")
    
    if os.path.exists(chk_path):
        chk = torch.load(chk_path, map_location=device, weights_only=False)
        model.load_state_dict(chk['model_state_dict'])
        if 'optimizer_state_dict' in chk:
            optimizer.load_state_dict(chk['optimizer_state_dict'])
        if 'epoch' in chk:
            start_epoch = chk['epoch'] + 1
        if 'torch_rng_state' in chk:
            torch.set_rng_state(chk['torch_rng_state'])
        if 'np_rng_state' in chk:
            np.random.set_state(chk['np_rng_state'])
        if 'best_val_loss' in chk:
            best_val_loss = chk['best_val_loss']
        print(f"Resumed V3-{variant} from epoch {start_epoch-1}")
    else:
        if os.path.exists(history_file):
            os.remove(history_file)
    
    print(f"--- Starting V3-{variant} Training ({epochs} epochs) ---")
    
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_imgs, batch_conds, _, _ in tqdm(train_loader, desc=f'Epoch {epoch} Train', leave=False):
            batch_imgs = batch_imgs.to(device)
            batch_conds = batch_conds.to(device)
            
            if np.random.rand() < 0.15:
                batch_conds = torch.zeros_like(batch_conds)
                
            optimizer.zero_grad()
            t = torch.randint(0, Config.TIMESTEPS, (batch_imgs.shape[0],), device=device).long()
            noise = torch.randn_like(batch_imgs)
            x_t = scheduler.q_sample(batch_imgs, t, noise)
            
            pred_noise = model(x_t, t, batch_conds)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_imgs, batch_conds, _, _ in tqdm(val_loader, desc=f'Epoch {epoch} Val', leave=False):
                batch_imgs = batch_imgs.to(device)
                batch_conds = batch_conds.to(device)
                t = torch.randint(0, Config.TIMESTEPS, (batch_imgs.shape[0],), device=device).long()
                noise = torch.randn_like(batch_imgs)
                x_t = scheduler.q_sample(batch_imgs, t, noise)
                pred_noise = model(x_t, t, batch_conds)
                val_loss += F.mse_loss(pred_noise, noise).item() * batch_imgs.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Iterative History
        hist_df = pd.DataFrame([{'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}])
        hist_df.to_csv(history_file, mode='a', header=not os.path.exists(history_file), index=False)
        
        # Atomic Checkpoint
        state = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'variant': variant,
            'best_val_loss': best_val_loss,
            'torch_rng_state': torch.get_rng_state(),
            'np_rng_state': np.random.get_state()
        }
        
        tmp_chk_path = os.path.join(out_dir, "checkpoint_tmp.pt")
        torch.save(state, tmp_chk_path)
        os.replace(tmp_chk_path, chk_path)
        
        # Best Validation Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_chk_path = os.path.join(out_dir, "best_val_checkpoint.pt")
            tmp_best = os.path.join(out_dir, "best_tmp.pt")
            torch.save(state, tmp_best)
            os.replace(tmp_best, best_chk_path)
            
    print(f"V3-{variant} Training Complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, choices=["A", "B", "C"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    train_v3(args.variant, args.epochs)
