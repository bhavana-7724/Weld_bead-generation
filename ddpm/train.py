import os
import argparse
import time
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from .config import Config
from .dataset import WeldDataset, get_interpolation_dataset
from .model import ConditionalUNet
from .diffusion import DiffusionScheduler
from .utils import set_seed, create_dirs

def run_smoke_test():
    print("=== RUNNING SMOKE TEST ===")
    set_seed()
    create_dirs()
    
    device = Config.DEVICE
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device used: {device}")
    
    dataset = WeldDataset(Config.TRAIN_CSV, Config.PROCESSED_IMG_DIR, is_train=True)
    dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # 1. Load one batch
    batch = next(iter(dataloader))
    imgs, conds, exp_ids, img_ids = batch
    imgs = imgs.to(device)
    conds = conds.to(device)
    
    # 2 & 3. Verifications
    assert imgs.shape == (imgs.shape[0], 1, 256, 256), f"Wrong image shape: {imgs.shape}"
    assert conds.shape == (conds.shape[0], Config.COND_DIM), f"Wrong condition shape: {conds.shape}"
    print(f"Batch size: {imgs.shape[0]}")
    print(f"Image shape: {imgs.shape}, Condition shape: {conds.shape}")
    print(f"Experiment IDs in batch: {exp_ids}")
    print(f"Image IDs in batch: {img_ids}")
    print(f"Image values min={imgs.min().item():.3f}, max={imgs.max().item():.3f}")
    
    # Forward diffusion
    t = torch.randint(0, Config.TIMESTEPS, (imgs.shape[0],), device=device).long()
    noise = torch.randn_like(imgs)
    noisy_imgs = scheduler.q_sample(imgs, t, noise)
    
    # Forward pass
    predicted_noise = model(noisy_imgs, t, conds)
    
    assert predicted_noise.shape == noise.shape, f"Predicted noise shape mismatch: {predicted_noise.shape} != {noise.shape}"
    
    # Loss calculation
    loss = F.mse_loss(predicted_noise, noise)
    print(f"Initial loss: {loss.item()}")
    assert torch.isfinite(loss), "Loss is not finite"
    
    # Backward pass & update
    loss.backward()
    optimizer.step()
    
    # Check updated loss (no need for it to be lower, just checking if forward works after update)
    predicted_noise_new = model(noisy_imgs, t, conds)
    loss_new = F.mse_loss(predicted_noise_new, noise)
    print(f"Post-update loss: {loss_new.item()}")
    
    # Verify checkpointing
    chk_path = os.path.join(Config.MODEL_DIR, "smoke_test_chk.pt")
    torch.save(model.state_dict(), chk_path)
    model.load_state_dict(torch.load(chk_path))
    
    # Reverse sampling
    sample_shape = (1, 1, 256, 256)
    sample_c = conds[0:1] # Just take the first condition
    sampled_img = scheduler.p_sample_loop(model, sample_shape, sample_c)
    
    assert sampled_img.shape == sample_shape, f"Sampled shape mismatch: {sampled_img.shape}"
    print(f"Sampled image finite min={sampled_img.min()}, max={sampled_img.max()}")
    print("SMOKE TEST PASSED.")

def train():
    set_seed()
    create_dirs()
    device = Config.DEVICE
    
    train_dataset = WeldDataset(Config.TRAIN_CSV, Config.PROCESSED_IMG_DIR, is_train=True)
    val_dataset = WeldDataset(Config.VAL_CSV, Config.PROCESSED_IMG_DIR, scaler=train_dataset.scaler, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()
    
    for epoch in range(1, Config.MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        
        for imgs, conds, _, _ in train_loader:
            imgs, conds = imgs.to(device), conds.to(device)
            optimizer.zero_grad()
            
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
        
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
        
        # Save latest
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': {
                'IMG_CHANNELS': Config.IMG_CHANNELS,
                'BASE_CHANNELS': Config.BASE_CHANNELS,
                'CHANNEL_MULT': Config.CHANNEL_MULT,
                'COND_DIM': Config.COND_DIM,
                'TIME_EMB_DIM': Config.TIME_EMB_DIM,
                'TIMESTEPS': Config.TIMESTEPS,
                'SEED': Config.SEED
            }
        }, os.path.join(Config.MODEL_DIR, "checkpoint_latest.pt"))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, os.path.join(Config.MODEL_DIR, "checkpoint_best.pt"))
            print(f"--> Saved best model at epoch {epoch}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
                
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes.")
    
    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(Config.OUTPUT_DIR, "ddpm_training_history.csv"), index=False)
    
    with open(os.path.join(Config.OUTPUT_DIR, "ddpm_config.json"), "w") as f:
        json.dump({k: v for k, v in Config.__dict__.items() if not k.startswith("__") and isinstance(v, (int, float, str, list))}, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    args = parser.parse_args()
    
    if args.smoke:
        run_smoke_test()
    else:
        train()
