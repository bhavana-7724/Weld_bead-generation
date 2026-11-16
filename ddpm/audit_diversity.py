import os
import torch
import numpy as np
import pandas as pd
from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.sample import generate_samples
from ddpm.dataset import get_interpolation_dataset

def main():
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(Config.DEVICE)
    
    chk_path = os.path.join(Config.MODEL_DIR, "checkpoint_best.pt")
    model.load_state_dict(torch.load(chk_path, map_location=Config.DEVICE)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=Config.DEVICE)
    
    ds = get_interpolation_dataset()
    img_tensor, cond_tensor, exp_id, img_id = ds[0] # Sample 5
    
    cond_tensor = cond_tensor.unsqueeze(0).to(Config.DEVICE)
    
    print("Generating samples for Diversity & Sensitivity Check...")
    
    # 1. Diversity: Generate 2 samples with fixed condition, different seeds
    # num_samples=2, generation_seed=42 will produce shape (2, 1, 256, 256) inside sample.py
    # wait, sample.py logic generates with a single seed but batch size N. Does a single batch of N give diverse outputs?
    # Yes, because torch.randn uses the RNG state, generating N independent noise vectors.
    out_diversity = generate_samples(model, scheduler, cond_tensor, num_samples_per_cond=2, seed=42)
    img_seed1 = out_diversity[0].squeeze().cpu().numpy()
    img_seed2 = out_diversity[1].squeeze().cpu().numpy()
    
    mse_diversity = np.mean((img_seed1 - img_seed2)**2)
    
    # 2. Sensitivity: Modify condition (e.g. change feed_rate which is index 0)
    cond_tensor_mod = cond_tensor.clone()
    # increase normalized feed_rate by 1.0 standard deviations
    cond_tensor_mod[0, 0] += 1.0 
    
    out_mod = generate_samples(model, scheduler, cond_tensor_mod, num_samples_per_cond=1, seed=42)
    # wait, if I use the same seed, the first sample in the batch will have the exact same starting noise!
    # generate_samples sets the manual_seed to 42 each time.
    img_mod = out_mod[0].squeeze().cpu().numpy()
    
    mse_sensitivity = np.mean((img_seed1 - img_mod)**2)
    
    print(f"MSE between two different random seeds (Diversity): {mse_diversity:.4f}")
    print(f"MSE between identical seed but modified condition (Sensitivity): {mse_sensitivity:.4f}")
    
    res = {
        "mse_diversity_fixed_cond": float(mse_diversity),
        "mse_sensitivity_fixed_seed": float(mse_sensitivity)
    }
    
    import json
    with open(os.path.join(Config.OUTPUT_DIR, "audit_diversity.json"), "w") as f:
        json.dump(res, f, indent=4)

if __name__ == "__main__":
    main()
