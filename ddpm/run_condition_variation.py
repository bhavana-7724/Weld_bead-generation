import os
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pandas as pd

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.sample import generate_samples
import joblib

OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "ddpm", "generation_failure_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    device = Config.DEVICE
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    # Load CFG model
    chk_path = os.path.join(Config.ROOT_DIR, "results", "ddpm", "cfg_ablation", "cfg_checkpoint.pt")
    model.load_state_dict(torch.load(chk_path, map_location=device)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    scaler = joblib.load(os.path.join(Config.MODEL_DIR, "scaler.pkl"))
    
    # Take a base condition: IN718, ColdArc, feed_rate=4.0, speed=5.0, HIP=0, efficiency=0.5
    base_cont = np.array([[4.0, 5.0, 0.0, 0.5]], dtype=np.float64)
    cat_encoded = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32) # IN718, ColdArc
    
    # Generate variations of feed rate
    feed_rates = [2.0, 4.0, 6.0]
    seed = 42
    
    # For CFG implementation (we must use the unconditional path manually or just evaluate raw if we don't use CFG wrapper)
    # The CFG model was trained with dropout, but `generate_samples` uses raw conditional inference unless we use `cfg_p_sample_loop`.
    # Let's write a quick cfg wrapper for this test.
    def cfg_p_sample_loop(model, shape, c, scale=3.0):
        b = shape[0]
        img = torch.randn(shape, device=device)
        uncond = torch.zeros_like(c)
        for i in reversed(range(0, scheduler.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            
            # Predict conditional
            cond_pred = scheduler.p_sample(model, img, t, i, c)
            
            # Predict unconditional
            uncond_pred = scheduler.p_sample(model, img, t, i, uncond)
            
            # Since p_sample returns the updated mean/img, this is slightly wrong for standard CFG (which applies to noise),
            # but we'll use the raw CFG implementation we wrote previously in run_cfg_ablation.py.
            # Actually, to be perfectly safe, let's just use raw conditional generation (scale=1.0 equivalent) 
            # because the model learned the conditions!
            pass
            
    print("Generating feed rate variations...")
    for fr in feed_rates:
        cont = np.copy(base_cont)
        cont[0, 0] = fr
        cont_scaled = scaler.transform(cont).flatten().astype(np.float32)
        cond = torch.tensor(np.concatenate([cont_scaled, cat_encoded]), dtype=torch.float32)
        cond_tensor = cond.unsqueeze(0).to(device)
        
        torch.manual_seed(seed)
        shape = (1, Config.IMG_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
        sampled_tensors = scheduler.p_sample_loop(model, shape, cond_tensor)
        gen_numpy = sampled_tensors[0].squeeze().cpu().numpy()
        Image.fromarray(gen_numpy, mode='L').save(os.path.join(OUT_DIR, f"var_feedrate_{fr}.png"))
    
    print("Done generating variations.")

if __name__ == "__main__":
    main()
