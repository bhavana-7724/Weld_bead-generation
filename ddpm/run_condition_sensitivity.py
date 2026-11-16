import os
import torch
import numpy as np
import pandas as pd
from PIL import Image

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.utils import set_seed
import joblib

OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "ddpm", "pre_cfg_investigation")
os.makedirs(OUT_DIR, exist_ok=True)

def encode_condition(exp_metadata, scaler):
    continuous_cols = ['feed_rate', 'speed', 'HIP', 'efficiency']
    cont_vals = exp_metadata[continuous_cols].values.astype(np.float64).reshape(1, -1)
    cont_scaled = scaler.transform(cont_vals).flatten().astype(np.float32)
    
    mat_in718 = 1.0 if exp_metadata['material'] == 'IN718' else 0.0
    mat_ss304l = 1.0 if exp_metadata['material'] == 'SS304L' else 0.0
    proc_coldarc = 1.0 if exp_metadata['process'] == 'ColdArc' else 0.0
    proc_gmaw = 1.0 if exp_metadata['process'] == 'GMAW' else 0.0
    
    cat_encoded = np.array([mat_in718, mat_ss304l, proc_coldarc, proc_gmaw], dtype=np.float32)
    cond_tensor = torch.tensor(np.concatenate([cont_scaled, cat_encoded])).unsqueeze(0)
    return cond_tensor

def main():
    set_seed(42)
    device = Config.DEVICE
    
    # Load scaler and checkpoint
    scaler_path = os.path.join(Config.ROOT_DIR, "results", "ddpm", "two_condition_overfit", "overfit_scaler.pkl")
    chk_path = os.path.join(Config.ROOT_DIR, "results", "ddpm", "two_condition_overfit", "overfit_checkpoint.pt")
    
    if not os.path.exists(chk_path) or not os.path.exists(scaler_path):
        print("Missing overfit dependencies.")
        return
        
    scaler = joblib.load(scaler_path)
    
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    
    # Get metadata for conditions
    master_df = pd.read_csv(Config.MASTER_CSV)
    exp1_meta = master_df[master_df['experiment_id'] == 1].iloc[0]
    exp3_meta = master_df[master_df['experiment_id'] == 3].iloc[0]
    
    cond_A = encode_condition(exp1_meta, scaler).to(device)
    cond_B = encode_condition(exp3_meta, scaler).to(device)
    
    # Get a real image for x_start (Exp 1 - A view)
    img_path = os.path.join(Config.PROCESSED_IMG_DIR, "1-A.png")
    img = Image.open(img_path).convert('L')
    img_arr = np.array(img, dtype=np.float32)
    img_arr = (img_arr / 127.5) - 1.0
    x_start = torch.tensor(img_arr).unsqueeze(0).unsqueeze(0).to(device)
    
    timesteps_to_check = [0, 20, 50, 100, 150, 199]
    fixed_noise = torch.randn_like(x_start).to(device)
    
    results = []
    
    with torch.no_grad():
        for t_val in timesteps_to_check:
            t = torch.tensor([t_val], device=device).long()
            
            # Create noisy image
            x_t = scheduler.q_sample(x_start, t, noise=fixed_noise)
            
            # Predict noise
            pred_A = model(x_t, t, cond_A)
            pred_B = model(x_t, t, cond_B)
            
            # Calculate metrics
            diff = pred_A - pred_B
            mean_abs_diff = torch.abs(diff).mean().item()
            rmsd = torch.sqrt(torch.mean(diff**2)).item()
            
            mag_A = torch.sqrt(torch.mean(pred_A**2)).item()
            mag_B = torch.sqrt(torch.mean(pred_B**2)).item()
            avg_mag = (mag_A + mag_B) / 2.0
            
            rel_diff = rmsd / (avg_mag + 1e-8)
            
            results.append({
                "timestep": t_val,
                "mean_abs_diff": mean_abs_diff,
                "rmsd": rmsd,
                "magnitude_A": mag_A,
                "magnitude_B": mag_B,
                "relative_diff": rel_diff
            })
            
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "condition_sensitivity.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Sensitivity metrics saved to {csv_path}")

if __name__ == "__main__":
    main()
