import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.sample import generate_samples

OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "ddpm", "interpolation_matched_seeds")
os.makedirs(OUT_DIR, exist_ok=True)

def evaluate_metrics(real_img, gen_img):
    mae = mean_absolute_error(real_img.flatten(), gen_img.flatten())
    rmse = np.sqrt(mean_squared_error(real_img.flatten(), gen_img.flatten()))
    
    data_range = 255
    s = ssim(real_img, gen_img, data_range=data_range)
    p = psnr(real_img, gen_img, data_range=data_range)
    
    return mae, rmse, s, p

def main():
    device = Config.DEVICE
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    chk_path = os.path.join(Config.MODEL_DIR, "checkpoint_best.pt")
    model.load_state_dict(torch.load(chk_path, map_location=device)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    
    holdout_exps = [5, 14, 22, 32]
    seeds = [100, 101, 102, 103, 104]
    
    master_df = pd.read_csv(Config.MASTER_CSV)
    eval_df = master_df[master_df['experiment_id'].isin(holdout_exps)].drop_duplicates(subset=['experiment_id']).copy()
    
    # Needs a scaler. We can load from TrainDataset like before, or joblib.
    import joblib
    scaler = joblib.load(os.path.join(Config.MODEL_DIR, "scaler.pkl"))
    continuous_cols = ['feed_rate', 'speed', 'HIP', 'efficiency']
    
    results = []
    
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
        cond_tensor = cond.unsqueeze(0).to(device)
        
        real_img_path = os.path.join(Config.PROCESSED_IMG_DIR, f"{exp_id}-A.png")
        real_img = Image.open(real_img_path).convert('L')
        real_img_numpy = np.array(real_img, dtype=np.uint8)
        
        for seed in seeds:
            out_name = f"exp{exp_id}_s{seed}_baseline.png"
            out_path = os.path.join(OUT_DIR, out_name)
            
            if not os.path.exists(out_path):
                gen_tensors = generate_samples(model, scheduler, cond_tensor, 1, seed)
                gen_img_numpy = gen_tensors[0].squeeze().cpu().numpy()
                Image.fromarray(gen_img_numpy, mode='L').save(out_path)
            else:
                gen_img_numpy = np.array(Image.open(out_path))
                
            mae, rmse, s, p = evaluate_metrics(real_img_numpy, gen_img_numpy)
            
            results.append({
                "experiment_id": exp_id,
                "seed": seed,
                "MAE": mae,
                "RMSE": rmse,
                "SSIM": s,
                "PSNR": p
            })
            
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "matched_baseline_metrics.csv"), index=False)
    
    print("Baseline Matched-Seed evaluation completed.")

if __name__ == "__main__":
    main()
