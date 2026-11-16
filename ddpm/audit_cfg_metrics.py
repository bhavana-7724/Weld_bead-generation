import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from ddpm.config import Config

OUT_DIR = os.path.join(Config.ROOT_DIR, "results", "ddpm", "cfg_ablation")
GEN_DIR = os.path.join(OUT_DIR, "generated")

def evaluate_metrics(real_img, gen_img):
    mae = mean_absolute_error(real_img.flatten(), gen_img.flatten())
    rmse = np.sqrt(mean_squared_error(real_img.flatten(), gen_img.flatten()))
    
    data_range = 255
    s = ssim(real_img, gen_img, data_range=data_range)
    p = psnr(real_img, gen_img, data_range=data_range)
    
    return mae, rmse, s, p

def main():
    holdout_exps = [5, 14, 22, 32]
    seeds = [100, 101, 102, 103, 104]
    scales = [1.0, 3.0, 5.0]
    
    results = []
    
    for exp_id in holdout_exps:
        real_img_path = os.path.join(Config.PROCESSED_IMG_DIR, f"{exp_id}-A.png")
        real_img = Image.open(real_img_path).convert('L')
        real_img_numpy = np.array(real_img, dtype=np.uint8)
        
        for scale in scales:
            for seed in seeds:
                gen_path = os.path.join(GEN_DIR, f"exp{exp_id}_s{seed}_cfg{scale}.png")
                
                if not os.path.exists(gen_path):
                    print(f"Missing: {gen_path}")
                    continue
                    
                gen_img = Image.open(gen_path)
                gen_img_numpy = np.array(gen_img, dtype=np.uint8)
                
                mae, rmse, s, p = evaluate_metrics(real_img_numpy, gen_img_numpy)
                
                results.append({
                    "experiment_id": exp_id,
                    "seed": seed,
                    "scale": scale,
                    "MAE": mae,
                    "RMSE": rmse,
                    "SSIM": s,
                    "PSNR": p
                })
                
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "cfg_absolute_metrics.csv"), index=False)
    
    summary = df.groupby('scale')[['MAE', 'RMSE', 'SSIM', 'PSNR']].mean()
    print("=== CFG Absolute Metrics Summary (vs Real A View) ===")
    print(summary)
    
if __name__ == "__main__":
    main()
