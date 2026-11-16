import pandas as pd
import json
import os
from ddpm.config import Config

def compute_summary():
    test_csv = os.path.join(Config.OUTPUT_DIR, "test_metrics.csv")
    interp_csv = os.path.join(Config.OUTPUT_DIR, "interpolation_metrics.csv")
    mem_csv = os.path.join(Config.OUTPUT_DIR, "memorization_check.csv")
    
    summary = {}
    
    if os.path.exists(test_csv):
        df_test = pd.read_csv(test_csv)
        summary['test'] = {
            'MAE': float(df_test['MAE'].mean()),
            'RMSE': float(df_test['RMSE'].mean()),
            'SSIM': float(df_test['SSIM'].mean()),
            'PSNR': float(df_test['PSNR'].mean())
        }
        
    if os.path.exists(interp_csv):
        df_interp = pd.read_csv(interp_csv)
        summary['interpolation'] = {
            'MAE': float(df_interp['MAE'].mean()),
            'RMSE': float(df_interp['RMSE'].mean()),
            'SSIM': float(df_interp['SSIM'].mean()),
            'PSNR': float(df_interp['PSNR'].mean())
        }
        
    if os.path.exists(mem_csv):
        df_mem = pd.read_csv(mem_csv)
        summary['memorization_flags'] = int(df_mem['is_memorized_flag'].sum())
        summary['total_samples'] = len(df_mem)
        
    with open(os.path.join(Config.OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f)

if __name__ == "__main__":
    compute_summary()
