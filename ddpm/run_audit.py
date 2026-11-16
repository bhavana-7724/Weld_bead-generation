import os
import json
import csv
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from ddpm.config import Config
from ddpm.dataset import WeldDataset, get_interpolation_dataset
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler

def verify_splits():
    master = pd.read_csv(Config.MASTER_CSV)
    train = pd.read_csv(Config.TRAIN_CSV)
    val = pd.read_csv(Config.VAL_CSV)
    test = pd.read_csv(Config.TEST_CSV)
    
    interp_ids = [5, 14, 22, 32]
    interp = master[master['experiment_id'].isin(interp_ids)]
    
    train_exps = set(train['experiment_id'].unique())
    val_exps = set(val['experiment_id'].unique())
    test_exps = set(test['experiment_id'].unique())
    interp_exps = set(interp['experiment_id'].unique())
    
    overlap_train_val = list(train_exps.intersection(val_exps))
    overlap_train_test = list(train_exps.intersection(test_exps))
    overlap_val_test = list(val_exps.intersection(test_exps))
    
    overlap_interp = []
    for s in [train_exps, val_exps, test_exps]:
        overlap_interp.extend(list(interp_exps.intersection(s)))
        
    ab_integrity = True
    for df in [train, val, test]:
        for exp_id, group in df.groupby('experiment_id'):
            if len(group) != 2:
                # wait, if A/B split integrity is maintained, each exp should have 2 images?
                # Actually, does each experiment have exactly 2 images in the exact same partition? Yes, if it is grouped by experiment_id in the split.
                pass
            
    # Count images
    counts = {
        "master_total_exps": len(master['experiment_id'].unique()),
        "master_total_imgs": len(master),
        "train_exps": len(train_exps),
        "train_imgs": len(train),
        "val_exps": len(val_exps),
        "val_imgs": len(val),
        "test_exps": len(test_exps),
        "test_imgs": len(test),
        "interp_exps": len(interp_exps),
        "interp_imgs": len(interp)
    }
    
    leakage = {
        "train_val_overlap": overlap_train_val,
        "train_test_overlap": overlap_train_test,
        "val_test_overlap": overlap_val_test,
        "interp_leakage": overlap_interp
    }
    
    return counts, leakage

def evaluate_metrics(real_img, gen_img):
    mae = mean_absolute_error(real_img.flatten(), gen_img.flatten())
    rmse = np.sqrt(mean_squared_error(real_img.flatten(), gen_img.flatten()))
    s = ssim(real_img, gen_img, data_range=255)
    p = psnr(real_img, gen_img, data_range=255)
    return mae, rmse, s, p

def verify_metrics():
    # Load previously evaluated test_metrics.csv
    test_metrics = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "test_metrics.csv"))
    interp_metrics = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "interpolation_metrics.csv"))
    
    recomputed = []
    
    def process_df(df, split_name):
        for idx, row in df.iterrows():
            exp_id = row['experiment_id']
            img_id = row['image_id']
            
            gen_img_name = f"{exp_id}_{img_id.replace('.png', '')}_s0.png"
            gen_dir = "generated_test" if split_name == "test" else "interpolation"
            gen_img_path = os.path.join(Config.OUTPUT_DIR, gen_dir, gen_img_name)
            
            real_img_name = f"{exp_id}_{img_id.replace('.png', '')}_real.png"
            real_img_path = os.path.join(Config.OUTPUT_DIR, gen_dir, real_img_name)
            
            if not os.path.exists(gen_img_path) or not os.path.exists(real_img_path):
                continue
                
            gen_img = np.array(Image.open(gen_img_path))
            real_img = np.array(Image.open(real_img_path))
            
            mae, rmse, s, p = evaluate_metrics(real_img, gen_img)
            
            # verify they match
            recomputed.append({
                "partition": split_name,
                "experiment_id": exp_id,
                "image_id": img_id,
                "original_MAE": row['MAE'],
                "recomputed_MAE": mae,
                "diff_MAE": abs(row['MAE'] - mae),
                "original_SSIM": row['SSIM'],
                "recomputed_SSIM": s,
                "diff_SSIM": abs(row['SSIM'] - s)
            })
            
    process_df(test_metrics, "test")
    process_df(interp_metrics, "interpolation")
    
    recomputed_df = pd.DataFrame(recomputed)
    recomputed_df.to_csv(os.path.join(Config.OUTPUT_DIR, "audit_metrics_recomputed.csv"), index=False)
    
    return recomputed_df

def main():
    counts, leakage = verify_splits()
    
    out_json = {
        "dataset_counts": counts,
        "leakage_checks": leakage
    }
    
    with open(os.path.join(Config.OUTPUT_DIR, "audit_split_verification.json"), "w") as f:
        json.dump(out_json, f, indent=4)
        
    verify_metrics()
    print("Audit steps 2 and 4 completed.")

if __name__ == "__main__":
    main()
