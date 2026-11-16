import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from .config import Config
from .dataset import WeldDataset

def check_memorization():
    train_dataset = WeldDataset(Config.TRAIN_CSV, Config.PROCESSED_IMG_DIR, is_train=True)
    interp_metrics_path = os.path.join(Config.OUTPUT_DIR, "interpolation_metrics.csv")
    test_metrics_path = os.path.join(Config.OUTPUT_DIR, "test_metrics.csv")
    
    gen_dir_test = os.path.join(Config.OUTPUT_DIR, "generated_test")
    gen_dir_interp = os.path.join(Config.OUTPUT_DIR, "interpolation")
    
    # Load all real training images into memory
    train_imgs = []
    train_ids = []
    for i in range(len(train_dataset)):
        img_tensor, _, exp_id, img_id = train_dataset[i]
        real_img_numpy = ((img_tensor.squeeze().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        train_imgs.append(real_img_numpy.flatten())
        train_ids.append((exp_id, img_id))
        
    train_imgs = np.array(train_imgs)
    
    # Check nearest neighbors for generated images
    memorization_results = []
    
    def evaluate_nearest(metrics_df, gen_dir, split_name):
        for idx, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc=f"Checking {split_name}"):
            exp_id = row['experiment_id']
            img_id = row['image_id']
            sample_id = row['generated_sample_id']
            gen_img_name = f"{exp_id}_{img_id.replace('.png', '')}_s{sample_id}.png"
            gen_img_path = os.path.join(gen_dir, gen_img_name)
            
            if not os.path.exists(gen_img_path):
                continue
                
            gen_img = np.array(Image.open(gen_img_path)).flatten()
            
            # Find closest train image by MSE
            mses = [mean_squared_error(gen_img, t_img) for t_img in train_imgs]
            min_idx = np.argmin(mses)
            min_mse = mses[min_idx]
            closest_train_exp, closest_train_img = train_ids[min_idx]
            
            memorization_results.append({
                "partition": split_name,
                "gen_experiment_id": exp_id,
                "gen_image_id": img_id,
                "sample_id": sample_id,
                "closest_train_exp": closest_train_exp,
                "closest_train_img": closest_train_img,
                "min_mse_to_train": min_mse,
                "is_memorized_flag": min_mse < 100.0 # arbitrary threshold for near duplicate
            })
            
    if os.path.exists(test_metrics_path):
        test_df = pd.read_csv(test_metrics_path)
        evaluate_nearest(test_df, gen_dir_test, "test")
        
    if os.path.exists(interp_metrics_path):
        interp_df = pd.read_csv(interp_metrics_path)
        evaluate_nearest(interp_df, gen_dir_interp, "interpolation")
        
    res_df = pd.DataFrame(memorization_results)
    out_path = os.path.join(Config.OUTPUT_DIR, "memorization_check.csv")
    res_df.to_csv(out_path, index=False)
    print(f"Memorization check saved to {out_path}")

if __name__ == "__main__":
    check_memorization()
