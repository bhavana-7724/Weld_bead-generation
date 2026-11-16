import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from ddpm.config import Config
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.dataset import WeldDataset, get_interpolation_dataset
from ddpm.sample import generate_samples

def get_metrics(real_img, gen_img):
    mae = mean_absolute_error(real_img.flatten(), gen_img.flatten())
    rmse = np.sqrt(mean_squared_error(real_img.flatten(), gen_img.flatten()))
    s = ssim(real_img, gen_img, data_range=255)
    p = psnr(real_img, gen_img, data_range=255)
    return mae, rmse, s, p

def task1_and_2():
    print("Running Task 1 & 2: Recomputing Pair Metrics")
    test_metrics = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "test_metrics.csv"))
    interp_metrics = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "interpolation_metrics.csv"))
    
    combined = pd.concat([test_metrics, interp_metrics], ignore_index=True)
    
    pair_metrics = []
    
    for idx, row in combined.iterrows():
        exp_id = row['experiment_id']
        img_id = row['image_id']
        split_name = row['partition']
        
        # Determine view
        view = "A" if "-A" in img_id.replace(" ", "") or "- A" in img_id else "B"
        
        gen_dir = "generated_test" if split_name == "test" else "interpolation"
        gen_img_name = f"{exp_id}_{img_id.replace('.png', '')}_s0.png"
        real_img_name = f"{exp_id}_{img_id.replace('.png', '')}_real.png"
        
        gen_img_path = os.path.join(Config.OUTPUT_DIR, gen_dir, gen_img_name)
        real_img_path = os.path.join(Config.OUTPUT_DIR, gen_dir, real_img_name)
        
        if not os.path.exists(gen_img_path) or not os.path.exists(real_img_path):
            continue
            
        gen_img = np.array(Image.open(gen_img_path))
        real_img = np.array(Image.open(real_img_path))
        
        mae, rmse, s, p = get_metrics(real_img, gen_img)
        
        pair_metrics.append({
            "experiment_id": exp_id,
            "partition": split_name,
            "view": view,
            "seed": row['generation_seed'],
            "generated_filename": gen_img_name,
            "reference_filename": real_img_name,
            "MAE": mae,
            "RMSE": rmse,
            "SSIM": s,
            "PSNR": p
        })
        
    df = pd.DataFrame(pair_metrics)
    out_csv = os.path.join(Config.OUTPUT_DIR, "evaluation_pair_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    
def task3_diversity():
    print("Running Task 3: Diversity Experiment")
    device = Config.DEVICE
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT, cond_dim=Config.COND_DIM, time_emb_dim=Config.TIME_EMB_DIM
    ).to(device)
    
    chk_path = os.path.join(Config.MODEL_DIR, "checkpoint_best.pt")
    model.load_state_dict(torch.load(chk_path, map_location=device)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=device)
    
    ds = get_interpolation_dataset() # Contains 4 experiments, we'll pick 3
    exps_to_test = [5, 14, 22]
    seeds_to_test = [100, 101, 102, 103, 104]
    
    cond_tensors = {}
    for i in range(len(ds)):
        img, cond, exp_id, _ = ds[i]
        if exp_id in exps_to_test and exp_id not in cond_tensors:
            cond_tensors[exp_id] = cond
    
    generated_images = {}
    for exp_id in exps_to_test:
        generated_images[exp_id] = {}
        cond_tensor = cond_tensors[exp_id].unsqueeze(0).to(device)
        for seed in seeds_to_test:
            out = generate_samples(model, scheduler, cond_tensor, num_samples_per_cond=1, seed=seed)
            generated_images[exp_id][seed] = out[0].squeeze().cpu().numpy()
            
    # Compute metrics
    diversity_records = []
    
    # Intra-condition (between seeds)
    for exp_id in exps_to_test:
        for i, s1 in enumerate(seeds_to_test):
            for s2 in seeds_to_test[i+1:]:
                img1 = generated_images[exp_id][s1]
                img2 = generated_images[exp_id][s2]
                _, _, ssim_val, _ = get_metrics(img1, img2)
                mse_val = np.mean((img1 - img2)**2)
                diversity_records.append({
                    "comparison_type": "intra_condition_diff_seed",
                    "exp_A": exp_id, "seed_A": s1,
                    "exp_B": exp_id, "seed_B": s2,
                    "MSE": mse_val, "SSIM": ssim_val
                })
                
    # Inter-condition (same seed)
    for seed in seeds_to_test:
        for i, e1 in enumerate(exps_to_test):
            for e2 in exps_to_test[i+1:]:
                img1 = generated_images[e1][seed]
                img2 = generated_images[e2][seed]
                _, _, ssim_val, _ = get_metrics(img1, img2)
                mse_val = np.mean((img1 - img2)**2)
                diversity_records.append({
                    "comparison_type": "inter_condition_same_seed",
                    "exp_A": e1, "seed_A": seed,
                    "exp_B": e2, "seed_B": seed,
                    "MSE": mse_val, "SSIM": ssim_val
                })
                
    df = pd.DataFrame(diversity_records)
    out_csv = os.path.join(Config.OUTPUT_DIR, "diversity_experiment.csv")
    df.to_csv(out_csv, index=False)
    
    # Plot grid
    fig, axes = plt.subplots(len(exps_to_test), len(seeds_to_test), figsize=(15, 9))
    for i, exp_id in enumerate(exps_to_test):
        for j, seed in enumerate(seeds_to_test):
            axes[i, j].imshow(generated_images[exp_id][seed], cmap='gray')
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(f"Seed {seed}")
            if j == 0:
                axes[i, j].text(-20, 128, f"Exp {exp_id}", va='center', ha='right', rotation=90, fontsize=12)
    
    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "ddpm", "diversity_grid.png")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_csv} and {fig_path}")
    
def task4_memorization():
    print("Running Task 4: Reassessing Memorization")
    train_ds = WeldDataset(Config.TRAIN_CSV, Config.PROCESSED_IMG_DIR, is_train=True)
    
    train_imgs = []
    train_info = []
    for i in range(len(train_ds)):
        img_tensor, _, exp_id, img_id = train_ds[i]
        real_img_numpy = ((img_tensor.squeeze().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        train_imgs.append(real_img_numpy)
        train_info.append({"exp_id": exp_id, "img_id": img_id})
        
    pair_df = pd.read_csv(os.path.join(Config.OUTPUT_DIR, "evaluation_pair_metrics.csv"))
    
    # Unique generated images
    unique_gens = pair_df.drop_duplicates(subset=['experiment_id', 'generated_filename'])
    
    nearest_records = []
    
    fig, axes = plt.subplots(len(unique_gens), 2, figsize=(6, 3 * len(unique_gens)))
    if len(unique_gens) == 1:
        axes = [axes]
        
    for i, (_, row) in enumerate(unique_gens.iterrows()):
        gen_dir = "generated_test" if row['partition'] == "test" else "interpolation"
        gen_img_path = os.path.join(Config.OUTPUT_DIR, gen_dir, row['generated_filename'])
        gen_img = np.array(Image.open(gen_img_path))
        
        best_mse = float('inf')
        best_ssim = -1.0
        best_train_idx = -1
        
        for j, t_img in enumerate(train_imgs):
            mae, rmse, s, p = get_metrics(t_img, gen_img)
            mse_val = rmse**2
            if mse_val < best_mse:
                best_mse = mse_val
                best_ssim = s
                best_train_idx = j
                
        nearest_records.append({
            "gen_experiment_id": row['experiment_id'],
            "gen_filename": row['generated_filename'],
            "nearest_train_experiment_id": train_info[best_train_idx]['exp_id'],
            "nearest_train_image_id": train_info[best_train_idx]['img_id'],
            "min_MSE": best_mse,
            "max_SSIM": best_ssim
        })
        
        ax1, ax2 = axes[i]
        ax1.imshow(gen_img, cmap='gray')
        ax1.set_title(f"Gen Exp {row['experiment_id']}")
        ax1.axis('off')
        
        ax2.imshow(train_imgs[best_train_idx], cmap='gray')
        ax2.set_title(f"Closest Train {train_info[best_train_idx]['exp_id']}")
        ax2.axis('off')
        
    df = pd.DataFrame(nearest_records)
    out_csv = os.path.join(Config.OUTPUT_DIR, "nearest_training_matches.csv")
    df.to_csv(out_csv, index=False)
    
    plt.tight_layout()
    fig_path = os.path.join("results", "figures", "ddpm", "nearest_training_matches.png")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_csv} and {fig_path}")

def main():
    task1_and_2()
    task3_diversity()
    task4_memorization()
    print("All tasks complete.")

if __name__ == "__main__":
    main()
