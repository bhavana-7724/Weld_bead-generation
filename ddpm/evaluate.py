import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

from .config import Config
from .dataset import WeldDataset, get_interpolation_dataset
from .model import ConditionalUNet
from .diffusion import DiffusionScheduler
from .sample import generate_samples

def evaluate_metrics(real_img, gen_img):
    mae = mean_absolute_error(real_img.flatten(), gen_img.flatten())
    rmse = np.sqrt(mean_squared_error(real_img.flatten(), gen_img.flatten()))
    
    data_range = 255
    s = ssim(real_img, gen_img, data_range=data_range)
    p = psnr(real_img, gen_img, data_range=data_range)
    
    return mae, rmse, s, p

def run_evaluation(model, scheduler, dataset, output_csv, output_img_dir, split_name="test"):
    device = Config.DEVICE
    os.makedirs(output_img_dir, exist_ok=True)
    
    results = []
    
    # Evaluate by unique experiment to save inference time (A/B have same conditions)
    evaluated_exps = set()
    
    for i in tqdm(range(len(dataset)), desc=f'Evaluating {split_name}'):
        img_tensor, cond_tensor, exp_id, img_id = dataset[i]
        
        real_img_numpy = ((img_tensor.squeeze().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
        cond_tensor = cond_tensor.unsqueeze(0).to(device) # [1, 8]
        
        if exp_id not in evaluated_exps:
            # Generate 1 sample per unique condition to save time
            num_samples = 1
            generation_seed = Config.SEED + i
            gen_tensors = generate_samples(model, scheduler, cond_tensor, num_samples, generation_seed)
            gen_img_numpy = gen_tensors[0].squeeze().cpu().numpy()
            
            # Save Image
            gen_img_pil = Image.fromarray(gen_img_numpy, mode='L')
            out_name = f"{exp_id}_{img_id.replace('.png', '')}_s0.png"
            gen_img_pil.save(os.path.join(output_img_dir, out_name))
            
            evaluated_exps.add(exp_id)
            # Store the generated image so we can compare it to the B image as well
            dataset._last_gen_img_numpy = gen_img_numpy
            dataset._last_gen_seed = generation_seed
        else:
            # Reuse the generated image for the B image comparison
            gen_img_numpy = dataset._last_gen_img_numpy
            generation_seed = dataset._last_gen_seed

        # Also save the original real image just once per condition for comparison grids
        real_img_pil = Image.fromarray(real_img_numpy, mode='L')
        real_img_pil.save(os.path.join(output_img_dir, f"{exp_id}_{img_id.replace('.png', '')}_real.png"))
            
        mae, rmse, s, p = evaluate_metrics(real_img_numpy, gen_img_numpy)
        row = dataset.metadata.iloc[i]
        
        results.append({
            "experiment_id": exp_id,
            "image_id": img_id,
            "partition": split_name,
            "material": row['material'],
            "process": row['process'],
            "feed_rate": row['feed_rate'],
            "speed": row['speed'],
            "HIP": row['HIP'],
            "efficiency": row['efficiency'],
            "generation_seed": generation_seed,
            "generated_sample_id": 0,
            "MAE": mae,
            "RMSE": rmse,
            "SSIM": s,
            "PSNR": p
        })
            
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")

def main():
    model = ConditionalUNet(
        in_channels=Config.IMG_CHANNELS, 
        base_channels=Config.BASE_CHANNELS,
        channel_mults=Config.CHANNEL_MULT,
        cond_dim=Config.COND_DIM,
        time_emb_dim=Config.TIME_EMB_DIM
    ).to(Config.DEVICE)
    
    chk_path = os.path.join(Config.MODEL_DIR, "checkpoint_best.pt")
    if not os.path.exists(chk_path):
        print(f"Error: No trained model found at {chk_path}")
        return
        
    model.load_state_dict(torch.load(chk_path, map_location=Config.DEVICE)['model_state_dict'])
    model.eval()
    
    scheduler = DiffusionScheduler(timesteps=Config.TIMESTEPS, device=Config.DEVICE)
    
    # 1. Evaluate Test Set
    test_dataset = WeldDataset(Config.TEST_CSV, Config.PROCESSED_IMG_DIR, is_train=False)
    run_evaluation(
        model, scheduler, test_dataset, 
        output_csv=os.path.join(Config.OUTPUT_DIR, "test_metrics.csv"),
        output_img_dir=os.path.join(Config.OUTPUT_DIR, "generated_test"),
        split_name="test"
    )
    
    # 2. Evaluate Interpolation Set
    interp_dataset = get_interpolation_dataset()
    run_evaluation(
        model, scheduler, interp_dataset, 
        output_csv=os.path.join(Config.OUTPUT_DIR, "interpolation_metrics.csv"),
        output_img_dir=os.path.join(Config.OUTPUT_DIR, "interpolation"),
        split_name="interpolation"
    )

if __name__ == "__main__":
    main()
