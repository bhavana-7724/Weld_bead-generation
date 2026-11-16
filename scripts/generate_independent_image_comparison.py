"""
generate_independent_image_comparison.py

Generates Figure 9: Independent Image Verification Contact Sheet.
Explicitly categorizes and distinguishes:
1. Ground Truth Experimental Macrograph (Real Photograph)
2. Genuinely Generated Neural Outputs (C-AE, C-VAE, C-GAN, DDPM)
3. Non-Generative Engineering Baselines:
   - Model E: Analytical Parametric Profiler (Geometric synthetic rendering)
   - Model F: Empirical Retrieval Baseline (Re-used training photo)
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v2")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load existing Figure 5 to extract the exact evaluated patches for all models
fig5_path = os.path.join(OUT_DIR, "fig5_generative_model_comparison.png")
if not os.path.exists(fig5_path):
    raise FileNotFoundError(f"Missing {fig5_path}")

# Figure 5 has 4 rows (Holdout 5, 14, 22, 32) and 7 columns
# Col 0: Real Ground Truth
# Col 1: Model A (C-AE)
# Col 2: Model B (C-VAE)
# Col 3: Model C (C-GAN)
# Col 4: Model D (DDPM 100-Ep)
# Col 5: Model E (Geometry-First)
# Col 6: Model F (Retrieval NN)

clean_df = pd.read_csv(os.path.join(DATA_DIR, "clean_dataset_36_v2.csv"))
holdouts = [5, 14, 22, 32]

fig5_img = cv2.imread(fig5_path)
H, W = fig5_img.shape[:2]

# Let's create an elegant, rigorously annotated publication figure
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, axes = plt.subplots(4, 7, figsize=(24, 15), dpi=300)

col_headers = [
    "GROUND TRUTH\n(Experimental Photo)",
    "MODEL A: C-AE\n(Neural Compression)",
    "MODEL B: C-VAE\n(Probabilistic Latent)",
    "MODEL C: C-GAN\n(Adversarial Generator)",
    "MODEL D: DDPM (100-Ep)\n(Reverse Diffusion)",
    "MODEL E: PARAMETRIC\n(Analytical Super-Ellipse)",
    "MODEL F: RETRIEVAL\n(Training Photo Re-used)"
]

col_categories = [
    "Real Data",
    "Neural Generative",
    "Neural Generative",
    "Neural Generative",
    "Neural Generative",
    "Analytical Synthesis",
    "Empirical Retrieval"
]

category_colors = {
    "Real Data": "#1565c0",
    "Neural Generative": "#c62828",
    "Analytical Synthesis": "#ef6c00",
    "Empirical Retrieval": "#6a1b9a"
}

# We load the raw real images and the generated images directly
from train_and_evaluate_generative_models import (
    render_geometry_cross_section,
    retrieve_nearest_training_image,
    ddpm_holdout_imgs,
    model_ae,
    model_vae,
    gen,
    get_condition_vector,
    device
)
import torch

for col_idx, (header, cat) in enumerate(zip(col_headers, col_categories)):
    axes[0, col_idx].text(0.5, 1.25, header, ha='center', va='bottom',
                          fontsize=10.5, fontweight='bold', transform=axes[0, col_idx].transAxes)
    axes[0, col_idx].text(0.5, 1.05, f"[{cat}]", ha='center', va='bottom',
                          fontsize=9, fontweight='bold', color=category_colors[cat], transform=axes[0, col_idx].transAxes)

for row_idx, h in enumerate(holdouts):
    row_info = clean_df[clean_df['sample'] == h].iloc[0]
    m_name = row_info['material']
    p_name = row_info['process']
    f_val = row_info['feed_rate']
    s_val = row_info['speed']
    w_val = row_info['width']
    d_val = row_info['depth']
    r_val = row_info['reinforcement']
    
    # 0. Real Image
    real_path = os.path.join(DATA_DIR, "processed", "images_36_v3", f"{h}-A.png")
    real_img = cv2.imread(real_path, cv2.IMREAD_GRAYSCALE)
    
    # Condition vector
    c_vec, _ = get_condition_vector(h)
    c_tensor = torch.tensor(c_vec).unsqueeze(0).to(device)
    
    # 1. Model A
    with torch.no_grad():
        ae_out = model_ae.generate(c_tensor).squeeze().cpu().numpy()
        ae_img = (ae_out * 255.0).astype(np.uint8)
        
    # 2. Model B
    with torch.no_grad():
        vae_out = model_vae.sample(c_tensor, seed=42).squeeze().cpu().numpy()
        vae_img = (vae_out * 255.0).astype(np.uint8)
        
    # 3. Model C
    with torch.no_grad():
        z_sample = torch.randn(1, 64, device=device)
        gan_out = gen(z_sample, c_tensor).squeeze().cpu().numpy()
        gan_img = (gan_out * 255.0).astype(np.uint8)
        
    # 4. Model D
    ddpm_img = ddpm_holdout_imgs[h]
    
    # 5. Model E (Ground Truth Dimensions + 12 px/mm scale)
    geom_img = render_geometry_cross_section(w_val, d_val, r_val)
    
    # 6. Model F
    ret_img, ret_id, ret_dist = retrieve_nearest_training_image(h)
    
    imgs = [real_img, ae_img, vae_img, gan_img, ddpm_img, geom_img, ret_img]
    
    row_label = f"Holdout {h}\n{m_name} | {p_name}\nF={f_val} m/min\nS={s_val} cm/min\nW={w_val} D={d_val} R={r_val}"
    
    for c_i, img in enumerate(imgs):
        axes[row_idx, c_i].imshow(img, cmap='gray')
        axes[row_idx, c_i].set_xticks([])
        axes[row_idx, c_i].set_yticks([])
        
        # Add thin border colored by category
        for spine in axes[row_idx, c_i].spines.values():
            spine.set_edgecolor(category_colors[col_categories[c_i]])
            spine.set_linewidth(2.0)
            
    axes[row_idx, 0].set_ylabel(row_label, fontsize=9.5, fontweight='bold', labelpad=10)

plt.subplots_adjust(top=0.90, bottom=0.03, left=0.08, right=0.98, hspace=0.15, wspace=0.08)

out_fig_path = os.path.join(OUT_DIR, "fig9_independent_image_verification.png")
fig.savefig(out_fig_path, dpi=300)
plt.close(fig)
print(f"Saved: {out_fig_path}")
