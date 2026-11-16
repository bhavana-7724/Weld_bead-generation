"""
generate_sample.py

Standalone, deterministic reproduction script for conditional weld cross-section generation.
Loads the trained GC-VAE checkpoint and generates a genuine weld macrograph cross-section
strictly from welding input parameters (without accessing ground-truth images or bead dimensions).

Usage examples:
  # Generate by experiment ID from dataset:
  python scripts/generate_sample.py --sample_id 15 --seed 42

  # Generate by specifying explicit welding parameters:
  python scripts/generate_sample.py --material IN718 --process GMAW --feed_rate 5.5 --speed 25.0 --seed 42 --output results/final_integration/meeting_demo/best_generated_weld.png
"""

import os
import sys
import argparse
import joblib
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_CKPT = os.path.join(BASE_DIR, "results", "final_integration", "meeting_demo", "generation_model_checkpoint.pt")
DEFAULT_OUT = os.path.join(BASE_DIR, "results", "final_integration", "meeting_demo", "best_generated_weld.png")

# ---------------------------------------------------------
# Architecture Definition (matches training checkpoint)
# ---------------------------------------------------------
class ConditionalWeldGenerator(nn.Module):
    def __init__(self, cond_dim=8, noise_dim=16):
        super().__init__()
        self.cond_dim = cond_dim
        self.noise_dim = noise_dim
        in_dim = cond_dim + noise_dim

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        return self.dec(h)

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            if seed is not None:
                g = torch.Generator(device=c.device)
                g.manual_seed(seed)
                z = torch.randn((c.size(0), self.noise_dim), generator=g, dtype=torch.float32, device=c.device)
            else:
                z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
            return self.forward(c, z)

def build_condition_vector(feed_rate, speed, hip, efficiency, material, process, scaler):
    c_cont = np.array([[feed_rate, speed, hip, efficiency]], dtype=np.float64)
    c_norm = scaler.transform(c_cont).flatten().astype(np.float32)
    c_cat = np.array([
        1.0 if material.upper() == 'IN718' else 0.0,
        1.0 if material.upper() == 'SS304L' else 0.0,
        1.0 if process.lower() == 'coldarc' else 0.0,
        1.0 if process.upper() == 'GMAW' else 0.0
    ], dtype=np.float32)
    return np.concatenate([c_norm, c_cat])

def main():
    parser = argparse.ArgumentParser(description="Deterministic conditional weld generator")
    parser.add_argument("--sample_id", type=int, default=None, help="Experiment ID from dataset (1 to 36)")
    parser.add_argument("--material", type=str, default="IN718", choices=["IN718", "SS304L"])
    parser.add_argument("--process", type=str, default="GMAW", choices=["ColdArc", "GMAW"])
    parser.add_argument("--feed_rate", type=float, default=5.5, help="Wire feed rate in m/min")
    parser.add_argument("--speed", type=float, default=25.0, help="Travel speed in cm/min")
    parser.add_argument("--hip", type=float, default=None, help="Heat input in J/mm (optional)")
    parser.add_argument("--efficiency", type=float, default=None, help="Process efficiency (optional)")
    parser.add_argument("--seed", type=int, default=42, help="Random latent seed")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT, help="Path to trained model weights")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="Path to save output image")
    args = parser.parse_args()

    # Load scaler
    scaler_path = os.path.join(BASE_DIR, "models", "ddpm", "scaler.pkl")
    scaler = joblib.load(scaler_path)

    clean_path = os.path.join(DATA_DIR, "clean_dataset_36_v2.csv")
    clean_df = pd.read_csv(clean_path)

    if args.sample_id is not None:
        row = clean_df[clean_df['sample'] == args.sample_id].iloc[0]
        material = row['material']
        process = row['process']
        feed_rate = float(row['feed_rate'])
        speed = float(row['speed'])
        hip = float(row['HIP'])
        eff = float(row['efficiency'])
        print(f"Loaded Sample {args.sample_id}: {material} {process} | F={feed_rate} m/min, S={speed} cm/min, HIP={hip:.1f} J/mm")
    else:
        material = args.material
        process = args.process
        feed_rate = args.feed_rate
        speed = args.speed
        eff = args.efficiency if args.efficiency is not None else (0.60 if process.lower() == 'coldarc' else 0.80)
        hip = args.hip if args.hip is not None else ((feed_rate * 25.0 * 60.0) / (speed * 10.0))
        print(f"Manual Condition: {material} {process} | F={feed_rate} m/min, S={speed} cm/min, HIP={hip:.1f} J/mm, Eff={eff:.2f}")

    c_vec = build_condition_vector(feed_rate, speed, hip, eff, material, process, scaler)
    c_tensor = torch.tensor(c_vec, dtype=torch.float32).unsqueeze(0)

    # Initialize and load model
    model = ConditionalWeldGenerator(cond_dim=8, noise_dim=16)
    # Check fallback if DEFAULT_CKPT does not exist yet but gc_vae_model.pt does
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        alt_ckpt = os.path.join(os.path.dirname(ckpt_path), "gc_vae_model.pt")
        if os.path.exists(alt_ckpt):
            ckpt_path = alt_ckpt

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint} or {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        gen_tensor = model.generate(c_tensor, seed=args.seed).squeeze(0)
        gen_np = gen_tensor.permute(1, 2, 0).numpy() # (H, W, 3) in [0, 1] RGB
        gen_np = np.clip(gen_np * 255.0, 0, 255).astype(np.uint8)

    # Save output image
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(gen_np, cv2.COLOR_RGB2BGR))
    print(f"Successfully generated 128x128 image saved to: {args.output}")

if __name__ == "__main__":
    main()
