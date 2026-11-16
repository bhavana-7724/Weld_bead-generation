"""
generate_sample_v6.py

Command-Line Interface (CLI) for Edge-Aware Conditional Weld Profile Generation (EC-WPG).
Allows users and evaluators to synthesize a cross-sectional weld profile image
from prospective welding process parameters alone, with zero experimental leakage.

Usage:
  python scripts/generate_sample_v6.py --material IN718 --process ColdArc --feed_rate 5.5 --speed 25.0 --output test_weld.png
  python scripts/generate_sample_v6.py --material SS304L --process GMAW --feed_rate 5.0 --speed 27.0 --predict_edge --output s32_synth.png
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn as nn

class ResUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_up = nn.BatchNorm2d(out_c)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.conv(self.act(self.bn_up(self.up(x))))

class EC_WPG(nn.Module):
    def __init__(self, cond_dim=6, noise_dim=16):
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
        self.up1 = ResUpBlock(128, 64)
        self.up2 = ResUpBlock(64, 32)
        self.up3 = ResUpBlock(32, 16)
        self.up4 = ResUpBlock(16, 16)
        
        self.head_img = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.head_edge = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, c, z=None):
        if z is None:
            z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
        inp = torch.cat([c, z], dim=-1)
        h = self.fc(inp).view(-1, 128, 8, 8)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        feat = self.up4(h)
        return self.head_img(feat), self.head_edge(feat)

    def generate(self, c, seed=42):
        self.eval()
        with torch.no_grad():
            if seed is not None:
                g = torch.Generator(device=c.device)
                g.manual_seed(seed)
                z = torch.randn((c.size(0), self.noise_dim), generator=g, dtype=torch.float32, device=c.device) * 0.05
            else:
                z = torch.zeros((c.size(0), self.noise_dim), dtype=torch.float32, device=c.device)
            return self.forward(c, z)

def main():
    parser = argparse.ArgumentParser(description="Generate weld bead profile image using trained EC-WPG model.")
    parser.add_argument("--material", type=str, required=True, choices=["IN718", "SS304L"],
                        help="Base material: IN718 or SS304L")
    parser.add_argument("--process", type=str, required=True, choices=["ColdArc", "GMAW"],
                        help="Welding process: ColdArc or GMAW")
    parser.add_argument("--feed_rate", type=float, required=True,
                        help="Wire feed rate in m/min (recommended: 4.5 - 6.0)")
    parser.add_argument("--speed", type=float, required=True,
                        help="Welding travel speed in cm/min (recommended: 20.0 - 30.0)")
    parser.add_argument("--checkpoint", type=str, default="results/final_integration/guide_demo_v6/ec_wpg_checkpoint.pt",
                        help="Path to trained EC-WPG checkpoint file")
    parser.add_argument("--output", type=str, default="generated_weld_v6.png",
                        help="Output path for synthesized RGB image")
    parser.add_argument("--predict_edge", action="store_true",
                        help="Also output auxiliary predicted edge boundary map")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for latent noise")
    
    args = parser.parse_args()

    # Locate checkpoint
    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt_path = os.path.join(base_dir, ckpt_path)
        if os.path.exists(alt_path):
            ckpt_path = alt_path
        else:
            print(f"Error: Checkpoint not found at {args.checkpoint} or {alt_path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 65)
    print("  EDGE-AWARE CONDITIONAL WELD PROFILE GENERATOR (EC-WPG) CLI  ")
    print("=" * 65)
    print(f"Loading checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    scaler = checkpoint["scaler_params"]
    
    # Normalize inputs
    f_val = args.feed_rate
    s_val = args.speed
    f_norm = (f_val - scaler["feed_rate_min"]) / (scaler["feed_rate_max"] - scaler["feed_rate_min"] + 1e-8)
    s_norm = (s_val - scaler["speed_min"]) / (scaler["speed_max"] - scaler["speed_min"] + 1e-8)
    
    c_cat = [
        1.0 if args.material == "IN718" else 0.0,
        1.0 if args.material == "SS304L" else 0.0,
        1.0 if args.process == "ColdArc" else 0.0,
        1.0 if args.process == "GMAW" else 0.0
    ]
    cond_vec = np.array([f_norm, s_norm] + c_cat, dtype=np.float32)
    cond_tensor = torch.tensor(cond_vec, dtype=torch.float32).unsqueeze(0)
    
    print("\nInput Welding Conditions:")
    print(f"  Material:         {args.material}")
    print(f"  Process:          {args.process}")
    print(f"  Wire Feed Rate:   {f_val:.2f} m/min (Normalized: {f_norm:.3f})")
    print(f"  Travel Speed:     {s_val:.2f} cm/min (Normalized: {s_norm:.3f})")
    print(f"  Heat Input Ratio: {f_val / s_val:.4f} (Feed / Speed proxy)")
    
    # Load model
    model = EC_WPG(cond_dim=checkpoint.get("cond_dim", 6), noise_dim=checkpoint.get("noise_dim", 16))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Generate
    with torch.no_grad():
        out_img_t, out_edge_t = model.generate(cond_tensor, seed=args.seed)
        out_img = out_img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_edge = out_edge_t.squeeze(0).squeeze(0).cpu().numpy()
        
    out_bgr = cv2.cvtColor((np.clip(out_img, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, out_bgr)
    print(f"\nGenerated RGB Weld Profile saved to: {args.output}")
    
    if args.predict_edge:
        edge_out_path = os.path.splitext(args.output)[0] + "_edge.png"
        edge_u8 = (np.clip(out_edge, 0.0, 1.0) * 255).astype(np.uint8)
        cv2.imwrite(edge_out_path, edge_u8)
        print(f"Auxiliary Boundary Map saved to:       {edge_out_path}")
        
    print("\nStandardized Physical Coordinates:")
    print("  - X = 64:  Workpiece plate front-face baseline")
    print("  - X > 64:  Base metal plate (solid workpiece substrate)")
    print("  - X < 64:  Projecting external weld crown / reinforcement")
    print("  - Y = 64:  Weld bead horizontal centerline")
    print("=" * 65)

if __name__ == "__main__":
    main()
