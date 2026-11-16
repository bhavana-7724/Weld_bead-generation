"""
gpu_cvae_test.py

CUDA and GPU Verification for Conditional VAE (CVAE):
- Verifies loading of permanent CVAE checkpoint.
- Runs forward inference with prospective conditioning on GPU/CPU.
- Evaluates both deterministic zero-latent decoding (z=0) and stochastic sampling (z ~ N(0, I)).
"""

import os
import sys
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.generate_all_36_samples import ConditionalVAE

def run_gpu_cvae_test():
    print("=" * 70)
    print("WELDING AI RESEARCH: CVAE GPU INFERENCE VERIFICATION")
    print("=" * 70)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")

    cvae_ckpt = os.path.join(BASE_DIR, "results", "final_integration", "guide_demo_v7_integrity_audit", "checkpoints", "cvae_checkpoint.pt")
    if not os.path.exists(cvae_ckpt):
        print(f"ERROR: CVAE checkpoint not found at {cvae_ckpt}")
        return False

    model = ConditionalVAE(cond_dim=6, latent_dim=24).to(device)
    ckpt = torch.load(cvae_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("CVAE Model State Dict successfully loaded.")

    # Test prospective condition c in R^6
    c = torch.tensor([[0.5, 0.5, 1.0, 0.0, 1.0, 0.0]], dtype=torch.float32, device=device)
    with torch.no_grad():
        out_det = model.generate(c)
        # Stochastic sampling with random z
        z_rand = torch.randn(1, 24, device=device)
        out_stoch = model.decode(z_rand, c)

    print(f"Deterministic Output Shape: {out_det.shape} | Range: [{out_det.min():.3f}, {out_det.max():.3f}]")
    print(f"Stochastic Output Shape:    {out_stoch.shape} | Range: [{out_stoch.min():.3f}, {out_stoch.max():.3f}]")
    diff = torch.abs(out_det - out_stoch).mean().item()
    print(f"Mean Difference (Det vs Stoch): {diff:.4f}")
    print("\nCVAE GPU Test Passed Successfully!")
    print("=" * 70)
    return True

if __name__ == "__main__":
    run_gpu_cvae_test()
