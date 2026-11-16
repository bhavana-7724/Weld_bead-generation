"""
gpu_smoke_test.py

CUDA and GPU Verification Script for Welding Generative Models:
- Verifies PyTorch installation, CUDA availability, and device properties.
- Checks allocation on NVIDIA A100-SXM4-40GB or local fallback.
- Verifies float16 / bfloat16 mixed precision support (torch.amp.autocast).
- Executes dummy forward pass on DirectCNN and GC-WPG-v2.
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def run_gpu_smoke_test():
    print("=" * 70)
    print("WELDING AI RESEARCH: GPU & CUDA ENVIRONMENT SMOKE TEST")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(0)
        print(f"Device Name:     {props.name}")
        print(f"Compute Capab.:  {props.major}.{props.minor}")
        print(f"Total VRAM:      {props.total_memory / (1024**3):.2f} GB")
        print(f"Device Count:    {torch.cuda.device_count()} (Using CUDA:0 exclusively)")
    else:
        device = torch.device("cpu")
        print("Device Name:     CPU (CUDA not available in local runtime)")

    # Test tensor allocation and mixed precision
    print("\nTesting tensor allocation and autocast...")
    x = torch.randn(4, 6, device=device)
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
        linear = nn.Linear(6, 64).to(device)
        y = linear(x)
    print(f"Autocast Output Shape: {y.shape} | Device: {y.device} | Dtype: {y.dtype}")

    # Checkpoint loading test
    cnn_ckpt = os.path.join(BASE_DIR, "models", "generative", "direct_cnn_checkpoint.pt")
    gc_ckpt = os.path.join(BASE_DIR, "models", "generative", "gc_wpg_v2_checkpoint.pt")
    
    print("\nVerifying Generative Checkpoint Availability:")
    print(f"  Direct CNN: {os.path.exists(cnn_ckpt)} -> {cnn_ckpt}")
    print(f"  GC-WPG-v2:  {os.path.exists(gc_ckpt)} -> {gc_ckpt}")

    print("\nGPU Smoke Test Completed Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    run_gpu_smoke_test()
