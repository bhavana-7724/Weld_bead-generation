# Conditional DDPM for Weld-Bead Image Generation

## Overview
This repository contains the implementation of a Conditional Denoising Diffusion Probabilistic Model (DDPM) to generate weld-bead images based on specified welding conditions (feed rate, speed, HIP, efficiency, material, process).

## Directory Structure
- `ddpm/config.py`: Configuration and hyperparameters.
- `ddpm/dataset.py`: PyTorch dataset for continuous/categorical conditioning and image scaling.
- `ddpm/model.py`: Conditional U-Net.
- `ddpm/diffusion.py`: Forward/reverse diffusion schedule.
- `ddpm/train.py`: Training script and smoke test.
- `ddpm/sample.py`: Sampling and inference utilities.
- `ddpm/evaluate.py`: Evaluation metrics (MAE, RMSE, SSIM, PSNR).
- `ddpm/utils.py`: Determinism and file-system setup.

## Usage

### 1. Smoke Test
Run the smoke test to verify forward/backward passes and output dimensions:
```bash
python -m ddpm.train --smoke
```

### 2. Full Training
Train the Conditional DDPM using the 20 predefined training experiments:
```bash
python -m ddpm.train
```
Best models and configuration will be saved in `models/ddpm/`.
Training history will be logged to `results/ddpm/ddpm_training_history.csv`.

### 3. Evaluation
After training, evaluate the model on the test partition and the held-out interpolation experiments (5, 14, 22, 32):
```bash
python -m ddpm.evaluate
```
Images and metrics will be saved in `results/ddpm/`.

## Dependencies
- `torch`
- `numpy`
- `pandas`
- `pillow`
- `scikit-learn`
- `scikit-image`
- `tqdm`
