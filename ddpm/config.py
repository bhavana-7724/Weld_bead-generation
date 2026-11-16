import os
import torch

class Config:
    # Random seed
    SEED = 42

    # Model architecture
    IMG_CHANNELS = 1
    IMG_SIZE = 256
    BASE_CHANNELS = 16 # Reduced from 32 for speed on CPU
    CHANNEL_MULT = [1, 2, 4]
    TIME_EMB_DIM = 128
    COND_DIM = 8

    # Diffusion schedule
    TIMESTEPS = 200
    BETA_START = 0.0001
    BETA_END = 0.02

    # Training
    BATCH_SIZE = 4
    LR = 2e-4 # slightly higher for faster convergence
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    MAX_EPOCHS = 40 # Reduced for execution time constraints
    PATIENCE = 15

    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    PROCESSED_IMG_DIR = os.path.join(DATA_DIR, "processed", "images_36")
    TRAIN_CSV = os.path.join(DATA_DIR, "image_train_metadata.csv")
    VAL_CSV = os.path.join(DATA_DIR, "image_val_metadata.csv")
    TEST_CSV = os.path.join(DATA_DIR, "image_test_metadata.csv")
    MASTER_CSV = os.path.join(DATA_DIR, "image_metadata_36.csv")

    OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "ddpm")
    MODEL_DIR = os.path.join(ROOT_DIR, "models", "ddpm")
    
    # Preprocessor save path
    SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
