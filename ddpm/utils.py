import random
import numpy as np
import torch
import os
from .config import Config

def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_dirs():
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "generated_test"), exist_ok=True)
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "interpolation"), exist_ok=True)
    os.makedirs(os.path.join(Config.ROOT_DIR, "results", "figures", "ddpm"), exist_ok=True)
