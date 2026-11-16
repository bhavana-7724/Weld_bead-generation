import os
import torch
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from .config import Config
from .model import ConditionalUNet
from .diffusion import DiffusionScheduler
from .dataset import WeldDataset

def generate_samples(model, scheduler, conds, num_samples_per_cond, seed):
    device = Config.DEVICE
    torch.manual_seed(seed)
    
    b = conds.shape[0]
    # We duplicate the conditions for num_samples_per_cond
    conds = conds.repeat_interleave(num_samples_per_cond, dim=0)
    
    shape = (b * num_samples_per_cond, Config.IMG_CHANNELS, Config.IMG_SIZE, Config.IMG_SIZE)
    sampled_tensors = scheduler.p_sample_loop(model, shape, conds)
    
    return sampled_tensors
