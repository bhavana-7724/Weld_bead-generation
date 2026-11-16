import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
from .config import Config

class WeldDataset(Dataset):
    def __init__(self, metadata_path, img_dir, scaler=None, is_train=False):
        self.metadata = pd.read_csv(metadata_path)
        self.img_dir = img_dir
        self.is_train = is_train
        
        self.continuous_cols = ['feed_rate', 'speed', 'HIP', 'efficiency']
        
        # Initialize or fit scaler
        if self.is_train and scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(self.metadata[self.continuous_cols].values)
            os.makedirs(os.path.dirname(Config.SCALER_PATH), exist_ok=True)
            joblib.dump(self.scaler, Config.SCALER_PATH)
        else:
            if scaler is not None:
                self.scaler = scaler
            else:
                self.scaler = joblib.load(Config.SCALER_PATH)

    def __len__(self):
        return len(self.metadata)
        
    def _encode_categorical(self, material, process):
        # Deterministic 4-element one-hot mapping
        mat_in718 = 1.0 if material == 'IN718' else 0.0
        mat_ss304l = 1.0 if material == 'SS304L' else 0.0
        proc_coldarc = 1.0 if process == 'ColdArc' else 0.0
        proc_gmaw = 1.0 if process == 'GMAW' else 0.0
        return [mat_in718, mat_ss304l, proc_coldarc, proc_gmaw]

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        exp_id = row['experiment_id']
        img_id = row['image_id']
        
        # Sometimes image_id in CSV has spaces or different extensions, fix it to .png as processed
        img_filename = f"{exp_id}-{row['A/B']}.png"
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Load image
        img = Image.open(img_path)
        if img.mode != 'L':
            img = img.convert('L')
            
        img_arr = np.array(img, dtype=np.float32)
        
        # Normalize to [-1, 1]
        img_arr = (img_arr / 127.5) - 1.0
        
        # [1, 256, 256]
        img_tensor = torch.tensor(img_arr).unsqueeze(0)
        
        # Continuous condition scaling
        cont_vals = row[self.continuous_cols].values.astype(np.float64).reshape(1, -1)
        cont_scaled = self.scaler.transform(cont_vals).flatten().astype(np.float32)
        
        # Categorical condition encoding
        cat_encoded = self._encode_categorical(row['material'], row['process'])
        cat_encoded = np.array(cat_encoded, dtype=np.float32)
        
        # Combine
        cond_tensor = torch.tensor(np.concatenate([cont_scaled, cat_encoded]))
        
        return img_tensor, cond_tensor, exp_id, img_id

def get_interpolation_dataset(scaler=None):
    # Extracts the 4 holdout experiments (8 images)
    master_df = pd.read_csv(Config.MASTER_CSV)
    holdout_ids = [5, 14, 22, 32]
    holdout_df = master_df[master_df['experiment_id'].isin(holdout_ids)].copy()
    
    # Save a temporary csv for dataset loading
    tmp_path = os.path.join(Config.DATA_DIR, "tmp_holdout_metadata.csv")
    holdout_df.to_csv(tmp_path, index=False)
    
    dataset = WeldDataset(tmp_path, Config.PROCESSED_IMG_DIR, scaler=scaler, is_train=False)
    return dataset
