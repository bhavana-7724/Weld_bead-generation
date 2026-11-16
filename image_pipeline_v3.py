import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('results/ddpm/v2_cfg_experiment', exist_ok=True)

meta_36 = pd.read_csv('data/image_metadata_36.csv')
train_meta = pd.read_csv('data/image_train_metadata.csv')

def process_padding(img, target_size=(256, 256)):
    h, w = img.shape[:2]
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    
    roi = img[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    max_dim = max(roi_h, roi_w)
    pad_top = (max_dim - roi_h) // 2
    pad_bottom = max_dim - roi_h - pad_top
    pad_left = (max_dim - roi_w) // 2
    pad_right = max_dim - roi_w - pad_left
    
    # 1. Constant Padding (Black)
    constant_pad = cv2.copyMakeBorder(gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    
    # 2. Reflection Padding
    reflect_pad = cv2.copyMakeBorder(gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT)
    
    # Resizing
    const_resized = cv2.resize(constant_pad, target_size, interpolation=cv2.INTER_AREA)
    reflect_resized = cv2.resize(reflect_pad, target_size, interpolation=cv2.INTER_AREA)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    const_final = clahe.apply(const_resized)
    reflect_final = clahe.apply(reflect_resized)
    
    # Normalized tensor emulation [-1, 1] mapped back to [0, 255] for visualization
    const_tensor_viz = (((const_final.astype(np.float32) / 127.5) - 1.0) + 1.0) / 2.0 * 255
    reflect_tensor_viz = (((reflect_final.astype(np.float32) / 127.5) - 1.0) + 1.0) / 2.0 * 255
    
    return gray, const_final, reflect_final, const_tensor_viz.astype(np.uint8), reflect_tensor_viz.astype(np.uint8)

def generate_contact_sheet():
    print("Generating Padding Comparison Contact Sheet...")
    # Sample 4 training images
    train_exps = train_meta['experiment_id'].unique()[:4]
    sampled = meta_36[meta_36['experiment_id'].isin(train_exps)].groupby('experiment_id').first().reset_index().to_dict('records')
        
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    cols = ['Original Crop', 'Constant Padded', 'Constant Tensor Viz', 'Reflection Padded', 'Reflect Tensor Viz']
    
    for col, title in enumerate(cols):
        axes[0, col].set_title(title)
        
    for i, row in enumerate(sampled):
        exp_id = row['experiment_id']
        path = row['image_path']
        zip_path, internal_path = path.split('.zip/')
        zip_path += '.zip'
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(internal_path) as f:
                img_bytes = f.read()
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
        gray, const_final, reflect_final, const_tensor_viz, reflect_tensor_viz = process_padding(img)
        
        axes[i, 0].imshow(gray, cmap='gray')
        axes[i, 0].set_ylabel(f'Exp {exp_id}')
        
        axes[i, 1].imshow(const_final, cmap='gray')
        axes[i, 2].imshow(const_tensor_viz, cmap='gray')
        axes[i, 3].imshow(reflect_final, cmap='gray')
        axes[i, 4].imshow(reflect_tensor_viz, cmap='gray')
        
        for ax in axes[i]: ax.set_xticks([]); ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig('results/ddpm/v2_cfg_experiment/padding_comparison.png')
    plt.close()
    print("Padding Comparison Complete.")

if __name__ == "__main__":
    generate_contact_sheet()
