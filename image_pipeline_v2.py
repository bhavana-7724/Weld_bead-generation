import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs('data/processed/images_36_v2', exist_ok=True)
os.makedirs('results/ddpm/v2_diagnostic', exist_ok=True)

meta_36 = pd.read_csv('data/image_metadata_36.csv')
train_meta = pd.read_csv('data/image_train_metadata.csv')

def preprocess_image_v2(img, target_size=(256, 256)):
    # 1. Raw image
    h, w = img.shape[:2]
    
    # 2. ROI Extraction (Reproducible Crop)
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    
    roi = img[y1:y2, x1:x2]
    roi_h, roi_w = roi.shape[:2]
    
    # 3. Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 4. Aspect-Ratio Preserving Pad to Square
    max_dim = max(roi_h, roi_w)
    pad_top = (max_dim - roi_h) // 2
    pad_bottom = max_dim - roi_h - pad_top
    pad_left = (max_dim - roi_w) // 2
    pad_right = max_dim - roi_w - pad_left
    
    padded = cv2.copyMakeBorder(gray, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    
    # 5. Resize to target
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    
    # 6. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    return roi, padded, resized, enhanced

def generate_v2_dataset():
    print("Generating Aspect-Ratio Preserving Dataset (images_36_v2)...")
    for idx, row in meta_36.iterrows():
        path = row['image_path']
        zip_path, internal_path = path.split('.zip/')
        zip_path += '.zip'
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open(internal_path) as f:
                    img_bytes = f.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None: continue
                    
                    _, _, _, enhanced = preprocess_image_v2(img)
                    
                    base_name = row['image_id'].split('.')[0] + '.png'
                    cv2.imwrite(os.path.join('data/processed/images_36_v2', base_name), enhanced)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            
def generate_contact_sheet():
    print("Generating Corrected Real Training Images Contact Sheet...")
    # Sample some training images to visualize
    train_exps = train_meta['experiment_id'].unique()
    groups = meta_36[meta_36['experiment_id'].isin(train_exps)].groupby(['material', 'process'])
    
    sampled = []
    for name, group in groups:
        sampled.extend(group.sample(min(3, len(group)), random_state=42).to_dict('records'))
        
    fig, axes = plt.subplots(len(sampled), 3, figsize=(12, 4 * len(sampled)))
    
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
                
        roi, padded, resized, enhanced = preprocess_image_v2(img)
        
        axes[i, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Raw Crop Exp {exp_id}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(padded, cmap='gray')
        axes[i, 1].set_title('Padded (1:1 Ratio)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(enhanced, cmap='gray')
        axes[i, 2].set_title('Final Enhanced 256x256')
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.savefig('results/ddpm/v2_diagnostic/corrected_training_images_sheet.png')
    plt.close()

if __name__ == "__main__":
    generate_v2_dataset()
    generate_contact_sheet()
