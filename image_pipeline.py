import os
import zipfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.makedirs('data/processed/images_36', exist_ok=True)
os.makedirs('data/processed/images_L27', exist_ok=True)
os.makedirs('results/figures/image_preprocessing', exist_ok=True)

meta_36 = pd.read_csv('data/image_metadata_36.csv')
meta_l27 = pd.read_csv('data/image_metadata_L27.csv')

def preprocess_image(img, target_size=(256, 256)):
    # 1. Raw image is img
    
    # 2. ROI Extraction (Reproducible Crop)
    # Assumption: The macrograph contains the weld near the horizontal center and vertical center/bottom.
    # We will crop the central 50% width, and 60% height starting from 20% from the top.
    # For a 6000x4000 image: 
    # height (y): 4000 * 0.2 = 800 to 4000 * 0.8 = 3200
    # width (x): 6000 * 0.25 = 1500 to 6000 * 0.75 = 4500
    h, w = img.shape[:2]
    
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    
    roi = img[y1:y2, x1:x2]
    
    # 3. Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 4. Resize
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # 5. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    
    # 6. Normalization [0, 1] for modeling (though we save as 0-255 image)
    # For saving, we just keep it as 0-255 uint8.
    
    # 7. Optional Otsu thresholding (for comparison)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return roi, resized, enhanced, thresh

# Create a preprocessing visualization for one sample
def save_preprocessing_example(meta_df, name):
    row = meta_df.iloc[0]
    path = row['image_path']
    zip_path, internal_path = path.split('.zip/')
    zip_path += '.zip'
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(internal_path) as f:
            img_bytes = f.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            roi, resized, enhanced, thresh = preprocess_image(img)
            
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Raw ({img.shape[1]}x{img.shape[0]})')
            axes[0].axis('off')
            
            axes[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'ROI ({roi.shape[1]}x{roi.shape[0]})')
            axes[1].axis('off')
            
            axes[2].imshow(resized, cmap='gray')
            axes[2].set_title(f'Resized (256x256)')
            axes[2].axis('off')
            
            axes[3].imshow(enhanced, cmap='gray')
            axes[3].set_title('CLAHE Enhanced')
            axes[3].axis('off')
            
            axes[4].imshow(thresh, cmap='gray')
            axes[4].set_title('Otsu Thresholding')
            axes[4].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'results/figures/image_preprocessing/preprocessing_example_{name}.png')
            plt.close()

save_preprocessing_example(meta_36, '36_exp')
save_preprocessing_example(meta_l27, 'L27')

# Generate Data Splits

# For 36-Experiment
all_exps = meta_36['experiment_id'].unique()
interpolation_holdouts = [5, 14, 22, 32]
train_val_test_exps = [e for e in all_exps if e not in interpolation_holdouts] # 32 exps

# Split 32 into Train (20), Val (6), Test (6)
np.random.seed(42)
np.random.shuffle(train_val_test_exps)
train_exps = train_val_test_exps[:20]
val_exps = train_val_test_exps[20:26]
test_exps = train_val_test_exps[26:]

meta_36['split'] = 'interpolation_holdout'
meta_36.loc[meta_36['experiment_id'].isin(train_exps), 'split'] = 'train'
meta_36.loc[meta_36['experiment_id'].isin(val_exps), 'split'] = 'val'
meta_36.loc[meta_36['experiment_id'].isin(test_exps), 'split'] = 'test'

meta_36[meta_36['split'] == 'train'].to_csv('data/image_train_metadata.csv', index=False)
meta_36[meta_36['split'] == 'val'].to_csv('data/image_val_metadata.csv', index=False)
meta_36[meta_36['split'] == 'test'].to_csv('data/image_test_metadata.csv', index=False)

# For L27
all_l27_exps = meta_l27['experiment_id'].unique()
np.random.shuffle(all_l27_exps) # 17 total
train_l27_exps = all_l27_exps[:13]
test_l27_exps = all_l27_exps[13:]

meta_l27['split'] = 'test'
meta_l27.loc[meta_l27['experiment_id'].isin(train_l27_exps), 'split'] = 'train'

meta_l27[meta_l27['split'] == 'train'].to_csv('data/l27_image_train_metadata.csv', index=False)
meta_l27[meta_l27['split'] == 'test'].to_csv('data/l27_image_test_metadata.csv', index=False)

# Actual Image Processing and Saving
def process_and_save_dataset(meta_df, output_dir):
    for idx, row in meta_df.iterrows():
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
                    
                    _, _, enhanced, thresh = preprocess_image(img)
                    
                    base_name = row['image_id'].split('.')[0] + '.png'
                    cv2.imwrite(os.path.join(output_dir, base_name), enhanced)
                    cv2.imwrite(os.path.join(output_dir, 'thresh_' + base_name), thresh)
        except Exception as e:
            print(f"Failed to process {path}: {e}")

print("Processing 36 dataset...")
process_and_save_dataset(meta_36, 'data/processed/images_36')
print("Processing L27 dataset...")
process_and_save_dataset(meta_l27, 'data/processed/images_L27')
print("Done processing.")
