"""
prepare_aligned_dataset.py

Extracts, physically registers, and aligns all 72 macrograph cross-sections (36 samples x 2 views A/B)
at 128x128 resolution with strict geometric alignment:
- Plate front-face baseline at X = 64 (midline).
- Base metal workpiece on the RIGHT (X > 64).
- Weld crown / reinforcement projecting laterally to the LEFT (X < 64).
- Penetration fusion extending into base metal on the RIGHT (X > 64).
- Isotropic aspect ratio (1:1 square crop from raw 2000x2000 camera window).
- Zero reflection padding artifacts.
- Saves images to data/processed/aligned_128/ as {sample}-{view}.png.
- Computes training-only mean and std (excluding holdouts 5, 14, 22, 32).
- Generates visual verification contact sheets.
"""

import os
import re
import zipfile
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_IMG_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

DEMO_DIR = os.path.join(BASE_DIR, "results", "final_integration", "meeting_demo")
os.makedirs(DEMO_DIR, exist_ok=True)

ZIP_1 = os.path.join(BASE_DIR, "..", "Dataset 3(1).zip")
ZIP_2 = os.path.join(BASE_DIR, "..", "Dataset 3(2).zip")

HOLDOUTS = [5, 14, 22, 32]

def find_file_in_zip(z, sid, view):
    # Matches patterns like "/1-A.JPG", "/1 - A.JPG", "1-A.jpg"
    pattern = re.compile(rf'(?:^|/){sid}\s*-\s*{view}\.JPG$', re.IGNORECASE)
    matches = [n for n in z.namelist() if pattern.search(n)]
    if not matches:
        # Fallback search
        pattern2 = re.compile(rf'{sid}\s*-\s*{view}', re.IGNORECASE)
        matches = [n for n in z.namelist() if pattern2.search(n) and n.lower().endswith('.jpg')]
    return matches[0] if matches else None

def main():
    print("Opening raw macrograph archives...")
    z1 = zipfile.ZipFile(ZIP_1, 'r')
    z2 = zipfile.ZipFile(ZIP_2, 'r')
    
    extracted_records = []
    
    for sid in range(1, 37):
        z = z1 if sid <= 18 else z2
        for view in ['A', 'B']:
            fname = find_file_in_zip(z, sid, view)
            if not fname:
                print(f"ERROR: Could not locate image for Sample {sid} View {view}")
                continue
                
            b = z.read(fname)
            raw = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            
            # Robust physical alignment crop:
            # 2000x2000 window: Y in [2000, 4000], X in [1200, 3200]
            crop_raw = raw[2000:4000, 1200:3200]
            crop_128 = cv2.resize(crop_raw, (128, 128), interpolation=cv2.INTER_AREA)
            
            save_name = f"{sid}-{view}.png"
            save_path = os.path.join(OUT_IMG_DIR, save_name)
            cv2.imwrite(save_path, crop_128)
            
            # Calculate intensity stats
            gray = cv2.cvtColor(crop_128, cv2.COLOR_BGR2GRAY)
            extracted_records.append({
                "sample": sid,
                "view": view,
                "filename": save_name,
                "path": save_path,
                "is_holdout": sid in HOLDOUTS,
                "mean_val": float(gray.mean()),
                "std_val": float(gray.std())
            })
            
    df_meta = pd.DataFrame(extracted_records)
    df_meta.to_csv(os.path.join(DATA_DIR, "aligned_128_metadata.csv"), index=False)
    print(f"Successfully processed and saved {len(df_meta)} images to {OUT_IMG_DIR}")
    
    # Compute training-set-only statistics
    train_meta = df_meta[~df_meta['is_holdout']].copy()
    holdout_meta = df_meta[df_meta['is_holdout']].copy()
    print(f"Training images: {len(train_meta)}, Holdout images: {len(holdout_meta)}")
    
    # Calculate channel-wise training mean and std
    train_imgs = []
    for _, r in train_meta.iterrows():
        im = cv2.imread(r['path']) # BGR
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        train_imgs.append(im)
    train_imgs = np.stack(train_imgs, axis=0) # (64, 128, 128, 3)
    
    rgb_mean = train_imgs.mean(axis=(0, 1, 2)).tolist()
    rgb_std = train_imgs.std(axis=(0, 1, 2)).tolist()
    print(f"RGB Training Mean: {rgb_mean}, Std: {rgb_std}")
    
    stats_dict = {
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "grayscale_mean": float(train_imgs.mean()),
        "grayscale_std": float(train_imgs.std()),
        "train_image_count": len(train_meta),
        "holdout_image_count": len(holdout_meta),
        "holdout_samples": HOLDOUTS
    }
    pd.Series(stats_dict).to_json(os.path.join(DATA_DIR, "aligned_128_stats.json"))
    
    # Generate verification contact sheet of all 36 samples (View A)
    fig, axes = plt.subplots(6, 6, figsize=(15, 15))
    fig.suptitle("Aligned 128x128 Physical Registration Grid (All 36 Samples, View A)\nWorkpiece on Right (X > 64), Reinforcement on Left (X < 64)", fontsize=13, y=0.99)
    
    for sid in range(1, 37):
        r_idx = (sid - 1) // 6
        c_idx = (sid - 1) % 6
        ax = axes[r_idx, c_idx]
        
        im_path = os.path.join(OUT_IMG_DIR, f"{sid}-A.png")
        im = cv2.imread(im_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        ax.imshow(im)
        ax.axvline(64, color='gold', linestyle=':', alpha=0.7, linewidth=1.2)
        ax.axhline(64, color='cyan', linestyle=':', alpha=0.5, linewidth=1.0)
        
        title_suffix = " (HOLDOUT)" if sid in HOLDOUTS else ""
        ax.set_title(f"S{sid}{title_suffix}", fontsize=9, color='red' if sid in HOLDOUTS else 'black')
        ax.axis('off')
        
    plt.tight_layout()
    contact_sheet_path = os.path.join(DEMO_DIR, "aligned_128_all_samples_contact_sheet.png")
    fig.savefig(contact_sheet_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved alignment contact sheet to {contact_sheet_path}")

if __name__ == "__main__":
    main()
