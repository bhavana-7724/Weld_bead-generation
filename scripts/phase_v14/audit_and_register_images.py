"""
audit_and_register_images.py

Phase V14: Stage 1 & Stage 2 - Comprehensive Macrograph Audit & Conservative Registration.
Audits the raw 6000x4000 source images and aligned 128x128 macrographs across all 36 physical experiments.
Applies conservative rigid registration (horizontal baseline alignment to X=64 and vertical reference tracking)
without non-linear distortion of ground-truth bead dimensions.
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")
ALIGNED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
GT_MASK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit", "gt_masks")

# External raw dataset path (on parent directory)
RAW_DATA_1 = os.path.join(os.path.dirname(BASE_DIR), "Dataset3_extracted", "Dataset 3(1)")
RAW_DATA_2 = os.path.join(os.path.dirname(BASE_DIR), "Dataset3_extracted", "Dataset 3(2)")

OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation")
AUDIT_DIR = os.path.join(OUT_BASE, "audit")
REG_DIR = os.path.join(OUT_BASE, "registered_images")
REG_COMP_DIR = os.path.join(REG_DIR, "comparisons")

os.makedirs(AUDIT_DIR, exist_ok=True)
os.makedirs(REG_DIR, exist_ok=True)
os.makedirs(REG_COMP_DIR, exist_ok=True)


def locate_raw_image(sample_id, view):
    """Finds raw 6000x4000 camera macrograph if available on disk."""
    folder = RAW_DATA_1 if sample_id <= 18 else RAW_DATA_2
    candidate = os.path.join(folder, f"{sample_id}-{view}.JPG")
    if os.path.exists(candidate):
        return candidate
    # Case-insensitive fallback
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower() == f"{sample_id}-{view}.jpg":
                return os.path.join(folder, f)
    return None


def detect_substrate_baseline(img_gray, mask=None):
    """
    Detects the true substrate front-face line in the aligned 128x128 image.
    In the established coordinate system:
    - Crown is at X < 64.
    - Plate substrate is at X >= 64.
    Uses horizontal gradient along columns 50 to 78, refined by mask boundary if available.
    """
    h, w = img_gray.shape
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # Analyze profile in search band [52, 75]
    search_min, search_max = 52, 76
    grad_profile = np.mean(np.abs(sobel_x[:, search_min:search_max]), axis=0)
    peak_x = search_min + int(np.argmax(grad_profile))
    
    # If mask is available, verify that crown does not exceed baseline
    if mask is not None and np.sum(mask > 0) > 20:
        mask_max_x = int(np.where(mask > 0)[1].max()) + 1
        # Baseline must be at or immediately adjacent to mask boundary
        if abs(peak_x - mask_max_x) <= 4:
            refined_x = mask_max_x
        else:
            refined_x = max(peak_x, mask_max_x)
    else:
        refined_x = peak_x
        
    return refined_x


def perform_conservative_registration(img_bgr, mask_gray, target_x=64):
    """
    Applies conservative rigid translation:
    - Shifts horizontally by dx so that the substrate baseline aligns exactly to target_x (64).
    - Preserves true physical bead dimensions and vertical scale intact.
    - Does NOT apply non-linear warping or artificial distortion.
    """
    h, w = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    current_x = detect_substrate_baseline(img_gray, mask_gray)
    dx = target_x - current_x
    dy = 0 # Vertical position preserved strictly to respect real alignment
    
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    
    # Border value: for substrate side (X > 64), extend plate intensity; for background, extend dark
    reg_img = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    
    reg_mask = None
    if mask_gray is not None:
        reg_mask = cv2.warpAffine(mask_gray, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # Ensure mask strictly terminates at target_x = 64
        reg_mask[:, target_x:] = 0
        
    return reg_img, reg_mask, current_x, dx, dy


def main():
    print("=" * 75)
    print("STAGE 1 & 2: MACROGRAPH RESOLUTION AUDIT & CONSERVATIVE RIGID REGISTRATION")
    print("=" * 75)
    
    if not os.path.exists(CLEAN_CSV):
        print(f"ERROR: Cannot locate {CLEAN_CSV}")
        sys.exit(1)
        
    df_clean = pd.read_csv(CLEAN_CSV)
    print(f"[AUDIT] Loaded {len(df_clean)} experimental conditions from {CLEAN_CSV}")
    
    audit_rows = []
    
    for idx, row in df_clean.iterrows():
        sid = int(row['sample'])
        scode = f"S{sid:02d}"
        mat = row['material']
        proc = row['process']
        feed = float(row['feed_rate'])
        speed = float(row['speed'])
        w_exp = float(row['width'])
        d_exp = float(row['depth'])
        r_exp = float(row['reinforcement'])
        
        for view in ['A', 'B']:
            img_code = f"{sid}-{view}"
            aligned_path = os.path.join(ALIGNED_DIR, f"{img_code}.png")
            gt_mask_path = os.path.join(GT_MASK_DIR, f"{img_code}.png")
            
            # Check raw source
            raw_path = locate_raw_image(sid, view)
            has_raw = raw_path is not None and os.path.exists(raw_path)
            raw_w, raw_h = (0, 0)
            if has_raw:
                try:
                    raw_sample = cv2.imread(raw_path)
                    if raw_sample is not None:
                        raw_h, raw_w = raw_sample.shape[:2]
                except Exception:
                    pass
            
            # Read aligned image and mask
            if not os.path.exists(aligned_path):
                print(f"[WARN] Aligned image missing: {aligned_path}")
                continue
                
            img_bgr = cv2.imread(aligned_path)
            h, w = img_bgr.shape[:2]
            
            mask_gray = None
            has_mask = False
            y_span = 0
            apex_x = 0
            apex_y = 0
            
            if os.path.exists(gt_mask_path):
                mask_gray = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_gray is not None and np.sum(mask_gray > 0) > 10:
                    has_mask = True
                    ys, xs = np.where(mask_gray > 0)
                    y_span = int(ys.max() - ys.min() + 1)
                    apex_idx = np.argmin(xs)
                    apex_x = int(xs[apex_idx])
                    apex_y = int(ys[apex_idx])
            
            # Conservative rigid registration
            reg_bgr, reg_mask, orig_baseline_x, dx, dy = perform_conservative_registration(img_bgr, mask_gray, target_x=64)
            
            # Save registered outputs
            save_reg_img_path = os.path.join(REG_DIR, f"{img_code}_registered.png")
            save_reg_mask_path = os.path.join(REG_DIR, f"{img_code}_registered_mask.png")
            cv2.imwrite(save_reg_img_path, reg_bgr)
            if reg_mask is not None:
                cv2.imwrite(save_reg_mask_path, reg_mask)
                
            # Create before/after visual comparison
            comp_fig_path = os.path.join(REG_COMP_DIR, f"{img_code}_registration_comparison.png")
            fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=150, facecolor='white')
            
            # Before
            ax0 = axes[0]
            ax0.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax0.axvline(orig_baseline_x, color='#EF4444', linestyle='--', linewidth=1.5, label=f'Detected Baseline (X={orig_baseline_x})')
            ax0.axvline(64, color='#3B82F6', linestyle=':', linewidth=1.2, label='Target (X=64)')
            if mask_gray is not None:
                ax0.contour(mask_gray, levels=[128], colors=['#10B981'], linewidths=1.2)
            ax0.set_title(f"Original Aligned: {scode}-{view}\nBaseline X = {orig_baseline_x}", fontsize=10, fontweight='bold')
            ax0.axis('off')
            ax0.legend(fontsize=7.5, loc='upper right')
            
            # After
            ax1 = axes[1]
            ax1.imshow(cv2.cvtColor(reg_bgr, cv2.COLOR_BGR2RGB))
            ax1.axvline(64, color='#10B981', linestyle='-', linewidth=1.5, label='Registered Baseline (X=64)')
            if reg_mask is not None:
                ax1.contour(reg_mask, levels=[128], colors=['#EAB308'], linewidths=1.2)
            ax1.set_title(f"Rigid Registered (dx={dx:+d} px)\nAligned Baseline X = 64", fontsize=10, fontweight='bold')
            ax1.axis('off')
            ax1.legend(fontsize=7.5, loc='upper right')
            
            plt.suptitle(f"Rigid Substrate Registration Audit — {scode} View {view} ({mat}, {proc})", fontsize=11, fontweight='bold')
            plt.tight_layout()
            plt.savefig(comp_fig_path, bbox_inches='tight', facecolor='white')
            plt.close()
            
            audit_rows.append({
                "sample_code": scode,
                "sample_id": sid,
                "view": view,
                "material": mat,
                "process": proc,
                "feed_rate_m_min": feed,
                "speed_cm_min": speed,
                "exp_width_mm": w_exp,
                "exp_depth_mm": d_exp,
                "exp_reinf_mm": r_exp,
                "raw_image_found": has_raw,
                "raw_resolution": f"{raw_w}x{raw_h}" if has_raw else "N/A",
                "aligned_resolution": f"{w}x{h}",
                "valid_crown_contour": has_mask,
                "detected_baseline_x": orig_baseline_x,
                "registration_shift_dx": dx,
                "target_baseline_x": 64,
                "crown_y_span_px": y_span,
                "crown_apex_x": apex_x,
                "crown_apex_y": apex_y,
                "scale_calibration_status": "Decoupled (Optical pixel scale varies; dimensions predicted via calibrated SVR)",
                "registered_image_path": save_reg_img_path,
                "comparison_figure_path": comp_fig_path
            })
            
    df_audit = pd.DataFrame(audit_rows)
    audit_csv_path = os.path.join(AUDIT_DIR, "IMAGE_RESOLUTION_AND_ALIGNMENT_AUDIT.csv")
    df_audit.to_csv(audit_csv_path, index=False)
    # Also save at root of results for convenient access
    df_audit.to_csv(os.path.join(OUT_BASE, "IMAGE_RESOLUTION_AND_ALIGNMENT_AUDIT.csv"), index=False)
    
    print(f"[AUDIT] Verified {len(df_audit)} macrographs ({len(df_audit[df_audit['view']=='A'])} View A, {len(df_audit[df_audit['view']=='B'])} View B).")
    print(f"[AUDIT] Raw 6000x4000 source images located: {df_audit['raw_image_found'].sum()} / {len(df_audit)}")
    print(f"[AUDIT] Baseline shift range: min dx = {df_audit['registration_shift_dx'].min()} px, max dx = {df_audit['registration_shift_dx'].max()} px.")
    print(f"[AUDIT] Saved audit report to: {audit_csv_path}")
    print(f"[AUDIT] Saved registered images to: {REG_DIR}")
    
    # Generate 36-sample registration contact sheet (View A)
    fig, axes = plt.subplots(6, 6, figsize=(16, 16), dpi=150, facecolor='white')
    fig.suptitle("Conservative Rigid Registration Grid (All 36 Samples, View A)\nBaseline Precisely Aligned to X = 64 (Green Line)", fontsize=13, fontweight='bold', y=0.99)
    
    for sid in range(1, 37):
        r_idx = (sid - 1) // 6
        c_idx = (sid - 1) % 6
        ax = axes[r_idx, c_idx]
        
        reg_im_path = os.path.join(REG_DIR, f"{sid}-A_registered.png")
        if os.path.exists(reg_im_path):
            im = cv2.imread(reg_im_path)
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.axvline(64, color='#10B981', linestyle='-', linewidth=1.2)
        ax.set_title(f"S{sid:02d}-A", fontsize=9, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    contact_path = os.path.join(REG_DIR, "all_36_registered_view_A_contact_sheet.png")
    plt.savefig(contact_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[AUDIT] Saved 36-sample contact sheet to: {contact_path}")


if __name__ == "__main__":
    main()
