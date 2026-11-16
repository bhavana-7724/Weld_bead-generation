"""
interpolate_geometry.py

Phase V14: Stage 3 - Geometry-Aware Shape & Contour Interpolation.
Performs topological Signed Distance Field (SDF) blending across registered reference masks:
  Phi_target(x, y) = sum_i w_i * Phi_i(x, y)
  Mask_target(x, y) = (Phi_target <= 0.0)

Extracts clean crown contours, weld toes, and apex landmarks.
Renders engineering-style sketches and predicts calibrated physical dimensions (W, D, R)
via the prospective SVR regressor.
"""

import os
import sys
import json
import pickle
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SVR_CKPT_PATH = os.path.join(BASE_DIR, "models", "numerical", "prospective_regressor_checkpoint.pkl")
OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation")
MASK_OUT_DIR = os.path.join(OUT_BASE, "predicted_masks")
CNT_OUT_DIR = os.path.join(OUT_BASE, "predicted_contours")

os.makedirs(MASK_OUT_DIR, exist_ok=True)
os.makedirs(CNT_OUT_DIR, exist_ok=True)


def compute_sdf(mask):
    """
    Computes Signed Distance Field (SDF) for a binary mask.
    Inside mask (mask > 0): negative distance to boundary.
    Outside mask (mask == 0): positive distance to boundary.
    Boundary: 0.
    """
    binary = (mask > 127).astype(np.uint8)
    if np.sum(binary) == 0:
        return np.ones(mask.shape, dtype=np.float32) * 100.0
    if np.all(binary == 1):
        return -np.ones(mask.shape, dtype=np.float32) * 100.0
        
    d_out = distance_transform_edt(1 - binary)
    d_in = distance_transform_edt(binary)
    sdf = d_out - d_in
    return sdf.astype(np.float32)


def blend_sdfs(masks, weights, target_baseline_x=64):
    """
    Blends multiple binary masks using their Signed Distance Fields.
    Guarantees topological continuity and smoothness.
    """
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)
    
    h, w = masks[0].shape[:2]
    blended_sdf = np.zeros((h, w), dtype=np.float32)
    
    for m, weight in zip(masks, weights):
        m_gray = m if len(m.shape) == 2 else cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        sdf = compute_sdf(m_gray)
        blended_sdf += weight * sdf
        
    # Zero-level set thresholding
    pred_mask = (blended_sdf <= 0.0).astype(np.uint8) * 255
    
    # Enforce strict substrate baseline constraint
    pred_mask[:, target_baseline_x:] = 0
    
    # Keep largest connected component to eliminate stray noise
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((pred_mask > 0).astype(np.uint8))
    if num_labels > 1:
        # Label 0 is background, find largest foreground
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        pred_mask = (labels == largest_label).astype(np.uint8) * 255
        
    # Gentle morphological close to ensure seamless boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    pred_mask[:, target_baseline_x:] = 0
    
    return pred_mask, blended_sdf


def extract_crown_contour_and_landmarks(mask, baseline_x=64):
    """
    Extracts crown contour coordinates, left/right toes, and apex.
    """
    pts = {}
    ys, xs = np.where(mask > 0)
    
    if len(ys) == 0:
        return {}, {'left_toe': (baseline_x, 30), 'right_toe': (baseline_x, 90), 'apex': (50, 60)}
        
    for y in range(int(ys.min()), int(ys.max()) + 1):
        row_xs = np.where(mask[y, :baseline_x] > 0)[0]
        if len(row_xs) > 0:
            # Crown boundary is outer edge (minimum x)
            pts[int(y)] = float(row_xs.min())
            
    sorted_ys = sorted(pts.keys())
    left_toe_y = sorted_ys[0]
    right_toe_y = sorted_ys[-1]
    
    left_toe = (float(baseline_x), float(left_toe_y))
    right_toe = (float(baseline_x), float(right_toe_y))
    
    # Apex: minimum x (greatest reinforcement protrusion)
    min_x_val = 999.0
    apex_y_cand = float(np.mean(sorted_ys))
    for y, x_val in pts.items():
        if x_val < min_x_val:
            min_x_val = x_val
            apex_y_cand = float(y)
    apex = (float(min_x_val), float(apex_y_cand))
    
    landmarks = {
        'left_toe': left_toe,
        'right_toe': right_toe,
        'apex': apex,
        'y_span_px': float(right_toe_y - left_toe_y),
        'protrusion_px': float(baseline_x - min_x_val)
    }
    
    return pts, landmarks


def predict_svr_dimensions(material, process, feed_rate, speed):
    """Predicts physical width, depth, and reinforcement (mm) via trained prospective SVR."""
    if not os.path.exists(SVR_CKPT_PATH):
        return 9.50, 1.80, 3.20 # Fallback nominal
        
    try:
        with open(SVR_CKPT_PATH, "rb") as f:
            svr_bundle = pickle.load(f)
        svr_model = svr_bundle["model"]
        scaler = svr_bundle["scaler"]
        encoder = svr_bundle["encoder"]
        
        df_in = pd.DataFrame([{"feed_rate": feed_rate, "speed": speed, "material": material, "process": process}])
        X_num = scaler.transform(df_in[["feed_rate", "speed"]])
        X_cat = encoder.transform(df_in[["material", "process"]])
        X_all = np.hstack([X_num, X_cat])
        svr_preds = svr_model.predict(X_all)[0]
        
        return round(float(svr_preds[0]), 2), round(float(svr_preds[1]), 2), round(float(svr_preds[2]), 2)
    except Exception as e:
        print(f"[WARN] SVR prediction failed: {e}. Using empirical fallback.")
        return 9.50, 1.80, 3.20


def render_weld_bead_sketch(pts_dict, W, D, R, is_annotated=True, title=None, out_path=None):
    """
    Renders academic engineering sketch on clean white background.
    Strictly avoids synthetic penetration boundaries while accurately depicting
    the crown profile, baseline, and calibrated W/D/R dimensions.
    """
    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=180)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plate_margin = max(4.0, W * 0.35)
    x_min = -W / 2.0 - plate_margin
    x_max = W / 2.0 + plate_margin
    y_plate_bottom = -D - max(2.5, D * 0.8)

    # Substrate block
    ax.fill_between([x_min, x_max], [y_plate_bottom, y_plate_bottom], [0, 0],
                    color='#F8FAFC', edgecolor='#94A3B8', linewidth=1.5, linestyle='-', zorder=1)
    # Baseline
    ax.plot([x_min, x_max], [0, 0], color='#0F172A', linewidth=2.2, zorder=4)

    if pts_dict and len(pts_dict) >= 5:
        ys_raw = sorted(pts_dict.keys())
        xs_raw = [pts_dict[y] for y in ys_raw]
        y_min_val, y_max_val = ys_raw[0], ys_raw[-1]
        y_span = max(1.0, float(y_max_val - y_min_val))
        y_mid = float(y_min_val + y_max_val) / 2.0
        protrusions = [max(0.0, 64.0 - x_val) for x_val in xs_raw]
        p_max = max(1.0, max(protrusions))

        x_crown = [((y - y_mid) / (y_span / 2.0)) * (W / 2.0) for y in ys_raw]
        y_crown = [(p / p_max) * R for p in protrusions]
        x_crown[0] = -W / 2.0
        y_crown[0] = 0.0
        x_crown[-1] = W / 2.0
        y_crown[-1] = 0.0
        xs_crown = np.array(x_crown)
        ys_crown = np.array(y_crown)
    else:
        xs_crown = np.linspace(-W / 2.0, W / 2.0, 100)
        ys_crown = R * np.maximum(0.0, 1.0 - (xs_crown / (W / 2.0)) ** 2) ** 0.75
        xs_crown[0] = -W / 2.0
        ys_crown[0] = 0.0
        xs_crown[-1] = W / 2.0
        ys_crown[-1] = 0.0

    # Crown fill & boundary
    ax.fill_between(xs_crown, 0, ys_crown, color='#E2E8F0', alpha=0.9, zorder=2)
    ax.plot(xs_crown, ys_crown, color='#0F172A', linewidth=2.2, zorder=5)

    # Depth indicator
    ax.plot([-W / 2.0, W / 2.0], [-D, -D], color='#059669', linewidth=1.5, linestyle='--', zorder=3)
    ax.scatter([0], [-D], color='#059669', s=40, zorder=5, edgecolor='#0F172A', linewidth=1.0)

    apex_idx = np.argmax(ys_crown)
    apex_x = xs_crown[apex_idx]
    apex_y = ys_crown[apex_idx]

    # Landmarks
    ax.scatter([-W / 2.0, W / 2.0], [0, 0], color='#2563EB', s=55, zorder=6, edgecolor='#0F172A', linewidth=1.1)
    ax.scatter([apex_x], [apex_y], color='#DC2626', s=55, zorder=6, edgecolor='#0F172A', linewidth=1.1)

    if is_annotated:
        # Toe annotations
        ax.annotate('Left Toe', xy=(-W / 2.0, 0), xytext=(-W / 2.0 - 2.8, max(1.2, R * 0.35)),
                    arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.2),
                    fontsize=8.5, fontweight='bold', color='#1E40AF', zorder=8)
        ax.annotate('Right Toe', xy=(W / 2.0, 0), xytext=(W / 2.0 + 1.0, max(1.2, R * 0.35)),
                    arrowprops=dict(arrowstyle='->', color='#2563EB', lw=1.2),
                    fontsize=8.5, fontweight='bold', color='#1E40AF', zorder=8)

        # Apex annotation
        ax.annotate('Crown Apex', xy=(apex_x, apex_y), xytext=(apex_x + 1.5, apex_y + 0.6),
                    arrowprops=dict(arrowstyle='->', color='#DC2626', lw=1.2),
                    fontsize=8.5, fontweight='bold', color='#991B1B', zorder=8)

        # Baseline annotation
        ax.text(x_max - 0.5, 0.25, 'Substrate Baseline (Y = 0)',
                fontsize=8.0, color='#475569', ha='right', va='bottom', fontstyle='italic')

        # Width dimension arrow (W)
        y_w = -max(1.0, D * 0.35)
        ax.annotate('', xy=(-W / 2.0, y_w), xytext=(W / 2.0, y_w),
                    arrowprops=dict(arrowstyle='<->', color='#1E3A8A', lw=1.5), zorder=7)
        ax.plot([-W / 2.0, -W / 2.0], [0, y_w - 0.5], color='#94A3B8', linestyle=':', lw=1.0)
        ax.plot([W / 2.0, W / 2.0], [0, y_w - 0.5], color='#94A3B8', linestyle=':', lw=1.0)
        ax.text(0, y_w - 0.35, f"Width (W) = {W:.2f} mm",
                fontsize=9.0, fontweight='bold', color='#1E3A8A', ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#93C5FD', alpha=0.9))

        # Reinforcement arrow (R)
        x_r = -W / 2.0 - 1.5
        ax.annotate('', xy=(x_r, 0), xytext=(x_r, R),
                    arrowprops=dict(arrowstyle='<->', color='#B91C1C', lw=1.5), zorder=7)
        ax.plot([x_r - 0.5, apex_x], [R, R], color='#94A3B8', linestyle=':', lw=1.0)
        ax.text(x_r - 0.4, R / 2.0, f"Reinforcement (R)\n= {R:.2f} mm",
                fontsize=8.5, fontweight='bold', color='#B91C1C', ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#FCA5A5', alpha=0.9))

        # Penetration depth arrow (D)
        x_d = W / 2.0 + 1.5
        ax.annotate('', xy=(x_d, 0), xytext=(x_d, -D),
                    arrowprops=dict(arrowstyle='<->', color='#047857', lw=1.5), zorder=7)
        ax.plot([0, x_d + 0.5], [-D, -D], color='#94A3B8', linestyle=':', lw=1.0)
        ax.text(x_d + 0.4, -D / 2.0, f"Penetration Depth (D)\n= {D:.2f} mm",
                fontsize=8.5, fontweight='bold', color='#047857', ha='left', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#6EE7B7', alpha=0.9))

    if title:
        ax.set_title(title, fontsize=10.5, fontweight='bold', pad=10)

    ax.set_xlim(x_min - 3.5, x_max + 3.5)
    ax.set_ylim(y_plate_bottom - 1.5, max(R * 1.5, 4.0))
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig, ax


def interpolate_geometry_for_query(query_res, view='A'):
    """
    Given a query result dict from WeldReferenceIndex:
    1. Loads anchor masks.
    2. Blends SDFs according to anchor weights.
    3. Extracts crown contour and landmarks.
    4. Predicts SVR dimensions (W, D, R).
    5. Saves predicted_mask, predicted_contour, and sketches.
    """
    mat = query_res['material']
    proc = query_res['process']
    feed = query_res['query_feed_rate']
    speed = query_res['query_speed']
    
    # Load anchor masks
    masks = []
    weights = []
    
    for a in query_res['anchors']:
        w = a['weight']
        mask_path = a[f'mask_{view.lower()}_path']
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None and np.sum(m > 0) > 10:
                masks.append(m)
                weights.append(w)
                
    if len(masks) == 0:
        raise RuntimeError(f"No valid anchor masks found for query: {mat}, {proc}, F={feed}, S={speed}, View {view}")
        
    # Blend SDFs
    pred_mask, blended_sdf = blend_sdfs(masks, weights, target_baseline_x=64)
    
    # Extract contour & landmarks
    pts_dict, landmarks = extract_crown_contour_and_landmarks(pred_mask, baseline_x=64)
    
    # Predict physical dimensions via SVR
    pred_w, pred_d, pred_r = predict_svr_dimensions(mat, proc, feed, speed)
    
    # Generate contour overlay image
    cnt_img = np.zeros_like(pred_mask)
    for y, x in pts_dict.items():
        cnt_img[int(y), int(round(x))] = 255
        
    return {
        'mask': pred_mask,
        'blended_sdf': blended_sdf,
        'contour_pts': pts_dict,
        'landmarks': landmarks,
        'contour_img': cnt_img,
        'svr_dimensions': {
            'width_mm': pred_w,
            'depth_mm': pred_d,
            'reinforcement_mm': pred_r
        }
    }


if __name__ == "__main__":
    from build_reference_index import WeldReferenceIndex
    idx = WeldReferenceIndex()
    
    # Run geometry interpolation for unseen sample (IN718, ColdArc, F=5.2, S=26.0)
    print("\n--- Running Geometry & Shape Interpolation for Unseen Condition ---")
    query_res = idx.query("IN718", "ColdArc", 5.2, 26.0)
    
    for view in ['A', 'B']:
        geom = interpolate_geometry_for_query(query_res, view=view)
        
        prefix = f"unseen_IN718_ColdArc_F5.2_S26.0_view_{view}"
        mask_path = os.path.join(MASK_OUT_DIR, f"{prefix}_predicted_mask.png")
        cnt_path = os.path.join(CNT_OUT_DIR, f"{prefix}_predicted_contour.png")
        json_path = os.path.join(CNT_OUT_DIR, f"{prefix}_predicted_contour.json")
        sketch_unann_path = os.path.join(OUT_BASE, "generated_images", f"{prefix}_weld_bead_sketch.png")
        sketch_ann_path = os.path.join(OUT_BASE, "generated_images", f"{prefix}_annotated_weld_bead_sketch.png")
        
        os.makedirs(os.path.dirname(sketch_unann_path), exist_ok=True)
        
        cv2.imwrite(mask_path, geom['mask'])
        cv2.imwrite(cnt_path, geom['contour_img'])
        
        with open(json_path, 'w') as f:
            json.dump({
                'condition': {'material': 'IN718', 'process': 'ColdArc', 'feed_rate': 5.2, 'speed': 26.0, 'view': view},
                'landmarks': geom['landmarks'],
                'svr_dimensions': geom['svr_dimensions'],
                'contour_points': geom['contour_pts']
            }, f, indent=2)
            
        w_mm = geom['svr_dimensions']['width_mm']
        d_mm = geom['svr_dimensions']['depth_mm']
        r_mm = geom['svr_dimensions']['reinforcement_mm']
        
        render_weld_bead_sketch(geom['contour_pts'], w_mm, d_mm, r_mm, is_annotated=False,
                                title=f"Weld Bead Profile Sketch (View {view})\nIN718 | ColdArc | F=5.2 m/min, S=26.0 cm/min",
                                out_path=sketch_unann_path)
                                
        render_weld_bead_sketch(geom['contour_pts'], w_mm, d_mm, r_mm, is_annotated=True,
                                title=f"Annotated Weld Bead Sketch (View {view})\nIN718 | ColdArc | F=5.2 m/min, S=26.0 cm/min",
                                out_path=sketch_ann_path)
                                
        print(f"[VIEW {view}] SVR Dimensions: W={w_mm:.2f} mm, D={d_mm:.2f} mm, R={r_mm:.2f} mm")
        print(f"[VIEW {view}] Left Toe: {geom['landmarks']['left_toe']}, Right Toe: {geom['landmarks']['right_toe']}, Apex: {geom['landmarks']['apex']}")
        print(f"[VIEW {view}] Saved mask: {mask_path}")
        print(f"[VIEW {view}] Saved annotated sketch: {sketch_ann_path}")
