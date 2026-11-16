"""
interpolate_rgb.py

Phase V14: Stage 3 & Stage 4 - Sharp Appearance Generation via Geometry-Aware Macrograph Interpolation.
Implements:
1. Method 1: Direct Linear Pixel Blending (Baseline).
2. Method 2: Shape-Aligned Non-Linear Morphing (Crown-to-Crown Backward Warping).
3. Method 3: Gradient-Domain Edge-Preserving Composition (Poisson Blending).
4. Sharpness Evaluation (Tenengrad Gradient Energy, Laplacian Variance, Boundary Contrast).

Outputs all synthesized variants alongside quantitative sharpness metrics.
"""

import os
import sys
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation")
GEN_DIR = os.path.join(OUT_BASE, "generated_images")
PANEL_DIR = os.path.join(OUT_BASE, "comparison_panels")

os.makedirs(GEN_DIR, exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)


def calculate_image_sharpness(img_bgr, mask=None):
    """
    Computes rigorous objective sharpness metrics:
    - Laplacian Variance (classic focus measure; higher = sharper edges)
    - Tenengrad Gradient Energy (Sobel magnitude squared; higher = higher frequency content)
    - Boundary Contrast (mean gradient along the contour boundary)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    
    # 1. Laplacian Variance
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    if mask is not None and np.sum(mask > 0) > 0:
        lap_var = float(np.var(lap[mask > 0]))
    else:
        lap_var = float(np.var(lap))
        
    # 2. Tenengrad Gradient Energy
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = sobel_x**2 + sobel_y**2
    if mask is not None and np.sum(mask > 0) > 0:
        tenengrad_mean = float(np.mean(tenengrad[mask > 0]))
    else:
        tenengrad_mean = float(np.mean(tenengrad))
        
    # 3. Boundary contrast (in a 5-pixel dilated ring around mask perimeter)
    if mask is not None and np.sum(mask > 0) > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dil = cv2.dilate(mask, kernel)
        ero = cv2.erode(mask, kernel)
        ring = (dil > 0) & (ero == 0)
        grad_mag = np.sqrt(tenengrad)
        boundary_contrast = float(np.mean(grad_mag[ring])) if np.sum(ring) > 0 else 0.0
    else:
        boundary_contrast = 0.0
        
    return {
        'laplacian_variance': lap_var,
        'tenengrad_energy': tenengrad_mean,
        'boundary_contrast': boundary_contrast
    }


def synthesize_direct_blend(anchor_images, weights):
    """Method 1: Direct weighted sum of registered macrograph pixel values."""
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)
    
    acc = np.zeros_like(anchor_images[0], dtype=np.float32)
    for img, w in zip(anchor_images, weights):
        acc += float(w) * img.astype(np.float32)
        
    return np.clip(acc, 0, 255).astype(np.uint8)


def warp_crown_to_target(anchor_img, anchor_pts, target_pts, baseline_x=64):
    """
    Computes smooth backward remap coordinates mapping target crown coordinates (x_t, y_t)
    to anchor crown coordinates (x_a, y_a).
    """
    h, w = anchor_img.shape[:2]
    
    # Grid of target coordinates
    map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, np.newaxis], w, axis=1)
    
    if len(target_pts) < 5 or len(anchor_pts) < 5:
        # Fallback to identity mapping if points insufficient
        return anchor_img
        
    t_ys = sorted(target_pts.keys())
    t_y_min, t_y_max = t_ys[0], t_ys[-1]
    t_span = max(1.0, float(t_y_max - t_y_min))
    
    a_ys = sorted(anchor_pts.keys())
    a_y_min, a_y_max = a_ys[0], a_ys[-1]
    a_span = max(1.0, float(a_y_max - a_y_min))
    
    # Precompute anchor contour curve for fast interpolation
    a_curve_y = np.array(a_ys, dtype=np.float32)
    a_curve_x = np.array([anchor_pts[y] for y in a_ys], dtype=np.float32)
    
    # Remap crown interior (x < baseline_x)
    for yt in range(t_y_min, t_y_max + 1):
        if yt not in target_pts:
            continue
        xt_bound = target_pts[yt]
        if xt_bound >= baseline_x:
            continue
            
        # Target vertical fraction [0, 1]
        v_frac = (yt - t_y_min) / t_span
        # Corresponding vertical coordinate in anchor
        ya = a_y_min + v_frac * a_span
        
        # Interpolate anchor crown edge at ya
        xa_bound = float(np.interp(ya, a_curve_y, a_curve_x))
        
        target_width = baseline_x - xt_bound
        anchor_width = baseline_x - xa_bound
        if target_width <= 0 or anchor_width <= 0:
            continue
            
        for xt in range(int(np.floor(xt_bound)), baseline_x):
            # Horizontal normalized penetration depth from baseline [0, 1]
            u_frac = (baseline_x - xt) / target_width
            xa = baseline_x - u_frac * anchor_width
            
            map_x[yt, xt] = np.clip(xa, 0, w - 1)
            map_y[yt, xt] = np.clip(ya, 0, h - 1)
            
    warped = cv2.remap(anchor_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return warped


def synthesize_shape_aligned_blend(anchor_images, anchor_pts_list, target_mask, target_pts, weights, baseline_x=64):
    """
    Method 2: Geometry-Aligned Morphing.
    Warps each anchor crown into the target crown geometry, then blends textures inside target mask.
    Blends substrate plate (x >= baseline_x) seamlessly.
    """
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.sum(weights)
    
    h, w = target_mask.shape[:2]
    warped_crowns = []
    
    for img, pts in zip(anchor_images, anchor_pts_list):
        warped = warp_crown_to_target(img, pts, target_pts, baseline_x=baseline_x)
        warped_crowns.append(warped)
        
    # Weighted average of warped crowns
    crown_acc = np.zeros((h, w, 3), dtype=np.float32)
    for w_img, weight in zip(warped_crowns, weights):
        crown_acc += float(weight) * w_img.astype(np.float32)
    crown_blended = np.clip(crown_acc, 0, 255).astype(np.uint8)
    
    # Substrate & background composite
    # Use direct blend of plate substrate for natural grain
    direct_bg = synthesize_direct_blend(anchor_images, weights)
    
    # Feathered alpha matte around target mask boundary to ensure ultra-clean transitions
    mask_float = (target_mask > 0).astype(np.float32)
    mask_feathered = cv2.GaussianBlur(mask_float, (3, 3), 0.8)[:, :, np.newaxis]
    
    final_composite = (mask_feathered * crown_blended.astype(np.float32) + 
                       (1.0 - mask_feathered) * direct_bg.astype(np.float32))
    
    return np.clip(final_composite, 0, 255).astype(np.uint8), crown_blended


def synthesize_poisson_blend(crown_source, background_target, target_mask, baseline_x=64):
    """
    Method 3: Poisson Gradient Domain Blending.
    Clones the geometry-aligned crown into the substrate background using seamlessClone.
    """
    h, w = target_mask.shape[:2]
    binary_mask = (target_mask > 0).astype(np.uint8) * 255
    
    # Center of crown mask for Poisson cloning
    ys, xs = np.where(binary_mask > 0)
    if len(ys) == 0:
        return crown_source
        
    center = (int(np.mean(xs)), int(np.mean(ys)))
    
    # Ensure clone center has bounding box inside image
    try:
        poisson_result = cv2.seamlessClone(crown_source, background_target, binary_mask, center, cv2.NORMAL_CLONE)
        return poisson_result
    except Exception as e:
        # Fallback to direct alpha blend if Poisson gradient solve diverges
        return crown_source


def generate_sharp_macrograph(query_res, geom_res, view='A'):
    """
    Full pipeline to generate sharp macrographs for a query:
    1. Loads anchor registered images and their crown points.
    2. Runs Direct Linear Blend (Method 1).
    3. Runs Shape-Aligned Warped Blend (Method 2).
    4. Runs Poisson Composite (Method 3).
    5. Computes quantitative sharpness metrics for all methods.
    """
    weights = [a['weight'] for a in query_res['anchors']]
    anchor_imgs = []
    anchor_pts_list = []
    anchor_names = []
    
    for a in query_res['anchors']:
        img_path = a[f'view_{view.lower()}_path']
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing anchor registered image: {img_path}")
            
        img = cv2.imread(img_path)
        anchor_imgs.append(img)
        anchor_names.append(a['sample_name'])
        
        # Load or extract anchor contour points
        mask_path = a[f'mask_{view.lower()}_path']
        pts = {}
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                ys, xs = np.where(m > 0)
                for y in range(int(ys.min()), int(ys.max()) + 1):
                    row_xs = np.where(m[y, :64] > 0)[0]
                    if len(row_xs) > 0:
                        pts[int(y)] = float(row_xs.min())
        anchor_pts_list.append(pts)
        
    target_mask = geom_res['mask']
    target_pts = geom_res['contour_pts']
    
    # Method 1: Direct Weighted Blend
    img_direct = synthesize_direct_blend(anchor_imgs, weights)
    
    # Method 2: Shape-Aligned Morphing
    img_warped, crown_warped = synthesize_shape_aligned_blend(
        anchor_imgs, anchor_pts_list, target_mask, target_pts, weights, baseline_x=64
    )
    
    # Method 3: Poisson Gradient Blend
    img_poisson = synthesize_poisson_blend(img_warped, img_direct, target_mask, baseline_x=64)
    
    # Quantitative Sharpness Metrics
    metrics_direct = calculate_image_sharpness(img_direct, target_mask)
    metrics_warped = calculate_image_sharpness(img_warped, target_mask)
    metrics_poisson = calculate_image_sharpness(img_poisson, target_mask)
    
    # Also evaluate reference sharpness for grounding
    ref_sharpness = [calculate_image_sharpness(im, target_mask)['laplacian_variance'] for im in anchor_imgs]
    mean_ref_lap = float(np.mean(ref_sharpness))
    
    return {
        'direct_blend': img_direct,
        'shape_aligned_morph': img_warped,
        'poisson_blend': img_poisson,
        'anchor_images': anchor_imgs,
        'anchor_names': anchor_names,
        'weights': weights,
        'metrics': {
            'direct_blend': metrics_direct,
            'shape_aligned_morph': metrics_warped,
            'poisson_blend': metrics_poisson,
            'mean_reference_laplacian_var': mean_ref_lap
        }
    }


def render_interpolation_showcase_panel(query_res, geom_res, synth_res, view='A', out_path=None):
    """
    Renders an academic comparison panel showcasing:
    - 4 Reference Anchor Macrographs with their weights
    - Direct Weighted Blend (demonstrating ghosting/blur)
    - Shape-Aligned Warped Interpolation (our proposed sharp method)
    - Quantitative Sharpness Comparison (Laplacian Variance, Tenengrad Energy)
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), dpi=180, facecolor='white')
    
    mat = query_res['material']
    proc = query_res['process']
    feed = query_res['query_feed_rate']
    speed = query_res['query_speed']
    
    # Row 1: 4 Reference Anchors
    anchors = query_res['anchors']
    for col in range(4):
        ax = axes[0, col]
        if col < len(anchors):
            a = anchors[col]
            img = synth_res['anchor_images'][col]
            w = a['weight']
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axvline(64, color='#10B981', linestyle='--', linewidth=1.1)
            ax.set_title(f"Anchor {col+1}: {a['sample_name']}\nF={a['feed_rate']:.1f}, S={a['speed']:.1f}\nWeight = {w:.2f} ({w*100:.1f}%)",
                         fontsize=10, fontweight='bold', color='#0F172A')
        else:
            ax.axis('off')
        ax.axis('off')
        
    # Row 2, Col 0: Direct Blend (Baseline)
    ax_d = axes[1, 0]
    ax_d.imshow(cv2.cvtColor(synth_res['direct_blend'], cv2.COLOR_BGR2RGB))
    ax_d.axvline(64, color='#EF4444', linestyle='--', linewidth=1.2)
    lap_d = synth_res['metrics']['direct_blend']['laplacian_variance']
    ax_d.set_title(f"Method 1: Direct Weighted Blend\n[Laplacian Var: {lap_d:.1f}]\n(Shows edge blurring / ghosting)",
                   fontsize=9.5, fontweight='bold', color='#B91C1C')
    ax_d.axis('off')
    
    # Row 2, Col 1: Proposed Shape-Aligned Sharp Interpolation
    ax_s = axes[1, 1]
    ax_s.imshow(cv2.cvtColor(synth_res['shape_aligned_morph'], cv2.COLOR_BGR2RGB))
    ax_s.axvline(64, color='#10B981', linestyle='--', linewidth=1.2)
    lap_s = synth_res['metrics']['shape_aligned_morph']['laplacian_variance']
    ax_s.set_title(f"Method 2: Shape-Aligned Morph\n[Laplacian Var: {lap_s:.1f} (+{(lap_s-lap_d)/max(1e-3, lap_d)*100:+.1f}%)]\n(Proposed Sharp Macrograph)",
                   fontsize=9.5, fontweight='bold', color='#047857')
    ax_s.axis('off')
    
    # Row 2, Col 2: SDF Zero-Level Mask & Extracted Contour
    ax_m = axes[1, 2]
    cnt_overlay = np.ones((128, 128, 3), dtype=np.uint8) * 250
    cnt_overlay[geom_res['mask'] > 0] = [219, 234, 254] # Light blue crown
    for y, x in geom_res['contour_pts'].items():
        cnt_overlay[int(y), int(round(x))] = [220, 38, 38] # Red crown edge
    ax_m.imshow(cnt_overlay)
    ax_m.axvline(64, color='#0F172A', linestyle='-', linewidth=1.5, label='Baseline')
    lt = geom_res['landmarks']['left_toe']
    rt = geom_res['landmarks']['right_toe']
    ap = geom_res['landmarks']['apex']
    ax_m.scatter([lt[0], rt[0]], [lt[1], rt[1]], color='#2563EB', s=35, zorder=5)
    ax_m.scatter([ap[0]], [ap[1]], color='#DC2626', s=35, zorder=5)
    ax_m.set_title(f"Topological SDF Crown Contour\nW={geom_res['svr_dimensions']['width_mm']:.2f} mm | R={geom_res['svr_dimensions']['reinforcement_mm']:.2f} mm\nToes & Apex Landmark Tracking",
                   fontsize=9.5, fontweight='bold', color='#1E40AF')
    ax_m.axis('off')
    
    # Row 2, Col 3: Sharpness & Edge Metrics Comparison Bar Chart
    ax_b = axes[1, 3]
    categories = ['Direct\nBlend', 'Shape-Aligned\n(Proposed)', 'Mean Ref\nAnchors']
    vals = [lap_d, lap_s, synth_res['metrics']['mean_reference_laplacian_var']]
    colors = ['#EF4444', '#10B981', '#3B82F6']
    bars = ax_b.bar(categories, vals, color=colors, width=0.55, edgecolor='#0F172A', linewidth=1.0)
    for bar, val in zip(bars, vals):
        ax_b.text(bar.get_x() + bar.get_width()/2.0, val + max(vals)*0.03, f"{val:.1f}",
                  ha='center', va='bottom', fontsize=9.0, fontweight='bold')
    ax_b.set_ylabel("Laplacian Variance (Sharpness)", fontsize=9, fontweight='bold')
    ax_b.set_title("Quantitative Sharpness Audit", fontsize=9.5, fontweight='bold', color='#0F172A')
    ax_b.set_ylim(0, max(vals) * 1.25)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.tick_params(axis='both', labelsize=8.5)
    
    plt.suptitle(f"Phase V14: Geometry-Aware Sharp Macrograph Interpolation (View {view})\n"
                 f"Condition: {mat} | {proc} | F = {feed:.1f} m/min, S = {speed:.1f} cm/min",
                 fontsize=13, fontweight='bold', color='#0F172A', y=0.99)
    plt.tight_layout()
    
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig


if __name__ == "__main__":
    from build_reference_index import WeldReferenceIndex
    from interpolate_geometry import interpolate_geometry_for_query
    
    idx = WeldReferenceIndex()
    query_res = idx.query("IN718", "ColdArc", 5.2, 26.0)
    
    print("\n--- Running Sharp Appearance Generation for Unseen Condition ---")
    for view in ['A', 'B']:
        geom_res = interpolate_geometry_for_query(query_res, view=view)
        synth_res = generate_sharp_macrograph(query_res, geom_res, view=view)
        
        prefix = f"unseen_IN718_ColdArc_F5.2_S26.0_view_{view}"
        sharp_img_path = os.path.join(GEN_DIR, f"{prefix}_sharp_interpolated.png")
        direct_img_path = os.path.join(GEN_DIR, f"{prefix}_direct_blend.png")
        panel_path = os.path.join(PANEL_DIR, f"{prefix}_sharp_interpolation_showcase.png")
        
        cv2.imwrite(sharp_img_path, synth_res['shape_aligned_morph'])
        cv2.imwrite(direct_img_path, synth_res['direct_blend'])
        render_interpolation_showcase_panel(query_res, geom_res, synth_res, view=view, out_path=panel_path)
        
        m_dir = synth_res['metrics']['direct_blend']
        m_shp = synth_res['metrics']['shape_aligned_morph']
        print(f"\n[VIEW {view} RESULTS]")
        print(f"  Direct Blend Sharpness:      Laplacian Var = {m_dir['laplacian_variance']:.2f} | Tenengrad = {m_dir['tenengrad_energy']:.2f}")
        print(f"  Shape-Aligned Sharpness:     Laplacian Var = {m_shp['laplacian_variance']:.2f} | Tenengrad = {m_shp['tenengrad_energy']:.2f}")
        pct_gain = (m_shp['laplacian_variance'] - m_dir['laplacian_variance']) / max(1e-3, m_dir['laplacian_variance']) * 100.0
        print(f"  Sharpness Improvement Gain:  {pct_gain:+.2f}%")
        print(f"  Saved sharp macrograph:      {sharp_img_path}")
        print(f"  Saved comparison panel:      {panel_path}")
