"""
generate_single_condition.py

Phase V14: Stage 4 - Single-Condition Generator & Comparative Panel Pipeline.
Generates sharp macrographs, topological geometry, calibrated SVR dimensions,
and comparison against M1 Direct CNN for any known or unseen welding condition.
Produces 9 standardized research artifacts per view.
"""

import os
import sys
import argparse
import json
import pickle
import torch
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local module imports
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
sys.path.append(BASE_DIR)
sys.path.append(CURRENT_DIR)

from build_reference_index import WeldReferenceIndex
from interpolate_geometry import (
    interpolate_geometry_for_query,
    render_weld_bead_sketch,
    predict_svr_dimensions
)
from interpolate_rgb import (
    generate_sharp_macrograph,
    calculate_image_sharpness
)
from scripts.phase_v9.geometry_models import DirectCNN

OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation")
M1_CKPT = os.path.join(BASE_DIR, "results", "phase_v12_finalized_project_stage", "checkpoints_used", "m1_direct_cnn_final.pt")
NORM_PARAMS = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "checkpoints", "normalization_params.json")


def load_m1_model():
    """Loads trained M1 Direct CNN for comparative evaluation."""
    if not os.path.exists(M1_CKPT) or not os.path.exists(NORM_PARAMS):
        return None, None
        
    model = DirectCNN(cond_dim=6, noise_dim=16)
    ckpt = torch.load(M1_CKPT, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    model.eval()
    
    with open(NORM_PARAMS, 'r') as f:
        norm_params = json.load(f)
        
    return model, norm_params


def run_m1_inference(model, norm_params, material, process, feed_rate, speed):
    """Generates 128x128 synthetic cross-section via M1 Direct CNN."""
    if model is None or norm_params is None:
        return None
        
    f_norm = (feed_rate - norm_params["f_min"]) / (norm_params["f_max"] - norm_params["f_min"] + 1e-8)
    s_norm = (speed - norm_params["s_min"]) / (norm_params["s_max"] - norm_params["s_min"] + 1e-8)
    
    c = torch.tensor([[
        f_norm, s_norm,
        1.0 if material == "IN718" else 0.0,
        1.0 if material == "SS304L" else 0.0,
        1.0 if process == "ColdArc" else 0.0,
        1.0 if process == "GMAW" else 0.0
    ]], dtype=torch.float32)
    
    with torch.no_grad():
        out = model(c)
        
    img_rgb = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


def render_master_comparison_panel(query_res, geom_res, synth_res, m1_bgr, view='A', out_path=None):
    """
    Renders the definitive 8-cell supervisor presentation panel:
    Top Row: 4 Reference Anchor Macrographs with weights & parameters.
    Bottom Row:
      1. M1 Direct CNN (Deep Learning baseline - blurry)
      2. Proposed Phase V14 Sharp Macrograph (Shape-aligned morphing)
      3. Topological SDF Crown Contour & Boundary overlay
      4. Calibrated Annotated Weld Bead Sketch with (W, D, R)
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.2), dpi=180, facecolor='white')
    
    mat = query_res['material']
    proc = query_res['process']
    feed = query_res['query_feed_rate']
    speed = query_res['query_speed']
    is_interp = query_res['is_interpolation']
    status_str = "Unseen Parameter Interpolation" if is_interp else "Out-of-Domain Extrapolation"
    
    # -------------------------------------------------------------
    # Row 1: Reference Anchors
    # -------------------------------------------------------------
    anchors = query_res['anchors']
    for col in range(4):
        ax = axes[0, col]
        if col < len(anchors):
            a = anchors[col]
            img = synth_res['anchor_images'][col]
            w = a['weight']
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axvline(64, color='#10B981', linestyle='--', linewidth=1.2)
            ax.set_title(f"Anchor {col+1}: {a['sample_name']} (Weight: {w:.2f})\n"
                         f"F={a['feed_rate']:.1f} m/min, S={a['speed']:.1f} cm/min\n"
                         f"W={a['width_mm']:.2f}, D={a['depth_mm']:.2f}, R={a['reinforcement_mm']:.2f} mm",
                         fontsize=9.5, fontweight='bold', color='#0F172A')
        else:
            ax.axis('off')
        ax.axis('off')
        
    # -------------------------------------------------------------
    # Row 2: Comparisons & Synthesized Outputs
    # -------------------------------------------------------------
    # (e) M1 Direct CNN
    ax_m1 = axes[1, 0]
    if m1_bgr is not None:
        ax_m1.imshow(cv2.cvtColor(m1_bgr, cv2.COLOR_BGR2RGB))
        ax_m1.axvline(64, color='#EAB308', linestyle='--', linewidth=1.2)
        m1_lap = calculate_image_sharpness(m1_bgr)['laplacian_variance']
        ax_m1.set_title(f"(e) M1 Direct CNN (Baseline)\n[Laplacian Var: {m1_lap:.1f}]\nDirect Deep Learning (Blurry Texture)",
                        fontsize=10, fontweight='bold', color='#B91C1C')
    else:
        ax_m1.text(0.5, 0.5, "M1 CNN Not Available", ha='center', va='center')
    ax_m1.axis('off')
    
    # (f) Proposed Phase V14 Sharp Macrograph
    ax_sharp = axes[1, 1]
    sharp_bgr = synth_res['shape_aligned_morph']
    ax_sharp.imshow(cv2.cvtColor(sharp_bgr, cv2.COLOR_BGR2RGB))
    ax_sharp.axvline(64, color='#10B981', linestyle='--', linewidth=1.2)
    sharp_lap = synth_res['metrics']['shape_aligned_morph']['laplacian_variance']
    tenengrad = synth_res['metrics']['shape_aligned_morph']['tenengrad_energy']
    ax_sharp.set_title(f"(f) Phase V14 Sharp Macrograph\n[Laplacian Var: {sharp_lap:.1f} | Ten: {tenengrad:.0f}]\nGeometry-Aware Sharp Interpolation",
                       fontsize=10, fontweight='bold', color='#047857')
    ax_sharp.axis('off')
    
    # (g) Topological SDF Crown Contour & Mask
    ax_mask = axes[1, 2]
    vis_mask = np.ones((128, 128, 3), dtype=np.uint8) * 252
    vis_mask[geom_res['mask'] > 0] = [219, 234, 254]
    for y, x in geom_res['contour_pts'].items():
        vis_mask[int(y), int(round(x))] = [220, 38, 38]
    ax_mask.imshow(vis_mask)
    ax_mask.axvline(64, color='#0F172A', linestyle='-', linewidth=1.5)
    lt = geom_res['landmarks']['left_toe']
    rt = geom_res['landmarks']['right_toe']
    ap = geom_res['landmarks']['apex']
    ax_mask.scatter([lt[0], rt[0]], [lt[1], rt[1]], color='#2563EB', s=45, zorder=5)
    ax_mask.scatter([ap[0]], [ap[1]], color='#DC2626', s=45, zorder=5)
    ax_mask.set_title(f"(g) Topological SDF Crown Contour\nApex: ({ap[0]:.1f}, {ap[1]:.1f}) | Y-Span: {geom_res['landmarks']['y_span_px']:.1f} px\nContinuous Zero-Level Level Set",
                      fontsize=10, fontweight='bold', color='#1E40AF')
    ax_mask.axis('off')
    
    # (h) Annotated Weld Bead Sketch
    ax_sketch = axes[1, 3]
    w_mm = geom_res['svr_dimensions']['width_mm']
    d_mm = geom_res['svr_dimensions']['depth_mm']
    r_mm = geom_res['svr_dimensions']['reinforcement_mm']
    
    # Render sketch into ax_sketch directly
    sketch_fig, _ = render_weld_bead_sketch(geom_res['contour_pts'], w_mm, d_mm, r_mm, is_annotated=True,
                                            title=f"W={w_mm:.2f} mm, D={d_mm:.2f} mm, R={r_mm:.2f} mm",
                                            out_path=None)
    # Convert sketch_fig canvas to array
    sketch_fig.canvas.draw()
    sketch_rgba = np.asarray(sketch_fig.canvas.buffer_rgba())
    ax_sketch.imshow(sketch_rgba)
    ax_sketch.set_title(f"(h) Calibrated Weld Dimensions\nW={w_mm:.2f} mm | D={d_mm:.2f} mm | R={r_mm:.2f} mm\nProspective SVR Predictions",
                        fontsize=10, fontweight='bold', color='#0F172A')
    ax_sketch.axis('off')
    plt.close(sketch_fig)
    
    plt.suptitle(f"Phase V14: Geometry-Aware Sharp Macrograph Generation (View {view})\n"
                 f"Target: {mat} | {proc} | Feed Rate: {feed:.1f} m/min | Speed: {speed:.1f} cm/min [{status_str}]",
                 fontsize=13.5, fontweight='bold', color='#0F172A', y=0.99)
    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        return fig


def generate_single_condition(material, process, feed_rate, speed, out_dir=None, views=('A', 'B')):
    """
    Full end-to-end generator for a single condition:
    Produces all 9 research artifacts per requested view.
    """
    if out_dir is None:
        out_dir = os.path.join(OUT_BASE, f"generation_{material}_{process}_F{feed_rate:.1f}_S{speed:.1f}")
    os.makedirs(out_dir, exist_ok=True)
    
    ref_idx = WeldReferenceIndex()
    query_res = ref_idx.query(material, process, feed_rate, speed)
    
    m1_model, norm_params = load_m1_model()
    m1_bgr = run_m1_inference(m1_model, norm_params, material, process, feed_rate, speed)
    
    results = {}
    
    for view in views:
        view_dir = os.path.join(out_dir, f"view_{view}")
        os.makedirs(view_dir, exist_ok=True)
        
        # 1. Geometry & Topological Contour
        geom_res = interpolate_geometry_for_query(query_res, view=view)
        
        # 2. Sharp Appearance
        synth_res = generate_sharp_macrograph(query_res, geom_res, view=view)
        
        # Save individual standardized artifacts
        # 1. raw_generated_rgb.png (Sharp macrograph)
        sharp_path = os.path.join(view_dir, "raw_generated_rgb.png")
        cv2.imwrite(sharp_path, synth_res['shape_aligned_morph'])
        
        # 2. direct_blend_rgb.png
        direct_path = os.path.join(view_dir, "direct_blend_rgb.png")
        cv2.imwrite(direct_path, synth_res['direct_blend'])
        
        # 3. m1_direct_cnn_rgb.png
        if m1_bgr is not None:
            m1_path = os.path.join(view_dir, "m1_direct_cnn_rgb.png")
            cv2.imwrite(m1_path, m1_bgr)
            
        # 4. predicted_mask.png
        mask_path = os.path.join(view_dir, "predicted_mask.png")
        cv2.imwrite(mask_path, geom_res['mask'])
        
        # 5. predicted_contour.png & JSON
        cnt_path = os.path.join(view_dir, "predicted_contour.png")
        cv2.imwrite(cnt_path, geom_res['contour_img'])
        json_path = os.path.join(view_dir, "predicted_contour.json")
        with open(json_path, 'w') as f:
            json.dump({
                'condition': {'material': material, 'process': process, 'feed_rate': feed_rate, 'speed': speed, 'view': view},
                'landmarks': geom_res['landmarks'],
                'svr_dimensions': geom_res['svr_dimensions'],
                'contour_points': geom_res['contour_pts']
            }, f, indent=2)
            
        # 6. weld_bead_sketch.png (unannotated)
        w_mm = geom_res['svr_dimensions']['width_mm']
        d_mm = geom_res['svr_dimensions']['depth_mm']
        r_mm = geom_res['svr_dimensions']['reinforcement_mm']
        sketch_unann_path = os.path.join(view_dir, "weld_bead_sketch.png")
        render_weld_bead_sketch(geom_res['contour_pts'], w_mm, d_mm, r_mm, is_annotated=False,
                                title=f"Weld Profile Sketch (View {view})\n{material} | {process} | F={feed_rate:.1f}, S={speed:.1f}",
                                out_path=sketch_unann_path)
                                
        # 7. annotated_weld_bead_sketch.png
        sketch_ann_path = os.path.join(view_dir, "annotated_weld_bead_sketch.png")
        render_weld_bead_sketch(geom_res['contour_pts'], w_mm, d_mm, r_mm, is_annotated=True,
                                title=f"Annotated Weld Profile Sketch (View {view})\n{material} | {process} | F={feed_rate:.1f}, S={speed:.1f}",
                                out_path=sketch_ann_path)
                                
        # 8. final_comparison_panel.png
        panel_path = os.path.join(view_dir, "final_comparison_panel.png")
        render_master_comparison_panel(query_res, geom_res, synth_res, m1_bgr, view=view, out_path=panel_path)
        
        # 9. metadata.json
        meta_path = os.path.join(view_dir, "metadata.json")
        meta = {
            'target_condition': {
                'material': material,
                'process': process,
                'feed_rate': feed_rate,
                'speed': speed,
                'view': view,
                'status': 'interpolation' if query_res['is_interpolation'] else 'extrapolation',
                'method': query_res['method']
            },
            'anchors': [
                {
                    'sample_name': a['sample_name'],
                    'feed_rate': a['feed_rate'],
                    'speed': a['speed'],
                    'weight': a['weight'],
                    'width_mm': a['width_mm'],
                    'depth_mm': a['depth_mm'],
                    'reinforcement_mm': a['reinforcement_mm']
                }
                for a in query_res['anchors']
            ],
            'svr_dimensions_mm': geom_res['svr_dimensions'],
            'landmarks_px': geom_res['landmarks'],
            'sharpness_metrics': {
                'shape_aligned_sharp': synth_res['metrics']['shape_aligned_morph'],
                'direct_blend': synth_res['metrics']['direct_blend'],
                'm1_direct_cnn': calculate_image_sharpness(m1_bgr) if m1_bgr is not None else None,
                'mean_reference': synth_res['metrics']['mean_reference_laplacian_var']
            },
            'saved_artifacts': {
                'raw_generated_rgb': sharp_path,
                'direct_blend_rgb': direct_path,
                'm1_direct_cnn_rgb': m1_path if m1_bgr is not None else None,
                'predicted_mask': mask_path,
                'predicted_contour_png': cnt_path,
                'predicted_contour_json': json_path,
                'weld_bead_sketch': sketch_unann_path,
                'annotated_weld_bead_sketch': sketch_ann_path,
                'final_comparison_panel': panel_path
            }
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
            
        # Verify metadata.json contains exact requested parameters
        with open(meta_path, 'r') as f_v:
            v_meta = json.load(f_v)
        t_cond = v_meta.get('target_condition', {})
        if (
            t_cond.get('material') != material or
            t_cond.get('process') != process or
            abs(t_cond.get('feed_rate', 0.0) - feed_rate) > 1e-4 or
            abs(t_cond.get('speed', 0.0) - speed) > 1e-4 or
            t_cond.get('view') != view
        ):
            raise ValueError(f"Metadata verification failed for {meta_path}: parameters mismatch!")
        print(f"[VERIFIED] metadata.json verified for View {view}: Material={material}, Process={process}, Feed={feed_rate}, Speed={speed}")

        # Print all generated artifact paths for this view
        print(f"\n[OUTPUT ARTIFACTS] View {view}:")
        for art_name, art_path in meta['saved_artifacts'].items():
            if art_path:
                print(f"  ├── {art_name:28s}: {art_path}")
        print(f"  └── {'metadata_json':28s}: {meta_path}")

        results[view] = meta
        print(f"[{material} {process} F={feed_rate} S={speed} VIEW {view}] Successfully generated 9 artifacts in {view_dir}")
        
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase V14: Single Condition Sharp Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--material",
        type=str,
        default="IN718",
        help="Base alloy material (e.g. IN718, SS304L)"
    )
    parser.add_argument(
        "--process",
        type=str,
        default="ColdArc",
        help="Welding process (e.g. ColdArc, GMAW)"
    )
    parser.add_argument(
        "--feed", "--feed_rate",
        dest="feed",
        type=float,
        default=5.2,
        help="Wire feed rate in m/min"
    )
    parser.add_argument(
        "--speed", "--travel_speed",
        dest="speed",
        type=float,
        default=26.0,
        help="Welding travel speed in cm/min"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="both",
        choices=["A", "B", "both"],
        help="Cross-sectional view to generate ('A', 'B', or 'both')"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Custom output directory (default: results/phase_v14_sharp_interpolation/generation_<mat>_<proc>_F<feed>_S<speed>)"
    )
    args = parser.parse_args()
    
    material = args.material.strip()
    process = args.process.strip()
    feed = float(args.feed)
    speed = float(args.speed)

    # Validate supported parameter combinations
    SUPPORTED_DOMAINS = [
        ("IN718", "ColdArc"),
        ("IN718", "GMAW"),
        ("SS304L", "ColdArc"),
        ("SS304L", "GMAW")
    ]

    if (material, process) not in SUPPORTED_DOMAINS:
        supported_str = ", ".join([f"('{m}', '{p}')" for m, p in SUPPORTED_DOMAINS])
        print(f"\n[ERROR] Unsupported parameter combination: material='{material}', process='{process}'.", file=sys.stderr)
        print(f"Supported material and process combinations are:\n  {supported_str}\n", file=sys.stderr)
        sys.exit(1)

    if feed <= 0 or speed <= 0:
        print(f"\n[ERROR] Non-physical parameter values: feed={feed} m/min and speed={speed} cm/min must be strictly positive.\n", file=sys.stderr)
        sys.exit(1)

    views = ['A', 'B'] if args.view == 'both' else [args.view]
    results = generate_single_condition(material, process, feed, speed, out_dir=args.out_dir, views=views)

    print("\n" + "=" * 75)
    print("SINGLE CONDITION GENERATION COMPLETED SUCCESSFULLY")
    print(f"Material: {material} | Process: {process} | Feed Rate: {feed} m/min | Speed: {speed} cm/min")
    print("=" * 75)


if __name__ == "__main__":
    main()
