"""
generate_demo_cases.py

Executes the 3 mandatory demonstration cases for Phase V8:
1. Case 1 (Reproducibility): IN718, ColdArc, F=5.0 m/min, S=25.0 cm/min (Sample 4 setpoint)
2. Case 2 (Unseen Interpolation): SS304L, GMAW, F=5.25 m/min, S=24.5 cm/min
3. Case 3 (Out-of-Domain): IN718, GMAW, F=6.5 m/min, S=35.0 cm/min (Extrapolation warning)

Saves:
- Raw images (Direct CNN & GC-WPG-v2)
- Contour overlay images
- Contour-only masks
- Multi-panel summary figures
- demo_cases_summary.json
- EXAMPLE_INPUTS.csv
"""

import os
import sys
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setup path
APP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from inference_engine import WeldPredictor

DEMO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "final_integration", "guide_demo_v8"
)
OUTPUT_DIR = os.path.join(DEMO_DIR, "demo_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEMO_CASES = [
    {
        "case_id": "case_1",
        "title": "Case 1: Known Experiment Setpoint (Reproducibility Validation)",
        "description": "Corresponds to historical Sample 4 setpoint (IN718, ColdArc, F=5.0, S=25.0). Validates that models reproduce recognizable bead deposit geometry matching training distribution.",
        "material": "IN718",
        "process": "ColdArc",
        "feed_rate": 5.0,
        "speed": 25.0,
        "category": "Known Benchmark"
    },
    {
        "case_id": "case_2",
        "title": "Case 2: Unseen In-Domain Parameter Interpolation (Generalization)",
        "description": "Novel parameter setpoint inside experimental domain envelope ([4.5, 6.0] m/min, [20.0, 32.0] cm/min) never observed during training. Tests smooth interpolation of profile geometry.",
        "material": "SS304L",
        "process": "GMAW",
        "feed_rate": 5.25,
        "speed": 24.5,
        "category": "In-Domain Interpolation"
    },
    {
        "case_id": "case_3",
        "title": "Case 3: Out-of-Domain Parameter Validation (Safety & Extrapolation)",
        "description": "Extreme parameter setpoint exceeding experimental boundaries (F=6.5 > 6.0 m/min, S=35.0 > 32.0 cm/min). Verifies automated out-of-domain warning flags and model stability under extrapolation.",
        "material": "IN718",
        "process": "GMAW",
        "feed_rate": 6.5,
        "speed": 35.0,
        "category": "Out-of-Domain Extrapolation"
    }
]

def run_demos():
    print("Initializing WeldPredictor...")
    predictor = WeldPredictor()
    results_summary = []

    for case in DEMO_CASES:
        c_id = case["case_id"]
        mat = case["material"]
        proc = case["process"]
        f = case["feed_rate"]
        s = case["speed"]
        print(f"\nExecuting {case['title']}...")

        full_res = predictor.run_full_inference(mat, proc, f, s, model_choice="both")
        
        num_res = full_res["numerical_predictions"]
        gen_res = full_res["generative_results"]
        
        case_summary = {
            "case_id": c_id,
            "title": case["title"],
            "description": case["description"],
            "category": case["category"],
            "inputs": {
                "material": mat,
                "process": proc,
                "feed_rate_m_min": f,
                "speed_cm_min": s
            },
            "is_out_of_domain": full_res["is_out_of_domain"],
            "warnings": full_res["warnings"],
            "numerical_predictions": num_res,
            "models": {}
        }

        # Process each model
        for m_key in ["direct_cnn", "gc_wpg_v2"]:
            m_data = gen_res[m_key]
            raw_rgb = m_data["raw_rgb_uint8"]
            c_data = m_data["contour_data"]
            overlay_rgb = c_data["overlay_rgb"]
            contour_rgb = c_data["contour_only_rgb"]

            # Save PNGs (Convert RGB to BGR for cv2.imwrite)
            raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
            contour_bgr = cv2.cvtColor(contour_rgb, cv2.COLOR_RGB2BGR)

            raw_path = os.path.join(OUTPUT_DIR, f"{c_id}_{m_key}_raw.png")
            overlay_path = os.path.join(OUTPUT_DIR, f"{c_id}_{m_key}_overlay.png")
            contour_path = os.path.join(OUTPUT_DIR, f"{c_id}_{m_key}_contour.png")

            cv2.imwrite(raw_path, raw_bgr)
            cv2.imwrite(overlay_path, overlay_bgr)
            cv2.imwrite(contour_path, contour_bgr)

            case_summary["models"][m_key] = {
                "model_name": m_data["model_name"],
                "inference_time_ms": m_data["inference_time_ms"],
                "reinf_projection_px": c_data["reinf_projection_px"],
                "active_area_px_count": c_data["active_area_px_count"],
                "apex_x_px": c_data["apex_x_px"],
                "apex_y_px": c_data["apex_y_px"],
                "is_failed": c_data["is_failed"],
                "raw_image": f"{c_id}_{m_key}_raw.png",
                "overlay_image": f"{c_id}_{m_key}_overlay.png",
                "contour_image": f"{c_id}_{m_key}_contour.png"
            }

        results_summary.append(case_summary)

        # Generate a comprehensive 2x3 visualization panel for this demo case
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), dpi=150)
        fig.suptitle(f"{case['title']}\nInputs: {mat} | {proc} | F = {f} m/min | S = {s} cm/min",
                     fontsize=13, fontweight='bold', y=0.98)

        # Direct CNN row
        axes[0, 0].imshow(gen_res["direct_cnn"]["raw_rgb_uint8"])
        axes[0, 0].set_title(f"Direct CNN: Raw Synthetic (128x128)\nTime: {gen_res['direct_cnn']['inference_time_ms']} ms", fontsize=10)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gen_res["direct_cnn"]["contour_data"]["overlay_rgb"])
        c_dir = gen_res["direct_cnn"]["contour_data"]
        axes[0, 1].set_title(f"Direct CNN: Contour Overlay\nProj: {c_dir['reinf_projection_px']} px | Area: {c_dir['active_area_px_count']} px²", fontsize=10)
        axes[0, 1].axis('off')

        axes[0, 2].imshow(gen_res["direct_cnn"]["contour_data"]["contour_only_rgb"])
        axes[0, 2].set_title(f"Direct CNN: Extracted Boundary\nApex: ({c_dir['apex_x_px']}, {c_dir['apex_y_px']})", fontsize=10)
        axes[0, 2].axis('off')

        # GC-WPG-v2 row
        axes[1, 0].imshow(gen_res["gc_wpg_v2"]["raw_rgb_uint8"])
        axes[1, 0].set_title(f"GC-WPG-v2: Raw Synthetic (128x128)\nTime: {gen_res['gc_wpg_v2']['inference_time_ms']} ms", fontsize=10)
        axes[1, 0].axis('off')

        axes[1, 1].imshow(gen_res["gc_wpg_v2"]["contour_data"]["overlay_rgb"])
        c_gc = gen_res["gc_wpg_v2"]["contour_data"]
        axes[1, 1].set_title(f"GC-WPG-v2: Contour Overlay\nProj: {c_gc['reinf_projection_px']} px | Area: {c_gc['active_area_px_count']} px²", fontsize=10)
        axes[1, 1].axis('off')

        axes[1, 2].imshow(gen_res["gc_wpg_v2"]["contour_data"]["contour_only_rgb"])
        axes[1, 2].set_title(f"GC-WPG-v2: Extracted Boundary\nApex: ({c_gc['apex_x_px']}, {c_gc['apex_y_px']})", fontsize=10)
        axes[1, 2].axis('off')

        # Add Numerical Box at the bottom
        num_text = (
            f"Prospective SVR Predictions (5-fold CV MAE = 0.681 mm):  "
            f"Width = {num_res['width_mm']:.2f} mm  |  Depth = {num_res['depth_mm']:.2f} mm  |  Reinforcement = {num_res['reinforcement_mm']:.2f} mm\n"
        )
        if full_res["is_out_of_domain"]:
            num_text += "WARNING: PARAMETERS ARE OUTSIDE THE EXPERIMENTAL DOMAIN (Extrapolation Warning Triggered)\n"
        num_text += "Physical Limitation: Generated images estimate crown surface morphology only. Internal penetration depth is estimated via SVR."

        plt.figtext(0.5, 0.02, num_text, ha='center', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3CD' if full_res["is_out_of_domain"] else '#E8F4F8',
                              edgecolor='#FFA000' if full_res["is_out_of_domain"] else '#3498DB', alpha=0.9))

        panel_path = os.path.join(OUTPUT_DIR, f"{c_id}_comparison_panel.png")
        plt.tight_layout(rect=[0, 0.08, 1, 0.94])
        plt.savefig(panel_path, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison panel: {panel_path}")

    # Save summary JSON
    json_path = os.path.join(DEMO_DIR, "demo_cases_summary.json")
    with open(json_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Saved demo summary: {json_path}")

    # Generate EXAMPLE_INPUTS.csv
    example_rows = []
    # Add demo cases
    for case in DEMO_CASES:
        example_rows.append({
            "Example_ID": case["case_id"],
            "Category": case["category"],
            "Material": case["material"],
            "Process": case["process"],
            "Wire_Feed_Rate_m_min": case["feed_rate"],
            "Travel_Speed_cm_min": case["speed"],
            "Notes": case["description"]
        })
    # Add a few additional reference test points
    example_rows.extend([
        {
            "Example_ID": "ref_1",
            "Category": "In-Domain Reference",
            "Material": "IN718",
            "Process": "ColdArc",
            "Wire_Feed_Rate_m_min": 4.5,
            "Travel_Speed_cm_min": 20.0,
            "Notes": "Domain lower bound for ColdArc"
        },
        {
            "Example_ID": "ref_2",
            "Category": "In-Domain Reference",
            "Material": "SS304L",
            "Process": "GMAW",
            "Wire_Feed_Rate_m_min": 6.0,
            "Travel_Speed_cm_min": 32.0,
            "Notes": "Domain upper bound for GMAW"
        },
        {
            "Example_ID": "ref_3",
            "Category": "In-Domain Interpolation",
            "Material": "IN718",
            "Process": "GMAW",
            "Wire_Feed_Rate_m_min": 5.5,
            "Travel_Speed_cm_min": 28.0,
            "Notes": "Mid-high power condition"
        }
    ])
    df_examples = pd.DataFrame(example_rows)
    csv_path = os.path.join(DEMO_DIR, "EXAMPLE_INPUTS.csv")
    df_examples.to_csv(csv_path, index=False)
    print(f"Saved example inputs table: {csv_path}")

if __name__ == "__main__":
    run_demos()
