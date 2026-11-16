"""
cli_infer.py

Command-Line Interface for AI-Based Weld Bead Geometry Prediction and Cross-Section Generation.
Supports both non-interactive execution via CLI arguments and interactive terminal wizard.
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.inference_engine import WeldPredictor

def main():
    parser = argparse.ArgumentParser(
        description="Predict Weld Bead Profile & Generate Cross-Sectional Macrograph (GMAW & Cold Arc)"
    )
    parser.add_argument("--material", type=str, choices=["IN718", "SS304L"], default=None, help="Base Material")
    parser.add_argument("--process", type=str, choices=["ColdArc", "GMAW"], default=None, help="Welding Process")
    parser.add_argument("--feed_rate", type=float, default=None, help="Wire feed rate in m/min (Domain: 4.5 - 6.0)")
    parser.add_argument("--speed", type=float, default=None, help="Travel speed in cm/min (Domain: 20.0 - 32.0)")
    parser.add_argument("--model", type=str, choices=["direct_cnn", "gc_wpg_v2", "both"], default="both", help="Generator Architecture")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to save generated output images")

    args = parser.parse_args()

    predictor = WeldPredictor()

    # Interactive input if any required parameter is omitted
    material = args.material
    process = args.process
    feed_rate = args.feed_rate
    speed = args.speed
    model_choice = args.model

    if material is None or process is None or feed_rate is None or speed is None:
        print("=" * 75)
        print("  AI-BASED WELD BEAD PREDICTION & CROSS-SECTION GENERATOR (CLI)")
        print("=" * 75)
        if material is None:
            print("\nSelect Base Material:")
            print("  1. IN718 (Inconel 718)")
            print("  2. SS304L (Stainless Steel 304L)")
            m_choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
            material = "SS304L" if m_choice == "2" else "IN718"

        if process is None:
            print("\nSelect Welding Process:")
            print("  1. ColdArc (Low Heat Input)")
            print("  2. GMAW (Conventional Gas Metal Arc)")
            p_choice = input("Enter choice (1 or 2) [default: 1]: ").strip()
            process = "GMAW" if p_choice == "2" else "ColdArc"

        if feed_rate is None:
            f_in = input("\nEnter Wire Feed Rate (m/min) [Domain: 4.5 - 6.0, default: 5.0]: ").strip()
            feed_rate = float(f_in) if f_in else 5.0

        if speed is None:
            s_in = input("\nEnter Travel Speed (cm/min) [Domain: 20.0 - 32.0, default: 25.0]: ").strip()
            speed = float(s_in) if s_in else 25.0

    print("\nRunning inference pipeline...")
    result = predictor.run_full_inference(material, process, feed_rate, speed, model_choice=model_choice)

    if not result["success"]:
        print("\n[ERROR] Input Validation Failed:")
        for err in result["errors"]:
            print(f"  - {err}")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("  SECTION A: SELECTED WELDING CONDITIONS")
    print("=" * 75)
    print(f"  Material:        {result['inputs']['material']}")
    print(f"  Welding Process: {result['inputs']['process']}")
    print(f"  Wire Feed Rate:  {result['inputs']['feed_rate_m_min']:.2f} m/min")
    print(f"  Travel Speed:    {result['inputs']['speed_cm_min']:.2f} cm/min")
    if result["is_out_of_domain"]:
        print("\n  [WARNING] Out-of-Domain Parameters Detected:")
        for w in result["warnings"]:
            print(f"    * {w}")

    print("\n" + "=" * 75)
    print("  SECTION B: PROSPECTIVE NUMERICAL PREDICTIONS")
    print("=" * 75)
    num = result["numerical_predictions"]
    if num["available"]:
        print(f"  Model:           {num['model_name']}")
        print(f"  Predicted Width:         {num['width_mm']:.2f} mm  (Documented CV MAE: {num['documented_cv_mae_mm']['width_mae']:.3f} mm)")
        print(f"  Predicted Depth:         {num['depth_mm']:.2f} mm  (Documented CV MAE: {num['documented_cv_mae_mm']['depth_mae']:.3f} mm)")
        print(f"  Predicted Reinforcement: {num['reinforcement_mm']:.2f} mm  (Documented CV MAE: {num['documented_cv_mae_mm']['reinf_mae']:.3f} mm)")
        print(f"  Inference Time:          {num['inference_time_ms']} ms")
    else:
        print(f"  {num['message']}")

    print("\n" + "=" * 75)
    print("  SECTION C & D: GENERATED CROSS-SECTIONS & CONTOUR EXTRACTION")
    print("=" * 75)
    for m_key, g in result["generative_results"].items():
        print(f"\n  [Model: {g['model_name']}]")
        print(f"    Checkpoint:     {os.path.basename(g['checkpoint_used'])}")
        print(f"    SHA256 Hash:    {g['checkpoint_sha256'][:16]}...")
        print(f"    Inference Time: {g['inference_time_ms']} ms")
        cdata = g["contour_data"]
        if cdata["is_failed"]:
            print("    Extraction Status: FAILED (Insufficient foreground contrast in weld crown)")
        else:
            print(f"    Apex Coordinate:        X = {cdata['apex_x_px']} px, Y = {cdata['apex_y_px']} px")
            print(f"    Reinforcement Proj:     {cdata['reinf_projection_px']} pixels (Reference Baseline: X = 64)")
            print(f"    Active Bead Area:       {cdata['active_area_px_count']} px^2 (Foreground count)")
            print(f"    Contour Continuity:     {cdata['extracted_rows']}/91 rows detected")
            print(f"    Adaptive Threshold:     T = {cdata['adaptive_threshold']:.1f}")

    print("\n" + "=" * 75)
    print("  SECTION E & F: PROVENANCE AND PHYSICAL LIMITATIONS")
    print("=" * 75)
    print(f"  Resolution:    {result['provenance']['image_resolution']}")
    print(f"  Calibration:   {result['provenance']['calibration_note']}")
    print(f"  Disclaimer:    {result['provenance']['disclaimer']}")

    # Save outputs if requested
    out_dir = args.output_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"{material}_{process}_F{feed_rate:.1f}_S{speed:.1f}"
        for m_key, g in result["generative_results"].items():
            raw_path = os.path.join(out_dir, f"{prefix}_{m_key}_raw.png")
            ovr_path = os.path.join(out_dir, f"{prefix}_{m_key}_overlay.png")
            msk_path = os.path.join(out_dir, f"{prefix}_{m_key}_contour_only.png")
            cv2.imwrite(raw_path, cv2.cvtColor(g["raw_rgb_uint8"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(ovr_path, cv2.cvtColor(g["contour_data"]["overlay_rgb"], cv2.COLOR_RGB2BGR))
            cv2.imwrite(msk_path, cv2.cvtColor(g["contour_data"]["contour_only_rgb"], cv2.COLOR_RGB2BGR))
            print(f"\n  Saved images for {m_key}:")
            print(f"    Raw Image:     {raw_path}")
            print(f"    Contour Ovr:   {ovr_path}")
            print(f"    Contour Only:  {msk_path}")
            
        json_path = os.path.join(out_dir, f"{prefix}_summary.json")
        save_dict = {
            "inputs": result["inputs"],
            "numerical_predictions": result["numerical_predictions"],
            "generative_summary": {
                m: {
                    "model_name": g["model_name"],
                    "apex_x_px": g["contour_data"]["apex_x_px"],
                    "reinf_projection_px": g["contour_data"]["reinf_projection_px"],
                    "active_area_px_count": g["contour_data"]["active_area_px_count"],
                    "inference_time_ms": g["inference_time_ms"]
                } for m, g in result["generative_results"].items()
            },
            "provenance": result["provenance"]
        }
        with open(json_path, "w") as fp:
            json.dump(save_dict, fp, indent=2)
        print(f"    Metadata JSON: {json_path}")
    print("\nInference completed successfully.")

if __name__ == "__main__":
    main()
