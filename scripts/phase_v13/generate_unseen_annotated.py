"""
generate_unseen_annotated.py

Phase V13: Interactive Unseen Parameter Weld Bead Sketch & Dimension Annotator
Project: Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning

Features:
1. Accepts prospective parameter inputs via CLI flags or interactive prompt:
   - Material: IN718 or SS304L
   - Process: ColdArc or GMAW
   - Feed Rate: 4.5 to 7.0 m/min
   - Travel Speed: 20.0 to 32.0 cm/min
2. Automatically audits condition against data/clean_dataset_36.csv:
   - Detects whether condition is KNOWN or UNSEEN
   - Automatically adapts dimension annotation:
     * Known: Width/Depth/Reinforcement with Exp vs Pred values
     * Unseen: Width/Depth/Reinforcement with Pred values only
3. Predicts physical dimensions using prospective numerical regressor
4. Synthesizes weld cross-section RGB using trained DirectCNN
5. Extracts refined toe-bounded crown contours and binary masks
6. Saves all 6 required artifacts:
   1. raw_generated_rgb.png
   2. predicted_mask.png
   3. predicted_contour.png
   4. weld_bead_sketch.png
   5. annotated_weld_bead_sketch.png
   6. final_annotated_panel.png
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v13.weld_sketch_annotator import (
    ProspectiveDimensionPredictor,
    GenerativeImageInference,
    process_and_save_sample,
    PROCESSED_DIR,
    GEN36_FALLBACK_DIR,
    CLEAN_CSV_PATH
)

UNSEEN_BASE_DIR = os.path.join(BASE_DIR, "results", "phase_v13_dimension_annotated", "unseen_conditions")
os.makedirs(UNSEEN_BASE_DIR, exist_ok=True)


def audit_condition(material, process, feed_rate, speed):
    """Audits condition against clean_dataset_36.csv."""
    if not os.path.exists(CLEAN_CSV_PATH):
        return "GENUINE_UNSEEN_INTERPOLATION", None, None

    df = pd.read_csv(CLEAN_CSV_PATH)
    match = df[
        (df["material"].str.strip().str.upper() == str(material).strip().upper()) &
        (df["process"].str.strip().str.upper() == str(process).strip().upper()) &
        (np.isclose(df["feed_rate"].astype(float), float(feed_rate), atol=1e-3)) &
        (np.isclose(df["speed"].astype(float), float(speed), atol=1e-3))
    ]

    if len(match) > 0:
        row = match.iloc[0]
        s_code = f"S{int(row['sample']):02d}"
        exp_dims = {
            "width": float(row["width"]),
            "depth": float(row["depth"]),
            "reinforcement": float(row["reinforcement"])
        }
        return "KNOWN_EXPERIMENTAL_CONDITION", s_code, exp_dims
    else:
        return "GENUINE_UNSEEN_INTERPOLATION", None, None


def run_unseen_generation(material, process, feed_rate, speed, out_dir=None):
    t_start = time.time()
    mat_clean = "IN718" if "718" in str(material).upper() else "SS304L"
    proc_clean = "ColdArc" if "COLD" in str(process).upper() else "GMAW"
    feed_clean = float(feed_rate)
    speed_clean = float(speed)

    cond_type, s_code, exp_dims = audit_condition(mat_clean, proc_clean, feed_clean, speed_clean)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"unseen_{timestamp}_{mat_clean}_{proc_clean}_F{feed_clean:.1f}_S{speed_clean:.1f}"
    if out_dir is None:
        target_dir = os.path.join(UNSEEN_BASE_DIR, folder_name)
    else:
        target_dir = out_dir
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 80)
    print("PHASE V13: PROSPECTIVE UNSEEN WELD BEAD SKETCH & DIMENSION ANNOTATION")
    print(f"Condition:    {mat_clean} | {proc_clean} | Feed: {feed_clean:.1f} m/min | Speed: {speed_clean:.1f} cm/min")
    print(f"Audit Status: {cond_type}" + (f" (Matches {s_code})" if s_code else " (Novel Parameter Setpoint)"))
    print(f"Output Dir:   {target_dir}")
    print("=" * 80)

    # Models
    dim_predictor = ProspectiveDimensionPredictor()
    img_generator = GenerativeImageInference(device="cpu")

    # Predict physical dimensions
    pred_dims = dim_predictor.predict(mat_clean, proc_clean, feed_clean, speed_clean)
    print(f"\nPredicted Dimensions:")
    print(f"  Width (W):         {pred_dims['width']:.2f} mm")
    print(f"  Depth (D):         {pred_dims['depth']:.2f} mm")
    print(f"  Reinforcement (R): {pred_dims['reinforcement']:.2f} mm")

    # Generate cross-section RGB
    gen_rgb = img_generator.generate(mat_clean, proc_clean, feed_clean, speed_clean)

    # Ground truth references if known
    real_a_rgb, real_b_rgb = None, None
    if s_code:
        s_num = int(s_code[1:])
        p_ra = os.path.join(PROCESSED_DIR, f"{s_num}-A.png")
        if not os.path.exists(p_ra):
            p_ra = os.path.join(GEN36_FALLBACK_DIR, s_code, "Real_A.png")
        if os.path.exists(p_ra):
            real_a_rgb = cv2.cvtColor(cv2.imread(p_ra), cv2.COLOR_BGR2RGB)

        p_rb = os.path.join(PROCESSED_DIR, f"{s_num}-B.png")
        if not os.path.exists(p_rb):
            p_rb = os.path.join(GEN36_FALLBACK_DIR, s_code, "Real_B.png")
        if os.path.exists(p_rb):
            real_b_rgb = cv2.cvtColor(cv2.imread(p_rb), cv2.COLOR_BGR2RGB)

    sample_label = s_code if s_code else f"Unseen ({mat_clean}_{proc_clean}_F{feed_clean:.1f}_S{speed_clean:.1f})"
    row_info = {
        "sample_code": sample_label,
        "material": mat_clean,
        "process": proc_clean,
        "feed_rate": feed_clean,
        "speed": speed_clean,
        "classification": cond_type
    }

    # Process and save all 6 artifacts
    meta = process_and_save_sample(
        sample_id_str=sample_label,
        row_info=row_info,
        pred_dims=pred_dims,
        exp_dims=exp_dims,
        gen_rgb=gen_rgb,
        out_dir=target_dir,
        real_a_rgb=real_a_rgb,
        real_b_rgb=real_b_rgb
    )

    print("\n[ARTIFACTS CREATED]")
    for k, v in meta["artifacts"].items():
        print(f"  {k:<30}: {os.path.join(target_dir, v)}")

    t_total = time.time() - t_start
    print(f"\nExecution finished in {t_total:.2f} s")
    print("=" * 80)
    return target_dir, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase V13 Unseen Parameter Sketch & Annotation")
    parser.add_argument("--material", type=str, default="IN718", choices=["IN718", "SS304L"], help="Material")
    parser.add_argument("--process", type=str, default="ColdArc", choices=["ColdArc", "GMAW"], help="Process")
    parser.add_argument("--feed-rate", type=float, default=5.2, help="Wire Feed Rate in m/min (e.g. 5.2)")
    parser.add_argument("--speed", type=float, default=26.0, help="Travel Speed in cm/min (e.g. 26.0)")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()

    run_unseen_generation(
        material=args.material,
        process=args.process,
        feed_rate=args.feed_rate,
        speed=args.speed,
        out_dir=args.out_dir
    )
