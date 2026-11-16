"""
run_dimension_annotation_demo.py

Phase V13: Demonstration of Weld Bead Sketch & Dimension Annotations
Project: Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning

Demonstrates:
1. Known Condition: Specimen S02 (IN718, ColdArc, Feed=5.5 m/min, Speed=20.0 cm/min)
   - Width: Exp 10.76 mm | Pred 10.66 mm
   - Depth: Exp 1.77 mm | Pred 1.67 mm
   - Reinforcement: Exp 4.09 mm | Pred 4.12 mm
   - Outputs: results/phase_v13_dimension_annotated/demo/known_S02/

2. Unseen Condition: Prospective Interpolation (IN718, ColdArc, Feed=5.2 m/min, Speed=26.0 cm/min)
   - Width: Pred 10.15 mm
   - Depth: Pred 1.58 mm
   - Reinforcement: Pred 3.25 mm
   - Outputs: results/phase_v13_dimension_annotated/demo/unseen_interpolation/

For each condition, generates and verifies all 6 mandatory deliverables:
- 1. raw_generated_rgb.png
- 2. predicted_mask.png
- 3. predicted_contour.png
- 4. weld_bead_sketch.png
- 5. annotated_weld_bead_sketch.png
- 6. final_annotated_panel.png
"""

import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from scripts.phase_v13.run_all_known_annotated import run_all_known
from scripts.phase_v13.generate_unseen_annotated import run_unseen_generation

DEMO_BASE_DIR = os.path.join(BASE_DIR, "results", "phase_v13_dimension_annotated", "demo")
DEMO_KNOWN_DIR = os.path.join(DEMO_BASE_DIR, "known_S02")
DEMO_UNSEEN_DIR = os.path.join(DEMO_BASE_DIR, "unseen_interpolation")


def run_demonstration():
    t_start = time.time()
    print("=" * 80)
    print("PHASE V13: WELD BEAD SKETCH & DIMENSION-ANNOTATION DEMONSTRATION")
    print("=" * 80)

    # 1. Known Condition Demo: Specimen S02
    print("\n>>> DEMO 1: KNOWN CONDITION (SPECIMEN S02)")
    print("Parameters: IN718 | ColdArc | Feed = 5.5 m/min | Speed = 20.0 cm/min")
    print(f"Output Directory: {DEMO_KNOWN_DIR}")
    print("-" * 75)

    from scripts.phase_v13.weld_sketch_annotator import (
        ProspectiveDimensionPredictor,
        GenerativeImageInference,
        process_and_save_sample,
        PROCESSED_DIR,
        GEN36_FALLBACK_DIR
    )
    import cv2

    dim_predictor = ProspectiveDimensionPredictor()
    img_generator = GenerativeImageInference(device="cpu")

    # S02 Experimental values
    exp_s02 = {"width": 10.76, "depth": 1.77, "reinforcement": 4.09}
    pred_s02 = dim_predictor.predict("IN718", "ColdArc", 5.5, 20.0)
    gen_rgb_s02 = img_generator.generate("IN718", "ColdArc", 5.5, 20.0)

    # Real views
    p_ra = os.path.join(PROCESSED_DIR, "2-A.png")
    if not os.path.exists(p_ra):
        p_ra = os.path.join(GEN36_FALLBACK_DIR, "S02", "Real_A.png")
    real_a_rgb = cv2.cvtColor(cv2.imread(p_ra), cv2.COLOR_BGR2RGB) if os.path.exists(p_ra) else None

    p_rb = os.path.join(PROCESSED_DIR, "2-B.png")
    if not os.path.exists(p_rb):
        p_rb = os.path.join(GEN36_FALLBACK_DIR, "S02", "Real_B.png")
    real_b_rgb = cv2.cvtColor(cv2.imread(p_rb), cv2.COLOR_BGR2RGB) if os.path.exists(p_rb) else None

    row_s02 = {
        "sample_id": 2,
        "sample_code": "S02",
        "material": "IN718",
        "process": "ColdArc",
        "feed_rate": 5.5,
        "speed": 20.0
    }

    meta_s02 = process_and_save_sample(
        sample_id_str="S02",
        row_info=row_s02,
        pred_dims=pred_s02,
        exp_dims=exp_s02,
        gen_rgb=gen_rgb_s02,
        out_dir=DEMO_KNOWN_DIR,
        real_a_rgb=real_a_rgb,
        real_b_rgb=real_b_rgb
    )
    print("S02 artifacts generated successfully:")
    for k, v in meta_s02["artifacts"].items():
        p = os.path.join(DEMO_KNOWN_DIR, v)
        print(f"  [OK] {v:<30} ({os.path.getsize(p)} bytes)")

    # 2. Unseen Condition Demo: Prospective Interpolation
    print("\n>>> DEMO 2: PROSPECTIVE UNSEEN INTERPOLATION")
    print("Parameters: IN718 | ColdArc | Feed = 5.2 m/min | Speed = 26.0 cm/min")
    print(f"Output Directory: {DEMO_UNSEEN_DIR}")
    print("-" * 75)

    _, meta_unseen = run_unseen_generation(
        material="IN718",
        process="ColdArc",
        feed_rate=5.2,
        speed=26.0,
        out_dir=DEMO_UNSEEN_DIR
    )

    print("\nUnseen artifacts generated successfully:")
    for k, v in meta_unseen["artifacts"].items():
        p = os.path.join(DEMO_UNSEEN_DIR, v)
        print(f"  [OK] {v:<30} ({os.path.getsize(p)} bytes)")

    # Verification checklist
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKLIST (ALL 6 DELIVERABLES PRESENT IN BOTH DEMOS)")
    print("=" * 80)
    for folder_name, folder_path in [("Known (S02)", DEMO_KNOWN_DIR), ("Unseen (F=5.2, S=26)", DEMO_UNSEEN_DIR)]:
        print(f"\nTarget: {folder_name} -> {os.path.relpath(folder_path, BASE_DIR)}")
        expected_files = [
            "raw_generated_rgb.png",
            "predicted_mask.png",
            "predicted_contour.png",
            "weld_bead_sketch.png",
            "annotated_weld_bead_sketch.png",
            "final_annotated_panel.png"
        ]
        all_ok = True
        for fn in expected_files:
            fp = os.path.join(folder_path, fn)
            exists = os.path.exists(fp)
            size = os.path.getsize(fp) if exists else 0
            status = "FOUND" if exists and size > 0 else "MISSING"
            print(f"  [{status}] {fn:<30} ({size} bytes)")
            if not exists or size == 0:
                all_ok = False
        if all_ok:
            print(f"  >>> All 6 required artifacts verified for {folder_name}!")
        else:
            print(f"  !!! Missing artifacts detected in {folder_name}!")

    print("\n" + "=" * 80)
    print(f"Demonstration completed in {time.time() - t_start:.2f} s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(run_demonstration())
