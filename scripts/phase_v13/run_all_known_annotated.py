"""
run_all_known_annotated.py

Phase V13: Generate Sketch and Dimension Annotations for All Known Experiments (S01 to S36)
Project: Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning

Features:
- Processes all 36 physical welding experiments from data/clean_dataset_36.csv
- Loads experimental Views A and B
- Generates raw cross-section RGB using top-performing M1 Direct Conditional CNN
- Predicts physical dimensions using prospective numerical regressor
- Extracts refined toe-bounded crown contours and binary masks
- Saves all 6 required artifacts per specimen:
  1. raw_generated_rgb.png
  2. predicted_mask.png
  3. predicted_contour.png
  4. weld_bead_sketch.png
  5. annotated_weld_bead_sketch.png
  6. final_annotated_panel.png
- Compiles master metrics CSV:
  results/phase_v13_dimension_annotated/known_conditions/ALL_KNOWN_ANNOTATED_METRICS.csv
- Builds interactive presentation gallery:
  results/phase_v13_dimension_annotated/known_conditions/GALLERY_KNOWN_ANNOTATED.html
"""

import os
import sys
import time
import json
import argparse
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

OUT_BASE_DIR = os.path.join(BASE_DIR, "results", "phase_v13_dimension_annotated", "known_conditions")
os.makedirs(OUT_BASE_DIR, exist_ok=True)


def run_all_known(sample_filter=None):
    t_start = time.time()
    print("=" * 80)
    print("PHASE V13: CLEAN WELD BEAD SKETCH & DIMENSION-ANNOTATION STAGE")
    print("Target: Known Experimental Conditions (S01 to S36)")
    print(f"Destination: {OUT_BASE_DIR}")
    print("=" * 80)

    # Initialize models
    print("\nInitializing prospective dimension regressor and generative renderer...")
    dim_predictor = ProspectiveDimensionPredictor()
    img_generator = GenerativeImageInference(device="cpu")

    df_clean = pd.read_csv(CLEAN_CSV_PATH)
    if sample_filter:
        df_clean = df_clean[df_clean['sample'].isin(sample_filter)].copy().reset_index(drop=True)
        print(f"Filtered execution for samples: {sample_filter}")
    else:
        print(f"Processing all {len(df_clean)} experimental conditions...")

    all_records = []

    for idx, row in df_clean.iterrows():
        s_id = int(row['sample'])
        s_code = f"S{s_id:02d}"
        s_out_dir = os.path.join(OUT_BASE_DIR, s_code)

        mat = str(row['material'])
        proc = str(row['process'])
        feed = float(row['feed_rate'])
        speed = float(row['speed'])

        exp_dims = {
            "width": float(row['width']),
            "depth": float(row['depth']),
            "reinforcement": float(row['reinforcement'])
        }

        # Predict dimensions with prospective regressor
        pred_dims = dim_predictor.predict(mat, proc, feed, speed)

        # Generate raw cross-section RGB
        gen_rgb = img_generator.generate(mat, proc, feed, speed)

        # Load Real Views A and B
        p_ra = os.path.join(PROCESSED_DIR, f"{s_id}-A.png")
        if not os.path.exists(p_ra):
            p_ra = os.path.join(GEN36_FALLBACK_DIR, s_code, "Real_A.png")
        real_a_rgb = cv2.cvtColor(cv2.imread(p_ra), cv2.COLOR_BGR2RGB) if os.path.exists(p_ra) else None

        p_rb = os.path.join(PROCESSED_DIR, f"{s_id}-B.png")
        if not os.path.exists(p_rb):
            p_rb = os.path.join(GEN36_FALLBACK_DIR, s_code, "Real_B.png")
        real_b_rgb = cv2.cvtColor(cv2.imread(p_rb), cv2.COLOR_BGR2RGB) if os.path.exists(p_rb) else None

        row_info = {
            "sample_id": s_id,
            "sample_code": s_code,
            "material": mat,
            "process": proc,
            "feed_rate": feed,
            "speed": speed
        }

        # Process and save all 6 artifacts
        meta = process_and_save_sample(
            sample_id_str=s_code,
            row_info=row_info,
            pred_dims=pred_dims,
            exp_dims=exp_dims,
            gen_rgb=gen_rgb,
            out_dir=s_out_dir,
            real_a_rgb=real_a_rgb,
            real_b_rgb=real_b_rgb
        )

        w_err = abs(exp_dims["width"] - pred_dims["width"])
        d_err = abs(exp_dims["depth"] - pred_dims["depth"])
        r_err = abs(exp_dims["reinforcement"] - pred_dims["reinforcement"])

        rec = {
            "sample": s_id,
            "sample_code": s_code,
            "material": mat,
            "process": proc,
            "feed_rate": feed,
            "speed": speed,
            "exp_width_mm": exp_dims["width"],
            "pred_width_mm": pred_dims["width"],
            "width_err_mm": round(w_err, 2),
            "width_pct_err": round(w_err / (exp_dims["width"] + 1e-5) * 100, 1),
            "exp_depth_mm": exp_dims["depth"],
            "pred_depth_mm": pred_dims["depth"],
            "depth_err_mm": round(d_err, 2),
            "depth_pct_err": round(d_err / (exp_dims["depth"] + 1e-5) * 100, 1),
            "exp_reinf_mm": exp_dims["reinforcement"],
            "pred_reinf_mm": pred_dims["reinforcement"],
            "reinf_err_mm": round(r_err, 2),
            "reinf_pct_err": round(r_err / (exp_dims["reinforcement"] + 1e-5) * 100, 1),
            "apex_x_px": meta["geometry_extraction"]["apex_x"],
            "apex_y_px": meta["geometry_extraction"]["apex_y"],
            "reinf_proj_px": meta["geometry_extraction"]["reinf_px"],
            "artifacts_dir": os.path.relpath(s_out_dir, BASE_DIR)
        }
        all_records.append(rec)

        print(f"  [{idx+1:02d}/{len(df_clean):02d}] {s_code} ({mat}, {proc}, F={feed}, S={speed}) | "
              f"W: {exp_dims['width']:.2f} vs {pred_dims['width']:.2f} mm | "
              f"D: {exp_dims['depth']:.2f} vs {pred_dims['depth']:.2f} mm | "
              f"R: {exp_dims['reinforcement']:.2f} vs {pred_dims['reinforcement']:.2f} mm")

    df_out = pd.DataFrame(all_records)
    csv_path = os.path.join(OUT_BASE_DIR, "ALL_KNOWN_ANNOTATED_METRICS.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved metrics summary table to: {csv_path}")

    # Build Interactive HTML Presentation Gallery
    print("Building presentation gallery HTML...")
    gallery_path = os.path.join(OUT_BASE_DIR, "GALLERY_KNOWN_ANNOTATED.html")

    panel_cards = ""
    for r in all_records:
        s_code = r["sample_code"]
        img_rel = f"{s_code}/final_annotated_panel.png"
        sketch_rel = f"{s_code}/annotated_weld_bead_sketch.png"
        panel_cards += f"""
        <div class="card" id="{s_code}">
            <div class="card-header">
                <h3>{s_code} &mdash; {r['material']} | {r['process']} (Feed: {r['feed_rate']} m/min, Speed: {r['speed']} cm/min)</h3>
                <span class="badge">W: Exp {r['exp_width_mm']} | Pred {r['pred_width_mm']} mm</span>
                <span class="badge">D: Exp {r['exp_depth_mm']} | Pred {r['pred_depth_mm']} mm</span>
                <span class="badge">R: Exp {r['exp_reinf_mm']} | Pred {r['pred_reinf_mm']} mm</span>
            </div>
            <div class="card-body">
                <img src="{img_rel}" alt="{s_code} Final Annotated Panel" loading="lazy">
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Weld Bead Profile Sketches & Dimension Annotations (Known Conditions)</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; background: #0F172A; color: #F8FAFC; margin: 0; padding: 24px; }}
        header {{ text-align: center; padding-bottom: 20px; border-bottom: 1px solid #334155; margin-bottom: 25px; }}
        h1 {{ color: #38BDF8; margin-bottom: 6px; }}
        p.sub {{ color: #94A3B8; font-size: 14px; margin: 0; }}
        .nav-chips {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 8px; margin: 20px 0 30px 0; }}
        .chip {{ background: #1E293B; color: #38BDF8; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12.5px; font-weight: bold; border: 1px solid #334155; }}
        .chip:hover {{ background: #0284C7; color: white; }}
        .gallery {{ display: flex; flex-direction: column; gap: 30px; }}
        .card {{ background: #1E293B; border-radius: 12px; padding: 18px; border: 1px solid #334155; }}
        .card-header {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin-bottom: 12px; border-bottom: 1px solid #334155; padding-bottom: 10px; }}
        .card-header h3 {{ color: #F1F5F9; font-size: 16px; margin: 0; }}
        .badge {{ background: #0F172A; color: #38BDF8; font-size: 12px; padding: 4px 10px; border-radius: 4px; border: 1px solid #334155; font-family: monospace; }}
        .card-body img {{ width: 100%; border-radius: 8px; background: white; }}
    </style>
</head>
<body>
    <header>
        <h1>Engineering Weld Bead Sketches & Dimension Annotations</h1>
        <p class="sub">Known Experimental Specimen Evaluation (S01 to S36) &mdash; Width (W), Depth (D), Reinforcement (R)</p>
    </header>

    <div class="nav-chips">
        {' '.join([f'<a class="chip" href="#{r["sample_code"]}">{r["sample_code"]}</a>' for r in all_records])}
    </div>

    <section class="gallery">
        {panel_cards}
    </section>
</body>
</html>
"""
    with open(gallery_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive gallery to: {gallery_path}")

    # Summary
    print("\n" + "=" * 80)
    print("PHASE V13 KNOWN-CONDITION SKETCH & ANNOTATION COMPLETE")
    print("=" * 80)
    print(f"Total Specimen Panels Created: {len(all_records)}")
    print(f"Mean Width Error:              {df_out['width_err_mm'].mean():.2f} mm ({df_out['width_pct_err'].mean():.1f}%)")
    print(f"Mean Depth Error:              {df_out['depth_err_mm'].mean():.2f} mm ({df_out['depth_pct_err'].mean():.1f}%)")
    print(f"Mean Reinforcement Error:      {df_out['reinf_err_mm'].mean():.2f} mm ({df_out['reinf_pct_err'].mean():.1f}%)")
    print(f"Runtime:                       {time.time() - t_start:.1f} s")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase V13 Known Condition Sketch Annotator")
    parser.add_argument("--samples", type=int, nargs="*", default=None, help="Specific sample IDs to process (e.g. --samples 2 4 33)")
    args = parser.parse_args()

    sys.exit(run_all_known(sample_filter=args.samples))
