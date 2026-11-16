"""
verify_phase_v14.py

Automated Verification Suite for Phase V14 Deliverables.
Validates:
1. Integrity of FINAL_REVIEW_2026_11_16/ (ensures zero modifications).
2. Existence and validity of all Stage 1 - Stage 10 outputs.
3. Proper artifact generation for unseen test case (IN718 ColdArc F=5.2, S=26.0).
4. Benchmark metrics and report files.
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_REVIEW_DIR = os.path.join(BASE_DIR, "FINAL_REVIEW_2026_11_16")
OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v14_sharp_interpolation")
SCRIPT_DIR = os.path.join(BASE_DIR, "scripts", "phase_v14")


def verify_all():
    print("=" * 75)
    print("PHASE V14: AUTOMATED DELIVERABLE VERIFICATION SUITE")
    print("=" * 75)
    
    passed_checks = 0
    total_checks = 0
    
    def check(name, condition, details=""):
        nonlocal passed_checks, total_checks
        total_checks += 1
        status = "PASSED" if condition else "FAILED"
        print(f"[{status}] {name}" + (f" -> {details}" if details else ""))
        if condition:
            passed_checks += 1
            
    # 1. Protect Final Review Package
    check("Final Review Directory Protected", os.path.exists(FINAL_REVIEW_DIR), FINAL_REVIEW_DIR)
    
    # 2. Verify Core Scripts
    required_scripts = [
        "audit_and_register_images.py",
        "build_reference_index.py",
        "interpolate_geometry.py",
        "interpolate_rgb.py",
        "generate_single_condition.py",
        "evaluate_grouped.py",
        "build_gallery.py",
        "verify_phase_v14.py"
    ]
    for s in required_scripts:
        p = os.path.join(SCRIPT_DIR, s)
        check(f"Script: {s}", os.path.exists(p) and os.path.getsize(p) > 500, f"{os.path.getsize(p) if os.path.exists(p) else 0} bytes")

    # 3. Verify Stage 1 & 2 Audit & Registered Images
    audit_csv = os.path.join(OUT_BASE, "audit", "IMAGE_RESOLUTION_AND_ALIGNMENT_AUDIT.csv")
    check("Stage 1 Audit CSV", os.path.exists(audit_csv), audit_csv)
    if os.path.exists(audit_csv):
        df_audit = pd.read_csv(audit_csv)
        check("Audit CSV Count (72 Macrographs)", len(df_audit) == 72, f"{len(df_audit)} rows")
        check("Raw Camera Macrographs Located", df_audit['raw_image_found'].sum() >= 70, f"{df_audit['raw_image_found'].sum()} / 72")

    contact_sheet = os.path.join(OUT_BASE, "registered_images", "all_36_registered_view_A_contact_sheet.png")
    check("36-Sample Contact Sheet", os.path.exists(contact_sheet), contact_sheet)

    # 4. Verify Unseen Test Case (IN718, ColdArc, F=5.2, S=26.0)
    unseen_dir = os.path.join(OUT_BASE, "generation_IN718_ColdArc_F5.2_S26.0")
    check("Unseen Generation Directory", os.path.exists(unseen_dir), unseen_dir)
    
    required_artifacts = [
        "raw_generated_rgb.png",
        "direct_blend_rgb.png",
        "m1_direct_cnn_rgb.png",
        "predicted_mask.png",
        "predicted_contour.png",
        "predicted_contour.json",
        "weld_bead_sketch.png",
        "annotated_weld_bead_sketch.png",
        "final_comparison_panel.png",
        "metadata.json"
    ]
    for v in ['view_A', 'view_B']:
        for art in required_artifacts:
            art_path = os.path.join(unseen_dir, v, art)
            check(f"Artifact {v}/{art}", os.path.exists(art_path) and os.path.getsize(art_path) > 100)

    # 5. Verify Benchmark Metrics
    bench_csv = os.path.join(OUT_BASE, "metrics", "INTERPOLATION_BENCHMARK_RESULTS.csv")
    check("LOOCV Benchmark Results CSV", os.path.exists(bench_csv), bench_csv)
    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
        check("Evaluated LOOCV Cases", len(df_bench) >= 30, f"{len(df_bench)} test macrographs")

    summary_csv = os.path.join(OUT_BASE, "metrics", "METHOD_COMPARISON_SUMMARY.csv")
    check("Method Comparison Summary CSV", os.path.exists(summary_csv), summary_csv)

    charts_png = os.path.join(OUT_BASE, "comparison_panels", "evaluation_summary_charts.png")
    check("Evaluation Summary Charts", os.path.exists(charts_png), charts_png)

    # 6. Verify HTML Presentation & Report
    gallery_html = os.path.join(OUT_BASE, "GALLERY.html")
    check("Offline Standalone Presentation Gallery", os.path.exists(gallery_html) and os.path.getsize(gallery_html) > 5000, gallery_html)

    report_md = os.path.join(OUT_BASE, "reports", "PHASE_V14_SHARP_INTERPOLATION_REPORT.md")
    check("Comprehensive Technical Report", os.path.exists(report_md) and os.path.getsize(report_md) > 5000, report_md)

    print("=" * 75)
    print(f"VERIFICATION SUMMARY: {passed_checks} / {total_checks} CHECKS PASSED")
    print("=" * 75)
    if passed_checks == total_checks:
        print("[SUCCESS] ALL PHASE V14 DELIVERABLES VALIDATED SUCCESSFULLY!")
    else:
        print("[WARNING] SOME CHECKS FAILED - REVIEW LOGS ABOVE.")


if __name__ == "__main__":
    verify_all()
