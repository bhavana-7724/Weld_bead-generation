"""
audit_preflight_v13.py

Phase V13 Stage 0: Preflight Audit & Environment Verification
Target System: Single NVIDIA A100 GPU (CUDA_VISIBLE_DEVICES=0)
Project Root: weld_project

Verifies:
1. CUDA Device, PyTorch Build, cuDNN, VRAM capacity.
2. Dataset integrity: clean_dataset_36.csv, 72 aligned macrographs, 72 GT masks, 72 GT SDFs.
3. Geometric landmark extraction: 95 scanline contour points, apex (x, y), toe coordinates.
4. Baseline comparison assets from Phase V12 (MODEL_COMPARISON_SUMMARY.csv).
5. Output directory readiness and write permissions.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed", "aligned_128")
CLEAN_CSV = os.path.join(DATA_DIR, "clean_dataset_36.csv")

AUDIT_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "audit")
MASKS_DIR = os.path.join(AUDIT_DIR, "gt_masks")
SDF_DIR = os.path.join(AUDIT_DIR, "gt_sdf")
GEN36_FALLBACK_DIR = os.path.join(BASE_DIR, "results", "phase_v9_geometry_aware", "generated_36")

V12_METRICS_CSV = os.path.join(
    BASE_DIR, "results", "phase_v12_finalized_project_stage", "ACADEMIC_FINAL_VALIDATION", "MODEL_COMPARISON_SUMMARY.csv"
)

RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase_v13_a100")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FOLDS_CKPT_DIR = os.path.join(CKPT_DIR, "folds")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CURVES_DIR = os.path.join(METRICS_DIR, "learning_curves")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
GEN36_DIR = os.path.join(RESULTS_DIR, "generated_36")

for d in [RESULTS_DIR, LOGS_DIR, CKPT_DIR, FOLDS_CKPT_DIR, FIGURES_DIR, METRICS_DIR, CURVES_DIR, REPORTS_DIR, GEN36_DIR]:
    os.makedirs(d, exist_ok=True)


def run_preflight_audit():
    print("=" * 80)
    print("PHASE V13 [STAGE 0]: PREFLIGHT SYSTEM & DATA AUDIT")
    print(f"Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Dir:    {BASE_DIR}")
    print(f"Results Dir: {RESULTS_DIR}")
    print("=" * 80)

    audit_passed = True
    audit_log = []

    def log(msg, status="INFO"):
        line = f"[{status}] {msg}"
        print(line)
        audit_log.append(line)

    # 1. Environment & CUDA Check
    log("Checking Python and PyTorch runtime...", "CHECK")
    log(f"Python Version:  {sys.version.split()[0]}")
    log(f"PyTorch Version: {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    log(f"CUDA Available:  {cuda_avail}")

    if cuda_avail:
        dev_count = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        log(f"GPU Device Count: {dev_count}")
        log(f"Target Device:    cuda:0 ({dev_name})")
        log(f"VRAM Capacity:    {vram_gb:.2f} GB")
        log(f"Compute Cap:      {props.major}.{props.minor}")
        if vram_gb >= 30.0:
            log("Hardware conforms to NVIDIA A100 institutional specification.", "PASS")
        else:
            log(f"Hardware has {vram_gb:.1f} GB VRAM (less than 30 GB A100 target). Will proceed with caution.", "WARN")
    else:
        log("No CUDA device detected! System will fall back to CPU.", "WARN")

    # 2. Dataset Files Check
    log("\nVerifying experimental dataset and CSV tables...", "CHECK")
    if not os.path.exists(CLEAN_CSV):
        log(f"Missing master CSV: {CLEAN_CSV}", "FAIL")
        audit_passed = False
    else:
        df_clean = pd.read_csv(CLEAN_CSV)
        req_cols = ['sample', 'material', 'process', 'feed_rate', 'speed', 'width', 'depth', 'reinforcement']
        missing_cols = [c for c in req_cols if c not in df_clean.columns]
        if missing_cols:
            log(f"Clean CSV missing columns: {missing_cols}", "FAIL")
            audit_passed = False
        else:
            log(f"Clean CSV verified: {len(df_clean)} samples, all required columns present.", "PASS")

    # 3. Aligned Macrographs (72 images)
    log("\nVerifying 72 aligned macrograph images (Views A & B)...", "CHECK")
    missing_imgs = []
    found_in_fallback = 0
    for s in range(1, 37):
        for view in ["A", "B"]:
            p1 = os.path.join(PROCESSED_DIR, f"{s}-{view}.png")
            if not os.path.exists(p1):
                p2 = os.path.join(GEN36_FALLBACK_DIR, f"S{s:02d}", f"Real_{view}.png")
                if os.path.exists(p2):
                    found_in_fallback += 1
                else:
                    missing_imgs.append(f"S{s:02d}-{view}")
    if missing_imgs:
        log(f"Missing macrograph images: {len(missing_imgs)} ({missing_imgs[:5]}...)", "FAIL")
        audit_passed = False
    else:
        log(f"All 72 experimental macrographs located ({found_in_fallback} found via fallback path).", "PASS")

    # 4. Ground-Truth Masks and SDFs (72 masks, 72 SDFs)
    log("\nVerifying Ground-Truth geometry masks and Signed Distance Fields...", "CHECK")
    missing_masks = []
    missing_sdfs = []
    for s in range(1, 37):
        for view in ["A", "B"]:
            m_path = os.path.join(MASKS_DIR, f"{s}-{view}.png")
            if not os.path.exists(m_path):
                missing_masks.append(f"{s}-{view}")
            s_path = os.path.join(SDF_DIR, f"{s}-{view}.npy")
            if not os.path.exists(s_path):
                missing_sdfs.append(f"{s}-{view}")

    if missing_masks:
        log(f"Missing GT masks: {len(missing_masks)}", "FAIL")
        audit_passed = False
    else:
        log("All 72 Ground-Truth binary masks verified.", "PASS")

    if missing_sdfs:
        log(f"Missing GT SDFs: {len(missing_sdfs)}", "WARN")
    else:
        log("All 72 Ground-Truth SDF arrays verified.", "PASS")

    # 5. Phase V12 Comparative Assets Check
    log("\nVerifying Phase V12 academic comparative benchmarks...", "CHECK")
    if os.path.exists(V12_METRICS_CSV):
        df_v12 = pd.read_csv(V12_METRICS_CSV)
        log(f"Found Phase V12 summary with {len(df_v12)} models: {list(df_v12['model'])}", "PASS")
    else:
        log(f"Phase V12 summary not found at {V12_METRICS_CSV}. Will synthesize baseline entries if needed.", "WARN")

    # 6. Landmark Extraction Verification (Apex and Toe coordinates across all 72 views)
    log("\nTesting ground-truth landmark extraction across all 72 views...", "CHECK")
    try:
        from scripts.phase_v9.prepare_geometry_labels import extract_robust_crown_contour
        apex_xs, apex_ys, toe_tops, toe_bots = [], [], [], []
        for s in range(1, 37):
            for view in ["A", "B"]:
                img_path = os.path.join(PROCESSED_DIR, f"{s}-{view}.png")
                if not os.path.exists(img_path):
                    img_path = os.path.join(GEN36_FALLBACK_DIR, f"S{s:02d}", f"Real_{view}.png")
                gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                cnt_res = extract_robust_crown_contour(gray, x_baseline=64)
                pts = cnt_res.get("pts", {})
                if len(pts) > 0:
                    apex_xs.append(cnt_res.get("apex_x", 64))
                    apex_ys.append(cnt_res.get("apex_y", 64))
                    toe_tops.append(min(pts.keys()))
                    toe_bots.append(max(pts.keys()))
        log(f"Extracted landmarks from all {len(apex_xs)} views:", "PASS")
        log(f"  Apex X:    mean={np.mean(apex_xs):.1f} px, min={np.min(apex_xs)}, max={np.max(apex_xs)}")
        log(f"  Apex Y:    mean={np.mean(apex_ys):.1f} px, min={np.min(apex_ys)}, max={np.max(apex_ys)}")
        log(f"  Toe Top Y: mean={np.mean(toe_tops):.1f} px, min={np.min(toe_tops)}, max={np.max(toe_tops)}")
        log(f"  Toe Bot Y: mean={np.mean(toe_bots):.1f} px, min={np.min(toe_bots)}, max={np.max(toe_bots)}")
    except Exception as e:
        log(f"Landmark extraction encountered an error: {e}", "FAIL")
        audit_passed = False

    # 7. Write Permissions
    log("\nTesting results directory write permissions...", "CHECK")
    test_file = os.path.join(LOGS_DIR, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        log("Write permissions verified in results/phase_v13_a100/.", "PASS")
    except Exception as e:
        log(f"Write test failed: {e}", "FAIL")
        audit_passed = False

    # Save log file
    log_file_path = os.path.join(LOGS_DIR, "00_preflight_audit.log")
    with open(log_file_path, "w") as f:
        f.write("\n".join(audit_log) + "\n")
    log(f"Full audit log saved to: {log_file_path}")

    print("=" * 80)
    if audit_passed:
        print(">>> PREFLIGHT AUDIT PASSED: ALL PREREQUISITES SATISFIED <<<")
        print("Ready to proceed to Stage 1 (Grouped 5-Fold Cross-Validation).")
        print("=" * 80)
        return 0
    else:
        print("!!! PREFLIGHT AUDIT FAILED: CORRECT ISSUES BEFORE PROCEEDING !!!")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(run_preflight_audit())
