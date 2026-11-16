"""
run_phase_v9.py

Master Entry Point for Phase V9: High-Fidelity Geometry-Aware Weld Profile Generation and Scientific Validation.

Usage:
  python run_phase_v9.py --stage audit
  python run_phase_v9.py --stage cv
  python run_phase_v9.py --stage generate_36
  python run_phase_v9.py --stage sensitivity
  python run_phase_v9.py --stage unseen
  python run_phase_v9.py --stage all
"""

import os
import sys
import argparse
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts", "phase_v9")

def run_cmd(script_name, desc):
    print("\n" + "=" * 80)
    print(f"EXECUTING STAGE: {desc}")
    print(f"Script: scripts/phase_v9/{script_name}")
    print("=" * 80)
    t0 = time.time()
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    res = subprocess.run([sys.executable, "-u", script_path], check=True)
    print(f"\n[DONE] {desc} completed in {time.time() - t0:.1f}s (Exit code: {res.returncode})\n")

def main():
    parser = argparse.ArgumentParser(description="Phase V9 Master Pipeline Orchestrator")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["audit", "cv", "generate_36", "sensitivity", "unseen", "all"],
        help="Pipeline stage to execute: audit, cv, generate_36, sensitivity, unseen, or all"
    )
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80)
    print("PHASE V9: HIGH-FIDELITY GEOMETRY-AWARE WELD PROFILE GENERATION")
    print(f"Target Stage: {args.stage.upper()}")
    print("=" * 80)

    if args.stage in ["audit", "all"]:
        run_cmd("prepare_geometry_labels.py", "Phase A: Dataset Audit & Crown Geometry Ground Truth")

    if args.stage in ["cv", "all"]:
        run_cmd("run_grouped_cv.py", "Phase C: 5-Fold Grouped Cross-Validation Benchmark")

    if args.stage in ["generate_36", "all"]:
        run_cmd("train_and_generate_36.py", "Phase D: Production Models & 36-Sample Benchmark")

    if args.stage in ["sensitivity", "unseen", "all"]:
        run_cmd("run_sensitivity_and_unseen.py", "Phases E & F: Sensitivity Analysis & Unseen Predictions")

    print("=" * 80)
    print(f"PHASE V9 PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t_start:.1f} SECONDS!")
    print("=" * 80)

if __name__ == "__main__":
    main()
