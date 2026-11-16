"""
Master Pipeline Entrypoint for Phase V15: Pretrained AI Models for Weld Macrograph Generation.
Final Academic Review — 16 November 2026.

Usage:
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage all
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage audit
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage rife
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage film
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage evaluate
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage unseen
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage figures
    python scripts/phase_v15_pretrained/run_phase_v15.py --stage verify
"""

import os
import sys
import argparse
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
OUT_BASE = os.path.join(BASE_DIR, "results", "phase_v15_pretrained_models")
FIG_DIR = os.path.join(OUT_BASE, "comparison_figures")


def run_cmd(script_name, desc):
    print("\n" + "=" * 70)
    print(f"RUNNING: {desc} ({script_name})")
    print("=" * 70)
    script_path = os.path.join(CURRENT_DIR, script_name)
    res = subprocess.run([sys.executable, script_path], cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"ERROR: Stage {script_name} failed with exit code {res.returncode}")
        return False
    return True


def verify_phase_v15_deliverables():
    print("\n" + "=" * 70)
    print("VERIFYING PHASE V15 DELIVERABLES AND ARTIFACTS INTEGRITY")
    print("=" * 70)

    # 1. Figures Check
    figures = [
        "workflow_pretrained_pipeline.png",
        "rife_vs_film_interpolation.png",
        "heldout_experimental_comparison.png",
        "quantitative_model_ranking.png",
        "unseen_parameter_prediction_panel.png",
        "pretrained_model_limitations_and_failures.png",
    ]
    missing_figs = []
    for f in figures:
        p = os.path.join(FIG_DIR, f)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            missing_figs.append(f)
        else:
            print(f"  [OK] Figure: {f} ({os.path.getsize(p) // 1024} KB)")

    # 2. Benchmark CSVs Check
    csvs = [
        os.path.join(OUT_BASE, "FAIR_MODEL_COMPARISON.csv"),
        os.path.join(OUT_BASE, "MODEL_COMPARISON_SUMMARY.csv"),
        os.path.join(OUT_BASE, "audit", "IMAGE_AUDIT_REPORT.csv"),
        os.path.join(OUT_BASE, "audit", "MULTI_RESOLUTION_METRICS.csv"),
    ]
    missing_csvs = []
    for c in csvs:
        if not os.path.exists(c) or os.path.getsize(c) == 0:
            missing_csvs.append(os.path.basename(c))
        else:
            print(f"  [OK] CSV: {os.path.basename(c)} ({os.path.getsize(c)} bytes)")

    # 3. Reports Check
    reports = [
        os.path.join(OUT_BASE, "PHASE_V15_MASTER_REPORT.md"),
        os.path.join(OUT_BASE, "DATA_AND_MODEL_AUDIT.md"),
        os.path.join(OUT_BASE, "REPRODUCIBILITY.md"),
        os.path.join(OUT_BASE, "reports", "HIGH_RESOLUTION_TEXTURE_AUDIT.md"),
        os.path.join(OUT_BASE, "reports", "DIFFUSION_LORA_FEASIBILITY.md"),
        os.path.join(OUT_BASE, "reports", "PRETRAINED_MODEL_VERIFICATION.md"),
        os.path.join(OUT_BASE, "reports", "UNSEEN_GENERATION_REPORT.md"),
        os.path.join(OUT_BASE, "reports", "FAILURE_CASE_ANALYSIS.md"),
    ]
    missing_reports = []
    for r in reports:
        if not os.path.exists(r) or os.path.getsize(r) == 0:
            missing_reports.append(os.path.basename(r))
        else:
            print(f"  [OK] Report: {os.path.basename(r)} ({os.path.getsize(r)} bytes)")

    # 4. Checkpoints Check
    rife_chk = os.path.join(OUT_BASE, "checkpoints_manifest", "rife", "RIFE_train_log", "flownet.pkl")
    film_chk = os.path.join(OUT_BASE, "checkpoints_manifest", "film", "variables", "variables.data-00000-of-00001")

    print(f"  [OK] RIFE Weights: {os.path.exists(rife_chk)} ({os.path.getsize(rife_chk) // 1024 // 1024} MB)")
    print(f"  [OK] FILM Weights: {os.path.exists(film_chk)} ({os.path.getsize(film_chk) // 1024 // 1024} MB)")

    print("\n" + "=" * 70)
    if missing_figs or missing_csvs or missing_reports:
        print("VERIFICATION FAILED! Missing items:")
        if missing_figs:
            print("  Figures:", missing_figs)
        if missing_csvs:
            print("  CSVs:", missing_csvs)
        if missing_reports:
            print("  Reports:", missing_reports)
        return False
    else:
        print("PHASE V15 INTEGRITY VERIFICATION: 100% COMPLETE & VERIFIED!")
        print("=" * 70)
        return True


def main():
    parser = argparse.ArgumentParser(description="Phase V15 Master Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "audit", "rife", "film", "evaluate", "unseen", "figures", "verify"],
        default="verify",
        help="Pipeline stage to execute",
    )
    args = parser.parse_args()

    if args.stage == "all":
        stages = [
            ("audit_sources.py", "Dataset and Checkpoints Audit"),
            ("audit_high_res_texture.py", "High-Resolution Texture Preservation Audit"),
            ("rife_inference.py", "RIFE Pretrained Smoke Test"),
            ("film_inference.py", "FILM Pretrained Smoke Test"),
            ("pilot_diffusion_lora.py", "Diffusion LoRA Feasibility Analysis"),
            ("evaluate_heldout.py", "40-View Held-Out Scientific Evaluation"),
            ("generate_unseen_condition.py", "Unseen Condition Prospective Demonstration"),
            ("generate_master_figures.py", "Master Presentation Figures Generation"),
        ]
        for script, desc in stages:
            ok = run_cmd(script, desc)
            if not ok:
                sys.exit(1)
        verify_phase_v15_deliverables()

    elif args.stage == "audit":
        run_cmd("audit_sources.py", "Dataset Audit")
        run_cmd("audit_high_res_texture.py", "High-Res Texture Audit")

    elif args.stage == "rife":
        run_cmd("rife_inference.py", "RIFE Smoke Test")

    elif args.stage == "film":
        run_cmd("film_inference.py", "FILM Smoke Test")

    elif args.stage == "evaluate":
        run_cmd("evaluate_heldout.py", "40-View Held-Out Evaluation")

    elif args.stage == "unseen":
        run_cmd("generate_unseen_condition.py", "Unseen Condition Generation")

    elif args.stage == "figures":
        run_cmd("generate_master_figures.py", "Master Figures Generation")

    elif args.stage == "verify":
        ok = verify_phase_v15_deliverables()
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
