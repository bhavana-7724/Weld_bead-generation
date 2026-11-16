#!/usr/bin/env bash
# ==============================================================================
# run_a100_full_pipeline.sh
# 
# Master Execution Script for Phase V10 on NVIDIA A100 AI Server
# Project: Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning
# 
# Hardware: Single Authorized NVIDIA A100-SXM4-40GB GPU (CUDA_VISIBLE_DEVICES=0)
# Virtual Environment: /raid/home/cs24b1007/weld_project/welding_env
# Remote Root: /raid/home/cs24b1007/weld_project
# ==============================================================================

set -euo pipefail

# Trap unexpected errors and print line number
trap 'echo "!!! [ERROR] Pipeline failed at line $LINENO in command: $BASH_COMMAND" >&2' ERR

# 1. Enforce Single Authorized GPU & Institutional Constraints
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Prepare Directory Structure
RESULTS_DIR="${PROJECT_ROOT}/results/phase_v10_a100"
LOGS_DIR="${RESULTS_DIR}/logs"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
FOLDS_CKPT_DIR="${CKPT_DIR}/folds"
FIGURES_DIR="${RESULTS_DIR}/figures"
METRICS_DIR="${RESULTS_DIR}/metrics"
CURVES_DIR="${METRICS_DIR}/learning_curves"
REPORTS_DIR="${RESULTS_DIR}/reports"
GEN36_DIR="${RESULTS_DIR}/generated_36"
UNSEEN_DIR="${RESULTS_DIR}/unseen_predictions"

mkdir -p "${LOGS_DIR}" "${FOLDS_CKPT_DIR}" "${FIGURES_DIR}" "${METRICS_DIR}" "${CURVES_DIR}" "${REPORTS_DIR}" "${GEN36_DIR}" "${UNSEEN_DIR}"

echo "=============================================================================="
echo "  PHASE V10: NVIDIA A100 DEEP LEARNING WELD PROFILE PIPELINE"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Host: $(hostname)"
echo "  Project Root: ${PROJECT_ROOT}"
echo "=============================================================================="

# 2. Virtual Environment Activation
VENV_PATH="/raid/home/cs24b1007/weld_project/welding_env"
if [ -d "${VENV_PATH}" ]; then
    echo "[SETUP] Activating virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
elif [ -d "${PROJECT_ROOT}/welding_env" ]; then
    echo "[SETUP] Activating local virtual environment: ${PROJECT_ROOT}/welding_env"
    source "${PROJECT_ROOT}/welding_env/bin/activate"
else
    echo "[WARNING] No dedicated welding_env found at standard paths. Using active Python: $(which python)"
fi

# 3. Hardware & Environment Audit
echo "[CHECK] Auditing NVIDIA GPU Hardware and PyTorch CUDA Environment..."
if command -v nvidia-smi &> /dev/null; then
    echo "--- NVIDIA-SMI Device Overview ---"
    nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
    echo "----------------------------------"
fi

python -c "
import torch, sys
print(f'  Python Version:     {sys.version.split()[0]}')
print(f'  PyTorch Version:    {torch.__version__}')
print(f'  CUDA Available:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device Count:       {torch.cuda.device_count()}')
    print(f'  Target GPU:         {torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})')
    print(f'  Compute Capability: {torch.cuda.get_device_capability(0)}')
    print(f'  Allocated Memory:   {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB')
else:
    print('  FATAL: CUDA is NOT available! Execution halted.')
    sys.exit(1)
"

TOTAL_START=$(date +%s)

# ==============================================================================
# STEP 0: PRE-FLIGHT AUDITS (GT CONTOUR AUDIT & GEOSDF RESPONSIVENESS)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STEP 0/5] RUNNING PRE-FLIGHT AUDITS"
echo "  - Auditing 72 Ground-Truth Contours (View A vs View B consistency)"
echo "  - Verifying GeoSDF-CVAE conditioning gradient responsiveness"
echo "=============================================================================="
python -u scripts/phase_v10/audit_gt_contour_labels.py 2>&1 | tee "${LOGS_DIR}/00_gt_contour_audit.log"
python -u scripts/phase_v10/audit_geosdf_responsiveness.py 2>&1 | tee "${LOGS_DIR}/00_geosdf_responsiveness.log"

# ==============================================================================
# STEP 1: LEAKAGE-FREE GROUPED 5-FOLD CROSS-VALIDATION WITH INNER VALIDATION
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STEP 1/5] 5-FOLD GROUPED CV WITH INNER VALIDATION & EARLY STOPPING"
echo "  Models: M1 (DirectCNN), M2 (CVAE), M3 (GC-WPG-v2), M4 (GeoSDF-CVAE),"
echo "          M5 (cGAN), M6 (GF-WPG), B1 (PCA-Contour Baseline)"
echo "  Outer Test Set: 100% Sequestered | Early Stopping Patience: 20 epochs"
echo "  Logging to: ${LOGS_DIR}/01_grouped_cv_a100.log"
echo "=============================================================================="
S1_START=$(date +%s)

python -u scripts/phase_v10/train_grouped_cv_a100.py 2>&1 | tee "${LOGS_DIR}/01_grouped_cv_a100.log"

S1_END=$(date +%s)
echo "[STEP 1 COMPLETE] 5-Fold Cross-Validation finished in $((S1_END - S1_START)) seconds."

# ==============================================================================
# STEP 2: FULL PRODUCTION TRAINING & 36-SAMPLE CANONICAL GENERATION
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STEP 2/5] PRODUCTION TRAINING & CANONICAL 36-SAMPLE GENERATION"
echo "  Training M1-M6 on full dataset; saving production checkpoints;"
echo "  Generating S01-S36 with explicit raw RGB / mask / overlay separation"
echo "  Logging to: ${LOGS_DIR}/02_production_training_a100.log"
echo "=============================================================================="
S2_START=$(date +%s)

python -u scripts/phase_v10/train_production_a100.py 2>&1 | tee "${LOGS_DIR}/02_production_training_a100.log"

S2_END=$(date +%s)
echo "[STEP 2 COMPLETE] Production training & 36-sample generation finished in $((S2_END - S2_START)) seconds."

# ==============================================================================
# STEP 3: CONTROLLED SENSITIVITY SWEEPS & UNSEEN IN-DOMAIN PREDICTIONS
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STEP 3/5] GROUP-SPECIFIC SENSITIVITY SWEEPS & UNSEEN PREDICTIONS"
echo "  Sweeping strictly in-domain; reporting true monotonic/non-monotonic slopes;"
echo "  Evaluating unseen setpoints U1-U4; authoring physical validation trials"
echo "  Logging to: ${LOGS_DIR}/03_sensitivity_and_unseen_a100.log"
echo "=============================================================================="
S3_START=$(date +%s)

python -u scripts/phase_v10/evaluate_sensitivity_and_unseen_a100.py 2>&1 | tee "${LOGS_DIR}/03_sensitivity_and_unseen_a100.log"

S3_END=$(date +%s)
echo "[STEP 3 COMPLETE] Sensitivity sweeps & unseen predictions finished in $((S3_END - S3_START)) seconds."

# ==============================================================================
# STEP 4: LATENT DIFFUSION FEASIBILITY STUDY
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STEP 4/5] LATENT DIFFUSION MATHEMATICAL FEASIBILITY STUDY"
echo "  Authoring comparative audit: DIFFUSION_FEASIBILITY_STUDY.md"
echo "  Logging to: ${LOGS_DIR}/04_diffusion_feasibility.log"
echo "=============================================================================="
S4_START=$(date +%s)

python -u scripts/phase_v10/diffusion_feasibility_study.py 2>&1 | tee "${LOGS_DIR}/04_diffusion_feasibility.log"

S4_END=$(date +%s)
echo "[STEP 4 COMPLETE] Diffusion feasibility study finished in $((S4_END - S4_START)) seconds."

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

# ==============================================================================
# FINAL VERIFICATION & ARTIFACT SUMMARY
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  PHASE V10 A100 PIPELINE EXECUTION SUMMARY"
echo "  Total Execution Time: ${TOTAL_DURATION} seconds ($((TOTAL_DURATION / 60)) min $((TOTAL_DURATION % 60)) sec)"
echo "=============================================================================="

echo "[ARTIFACTS INVENTORY]"
echo "  Fold Checkpoints (Expected 30):      $(ls -1 ${FOLDS_CKPT_DIR}/*.pt 2>/dev/null | wc -l)"
echo "  Production Checkpoints (Expected 6):  $(ls -1 ${CKPT_DIR}/*_final.pt 2>/dev/null | wc -l)"
echo "  Learning Curve CSVs:                 $(ls -1 ${CURVES_DIR}/*.csv 2>/dev/null | wc -l)"
echo "  Generated 36 Sample Folders:         $(ls -d ${GEN36_DIR}/S* 2>/dev/null | wc -l)"
echo "  Unseen Prediction Folders:           $(ls -d ${UNSEEN_DIR}/U* 2>/dev/null | wc -l)"
echo "  Figure Files:                        $(ls -1 ${FIGURES_DIR}/*.png 2>/dev/null | wc -l)"
echo "  Metrics CSVs:                        $(ls -1 ${METRICS_DIR}/*.csv 2>/dev/null | wc -l)"
echo "  Scientific Reports:                  $(ls -1 ${REPORTS_DIR}/*.md 2>/dev/null | wc -l)"

echo ""
echo "All Phase V10 tasks completed successfully on NVIDIA A100 GPU!"
