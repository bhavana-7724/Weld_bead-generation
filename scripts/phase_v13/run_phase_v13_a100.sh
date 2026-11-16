#!/usr/bin/env bash
# ==============================================================================
# run_phase_v13_a100.sh
# 
# Master Execution Pipeline for Phase V13 on NVIDIA A100 AI Server
# Project: Predicting Weld Bead Profile for GMAW and Cold Arc Welding Using Deep Learning
# Model: Enhanced Geometry-First Weld Profile Generator v2 (GF-WPG v2)
#
# Hardware: Single Authorized NVIDIA A100-SXM4-40GB GPU (CUDA_VISIBLE_DEVICES=0)
# Virtual Environment: /raid/home/cs24b1007/weld_project/welding_env
# Remote Root: /raid/home/cs24b1007/weld_project
# ==============================================================================

set -euo pipefail

# Error trap to report exact failure line
trap 'echo "!!! [ERROR] Phase V13 pipeline failed at line $LINENO in command: $BASH_COMMAND" >&2' ERR

# 1. Enforce Single Authorized GPU & Institutional Constraints
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Prepare Directory Structure
RESULTS_DIR="${PROJECT_ROOT}/results/phase_v13_a100"
LOGS_DIR="${RESULTS_DIR}/logs"
CKPT_DIR="${RESULTS_DIR}/checkpoints"
FOLDS_CKPT_DIR="${CKPT_DIR}/folds"
FIGURES_DIR="${RESULTS_DIR}/figures"
METRICS_DIR="${RESULTS_DIR}/metrics"
CURVES_DIR="${METRICS_DIR}/learning_curves"
REPORTS_DIR="${RESULTS_DIR}/reports"
GEN36_DIR="${RESULTS_DIR}/generated_36"
PANELS_DIR="${RESULTS_DIR}/panels"

mkdir -p "${LOGS_DIR}" "${FOLDS_CKPT_DIR}" "${FIGURES_DIR}" "${METRICS_DIR}" "${CURVES_DIR}" "${REPORTS_DIR}" "${GEN36_DIR}" "${PANELS_DIR}"

echo "=============================================================================="
echo "  PHASE V13: NVIDIA A100 ENHANCED GEOMETRY-FIRST WELD MODEL PIPELINE"
echo "  Model: GF-WPG v2 (Geometry Regressor + Toe-Gated Rasterizer + Neural Renderer)"
echo "  Date:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Host:  $(hostname)"
echo "  Root:  ${PROJECT_ROOT}"
echo "=============================================================================="

# 2. Virtual Environment Activation
VENV_PATH="/raid/home/cs24b1007/weld_project/welding_env"
if [ -d "${VENV_PATH}" ]; then
    echo "[SETUP] Activating institutional virtual environment: ${VENV_PATH}"
    source "${VENV_PATH}/bin/activate"
elif [ -d "${PROJECT_ROOT}/welding_env" ]; then
    echo "[SETUP] Activating local virtual environment: ${PROJECT_ROOT}/welding_env"
    source "${PROJECT_ROOT}/welding_env/bin/activate"
else
    echo "[WARNING] No dedicated welding_env found at standard paths. Using active Python: $(which python)"
fi

# 3. Hardware & PyTorch CUDA Verification
echo ""
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
    print('  WARNING: CUDA is NOT available! Execution will proceed on CPU.')
"

TOTAL_START=$(date +%s)

# ==============================================================================
# STAGE 0: PREFLIGHT SYSTEM & DATA AUDIT
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STAGE 0/5] RUNNING PREFLIGHT AUDITS"
echo "  - Checking 72 macrographs, 72 GT masks, 72 GT SDFs"
echo "  - Verifying ground-truth landmark extraction (Apex & Toe attachment)"
echo "  - Logging to: ${LOGS_DIR}/00_preflight_audit.log"
echo "=============================================================================="
python -u scripts/phase_v13/audit_preflight_v13.py 2>&1 | tee "${LOGS_DIR}/00_preflight_audit.log"

# ==============================================================================
# STAGE 1: LEAKAGE-FREE GROUPED 5-FOLD CROSS-VALIDATION
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STAGE 1/5] 5-FOLD GROUPED CV WITH INNER VALIDATION & EARLY STOPPING"
echo "  Model: GF-WPG v2 (Enhanced Geometry-First Weld Profile Generator)"
echo "  Outer Test Set: 100% Sequestered | Early Stopping Patience: 20 epochs"
echo "  Logging to: ${LOGS_DIR}/01_grouped_cv_a100.log"
echo "=============================================================================="
S1_START=$(date +%s)

python -u scripts/phase_v13/train_grouped_cv_v13.py --max-epochs 80 --patience 20 2>&1 | tee "${LOGS_DIR}/01_grouped_cv_a100.log"

S1_END=$(date +%s)
echo "[STAGE 1 COMPLETE] 5-Fold Cross-Validation finished in $((S1_END - S1_START)) seconds."

# ==============================================================================
# STAGE 2: FULL PRODUCTION TRAINING ON ALL 36 EXPERIMENTS
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STAGE 2/5] FULL DATASET PRODUCTION TRAINING"
echo "  Training GF-WPG v2 on all 36 physical conditions (72 macrograph views)"
echo "  Target: results/phase_v13_a100/checkpoints/gfwpg_v2_final.pt"
echo "  Logging to: ${LOGS_DIR}/02_production_training_a100.log"
echo "=============================================================================="
S2_START=$(date +%s)

python -u scripts/phase_v13/train_production_v13.py --epochs 80 2>&1 | tee "${LOGS_DIR}/02_production_training_a100.log"

S2_END=$(date +%s)
echo "[STAGE 2 COMPLETE] Production training finished in $((S2_END - S2_START)) seconds."

# ==============================================================================
# STAGE 3: CANONICAL 36-SAMPLE GENERATION & ACADEMIC PANELS
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STAGE 3/5] CANONICAL 36-SAMPLE GENERATION & ACADEMIC PANELS"
echo "  - Synthesizing S01 to S36 with explicit raw RGB / mask / contour separation"
echo "  - Producing 36 individual publication panels (S01_academic_panel.png to S36_academic_panel.png)"
echo "  - Logging to: ${LOGS_DIR}/03_canonical_generation_a100.log"
echo "=============================================================================="
S3_START=$(date +%s)

python -u scripts/phase_v13/generate_canonical_v13.py 2>&1 | tee "${LOGS_DIR}/03_canonical_generation_a100.log"

S3_END=$(date +%s)
echo "[STAGE 3 COMPLETE] Canonical 36-sample generation finished in $((S3_END - S3_START)) seconds."

# ==============================================================================
# STAGE 4: COMPARATIVE BENCHMARKING AGAINST M1, M3, M5 & REPORTING
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STAGE 4/5] COMPARATIVE EVALUATION & SCIENTIFIC REPORTING"
echo "  - Reconciling GF-WPG v2 against Phase V12 results (M1, M3, M5, M2, M4)"
echo "  - Generating publication Figures 1-4"
echo "  - Authoring PHASE_V13_FINAL_RESEARCH_REPORT.md & GALLERY_V13.html"
echo "  - Logging to: ${LOGS_DIR}/04_evaluation_and_comparison_a100.log"
echo "=============================================================================="
S4_START=$(date +%s)

python -u scripts/phase_v13/evaluate_and_compare_v13.py 2>&1 | tee "${LOGS_DIR}/04_evaluation_and_comparison_a100.log"

S4_END=$(date +%s)
echo "[STAGE 4 COMPLETE] Comparative evaluation finished in $((S4_END - S4_START)) seconds."

# ==============================================================================
# STAGE 5: RESULTS PACKAGING & INTEGRITY VERIFICATION
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  [STAGE 5/5] PACKAGING & INTEGRITY VERIFICATION"
echo "  - Compressing results into phase_v13_results_bundle.tar.gz"
echo "  - Logging to: ${LOGS_DIR}/05_package_results.log"
echo "=============================================================================="
S5_START=$(date +%s)

python -u scripts/phase_v13/package_results_v13.py 2>&1 | tee "${LOGS_DIR}/05_package_results.log"

S5_END=$(date +%s)
echo "[STAGE 5 COMPLETE] Packaging finished in $((S5_END - S5_START)) seconds."

TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

# ==============================================================================
# FINAL EXECUTION SUMMARY & INVENTORY
# ==============================================================================
echo ""
echo "=============================================================================="
echo "  PHASE V13 A100 PIPELINE EXECUTION COMPLETED SUCCESSFULLY"
echo "  Total Execution Time: ${TOTAL_DURATION} seconds ($((TOTAL_DURATION / 60)) min $((TOTAL_DURATION % 60)) sec)"
echo "=============================================================================="

echo "[ARTIFACTS INVENTORY]"
echo "  Fold Checkpoints:        $(ls -1 ${FOLDS_CKPT_DIR}/*.pt 2>/dev/null | wc -l)"
echo "  Production Checkpoints:  $(ls -1 ${CKPT_DIR}/*.pt 2>/dev/null | wc -l)"
echo "  Learning Curve CSVs:     $(ls -1 ${CURVES_DIR}/*.csv 2>/dev/null | wc -l)"
echo "  Generated 36 Folders:    $(ls -d ${GEN36_DIR}/S* 2>/dev/null | wc -l)"
echo "  Academic 36 Panels:      $(ls -1 ${PANELS_DIR}/*.png 2>/dev/null | wc -l)"
echo "  Publication Figures:     $(ls -1 ${FIGURES_DIR}/*.png 2>/dev/null | wc -l)"
echo "  Metrics CSVs:            $(ls -1 ${METRICS_DIR}/*.csv 2>/dev/null | wc -l)"
echo "  Scientific Reports:      $(ls -1 ${REPORTS_DIR}/*.md 2>/dev/null | wc -l)"
echo "  Interactive Gallery:     $(ls -1 ${RESULTS_DIR}/GALLERY_V13.html 2>/dev/null | wc -l)"
echo "  Results Tarball:         $(ls -lh ${PROJECT_ROOT}/phase_v13_results_bundle.tar.gz 2>/dev/null | awk '{print $5, $9}')"
echo ""
echo "=============================================================================="
echo "All Phase V13 stages completed on NVIDIA A100 GPU!"
echo "=============================================================================="
