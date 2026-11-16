#!/usr/bin/env bash
# ==============================================================================
# run_phase_v12_full_pipeline.sh
# Phase V12 Finalized Project Stage: Automated Master Pipeline Runner
# Safe, unattended execution with error halting and verbose logging.
# Compatible with Linux AI Server (/raid/home/cs24b1007/weld_project) and local bash.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOG_DIR="${PROJECT_ROOT}/results/phase_v12_finalized_project_stage/logs"
mkdir -p "${LOG_DIR}"

LOG_FILE="${LOG_DIR}/phase_v12_pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "=============================================================================="
echo "PHASE V12 MASTER PIPELINE EXECUTION STARTED"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Project Root: ${PROJECT_ROOT}"
echo "Log File: ${LOG_FILE}"
echo "=============================================================================="

# 1. Environment Detection & Activation
if [ -d "/raid/home/cs24b1007/weld_project/welding_env" ]; then
    echo "[ENV] Activating remote Linux AI server environment..."
    source "/raid/home/cs24b1007/weld_project/welding_env/bin/activate"
elif [ -d "${PROJECT_ROOT}/welding_env" ]; then
    echo "[ENV] Activating local virtual environment..."
    source "${PROJECT_ROOT}/welding_env/bin/activate"
else
    echo "[ENV] Using system Python: $(which python3 || which python)"
fi

PYTHON_CMD=$(which python3 || which python)
echo "[ENV] Python version: $($PYTHON_CMD --version)"

# 2. Step 1: Asset Audit & Manifest Generation
echo ""
echo "------------------------------------------------------------------------------"
echo "STEP 1: Executing Asset Audit & Manifest Generation..."
echo "------------------------------------------------------------------------------"
$PYTHON_CMD "${SCRIPT_DIR}/create_manifest_and_audit.py"

# 3. Step 2 & 3: Master Runner for All 5 Generative Models (180 Panels + Metrics)
echo ""
echo "------------------------------------------------------------------------------"
echo "STEP 2 & 3: Generating All 180 Sample Panels, Compiling Metrics & Galleries..."
echo "------------------------------------------------------------------------------"
$PYTHON_CMD "${SCRIPT_DIR}/run_all_models_36_panels.py"

# 4. Step 4: Summary Figure Generation
echo ""
echo "------------------------------------------------------------------------------"
echo "STEP 4: Rendering High-Impact Publication Figures (Figs 1 to 5)..."
echo "------------------------------------------------------------------------------"
$PYTHON_CMD "${SCRIPT_DIR}/generate_stage_figures.py"

# 5. Step 5: Test Prospective Unseen Parameter Generation
echo ""
echo "------------------------------------------------------------------------------"
echo "STEP 5: Testing Unseen Parameter Prospective Generation on CPU..."
echo "------------------------------------------------------------------------------"
$PYTHON_CMD "${SCRIPT_DIR}/generate_unseen_interactive.py" \
    --material IN718 \
    --process ColdArc \
    --feed_rate 5.2 \
    --speed 26.0

$PYTHON_CMD "${SCRIPT_DIR}/generate_unseen_interactive.py" \
    --material SS304L \
    --process GMAW \
    --feed_rate 5.8 \
    --speed 28.0

echo ""
echo "=============================================================================="
echo "PHASE V12 MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY!"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Results available at: ${PROJECT_ROOT}/results/phase_v12_finalized_project_stage/"
echo "=============================================================================="
