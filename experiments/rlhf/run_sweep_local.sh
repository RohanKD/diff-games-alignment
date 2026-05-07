#!/bin/bash
# Run the full experiment sweep on a local machine with 2 GPUs.
# Runs 2 jobs in parallel (one per GPU).
#
# Usage: bash run_sweep_local.sh
# Quick test: bash run_sweep_local.sh --quick

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results/rlhf"
mkdir -p "${RESULTS_DIR}"

# Parse args
TOTAL_STEPS=500
if [[ "$1" == "--quick" ]]; then
    TOTAL_STEPS=100
    echo "*** QUICK MODE: 100 steps per run ***"
fi

BETAS=(0.01 0.02 0.05 0.1 0.2 0.5)
SEEDS=(42 123 456)

# Build job queue: (beta, condition_flag, seed)
JOBS=()
for BETA in "${BETAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        JOBS+=("${BETA} frozen ${SEED}")
        JOBS+=("${BETA} online ${SEED}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
echo "Total jobs: ${TOTAL_JOBS}"
echo "Results dir: ${RESULTS_DIR}"
echo ""

# Track running PIDs per GPU
PID_GPU0=""
PID_GPU1=""
JOB_IDX=0

run_job() {
    local GPU=$1
    local BETA=$2
    local CONDITION=$3
    local SEED=$4

    local ONLINE_FLAG=""
    if [[ "$CONDITION" == "online" ]]; then
        ONLINE_FLAG="--online_rm"
    fi

    local LOG_FILE="${RESULTS_DIR}/log_${CONDITION}_b${BETA}_s${SEED}.stdout"

    echo "[GPU ${GPU}] Starting: beta=${BETA}, ${CONDITION}, seed=${SEED}" >&2

    CUDA_VISIBLE_DEVICES=${GPU} python "${SCRIPT_DIR}/run_ppo.py" \
        --beta ${BETA} \
        ${ONLINE_FLAG} \
        --seed ${SEED} \
        --total_steps ${TOTAL_STEPS} \
        --output_dir "${RESULTS_DIR}" \
        > "${LOG_FILE}" 2>&1 &

    echo $!
}

# Process jobs 2 at a time
while [ $JOB_IDX -lt $TOTAL_JOBS ]; do
    # Check if GPU 0 is free
    if [ -z "$PID_GPU0" ] || ! kill -0 "$PID_GPU0" 2>/dev/null; then
        if [ $JOB_IDX -lt $TOTAL_JOBS ]; then
            read BETA CONDITION SEED <<< "${JOBS[$JOB_IDX]}"
            PID_GPU0=$(run_job 0 $BETA $CONDITION $SEED)
            JOB_IDX=$((JOB_IDX + 1))
        fi
    fi

    # Check if GPU 1 is free
    if [ -z "$PID_GPU1" ] || ! kill -0 "$PID_GPU1" 2>/dev/null; then
        if [ $JOB_IDX -lt $TOTAL_JOBS ]; then
            read BETA CONDITION SEED <<< "${JOBS[$JOB_IDX]}"
            PID_GPU1=$(run_job 1 $BETA $CONDITION $SEED)
            JOB_IDX=$((JOB_IDX + 1))
        fi
    fi

    # Wait a bit before checking again
    sleep 30

    # Progress
    DONE=$((JOB_IDX))
    RUNNING=0
    [ -n "$PID_GPU0" ] && kill -0 "$PID_GPU0" 2>/dev/null && RUNNING=$((RUNNING+1))
    [ -n "$PID_GPU1" ] && kill -0 "$PID_GPU1" 2>/dev/null && RUNNING=$((RUNNING+1))
    echo "  [Progress: ${DONE}/${TOTAL_JOBS} dispatched, ${RUNNING} running]"
done

# Wait for final jobs
echo "All jobs dispatched. Waiting for final jobs to complete..."
[ -n "$PID_GPU0" ] && wait $PID_GPU0
[ -n "$PID_GPU1" ] && wait $PID_GPU1

echo ""
echo "========================================"
echo "All ${TOTAL_JOBS} jobs complete!"
echo "========================================"
echo ""
echo "Run analysis:"
echo "  python ${SCRIPT_DIR}/analyze_results.py --results_dir ${RESULTS_DIR}"
