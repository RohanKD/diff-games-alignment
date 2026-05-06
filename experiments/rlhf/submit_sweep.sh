#!/bin/bash
# Submit the full experiment sweep on PACE.
# Usage: bash submit_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results/rlhf"
mkdir -p "${RESULTS_DIR}"

BETAS=(0.01 0.02 0.05 0.1 0.2 0.5)
SEEDS=(42 123 456)

for BETA in "${BETAS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        # Frozen RM condition
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=rlhf_frozen_b${BETA}_s${SEED}
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=${RESULTS_DIR}/slurm_frozen_b${BETA}_s${SEED}_%j.out

module load anaconda3 cuda/12.1
conda activate rlhf

cd ${SCRIPT_DIR}
python run_ppo.py \\
    --beta ${BETA} \\
    --seed ${SEED} \\
    --total_steps 500 \\
    --output_dir ${RESULTS_DIR}
EOF

        # Online RM condition
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=rlhf_online_b${BETA}_s${SEED}
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=${RESULTS_DIR}/slurm_online_b${BETA}_s${SEED}_%j.out

module load anaconda3 cuda/12.1
conda activate rlhf

cd ${SCRIPT_DIR}
python run_ppo.py \\
    --beta ${BETA} \\
    --online_rm \\
    --seed ${SEED} \\
    --total_steps 500 \\
    --output_dir ${RESULTS_DIR}
EOF
    done
done

echo "Submitted $(( ${#BETAS[@]} * ${#SEEDS[@]} * 2 )) jobs"
echo "Monitor with: squeue -u \$USER"
echo "After completion, run: python analyze_results.py --results_dir ${RESULTS_DIR}"
