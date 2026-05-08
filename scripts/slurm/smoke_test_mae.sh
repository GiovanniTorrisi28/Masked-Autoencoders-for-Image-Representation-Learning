#!/bin/bash
#SBATCH --job-name=smoke_mae
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e

cd ~/Masked-Autoencoders-for-Image-Representation-Learning

mkdir -p slurm_logs experiments/logs experiments/checkpoints

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "=== Smoke test: MAE Pre-training (2 epochs) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

apptainer exec --nv /shared/sifs/latest.sif \
    python train_mae.py \
        --config experiments/configs/smoke_mae_cluster.yaml

echo "Done: $(date)"
echo "Run 'bash scripts/slurm/sync_wandb.sh' from the login node to upload W&B logs."
