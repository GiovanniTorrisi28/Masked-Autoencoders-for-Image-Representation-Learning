#!/bin/bash
#SBATCH --job-name=smoke_supervised
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

# Load WANDB_API_KEY from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Compute nodes have no internet: W&B runs offline and syncs later from the login node
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

echo "=== Smoke test: Supervised ViT (2 epochs) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

apptainer exec --nv /shared/sifs/latest.sif \
    python train_supervised.py \
        --config experiments/configs/smoke_supervised_cluster.yaml

echo "Done: $(date)"
echo "Run 'bash scripts/slurm/sync_wandb.sh' from the login node to upload W&B logs."
