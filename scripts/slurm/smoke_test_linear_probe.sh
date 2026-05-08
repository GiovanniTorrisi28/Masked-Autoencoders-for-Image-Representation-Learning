#!/bin/bash
#SBATCH --job-name=smoke_linear_probe
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

MAE_CHECKPOINT="experiments/checkpoints/smoke_mae/checkpoint_best.pth"
if [ ! -f "${MAE_CHECKPOINT}" ]; then
    echo "ERROR: MAE smoke checkpoint not found at ${MAE_CHECKPOINT}"
    echo "Run sbatch scripts/slurm/smoke_test_mae.sh first."
    exit 1
fi

echo "=== Smoke test: Linear Probe (2 epochs) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

apptainer exec --nv /shared/sifs/latest.sif \
    python train_linear_probe.py \
        --config experiments/configs/smoke_linear_probe_cluster.yaml \
        --mae-checkpoint "${MAE_CHECKPOINT}"

echo "Done: $(date)"
echo "Run 'bash scripts/slurm/sync_wandb.sh' from the login node to upload W&B logs."
