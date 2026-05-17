#!/bin/bash
#SBATCH --job-name=linear_probe_20pct
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
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

echo "=== Linear Probe 20% Dataset (100 epochs) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

MAE_CHECKPOINT="experiments/checkpoints/mae_pretrain/checkpoint_best.pth"
if [ ! -f "${MAE_CHECKPOINT}" ]; then
    echo "ERROR: MAE checkpoint not found at ${MAE_CHECKPOINT}"
    exit 1
fi

apptainer exec --nv /shared/sifs/latest.sif \
    python train_linear_probe.py \
        --config "experiments/configs/linear_probe_small_dataset_20%_cluster.yaml" \
        --mae-checkpoint "${MAE_CHECKPOINT}"

echo "Done: $(date)"
echo "Run 'bash scripts/slurm/sync_wandb.sh' from the login node to upload W&B logs."
