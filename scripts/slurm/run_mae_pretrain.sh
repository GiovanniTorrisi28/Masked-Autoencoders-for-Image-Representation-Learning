#!/bin/bash
#SBATCH --job-name=mae_pretrain
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

echo "=== MAE Pre-training (200 epochs) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

apptainer exec --nv /shared/sifs/latest.sif \
    python train_mae.py \
        --config experiments/configs/mae_pretrain_cluster.yaml

echo "Done: $(date)"
echo "MAE checkpoint saved to: experiments/checkpoints/mae_pretrain/"
echo "Next step: sbatch scripts/slurm/run_linear_probe.sh"
echo "Run 'bash scripts/slurm/sync_wandb.sh' from the login node to upload W&B logs."
