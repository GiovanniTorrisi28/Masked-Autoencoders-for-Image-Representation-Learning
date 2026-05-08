#!/bin/bash
#SBATCH --job-name=smoke_evaluate
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%j_%x.out
#SBATCH --error=slurm_logs/%j_%x.err

set -e

cd ~/Masked-Autoencoders-for-Image-Representation-Learning

mkdir -p slurm_logs

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONUNBUFFERED=1

echo "=== Smoke Evaluation ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

# Evaluate supervised ViT smoke checkpoint
apptainer exec --nv /shared/sifs/latest.sif \
    python evaluate.py \
        --config experiments/configs/smoke_supervised_cluster.yaml \
        --checkpoint experiments/checkpoints/smoke_supervised/checkpoint_best.pth \
        --label "Supervised ViT (smoke, 2 epochs)"

# Evaluate MAE + linear probe smoke checkpoint
apptainer exec --nv /shared/sifs/latest.sif \
    python evaluate.py \
        --config experiments/configs/smoke_linear_probe_cluster.yaml \
        --checkpoint experiments/checkpoints/smoke_linear_probe/checkpoint_best.pth \
        --label "MAE + Linear Probe (smoke, 2 epochs)"

echo "Done: $(date)"
