#!/bin/bash
#SBATCH --job-name=evaluate
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

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONUNBUFFERED=1

echo "=== Valutazione finale ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "Start:  $(date)"

apptainer exec --nv /shared/sifs/latest.sif \
    python evaluate.py \
        --config experiments/configs/supervised_vit_cluster.yaml \
        --checkpoint experiments/checkpoints/supervised_vit_baseline_200/checkpoint_best.pth \
        --label "Supervised ViT Baseline (200 epoche)"

apptainer exec --nv /shared/sifs/latest.sif \
    python evaluate.py \
        --config experiments/configs/linear_probe_cluster.yaml \
        --checkpoint experiments/checkpoints/linear_probe_200/checkpoint_best.pth \
        --label "MAE + Linear Probe (500 + 200 epoche)"

echo "Done: $(date)"
