#!/bin/bash
#SBATCH --job-name=evaluate_per_class
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

mkdir -p slurm_logs figures experiments/results

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

export PYTHONUNBUFFERED=1

echo "=== Valutazione per-classe — tutti i modelli ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   ${SLURM_NODELIST}"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start:  $(date)"

apptainer exec --nv /shared/sifs/latest.sif \
    python evaluate_per_class.py

echo "Done: $(date)"
echo "Scarica i risultati con:"
echo "  scp TRRGNN02A28C351N@gcluster.dmi.unict.it:~/Masked-Autoencoders-for-Image-Representation-Learning/figures/per_class_accuracy* figures/"
