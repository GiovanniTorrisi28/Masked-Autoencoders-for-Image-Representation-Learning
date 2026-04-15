#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <account_partition> [email]"
  echo "Example: $0 dl-course-q1"
  echo "Example: $0 dl-course-q1 my.email@example.com"
  exit 1
fi

ACCOUNT="$1"
EMAIL="${2:-}"
SCRIPT="scripts/slurm/run_baseline_gcluster.sh"

if [ ! -f "$SCRIPT" ]; then
  echo "SLURM script not found: $SCRIPT"
  exit 1
fi

if [ -n "$EMAIL" ]; then
  sbatch \
    --account="$ACCOUNT" \
    --partition="$ACCOUNT" \
    --mail-type=END,FAIL \
    --mail-user="$EMAIL" \
    "$SCRIPT"
else
  sbatch \
    --account="$ACCOUNT" \
    --partition="$ACCOUNT" \
    "$SCRIPT"
fi
