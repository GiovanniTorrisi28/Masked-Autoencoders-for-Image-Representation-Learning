# Download checkpoints e log dal cluster.
# Usage: .\scripts\sync_from_cluster.ps1 -User TRRGNN02A28C351N

param(
    [Parameter(Mandatory=$true)]
    [string]$User
)

$HOST_CLUSTER = "gcluster.dmi.unict.it"
$REMOTE = "${User}@${HOST_CLUSTER}:~/Masked-Autoencoders-for-Image-Representation-Learning"

Write-Host "Downloading from ${REMOTE} ..."

# Download experiments (checkpoints + logs)
Write-Host "Downloading experiments/ ..."
scp -r "${REMOTE}/experiments" .

# Download wandb offline runs
Write-Host "Downloading wandb/ ..."
scp -r "${REMOTE}/wandb" . 2>$null

Write-Host ""
Write-Host "Done."
Write-Host "To view TensorBoard: tensorboard --logdir experiments/logs"
Write-Host "To sync W&B: wandb sync wandb/offline-run-*/"
