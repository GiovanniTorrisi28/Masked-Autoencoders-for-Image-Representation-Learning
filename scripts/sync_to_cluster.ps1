# Upload del progetto sul cluster usando scp (nativo su Windows 10/11).
# Usage: .\scripts\sync_to_cluster.ps1 -User TRRGNN02A28C351N
#
# Eseguire da PowerShell nella root del progetto:
#   cd C:\Users\giova\Universita\deep_learning\Masked-Autoencoders-for-Image-Representation-Learning
#   .\scripts\sync_to_cluster.ps1 -User TRRGNN02A28C351N

param(
    [Parameter(Mandatory=$true)]
    [string]$User
)

$HOST_CLUSTER = "gcluster.dmi.unict.it"
$REMOTE = "${User}@${HOST_CLUSTER}:~/Masked-Autoencoders-for-Image-Representation-Learning"

Write-Host "Uploading to ${REMOTE} ..."
Write-Host ""

# Directories to upload (destination = root of remote project)
$dirs = @(
    "src",
    "scripts",
    "notebooks"
)

# Root-level Python files
$rootFiles = @(
    "train_mae.py",
    "train_supervised.py",
    "train_linear_probe.py",
    "evaluate.py"
)

# Ensure remote directories exist
Write-Host "Creating remote directory structure..."
ssh "${User}@${HOST_CLUSTER}" "mkdir -p ~/Masked-Autoencoders-for-Image-Representation-Learning/experiments/configs ~/Masked-Autoencoders-for-Image-Representation-Learning/scripts/slurm ~/Masked-Autoencoders-for-Image-Representation-Learning/slurm_logs"

# Upload directories
foreach ($dir in $dirs) {
    if (Test-Path $dir) {
        Write-Host "Uploading $dir/ ..."
        scp -r $dir "${REMOTE}/"
    }
}

# experiments/configs needs explicit destination to land in the right place
Write-Host "Uploading experiments/configs/ ..."
scp -r experiments\configs "${User}@${HOST_CLUSTER}:~/Masked-Autoencoders-for-Image-Representation-Learning/experiments/"

# Upload root-level Python files
foreach ($file in $rootFiles) {
    if (Test-Path $file) {
        Write-Host "Uploading $file ..."
        scp $file "${REMOTE}/"
    }
}

# Upload .env (contains WANDB_API_KEY)
if (Test-Path ".env") {
    Write-Host "Uploading .env ..."
    scp ".env" "${REMOTE}/"
}

Write-Host ""
Write-Host "Done. Files uploaded to ${REMOTE}"
Write-Host ""
Write-Host "Next steps (from the cluster login node):"
Write-Host "  ssh ${User}@${HOST_CLUSTER}"
Write-Host "  cd ~/Masked-Autoencoders-for-Image-Representation-Learning"
Write-Host "  sbatch scripts/slurm/smoke_test_supervised.sh"
Write-Host "  sbatch scripts/slurm/smoke_test_mae.sh"
