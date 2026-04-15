param(
    [Parameter(Mandatory = $true)]
    [string]$Username
)

$ErrorActionPreference = "Stop"

$LocalRepo = "C:/Users/giova/Universita/deep_learning/Masked-Autoencoders-for-Image-Representation-Learning/"
$RemoteHost = "$Username@gcluster.dmi.unict.it"
$RemoteRepo = "Masked-Autoencoders-for-Image-Representation-Learning"

Write-Host "Syncing results from cluster to local..."

$useRsync = $null -ne (Get-Command rsync -ErrorAction SilentlyContinue)

if ($useRsync) {
    # Pull back only logs/checkpoints/processed reports.
    rsync -avz "$RemoteHost`:$RemoteRepo/experiments/logs/" "$LocalRepo/experiments/logs/"
    rsync -avz "$RemoteHost`:$RemoteRepo/experiments/checkpoints/" "$LocalRepo/experiments/checkpoints/"
    rsync -avz "$RemoteHost`:$RemoteRepo/data/processed/" "$LocalRepo/data/processed/"
}
else {
    Write-Host "rsync not found. Using scp fallback."
    New-Item -ItemType Directory -Force -Path "$LocalRepo/experiments/logs" | Out-Null
    New-Item -ItemType Directory -Force -Path "$LocalRepo/experiments/checkpoints" | Out-Null
    New-Item -ItemType Directory -Force -Path "$LocalRepo/data/processed" | Out-Null

    scp -r "$RemoteHost`:$RemoteRepo/experiments/logs/*" "$LocalRepo/experiments/logs/"
    scp -r "$RemoteHost`:$RemoteRepo/experiments/checkpoints/*" "$LocalRepo/experiments/checkpoints/"
    scp -r "$RemoteHost`:$RemoteRepo/data/processed/*" "$LocalRepo/data/processed/"
}

Write-Host "Sync complete."
