param(
    [Parameter(Mandatory = $true)]
    [string]$Username,

    [switch]$IncludeData
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:/Users/giova/Universita/deep_learning/Masked-Autoencoders-for-Image-Representation-Learning/"
$RemoteHost = "$Username@gcluster.dmi.unict.it"
$RemoteDir = "Masked-Autoencoders-for-Image-Representation-Learning"

function Test-IsExcluded {
    param(
        [string]$RelativePath,
        [bool]$IncludeDataFlag
    )

    $p = $RelativePath.Replace('\', '/')

    if ($p -eq ".git" -or $p.StartsWith(".git/")) { return $true }
    if ($p -eq ".venv" -or $p.StartsWith(".venv/")) { return $true }
    if ($p -eq "experiments/checkpoints" -or $p.StartsWith("experiments/checkpoints/")) { return $true }
    if ($p -eq "experiments/logs" -or $p.StartsWith("experiments/logs/")) { return $true }

    if (-not $IncludeDataFlag) {
        if ($p -eq "data" -or $p.StartsWith("data/")) { return $true }
    }
    else {
        if ($p -eq "data/processed" -or $p.StartsWith("data/processed/")) { return $true }
    }

    return $false
}

function Get-IncludedFileCount {
    param(
        [string]$RootPath,
        [bool]$IncludeDataFlag
    )

    $count = 0
    $normalizedRoot = ($RootPath.TrimEnd([char[]]@('\', '/')))
    $prefixLength = $normalizedRoot.Length
    Get-ChildItem -Path $RootPath -Recurse -File | ForEach-Object {
        $full = $_.FullName
        if ($full.StartsWith($normalizedRoot)) {
            $rel = $full.Substring($prefixLength).TrimStart([char[]]@('\', '/'))
        }
        else {
            $rel = $full
        }
        if (-not (Test-IsExcluded -RelativePath $rel -IncludeDataFlag $IncludeDataFlag)) {
            $count += 1
        }
    }
    return $count
}

function Format-Bytes {
    param([double]$Bytes)

    if ($Bytes -ge 1GB) { return "{0:N2} GB" -f ($Bytes / 1GB) }
    if ($Bytes -ge 1MB) { return "{0:N2} MB" -f ($Bytes / 1MB) }
    if ($Bytes -ge 1KB) { return "{0:N2} KB" -f ($Bytes / 1KB) }
    return "{0:N0} B" -f $Bytes
}

Write-Host "Syncing repository to cluster..."

$useRsync = $null -ne (Get-Command rsync -ErrorAction SilentlyContinue)

if ($IncludeData) {
    Write-Host "Including dataset folder in sync (data/raw/imagenet100)."
    $excludeArgs = @("--exclude=.git", "--exclude=.venv", "--exclude=data/processed", "--exclude=experiments/checkpoints", "--exclude=experiments/logs")
}
else {
    Write-Host "Code-only sync (dataset excluded). Use -IncludeData to copy dataset too."
    $excludeArgs = @("--exclude=.git", "--exclude=.venv", "--exclude=data", "--exclude=experiments/checkpoints", "--exclude=experiments/logs")
}

if ($useRsync) {
    Write-Host "Using rsync for incremental sync."
    rsync -avz --delete @excludeArgs "$RepoRoot" "$RemoteHost`:$RemoteDir/"
}
else {
    Write-Host "rsync not found. Using tar+scp+ssh fallback (progress visible during upload)."
    ssh $RemoteHost "mkdir -p '$RemoteDir'"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create remote directory '$RemoteDir' on cluster."
    }

    $tarExcludes = $excludeArgs | ForEach-Object {
        $pattern = $_.Replace("--exclude=", "")
        "--exclude=$pattern"
    }

    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $localArchive = Join-Path $env:TEMP "mae-sync-$timestamp.tar.gz"
    $remoteArchive = "/tmp/mae-sync-$timestamp.tar.gz"
    $tarLog = Join-Path $env:TEMP "mae-sync-$timestamp-tar.log"
    $tarErr = Join-Path $env:TEMP "mae-sync-$timestamp-tar.err"
    $tarProc = $null
    $scpProc = $null

    $totalFiles = Get-IncludedFileCount -RootPath $RepoRoot -IncludeDataFlag $IncludeData.IsPresent
    if ($totalFiles -le 0) {
        throw "No files found to archive after applying exclusions."
    }

    try {
        Write-Host "Creating compressed archive (progress estimated by processed files)..."
        $tarArgs = @("-czvf", "$localArchive") + $tarExcludes + @(".")
        $tarProc = Start-Process -FilePath "tar" -ArgumentList $tarArgs -WorkingDirectory $RepoRoot -NoNewWindow -PassThru -RedirectStandardOutput $tarLog -RedirectStandardError $tarErr
        $lastPrintedPercent = -1

        while (-not $tarProc.HasExited) {
            $processedLines = 0
            if (Test-Path $tarLog) {
                try {
                    $processedLines = (Get-Content $tarLog -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
                }
                catch {
                    $processedLines = 0
                }
            }
            $processedFiles = [Math]::Min($processedLines, $totalFiles)
            $percent = [int](($processedFiles * 100.0) / $totalFiles)
            if ($percent -ge 100) { $percent = 99 }
            Write-Progress -Activity "Creating compressed archive" -Status "$percent% ($processedFiles/$totalFiles files)" -PercentComplete $percent
            if ($percent -ge ($lastPrintedPercent + 5)) {
                Write-Host "Archive progress: $percent% ($processedFiles/$totalFiles files)"
                $lastPrintedPercent = $percent
            }
            Start-Sleep -Milliseconds 500
            $tarProc.Refresh()
        }

        try {
            $tarProc.WaitForExit()
            $tarProc.Refresh()
        }
        catch {
        }

        $tarExitCode = 0
        if ($null -ne $tarProc.ExitCode) {
            try {
                $tarExitCode = [int]$tarProc.ExitCode
            }
            catch {
                $tarExitCode = 0
            }
        }

        if ($tarExitCode -ne 0) {
            $errText = ""
            if (Test-Path $tarErr) {
                $errText = (Get-Content $tarErr -TotalCount 30) -join "`n"
            }
            throw "Archive creation failed with exit code $tarExitCode. $errText"
        }
        Write-Progress -Activity "Creating compressed archive" -Completed
        Write-Host "Archive progress: 100% ($totalFiles/$totalFiles files)"

        Write-Host "Uploading archive with progress..."
        $archiveSize = (Get-Item "$localArchive").Length
        Write-Host "Archive size: $(Format-Bytes $archiveSize)"

        # IMPORTANT: avoid opening extra SSH sessions while scp is running.
        # With password auth, concurrent prompts can break authentication.
        scp "$localArchive" "$RemoteHost`:$remoteArchive"
        if ($LASTEXITCODE -ne 0) {
            throw "Upload failed with exit code $LASTEXITCODE."
        }
        Write-Host "Upload completed ($(Format-Bytes $archiveSize))."

        Write-Host "Extracting archive on cluster..."
        $extractOutput = ssh $RemoteHost "set -e; mkdir -p '$RemoteDir'; tar -xzf '$remoteArchive' -C '$RemoteDir'; rm -f '$remoteArchive'; find '$RemoteDir' -type f | wc -l"
        if ($LASTEXITCODE -ne 0) {
            throw "Archive extraction failed on cluster."
        }

        $remoteFileCount = 0
        if ($extractOutput) {
            $lastLine = ($extractOutput | Select-Object -Last 1).ToString().Trim()
            [void][int]::TryParse($lastLine, [ref]$remoteFileCount)
        }

        if ($remoteFileCount -le 0) {
            throw "Sync completed but remote folder appears empty (0 files)."
        }

        Write-Host "Remote sync completed with $remoteFileCount files in '$RemoteDir'."
    }
    finally {
        if ($null -ne $tarProc -and -not $tarProc.HasExited) {
            try {
                $tarProc.WaitForExit(2000) | Out-Null
            }
            catch {
            }
        }
        if ($null -ne $scpProc -and -not $scpProc.HasExited) {
            try {
                $scpProc.WaitForExit(2000) | Out-Null
            }
            catch {
            }
        }

        if (Test-Path "$localArchive") {
            try {
                Remove-Item "$localArchive" -Force -ErrorAction Stop
            }
            catch {
                Write-Warning "Could not remove temp archive: $localArchive"
            }
        }
        if (Test-Path "$tarLog") {
            try {
                Remove-Item "$tarLog" -Force -ErrorAction Stop
            }
            catch {
                Write-Warning "Could not remove temp log: $tarLog"
            }
        }
        if (Test-Path "$tarErr") {
            try {
                Remove-Item "$tarErr" -Force -ErrorAction Stop
            }
            catch {
                Write-Warning "Could not remove temp error log: $tarErr"
            }
        }
    }
}

Write-Host "Sync complete."
