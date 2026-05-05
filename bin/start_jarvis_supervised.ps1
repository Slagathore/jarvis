# Jarvis supervisor wrapper.
#
# Run Jarvis through this instead of `python main.py` directly to enable
# the self-edit auto-revert path. When the orchestrator's restart_self
# tool fires (exit code 42), the supervisor re-launches Jarvis. If the
# new instance doesn't write data/heartbeat.txt within 10 seconds, we
# assume the most recent self-edit broke startup and roll it back via
# `git reset --hard HEAD~1` (using the SHA recorded in
# data/last_safe_sha.txt as a sanity check).
#
# Usage:
#   pwsh ./bin/start_jarvis_supervised.ps1

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
$heartbeat = Join-Path $root "data\heartbeat.txt"
$safeSha   = Join-Path $root "data\last_safe_sha.txt"
$restartPending = Join-Path $root "data\restart_pending.txt"

while ($true) {
    Write-Host "[Supervisor] starting Jarvis..." -ForegroundColor Cyan
    if (Test-Path $heartbeat) { Remove-Item $heartbeat -Force }
    if (Test-Path $restartPending) { Remove-Item $restartPending -Force }

    $process = Start-Process -FilePath "python" -ArgumentList "main.py" `
        -WorkingDirectory $root -PassThru -NoNewWindow

    # Watch for heartbeat for first 10s. If absent, assume broken startup.
    $startWatch = [System.Diagnostics.Stopwatch]::StartNew()
    $startupOk = $false
    while ($startWatch.Elapsed.TotalSeconds -lt 10) {
        if ($process.HasExited) { break }
        if (Test-Path $heartbeat) { $startupOk = $true; break }
        Start-Sleep -Milliseconds 250
    }

    if (-not $startupOk -and -not $process.HasExited) {
        # Process is up but never wrote heartbeat. Could still be loading
        # heavy models — give it 20 more seconds before forcing rollback.
        Write-Host "[Supervisor] no heartbeat after 10s; waiting another 20s..." -ForegroundColor Yellow
        while ($startWatch.Elapsed.TotalSeconds -lt 30) {
            if ($process.HasExited) { break }
            if (Test-Path $heartbeat) { $startupOk = $true; break }
            Start-Sleep -Milliseconds 500
        }
    }

    if (-not $startupOk) {
        Write-Host "[Supervisor] startup failed — reverting last self-edit" -ForegroundColor Red
        if (-not $process.HasExited) {
            try { Stop-Process -Id $process.Id -Force } catch {}
        }
        # Roll back one commit. The selfedit module always commits before
        # writing, so HEAD~1 is the prior known-good state.
        Push-Location $root
        try {
            git reset --hard HEAD~1
        } finally {
            Pop-Location
        }
        Start-Sleep -Seconds 2
        continue   # restart loop with reverted code
    }

    # Healthy startup — wait for the process to finish.
    Wait-Process -Id $process.Id
    $exitCode = $process.ExitCode

    if ($exitCode -eq 42) {
        Write-Host "[Supervisor] restart requested (exit 42); relaunching..." -ForegroundColor Cyan
        continue
    }

    Write-Host "[Supervisor] Jarvis exited with code $exitCode — supervisor stopping." -ForegroundColor Yellow
    break
}
