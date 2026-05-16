# stop.ps1 — Stop a running Jarvis instance.
#
# Writes data/stop.flag, which the orchestrator's stop-flag watcher turns
# into a real graceful shutdown (cameras released, DB closed) — and which
# the supervisor (bin/start_jarvis_supervised.ps1) consumes so it stops
# respawning instead of treating the stop as a crash.
#
# Force-kill is only a fallback, used if the graceful path doesn't finish
# within the shutdown budget.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile   = Join-Path $scriptDir "data\jarvis.pid"
$stopFlag  = Join-Path $scriptDir "data\stop.flag"

# 1. Signal graceful shutdown.
New-Item -ItemType File -Path $stopFlag -Force | Out-Null
Write-Host "Requested graceful shutdown (data/stop.flag)..." -ForegroundColor Yellow

# 2. Identify the Jarvis process — PID file first, then port 7070.
$jarvisPid = $null
if (Test-Path $pidFile) {
    try { $jarvisPid = [int](Get-Content $pidFile -Raw).Trim() } catch {}
}
if (-not $jarvisPid) {
    $conn = netstat -ano | Select-String ":7070\s+.*LISTENING"
    if ($conn) { $jarvisPid = [int](($conn -split '\s+')[-1]) }
}

$proc = $null
if ($jarvisPid) { $proc = Get-Process -Id $jarvisPid -ErrorAction SilentlyContinue }

if (-not $proc) {
    Write-Host "Jarvis is not running." -ForegroundColor Gray
    Remove-Item $stopFlag -Force -ErrorAction SilentlyContinue
    Remove-Item $pidFile  -Force -ErrorAction SilentlyContinue
    return
}

# 3. Wait for the orchestrator to run _shutdown and exit on its own.
#    Budget covers the 5s task-cancel + 8s _shutdown windows plus slack.
Write-Host "Waiting for Jarvis (PID $jarvisPid) to shut down gracefully..." -ForegroundColor Yellow
$exited = $proc.WaitForExit(20000)

# 4. Force-kill fallback.
if ($exited) {
    Write-Host "Jarvis stopped gracefully." -ForegroundColor Green
} else {
    Write-Host "Graceful shutdown timed out (20s) — force-killing." -ForegroundColor Red
    try { Stop-Process -Id $jarvisPid -Force } catch {}
    Write-Host "Jarvis stopped (forced)." -ForegroundColor Yellow
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
# stop.flag is left in place when we had to force-kill: the supervisor reads
# it to tell a real stop from a crash. The orchestrator deletes it itself on
# a graceful exit, and main.py clears any stale flag on the next start.
