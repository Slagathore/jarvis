# stop.ps1 — Stop a running Jarvis instance.
# Tries the PID file first, then falls back to finding the process by port.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile   = Join-Path $scriptDir "data\jarvis.pid"

$killed = $false

# Method 1: PID file
if (Test-Path $pidFile) {
    $pid = [int](Get-Content $pidFile -Raw).Trim()
    $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Stopping Jarvis (PID $pid)..." -ForegroundColor Yellow
        Stop-Process -Id $pid -Force
        $killed = $true
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
}

# Method 2: Find any python.exe holding port 7070
if (-not $killed) {
    $conn = netstat -ano | Select-String ":7070\s+.*LISTENING"
    if ($conn) {
        $portPid = ($conn -split '\s+')[-1]
        $proc = Get-Process -Id $portPid -ErrorAction SilentlyContinue
        if ($proc -and $proc.Name -like "python*") {
            Write-Host "Stopping Jarvis on port 7070 (PID $portPid)..." -ForegroundColor Yellow
            Stop-Process -Id $portPid -Force
            $killed = $true
        }
    }
}

if ($killed) {
    Write-Host "Jarvis stopped." -ForegroundColor Green
} else {
    Write-Host "Jarvis is not running." -ForegroundColor Gray
}
