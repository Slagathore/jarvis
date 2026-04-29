# start.ps1 — Launch Jarvis in the current terminal window.
# Activate the venv if not already active, then run main.py.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy    = Join-Path $scriptDir ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Error "Virtual environment not found at .venv\. Run: python -m venv .venv && .venv\Scripts\Activate.ps1 && pip install -r requirements.txt"
    exit 1
}

# Kill any existing instance first
& "$scriptDir\stop.ps1"

Write-Host "Starting Jarvis..." -ForegroundColor Cyan
& $venvPy "$scriptDir\main.py"
