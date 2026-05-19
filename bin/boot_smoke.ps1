# boot_smoke.ps1 — one-shot live boot smoke test.
#
# Boots Jarvis, waits up to 150 s for the dashboard, probes a few HTTP
# endpoints (including the new Review-tab API), then force-stops it.
# Orchestrator output is captured to data/boot_smoke.{out,err}.log for
# inspection. Read-only-ish: it starts and stops Jarvis, nothing else.
#
# Usage:  pwsh -ExecutionPolicy Bypass -File bin\boot_smoke.ps1

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'

Write-Host '[boot-smoke] pre-clean (force_stop any leftover)...'
& (Join-Path $root 'force_stop.ps1') | Out-Null

$outLog = Join-Path $root 'data\boot_smoke.out.log'
$errLog = Join-Path $root 'data\boot_smoke.err.log'
foreach ($f in @($outLog, $errLog)) {
    if (Test-Path $f) { Remove-Item $f -Force }
}

$proc = Start-Process -FilePath $py -ArgumentList 'main.py' `
    -WorkingDirectory $root `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
    -PassThru -WindowStyle Hidden
Write-Host "[boot-smoke] launched main.py pid=$($proc.Id), waiting for dashboard..."

# Wait through the full cold-start model-load (~2 min: YOLO / InsightFace /
# CLIP / Whisper / YAMNet) before probing. uvicorn binds the port early but
# the shared event loop is saturated until orchestrator init finishes, so an
# early probe just times out.
$crashed = $false
for ($i = 0; $i -lt 35; $i++) {
    Start-Sleep -Seconds 5
    if ($proc.HasExited) { $crashed = $true; break }
}
$port = Get-NetTCPConnection -LocalPort 7070 -State Listen `
    -ErrorAction SilentlyContinue

if ($crashed) {
    Write-Host ("[boot-smoke] RESULT: main.py EXITED during startup " +
        "(exit=$($proc.ExitCode)) -- boot FAILED")
} elseif (-not $port) {
    Write-Host ('[boot-smoke] RESULT: dashboard port 7070 not listening ' +
        'after 175s -- boot FAILED')
} else {
    Write-Host '[boot-smoke] init window elapsed; probing endpoints...'
    foreach ($ep in @('/api/health', '/api/object_vocab/review',
                      '/api/sound_vocab/review',
                      '/api/world_model/beliefs', '/api/state')) {
        try {
            $r = Invoke-WebRequest -Uri ("http://127.0.0.1:7070" + $ep) `
                -UseBasicParsing -TimeoutSec 10
            $body = [string]$r.Content
            if ($body.Length -gt 220) { $body = $body.Substring(0, 220) }
            Write-Host "[boot-smoke] GET $ep -> HTTP $($r.StatusCode)  $body"
        } catch {
            Write-Host "[boot-smoke] GET $ep -> ERROR $_"
        }
    }
    Write-Host '[boot-smoke] RESULT: dashboard up + endpoints probed'
}

Write-Host '[boot-smoke] stopping Jarvis...'
& (Join-Path $root 'force_stop.ps1')
Write-Host '[boot-smoke] DONE'
