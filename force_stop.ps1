# force_stop.ps1 -- Guaranteed Jarvis shutdown. The hammer.
#
# stop.ps1 asks nicely (writes data/stop.flag and waits for the orchestrator
# to unwind). When that does not work -- a wedged/starved event loop, or
# orphaned child processes left behind by an earlier crash or force-kill --
# this script finds EVERY process that belongs to Jarvis and kills it, and
# is careful to touch nothing else.
#
# HOW IT IDENTIFIES A "JARVIS PROCESS" -- the union of two independent signals:
#
#   Signal A -- the live process tree. Find the main run (its PID file, or
#     whatever is listening on the dashboard port, or a python command line
#     running main.py from THIS project dir) and expand it to every
#     descendant. Catches live children including non-python ones (piper.exe).
#
#   Signal B -- the interpreter. Any python.exe whose executable IS this
#     project's own venv interpreter (.venv\Scripts\python.exe) is Jarvis:
#     the main process and every python child/orphan it ever spawned all run
#     on that interpreter. A process from a *previous* run whose parent has
#     already died (so Signal A cannot see it) is still caught here.
#
# WHAT IT WILL NOT TOUCH: MCP servers run from a different venv; system and
# other-project python have different executables -- none match Signal B.
# VS Code's Python extension can transiently borrow this venv to probe the
# interpreter; those are excluded by their command line (vscode / pylance /
# ms-python / debugpy). Non-python processes are never considered.
#
# Usage:  .\force_stop.ps1
#
# ASCII-ONLY ON PURPOSE: a .ps1 with non-ASCII characters and no byte-order
# mark is mis-decoded by Windows PowerShell 5.1 (it assumes the ANSI code
# page), which corrupts the parse. Keep this file 7-bit ASCII so it runs
# identically under Windows PowerShell 5.1 and PowerShell 7+.

$ErrorActionPreference = 'SilentlyContinue'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile   = Join-Path $scriptDir 'data\jarvis.pid'
$stopFlag  = Join-Path $scriptDir 'data\stop.flag'
$dashPort  = 7070

# Canonical path to this project's venv python -- the Signal B fingerprint.
$venvPy = $null
$venvCandidate = Join-Path $scriptDir '.venv\Scripts\python.exe'
if (Test-Path $venvCandidate) {
    $venvPy = (Resolve-Path $venvCandidate).Path
} else {
    Write-Host "[force_stop] note: $venvCandidate not found - using process-tree signal only." -ForegroundColor DarkYellow
}

function Test-DevTooling($cmd) {
    # True for VS Code / language-server / debug processes that may borrow
    # the project venv -- never kill these.
    if (-not $cmd) { return $false }
    return ($cmd -match 'vscode' -or $cmd -match 'ms-python' -or
            $cmd -match 'pylance' -or $cmd -match 'debugpy' -or
            $cmd -match 'Claude Extensions' -or $cmd -match 'language.?server')
}

# -- 1. Give the graceful path a brief, last courtesy -----------------------
New-Item -ItemType File -Path $stopFlag -Force | Out-Null
Write-Host '[force_stop] wrote data/stop.flag - 5s grace for a clean exit...' -ForegroundColor Yellow
Start-Sleep -Seconds 5

# -- 2. Snapshot every python.exe (PID, parent, command line, executable) ---
$allPy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'")

# Signal A -- roots of the live Jarvis process tree.
$rootPids = New-Object System.Collections.Generic.List[int]
if (Test-Path $pidFile) {
    $parsed = 0
    if ([int]::TryParse((Get-Content $pidFile -Raw).Trim(), [ref]$parsed)) {
        $rootPids.Add($parsed)
    }
}
$conn = Get-NetTCPConnection -LocalPort $dashPort -State Listen -ErrorAction SilentlyContinue
if ($conn) { foreach ($c in $conn) { $rootPids.Add([int]$c.OwningProcess) } }
$escDir = [regex]::Escape($scriptDir)
foreach ($p in $allPy) {
    if ($p.CommandLine -and $p.CommandLine -match 'main\.py' -and
        $p.CommandLine -match $escDir) {
        $rootPids.Add([int]$p.ProcessId)
    }
}

# Expand each root PID to its full descendant tree.
$childrenOf = @{}
foreach ($p in $allPy) {
    $par = [int]$p.ParentProcessId
    if (-not $childrenOf.ContainsKey($par)) { $childrenOf[$par] = New-Object System.Collections.Generic.List[int] }
    $childrenOf[$par].Add([int]$p.ProcessId)
}
$victims = New-Object System.Collections.Generic.HashSet[int]
function Add-Subtree([int]$root, $childrenOf, $victims) {
    if (-not $victims.Add($root)) { return }
    if ($childrenOf.ContainsKey($root)) {
        foreach ($child in $childrenOf[$root]) { Add-Subtree $child $childrenOf $victims }
    }
}
foreach ($r in $rootPids) { Add-Subtree $r $childrenOf $victims }

# Signal B -- any python on this project's venv interpreter (minus dev tooling).
if ($venvPy) {
    foreach ($p in $allPy) {
        if ($p.ExecutablePath -and ($p.ExecutablePath -ieq $venvPy) -and
            -not (Test-DevTooling $p.CommandLine)) {
            [void]$victims.Add([int]$p.ProcessId)
        }
    }
}

# -- 3. Kill every identified process ---------------------------------------
if ($victims.Count -eq 0) {
    Write-Host '[force_stop] No Jarvis processes found - already stopped.' -ForegroundColor Green
} else {
    Write-Host "[force_stop] killing $($victims.Count) Jarvis process(es): $($victims -join ', ')" -ForegroundColor Red
    foreach ($vp in $victims) {
        Stop-Process -Id $vp -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# -- 4. Clear the runtime files so the next start is clean ------------------
Remove-Item $pidFile, $stopFlag -Force -ErrorAction SilentlyContinue

# -- 5. Verify nothing survived ---------------------------------------------
$stillListening = [bool](Get-NetTCPConnection -LocalPort $dashPort -State Listen -ErrorAction SilentlyContinue)
$leftover = 0
if ($venvPy) {
    $leftover = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.ExecutablePath -ieq $venvPy -and -not (Test-DevTooling $_.CommandLine) }).Count
}
if ($stillListening -or $leftover -gt 0) {
    Write-Host "[force_stop] WARNING - survivors: port ${dashPort} listening=$stillListening, venv-python leftover=$leftover" -ForegroundColor Red
    Write-Host '[force_stop] re-run, or investigate - something is respawning Jarvis.' -ForegroundColor Red
    exit 1
}
Write-Host '[force_stop] Jarvis is fully stopped. Clean slate.' -ForegroundColor Green
