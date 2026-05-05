# Jarvis supervisor wrapper.
#
# Run Jarvis through this instead of `python main.py` directly to enable
# auto-restart and the self-edit auto-revert path.
#
# Behavior:
#   - Cold start (you just ran the script): launch Jarvis and wait. No
#     time limit on startup; the supervisor never touches git here.
#   - Crash during normal running: just relaunch. No git changes.
#   - Self-edit restart (Jarvis exited with code 42): relaunch and watch
#     for data/heartbeat.txt for 60 seconds. If absent AND the most
#     recent commit message starts with 'selfedit:', revert that one
#     commit and try again. ANY other state — non-selfedit last commit,
#     unclear cause — supervisor refuses to revert and exits so Cole
#     can investigate manually.
#
# Usage:
#   pwsh ./bin/start_jarvis_supervised.ps1

$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot/.."
$heartbeat = Join-Path $root "data\heartbeat.txt"
$safeSha   = Join-Path $root "data\last_safe_sha.txt"
$restartPending = Join-Path $root "data\restart_pending.txt"

# Outer-loop state. $justSelfEdited is the ONLY signal that gates the
# heartbeat-or-revert dance. Set true only when Jarvis exits with code 42.
$justSelfEdited = $false

while ($true) {
    Write-Host "[Supervisor] starting Jarvis..." -ForegroundColor Cyan
    if (Test-Path $heartbeat) { Remove-Item $heartbeat -Force }
    if (Test-Path $restartPending) { Remove-Item $restartPending -Force }

    $process = Start-Process -FilePath "python" -ArgumentList "main.py" `
        -WorkingDirectory $root -PassThru -NoNewWindow

    if ($justSelfEdited) {
        # Post-selfedit restart: enforce a heartbeat within 60 seconds. If
        # the new code can't even boot, the self-edit broke something.
        Write-Host "[Supervisor] post-selfedit restart — watching for heartbeat (60s)..." -ForegroundColor Yellow
        $startWatch = [System.Diagnostics.Stopwatch]::StartNew()
        $startupOk = $false
        while ($startWatch.Elapsed.TotalSeconds -lt 60) {
            if ($process.HasExited) { break }
            if (Test-Path $heartbeat) { $startupOk = $true; break }
            Start-Sleep -Milliseconds 500
        }

        if (-not $startupOk) {
            Write-Host "[Supervisor] post-selfedit startup failed — examining HEAD" -ForegroundColor Red
            if (-not $process.HasExited) {
                try { Stop-Process -Id $process.Id -Force } catch {}
            }

            # CRITICAL: only revert if the most recent commit was authored
            # by the self-edit module itself. Cole's commits never get
            # touched by this supervisor. The selfedit module always commits
            # with a message starting 'selfedit:' (see modules/selfedit/edit.py
            # _git_snapshot), so that prefix is the safe-to-revert marker.
            Push-Location $root
            try {
                $lastMsg = git log -1 --pretty=%s
                if ($lastMsg -like "selfedit:*") {
                    Write-Host "[Supervisor] reverting selfedit commit: $lastMsg" -ForegroundColor Red
                    git reset --hard HEAD~1
                    $justSelfEdited = $false
                    Pop-Location
                    Start-Sleep -Seconds 2
                    continue
                } else {
                    Write-Host "[Supervisor] last commit is NOT a selfedit ('$lastMsg') — refusing to revert." -ForegroundColor Red
                    Write-Host "[Supervisor] something else is wrong with startup. Stopping supervisor; investigate manually." -ForegroundColor Red
                    Pop-Location
                    break
                }
            } catch {
                Pop-Location
                break
            }
        }

        # Heartbeat received — selfedit didn't break startup, clear the flag
        $justSelfEdited = $false
        Write-Host "[Supervisor] heartbeat received; selfedit applied successfully." -ForegroundColor Green
    } else {
        # Cold start or crash recovery: do not enforce a heartbeat deadline.
        # Whisper + sentence-transformers + face-encoder loads can take a
        # full minute on a fresh boot, and we never want the supervisor to
        # mistake slow startup for a broken self-edit.
        Write-Host "[Supervisor] cold start — Jarvis owns its own startup time. No git enforcement." -ForegroundColor DarkGray
    }

    # Wait for the process to finish, no matter how long that takes.
    Wait-Process -Id $process.Id
    $exitCode = $process.ExitCode

    if ($exitCode -eq 42) {
        # Self-edit's restart_self — engage the heartbeat-or-revert dance.
        Write-Host "[Supervisor] self-edit restart requested; relaunching with heartbeat watch..." -ForegroundColor Cyan
        $justSelfEdited = $true
        continue
    }

    if ($exitCode -eq 43) {
        # Dashboard-triggered restart — plain relaunch. NO heartbeat
        # enforcement, NO revert path. The user explicitly asked Jarvis to
        # restart; nothing about that should ever touch git.
        Write-Host "[Supervisor] dashboard restart requested; plain relaunch (no heartbeat watch, no revert)..." -ForegroundColor Cyan
        $justSelfEdited = $false
        continue
    }

    if ($exitCode -ne 0) {
        Write-Host "[Supervisor] Jarvis crashed (exit $exitCode); relaunching after 3s." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
        continue
    }

    Write-Host "[Supervisor] Jarvis exited cleanly. Supervisor stopping." -ForegroundColor Yellow
    break
}
