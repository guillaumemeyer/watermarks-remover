# Windows: Auto-start the service at login

Run the service silently in the background on every login, without a terminal window or manual start.

## 1. Clone the repo somewhere permanent

Replace `<path-to-clone>` below with wherever you want the repo to live (e.g. `C:\watermarks-remover` or `C:\Users\<you>\watermarks-remover`). Use the same path consistently in every step below.

```powershell
git clone https://github.com/guillaumemeyer/watermarks-remover.git <path-to-clone>
```

For text cleaning (Layer B) the service needs a rewrite backend. Put its
configuration in `<path-to-clone>\.env` (see `.env.example`), for example a local
Ollama model:

```ini
WATERMARKS_REWRITE_BACKEND=ollama
WATERMARKS_REWRITE_MODEL=gemma4:26b-cpu
WATERMARKS_REWRITE_BASE_URL=http://127.0.0.1:11434
WATERMARKS_REWRITE_TIMEOUT=900
WATERMARKS_REWRITE_REASONING_EFFORT=none
```

Variables already set in the user environment win over `.env`. If you want the
optional `mlm` tactic, create `.venv` and run `make bootstrap-mlm` (or the pip
commands in the README) first: the launcher below prefers `.venv\Scripts\python.exe`
when it exists (`-Python <exe>` or `WATERMARKS_SERVICE_PYTHON` name another
interpreter, for instance the `.venv` of a different checkout).

## 2. The launcher

`service\scripts\start_service.ps1` starts the server from the repo root with
the `.venv` interpreter (or `python` on PATH), loads `.env`, sets
`HF_HUB_OFFLINE=1` unless told otherwise, and **refuses to start when another
process already listens on the port** (a second server would bind silently and
the older one would keep answering); with `-LogFile` the refusal is written to
the log too. Try it once by hand:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File <path-to-clone>\service\scripts\start_service.ps1 -LogFile logs\service.log
```

## 3a. Register a scheduled task

This does **not** require Administrator privileges — `-AtLogOn` with a user-level trigger runs under your own account, so a regular PowerShell window is enough.

```powershell
$repo = "<path-to-clone>"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$repo\service\scripts\start_service.ps1`" -LogFile logs\service.log" `
  -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName "WatermarksRemoverService" -Action $action -Trigger $trigger -Settings $settings -Description "Auto-starts the watermarks-remover HTTP service at login"
```

`-ExecutionTimeLimit ([TimeSpan]::Zero)` matters: the default stops a task after 72 hours.
`schtasks.exe /Create` also works, but its `/TR` value is limited to 261
characters, so point it at a short `.vbs` wrapper (see 3b) instead of the full
PowerShell command line.

## 3b. Or: a Startup-folder launcher (no Task Scheduler)

On machines where `Register-ScheduledTask` / `schtasks` answers "Access is
denied" (managed devices, sandboxed shells), a script in the per-user Startup
folder does the same job and needs no privilege at all. Save this as
`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\WatermarksRemoverService.vbs`:

```vbscript
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "<path-to-clone>"
sh.Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File ""<path-to-clone>\service\scripts\start_service.ps1"" -LogFile logs\service.log", 0, False
```

Windows runs it at every login of that user; delete the file to stop
auto-starting. Add `-Python ""<exe>""` before the closing quote to pin an
interpreter.

## 4. Start it immediately (no reboot needed)

```powershell
Start-ScheduledTask -TaskName "WatermarksRemoverService"   # 3a
wscript.exe "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup\WatermarksRemoverService.vbs"   # 3b
```

## 5. Verify

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
Invoke-RestMethod http://127.0.0.1:8765/capabilities | Select-Object -ExpandProperty layer_b
```

`/health` should return `{"ok": true, "version": "..."}`, and `layer_b.default_strategy_usable` should be `true` once the rewrite backend is configured.

## Notes

- Requires Python 3.10+ (on PATH, or in the repo's `.venv`).
- The launcher runs at every login going forward — no manual start needed. To restart after a `git pull`, stop the `python.exe` that serves the port (`Get-NetTCPConnection -LocalPort 8765` shows its PID) and run the launcher again.
- To stop auto-starting: `Unregister-ScheduledTask -TaskName "WatermarksRemoverService"` (3a) or delete the `.vbs` from the Startup folder (3b).
- If the launcher reports success but `/health` fails, read `logs\service.log`; a `port 8765 already taken` line names the process that got there first.
