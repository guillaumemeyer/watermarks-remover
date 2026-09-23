<#
.SYNOPSIS
Start the watermarks-remover HTTP service on Windows with the right interpreter
and environment, refusing to double-bind the port.

.DESCRIPTION
Runs service/scripts/server.py from the repository root so the default Layer B
strategy (config/clean_strategy.json) resolves. Picks the repo's .venv
interpreter when it exists (that is where `make bootstrap-mlm` installs the
`mlm` stack), otherwise the first `python` on PATH; -Python or the
WATERMARKS_SERVICE_PYTHON variable names another interpreter explicitly.

Environment, in order of precedence: the caller's environment, then the
repository's `.env` file (KEY=VALUE lines; `#` comments; already-set variables
are never overwritten). HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE default to 1 so a
cached roberta-large never triggers a Hub round trip (TLS-intercepted networks
break it); unset them in .env or the environment to allow downloads.

A second server on an already-listening port binds silently on Windows and the
older process keeps answering, so this script exits with status 3 when the port
is taken and prints the owning process. Use -Force to skip that check.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File service\scripts\start_service.ps1
.EXAMPLE
service\scripts\start_service.ps1 -Port 18765 -LogFile logs\service-18765.log
#>
[CmdletBinding()]
param(
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 8765,
    [string]$LogFile,
    [string]$Python,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
Set-Location -LiteralPath $root

# --- .env (never overrides what the caller already set) ---------------------
$envFile = Join-Path $root '.env'
if (Test-Path -LiteralPath $envFile) {
    foreach ($line in Get-Content -LiteralPath $envFile) {
        $trim = $line.Trim()
        if (-not $trim -or $trim.StartsWith('#')) { continue }
        $eq = $trim.IndexOf('=')
        if ($eq -lt 1) { continue }
        $key = $trim.Substring(0, $eq).Trim()
        $value = $trim.Substring($eq + 1).Trim()
        if ($value.Length -ge 2 -and (($value[0] -eq '"' -and $value[-1] -eq '"') -or ($value[0] -eq "'" -and $value[-1] -eq "'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not $value) { continue }
        if (-not [Environment]::GetEnvironmentVariable($key, 'Process')) {
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
}
foreach ($offline in 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE') {
    if (-not [Environment]::GetEnvironmentVariable($offline, 'Process')) {
        [Environment]::SetEnvironmentVariable($offline, '1', 'Process')
    }
}
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'

# --- interpreter -------------------------------------------------------------
# -Python, then WATERMARKS_SERVICE_PYTHON, then the repo's .venv, then PATH.
$python = if ($Python) { $Python } elseif ($env:WATERMARKS_SERVICE_PYTHON) { $env:WATERMARKS_SERVICE_PYTHON } else { Join-Path $root '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $python)) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw 'No .venv\Scripts\python.exe in the repository and no python on PATH.' }
    $python = $cmd.Source
}

# --- port check ----------------------------------------------------------------
if (-not $Force) {
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($listeners.Count -gt 0) {
        foreach ($l in $listeners) {
            $p = Get-CimInstance Win32_Process -Filter "ProcessId=$($l.OwningProcess)" -ErrorAction SilentlyContinue
            Write-Error -ErrorAction Continue ("port {0} is already taken by PID {1}: {2}" -f $Port, $l.OwningProcess, $p.CommandLine)
        }
        Write-Error -ErrorAction Continue 'Refusing to start a second server on the same port (the older one would keep answering). Stop it first or pass -Force.'
        if ($LogFile) {
            $logPath = if ([IO.Path]::IsPathRooted($LogFile)) { $LogFile } else { Join-Path $root $LogFile }
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
            Add-Content -LiteralPath $logPath -Value ("[{0}] start_service.ps1: port {1} already taken by PID {2}; not starting a second server" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Port, (($listeners | ForEach-Object { $_.OwningProcess }) -join ','))
        }
        exit 3
    }
}

# --- start -----------------------------------------------------------------------
# Native stderr lines (the server's own warnings) must not become terminating errors.
$ErrorActionPreference = 'Continue'
$args = @((Join-Path $root 'service\scripts\server.py'), '--host', $BindHost, '--port', "$Port")
if ($LogFile) {
    $logPath = if ([IO.Path]::IsPathRooted($LogFile)) { $LogFile } else { Join-Path $root $LogFile }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null
    Write-Host ("watermarks-remover: {0} on http://{1}:{2} (model {3}; log {4})" -f $python, $BindHost, $Port, $env:WATERMARKS_REWRITE_MODEL, $logPath)
    & $python -X utf8 @args *>> $logPath
} else {
    Write-Host ("watermarks-remover: {0} on http://{1}:{2} (model {3})" -f $python, $BindHost, $Port, $env:WATERMARKS_REWRITE_MODEL)
    & $python -X utf8 @args
}
exit $LASTEXITCODE
