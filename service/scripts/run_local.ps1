param(
    [Parameter(Mandatory=$true, Position=0)]
    [ValidateSet('inspect','clean','academic','serve','check')]
    [string]$Mode,
    [Parameter(Position=1)][string]$Path,
    [string]$Output,
    [string]$Model = 'llama3.2:latest',
    [int]$Port = 8765
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$localPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $localPython)) {
    throw 'O ambiente .venv do projeto não foi encontrado.'
}
$arguments = @((Join-Path $PSScriptRoot 'local_workflow.py'), $Mode, '--model', $Model, '--port', "$Port")
if ($Path) { $arguments += $Path }
if ($Output) { $arguments += @('--output', $Output) }
& $localPython @arguments
exit $LASTEXITCODE
