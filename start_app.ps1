param(
    [ValidateRange(1024, 65535)]
    [int]$Port = 8510
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Run setup.ps1 first.' }
if (-not (Test-Path -LiteralPath 'artifacts\model_bundle\model.pt')) { throw 'Train and package a model first.' }
Write-Output "Open http://127.0.0.1:$Port to use the classifier."
& $taskPython -m streamlit run app.py --server.port $Port
if ($LASTEXITCODE -ne 0) { throw "The app exited with code $LASTEXITCODE" }
