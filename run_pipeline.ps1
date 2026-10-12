param(
    [ValidateSet("pilot", "full")]
    [string]$Profile = "pilot",
    [switch]$SkipDownload
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $taskPython)) { throw "Run setup.ps1 first." }
function Invoke-ProjectPython {
    param([string[]]$PythonArguments)
    & $taskPython @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline command failed with exit code $LASTEXITCODE" }
}
if (-not $SkipDownload) { Invoke-ProjectPython -PythonArguments @("-u", "scripts/download_data.py") }
Invoke-ProjectPython -PythonArguments @("-u", "-m", "src.prepare_data")
$taskConfig = if ($Profile -eq "pilot") { "configs/pilot.yaml" } else { "configs/train.yaml" }
$taskTrainingArgs = @("-u", "scripts/run_training.py", "--config", $taskConfig, "--name", $Profile)
if ($Profile -eq "pilot") { $taskTrainingArgs += @("--max-test-per-class", "20") }
Invoke-ProjectPython -PythonArguments $taskTrainingArgs
Write-Output "Pipeline complete. Start the interface with .venv\Scripts\python.exe -m streamlit run app.py and open http://127.0.0.1:8510"
