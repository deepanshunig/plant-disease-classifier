param(
    [string]$PythonExecutable = "py",
    [ValidateSet("cpu", "cu126", "cu128", "cu130")]
    [string]$Compute = "cpu"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Invoke-CheckedPython {
    param([string]$Executable, [string[]]$PythonArguments)
    & $Executable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Invoke-CheckedPython -Executable $PythonExecutable -PythonArguments @("-m", "venv", ".venv")
}

$taskVenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Invoke-CheckedPython -Executable $taskVenvPython -PythonArguments @("-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--force-reinstall", "--no-deps", "torch==2.9.1", "torchvision==0.24.1", "--index-url", "https://download.pytorch.org/whl/$Compute")
Invoke-CheckedPython -Executable $taskVenvPython -PythonArguments @("-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "-r", "requirements.txt")
Invoke-CheckedPython -Executable $taskVenvPython -PythonArguments @("-m", "pip", "check")
Invoke-CheckedPython -Executable $taskVenvPython -PythonArguments @("scripts/check_environment.py")
Write-Output "Setup complete. Use .venv\Scripts\python.exe to run project commands."
