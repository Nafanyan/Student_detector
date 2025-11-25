param(
    [string]$PythonExe = "python"
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $projectRoot ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment in $venvPath"
    & $PythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirements = Join-Path $projectRoot "requirements.txt"

Write-Host "Upgrading pip in virtual environment..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from $requirements ..."
& $venvPython -m pip install -r $requirements



