param(
    [string]$PythonExe = ""
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$venvPath = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    if ($PythonExe -ne "") {
        Write-Warning "Virtual environment not found, falling back to $PythonExe. Run scripts/install_dependencies.ps1 to create .venv."
        $venvPython = $PythonExe
    }
    else {
        Write-Error "Virtual environment not found at $venvPython. Run scripts/install_dependencies.ps1 first."
        exit 1
    }
}

Set-Location $projectRoot
& $venvPython -m app.main


