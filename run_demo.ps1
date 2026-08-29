param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $PythonExecutable) {
    $LocalPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $CondaPython = Join-Path $env:USERPROFILE "anaconda3\python.exe"

    if (Test-Path -LiteralPath $LocalPython) {
        $PythonExecutable = $LocalPython
    }
    elseif (Test-Path -LiteralPath $CondaPython) {
        $PythonExecutable = $CondaPython
    }
    else {
        $PythonExecutable = "python"
    }
}

Write-Host "Using Python: $PythonExecutable"
& $PythonExecutable -c "import numpy, pandas, sklearn, matplotlib, seaborn, joblib"
& $PythonExecutable (Join-Path $ProjectRoot "src\main.py")

