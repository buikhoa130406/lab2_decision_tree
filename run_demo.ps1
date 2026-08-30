param(
    [string]$PythonExecutable = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not $PythonExecutable) {
    $WindowsVenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    $UnixVenvPython = Join-Path $ProjectRoot ".venv/bin/python"

    if (Test-Path -LiteralPath $WindowsVenvPython) {
        $PythonExecutable = $WindowsVenvPython
    }
    elseif (Test-Path -LiteralPath $UnixVenvPython) {
        $PythonExecutable = $UnixVenvPython
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $PythonExecutable = "python"
    }
    elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $PythonExecutable = "python3"
    }
    else {
        throw "Python was not found. Install Python 3.10+ or pass -PythonExecutable with a valid path."
    }
}

Write-Host "Using Python: $PythonExecutable"
& $PythonExecutable -c "import numpy, pandas, sklearn, matplotlib, seaborn, joblib"
if ($LASTEXITCODE -ne 0) {
    throw "Project dependencies are missing. Run: python -m pip install -r requirements.txt"
}
& $PythonExecutable (Join-Path $ProjectRoot "src\main.py")
exit $LASTEXITCODE
