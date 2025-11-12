param(
    [string]$PythonExe = "python"
)

$scriptPath = Join-Path $PSScriptRoot "prewarm_metadata.py"
& $PythonExe $scriptPath @args

