param(
    [string]$PythonExe = "python"
)

$scriptPath = Join-Path $PSScriptRoot "show_metadata.py"
& $PythonExe $scriptPath @args

