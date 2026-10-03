$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& ".\.venv\Scripts\python.exe" scripts\smoke_test.py