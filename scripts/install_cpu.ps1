# Create a local CPU environment for MingJian.
# Run this only when network access and disk space are available.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
Write-Output "CPU environment ready."
Write-Output "Activate with: .\.venv\Scripts\Activate.ps1"