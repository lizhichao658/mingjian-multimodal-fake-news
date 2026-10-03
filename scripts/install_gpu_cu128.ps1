# Create a CUDA 12.8 environment for MingJian.
# Suitable for NVIDIA RTX 50-series / Blackwell GPUs when the driver supports CUDA 12.8+.
# Run this only when network access and disk space are available.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& ".\.venv\Scripts\python.exe" -m pip install -e ".[dev]"
Write-Output "CUDA 12.8 environment ready."
Write-Output "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Output "Verify with: .\.venv\Scripts\python.exe -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""