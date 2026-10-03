# Create a CUDA 12.8 environment for MingJian.
# Suitable for NVIDIA RTX 50-series / Blackwell GPUs when the driver supports CUDA 12.8+.
# Run this only when network access and disk space are available.
param(
    [switch]$BypassProxy
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($BypassProxy) {
    Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
    Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue
}

function Invoke-Checked {
    param([scriptblock]$Command)

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked { python -m venv .venv }
Invoke-Checked { & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip }
Invoke-Checked { & ".\.venv\Scripts\python.exe" -m pip install --retries 5 --timeout 60 torch torchvision --index-url https://download.pytorch.org/whl/cu128 }
Invoke-Checked { & ".\.venv\Scripts\python.exe" -m pip install --retries 5 --timeout 60 -e ".[dev]" }
Write-Output "CUDA 12.8 environment ready."
Write-Output "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Output 'Verify with: .\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
