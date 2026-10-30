# Install the reviewed CPU runtime in a fresh environment; never activate a shell.
param(
    [string]$EnvironmentPath = ".venv-cpu",
    [string]$PythonExecutable = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$environmentFullPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EnvironmentPath)
if (Test-Path -LiteralPath $environmentFullPath) {
    throw "EnvironmentPath already exists. Choose a new path; existing environments are not modified."
}

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed with exit code $LASTEXITCODE. No later step was run."
    }
}

Invoke-Checked { & $PythonExecutable -m venv $environmentFullPath }
$environmentPython = Join-Path $environmentFullPath "Scripts\python.exe"
Invoke-Checked { & $environmentPython -m pip install --retries 5 --timeout 60 "setuptools>=69" wheel }
Invoke-Checked { & $environmentPython -m pip install --retries 5 --timeout 60 "torch==2.11.0" "torchvision==0.26.0" --index-url https://download.pytorch.org/whl/cpu }
Invoke-Checked { & $environmentPython -m pip install --retries 5 --timeout 60 -r requirements-cpu.lock }
Invoke-Checked { & $environmentPython -m pip install -e ".[dev]" --no-deps --no-build-isolation }
Invoke-Checked { & $environmentPython -c "import torch, mingjian; assert torch.version.cuda is None; print('CPU runtime ready:', torch.__version__)" }
Write-Output "CPU environment ready. Run its Scripts/python.exe directly; activation is unnecessary."
Write-Output "Next: Scripts/python.exe scripts/prepare_local_model.py --help"
