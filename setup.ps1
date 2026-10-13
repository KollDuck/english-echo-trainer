# =============================================================
# English Echo Trainer - one-shot environment setup (Windows)
# Idempotent: safe to re-run. Designed for humans AND AI agents.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File setup.ps1 -Mirror   (China, recommended)
#   powershell -ExecutionPolicy Bypass -File setup.ps1           (direct HF access)
# =============================================================
param(
    [switch]$Mirror
)
$ErrorActionPreference = "Stop"
$RepoDir = $PSScriptRoot

Write-Host "== [1/5] Miniconda ==" -ForegroundColor Cyan
$condaDir = "$env:USERPROFILE\miniconda3"
$conda = "$condaDir\Scripts\conda.exe"
if (-not (Test-Path $conda)) {
    Write-Host "Installing Miniconda via winget ..."
    winget install --id=Anaconda.Miniconda3 -e --silent --accept-package-agreements --accept-source-agreements
}
if (-not (Test-Path $conda)) {
    throw "conda.exe not found at $conda - install Miniconda manually (see README.md)."
}
Write-Host "conda found: $conda"

Write-Host "== [2/5] conda env 'english' (python 3.11 + ffmpeg) ==" -ForegroundColor Cyan
$envPy = "$condaDir\envs\english\python.exe"
if (-not (Test-Path $envPy)) {
    & $conda create -n english python=3.11 ffmpeg --override-channels -c conda-forge -y
    if ($LASTEXITCODE -ne 0) { throw "conda create failed" }
}
Write-Host "python: $envPy"

Write-Host "== [3/5] pip dependencies ==" -ForegroundColor Cyan
# Network is slow? Use a mirror instead:
#   & $envPy -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
& $envPy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

Write-Host "== [4/5] pre-download Whisper model (large-v3) ==" -ForegroundColor Cyan
$modelArgs = @("large-v3")
if ($Mirror) { $modelArgs += "--mirror" }
& $envPy (Join-Path $RepoDir "scripts\download_model.py") @modelArgs
if ($LASTEXITCODE -ne 0) { throw "model download failed" }

Write-Host "== [5/5] GPU check ==" -ForegroundColor Cyan
& $envPy (Join-Path $RepoDir "scripts\check_gpu.py")

Write-Host ""
Write-Host "Setup complete! Launch the app:" -ForegroundColor Green
Write-Host "  - double-click start.bat, or"
Write-Host "  - `"$envPy`" -m streamlit run app.py"
