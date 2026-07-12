Write-Host "WARNING: This script should only be run on a machine with an NVIDIA GPU." -ForegroundColor Yellow
Write-Host "WARNING: It requires CUDA Toolkit, nvcc, and Visual Studio Build Tools C++ or Developer PowerShell." -ForegroundColor Yellow
Write-Host "WARNING: Make sure you are running this from a suitable Python environment." -ForegroundColor Yellow

$env:DISTUTILS_USE_SDK = "1"

Write-Host "Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing diff-gaussian-rasterization..." -ForegroundColor Cyan
cd gaussian-splatting
pip install -e submodules/diff-gaussian-rasterization

Write-Host "Installing simple-knn..." -ForegroundColor Cyan
pip install -e submodules/simple-knn
cd ..

Write-Host "Running environment check..." -ForegroundColor Cyan
python tools/check_env.py

Write-Host "If import still fails, please check:" -ForegroundColor Yellow
Write-Host "- nvcc --version output"
Write-Host "- where cl (make sure Visual Studio C++ build tools are active)"
Write-Host "- torch CUDA version matches your CUDA Toolkit"
