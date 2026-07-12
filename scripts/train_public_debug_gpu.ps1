Write-Host "Checking environment status..." -ForegroundColor Cyan
python tools/check_env.py

$envStatus = python tools/check_env.py | Select-String "CUDA_ENV_STATUS =" | Select-Object -Last 1
if ($envStatus -notmatch "READY" -and $envStatus -notmatch "NEED_BUILD") {
    Write-Host "CUDA environment not ready. Please follow docs/GPU_SETUP.md first." -ForegroundColor Red
    exit 1
}

if ($envStatus -match "READY") {
    Write-Host "Environment is READY. Starting training..." -ForegroundColor Green
    python tools/train_one_scene.py --scene_train_path ".\work_undistorted\phase1\public_set\hcm0031\train" --model_path "outputs\public_hcm0031_undistorted_low" --iterations 1000 --resolution 8 --data_device cpu
} else {
    Write-Host "Environment is NEED_BUILD. Please run scripts/setup_cuda_extensions_windows.ps1 first." -ForegroundColor Yellow
}
