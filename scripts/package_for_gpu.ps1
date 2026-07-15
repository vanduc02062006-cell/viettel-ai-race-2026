param (
    [string]$TargetDir = "gpu_handoff_prod_v3",
    [switch]$include_original_dataset
)

if (Test-Path $TargetDir) {
    Remove-Item -Recurse -Force $TargetDir
}
New-Item -ItemType Directory -Force -Path $TargetDir

Write-Host "Copying directories..." -ForegroundColor Cyan
Copy-Item -Recurse -Force "tools" -Destination "$TargetDir\tools"
Copy-Item -Recurse -Force "tests" -Destination "$TargetDir\tests"
Copy-Item -Recurse -Force "configs" -Destination "$TargetDir\configs"
Copy-Item -Recurse -Force "docs" -Destination "$TargetDir\docs"
Copy-Item -Recurse -Force "scripts" -Destination "$TargetDir\scripts"
Copy-Item -Recurse -Force "gaussian-splatting" -Destination "$TargetDir\gaussian-splatting"
Copy-Item -Recurse -Force "work_undistorted" -Destination "$TargetDir\work_undistorted"

Write-Host "Copying files..." -ForegroundColor Cyan
Copy-Item -Force "README.md" -Destination "$TargetDir\README.md"
Copy-Item -Force "KAGGLE_RUNBOOK.md" -Destination "$TargetDir\KAGGLE_RUNBOOK.md"
Copy-Item -Force ".gitignore" -Destination "$TargetDir\.gitignore"
New-Item -ItemType Directory -Force -Path "$TargetDir\logs" | Out-Null
if (Test-Path "outputs\evaluation\private_distortion_roundtrip.csv") {
    Copy-Item -Force "outputs\evaluation\private_distortion_roundtrip.csv" -Destination "$TargetDir\logs\distortion_roundtrip_reference.csv"
}

if ($include_original_dataset) {
    Write-Host "Copying original dataset (VAI_NVS_DATA)..." -ForegroundColor Cyan
    Copy-Item -Recurse -Force "VAI_NVS_DATA" -Destination "$TargetDir\VAI_NVS_DATA"
}

# Create RUN_FIRST_ON_GPU.md
$runFirstContent = @"
# GPU Execution Workflow

1. Check environment:
   `python tools/check_env.py`

2. Build extensions:
   - Windows: `powershell -ExecutionPolicy Bypass -File scripts/setup_cuda_extensions_windows.ps1`
   - Linux: `bash scripts/setup_cuda_extensions_linux.sh`

3. Verify environment is READY:
   `python tools/check_env.py`

4. Verify corrected camera metadata and test poses:
   `python tools/inspect_undistorted_dataset.py --data_root work_undistorted`

5. Run the public production ablation:
   `python tools/run_public_ablation.py --scene hcm0031 --tag r1_it30000_aa_dssim020 --iterations 30000 --resolution 1 --lambda_dssim 0.2`

6. Train private production models:
   `python tools/train_all_scenes.py --split private_set1 --iterations 30000 --resolution 1 --lambda_dssim 0.2`

7. Follow KAGGLE_RUNBOOK.md to render and validate the ZIP.
"@
Set-Content -Path "$TargetDir\RUN_FIRST_ON_GPU.md" -Value $runFirstContent -Encoding UTF8

# Remove unwanted
Write-Host "Cleaning up unwanted files in package..." -ForegroundColor Cyan
Get-ChildItem -Path $TargetDir -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $TargetDir -Recurse -Filter ".DS_Store" | Remove-Item -Force
Get-ChildItem -Path $TargetDir -Recurse -Filter "__MACOSX" | Remove-Item -Recurse -Force

Write-Host "Package created at $TargetDir" -ForegroundColor Green
