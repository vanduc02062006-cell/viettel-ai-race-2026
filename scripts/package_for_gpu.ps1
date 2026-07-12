param (
    [switch]$include_original_dataset
)

$targetDir = "gpu_handoff"
if (Test-Path $targetDir) {
    Remove-Item -Recurse -Force $targetDir
}
New-Item -ItemType Directory -Force -Path $targetDir

Write-Host "Copying directories..." -ForegroundColor Cyan
Copy-Item -Recurse -Force "tools" -Destination "$targetDir\tools"
Copy-Item -Recurse -Force "configs" -Destination "$targetDir\configs"
Copy-Item -Recurse -Force "docs" -Destination "$targetDir\docs"
Copy-Item -Recurse -Force "scripts" -Destination "$targetDir\scripts"
Copy-Item -Recurse -Force "gaussian-splatting" -Destination "$targetDir\gaussian-splatting"
Copy-Item -Recurse -Force "work_undistorted" -Destination "$targetDir\work_undistorted"

Write-Host "Copying files..." -ForegroundColor Cyan
Copy-Item -Force "README.md" -Destination "$targetDir\README.md"
Copy-Item -Force ".gitignore" -Destination "$targetDir\.gitignore"

if ($include_original_dataset) {
    Write-Host "Copying original dataset (VAI_NVS_DATA)..." -ForegroundColor Cyan
    Copy-Item -Recurse -Force "VAI_NVS_DATA" -Destination "$targetDir\VAI_NVS_DATA"
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

4. Train public debug scene:
   `powershell -ExecutionPolicy Bypass -File scripts/train_public_debug_gpu.ps1`

5. (Next steps) Render/evaluate public.
6. (Final steps) Private render.
"@
Set-Content -Path "$targetDir\RUN_FIRST_ON_GPU.md" -Value $runFirstContent -Encoding UTF8

# Remove unwanted
Write-Host "Cleaning up unwanted files in package..." -ForegroundColor Cyan
Get-ChildItem -Path $targetDir -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force
Get-ChildItem -Path $targetDir -Recurse -Filter ".DS_Store" | Remove-Item -Force
Get-ChildItem -Path $targetDir -Recurse -Filter "__MACOSX" | Remove-Item -Recurse -Force

Write-Host "Package created at $targetDir" -ForegroundColor Green
