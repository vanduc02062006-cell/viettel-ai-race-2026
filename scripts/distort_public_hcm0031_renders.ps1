Write-Host "Distorting public_set/hcm0031 renders back to original camera frame..." -ForegroundColor Cyan

python tools/distort_back_renders.py `
  --split public_set `
  --scene hcm0031 `
  --render_root "outputs\test_pose_renders" `
  --output_root "outputs\test_pose_renders_distorted" `
  --keep_original_extension
