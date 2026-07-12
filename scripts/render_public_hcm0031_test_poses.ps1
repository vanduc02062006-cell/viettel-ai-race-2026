Write-Host "Rendering public_set/hcm0031 test poses..." -ForegroundColor Cyan

python tools/render_test_poses.py `
  --split public_set `
  --scene hcm0031 `
  --model_path "outputs\public_hcm0031_undistorted_low" `
  --output_root "outputs\test_pose_renders"
