param (
    [string]$ModelTemplate = "{short_split}_{scene}",
    [string]$ZipPath = "submission_round1.zip"
)

Write-Host "Rendering private_set1 test poses..." -ForegroundColor Cyan
python tools/render_test_poses.py `
  --split private_set1 `
  --model_root "outputs" `
  --model_template $ModelTemplate `
  --output_root "outputs\test_pose_renders"

Write-Host "Distorting private renders into submission_round1..." -ForegroundColor Cyan
python tools/distort_back_renders.py `
  --split private_set1 `
  --render_root "outputs\test_pose_renders" `
  --output_root "submission_round1" `
  --layout split_scene `
  --keep_original_extension

Write-Host "Validating and zipping private submission..." -ForegroundColor Cyan
python tools/validate_submission.py `
  --split private_set1 `
  --submission_root "submission_round1" `
  --layout split_scene `
  --zip_path $ZipPath
