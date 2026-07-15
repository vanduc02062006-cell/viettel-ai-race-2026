param (
    [string]$ModelTemplate = "prod_{short_split}_{scene}",
    [string]$RenderRoot = "outputs\production_test_pose_renders",
    [string]$SubmissionRoot = "submission_round1_prod",
    [string]$ZipPath = "submission_round1_prod.zip"
)

Write-Host "Rendering private_set1 test poses..." -ForegroundColor Cyan
python tools/render_test_poses.py `
  --split private_set1 `
  --model_root "outputs" `
  --model_template $ModelTemplate `
  --output_root $RenderRoot `
  --output_format png `
  --antialiasing

Write-Host "Distorting private renders into submission_round1..." -ForegroundColor Cyan
python tools/distort_back_renders.py `
  --split private_set1 `
  --render_root $RenderRoot `
  --output_root $SubmissionRoot `
  --layout scene_only `
  --keep_original_extension

Write-Host "Validating and zipping private submission..." -ForegroundColor Cyan
python tools/validate_submission.py `
  --split private_set1 `
  --submission_root $SubmissionRoot `
  --layout scene_only `
  --zip_path $ZipPath
