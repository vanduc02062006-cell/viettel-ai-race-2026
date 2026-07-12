Write-Host "Evaluating public_set/hcm0031 distorted renders..." -ForegroundColor Cyan

python tools/evaluate_public_renders.py `
  --scene hcm0031 `
  --pred_root "outputs\test_pose_renders_distorted" `
  --output_csv "outputs\evaluation\public_hcm0031_metrics.csv"
