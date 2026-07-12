# Step-by-step GPU workflow

Run these commands from the repository root.

## 1. Check GPU environment

```powershell
python tools/check_env.py
```

Expected final status after setup: `CUDA_ENV_STATUS = READY`.

## 2. Build CUDA extensions

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_cuda_extensions_windows.ps1
```

Linux:

```bash
bash scripts/setup_cuda_extensions_linux.sh
```

Run the environment check again after building.

## 3. Train one public debug scene

```powershell
powershell -ExecutionPolicy Bypass -File scripts/train_public_debug_gpu.ps1
```

This writes the model to:

```text
outputs/public_hcm0031_undistorted_low
```

## 4. Render public test poses

```powershell
powershell -ExecutionPolicy Bypass -File scripts/render_public_hcm0031_test_poses.ps1
```

Raw pinhole renders are written to:

```text
outputs/test_pose_renders/public_set/hcm0031
```

## 5. Distort public renders back to the original frame

```powershell
powershell -ExecutionPolicy Bypass -File scripts/distort_public_hcm0031_renders.ps1
```

Distorted renders are written to:

```text
outputs/test_pose_renders_distorted/public_set/hcm0031
```

## 6. Evaluate public debug renders

```powershell
powershell -ExecutionPolicy Bypass -File scripts/evaluate_public_hcm0031.ps1
```

Metrics are written to:

```text
outputs/evaluation/public_hcm0031_metrics.csv
```

## 7. Train all private scenes

Train each private scene into a model folder matching this template:

```text
outputs/private_<scene>
```

Examples:

```text
outputs/private_HCM0249
outputs/private_HCM0254
outputs/private_HNI0437
```

## 8. Render, distort, validate, and zip private submission

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_private_submission.ps1
```

By default this expects model folders named with template:

```text
{short_split}_{scene}
```

For `private_set1/HCM0249`, that becomes:

```text
outputs/private_HCM0249
```

The final zip is:

```text
submission_round1.zip
```
