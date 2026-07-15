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

## 3. Verify the corrected camera pipeline

```powershell
python tools/audit_distortion_roundtrip.py --split private_set1 --max_images 4
```

`HNI0131` and `HNI0265` must use `full_fov` and report less than 1% near-black pixels.

## 4. Run one public production ablation

```powershell
python tools/run_public_ablation.py --scene hcm0031 --tag r1_it30000_aa_dssim020 --iterations 30000 --resolution 1 --lambda_dssim 0.2
```

This performs train → lossless render → distort-back → public evaluation. Metrics are written to:

```text
outputs/ablations/r1_it30000_aa_dssim020/metrics.csv
```

## 5. Train all private scenes

```powershell
python tools/train_all_scenes.py --split private_set1 --iterations 30000 --resolution 1 --lambda_dssim 0.2
```

Production defaults:

- full input resolution (`-r 1`)
- 30,000 iterations
- antialiasing enabled for both training and rendering
- checkpoints at 15,000 and 30,000 iterations
- per-model and per-batch manifests

Models use this template:

```text
outputs/prod_private_<scene>
```

## 6. Render, distort, validate, and zip private submission

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_private_submission.ps1
```

The script uses PNG intermediates, rejects excessive black borders, includes exactly the expected
target images, and creates:

```text
submission_round1_prod.zip
```
