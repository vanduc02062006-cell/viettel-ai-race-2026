# Kaggle Runbook

This repository contains the code pipeline for Viettel AI Race 2026 Novel View Synthesis.

Large competition data, trained models, Kaggle handoff zips, outputs, and submissions are intentionally not stored in GitHub.

## Expected Kaggle Input

Upload the clean Kaggle dataset package separately, for example:

```text
vai_nvs_kaggle_clean_v2.zip
```

Kaggle may auto-extract the zip under `/kaggle/input`.

## Setup

```python
from pathlib import Path
import shutil

input_root = Path("/kaggle/input")
work_dir = Path("/kaggle/working/vai_nvs")

if work_dir.exists():
    shutil.rmtree(work_dir)

candidates = []
for p in input_root.rglob("gaussian-splatting"):
    parent = p.parent
    if (parent / "tools").exists() and (parent / "scripts").exists() and (parent / "work_undistorted").exists():
        candidates.append(parent)

assert candidates, "Could not find extracted project under /kaggle/input"
shutil.copytree(candidates[0], work_dir)
```

```python
%cd /kaggle/working/vai_nvs
```

```bash
python -m pip install -U pip "setuptools<82" wheel ninja
pip install tqdm plyfile opencv-python pillow scipy matplotlib
bash scripts/setup_cuda_extensions_linux.sh
```

## Prune COLMAP Image Entries

```python
from pathlib import Path
import subprocess

root = Path("/kaggle/working/vai_nvs/work_undistorted/phase1")
for split in ["public_set", "private_set1"]:
    for scene in sorted((root / split).iterdir()):
        if scene.is_dir():
            subprocess.run([
                "python", "tools/prune_colmap_images_to_existing.py",
                "--scene_train_path", str(scene / "train"),
            ], check=True)
```

## Public Smoke Test

```bash
python gaussian-splatting/train.py \
  -s work_undistorted/phase1/public_set/hcm0031/train \
  -m outputs/public_hcm0031_r4_it7000 \
  --iterations 7000 \
  --save_iterations 7000 \
  -r 4 \
  --data_device cpu

python tools/render_test_poses.py \
  --split public_set \
  --scene hcm0031 \
  --model_path outputs/public_hcm0031_r4_it7000 \
  --output_root outputs/test_pose_renders

python tools/distort_back_renders.py \
  --split public_set \
  --scene hcm0031 \
  --render_root outputs/test_pose_renders \
  --output_root outputs/test_pose_renders_distorted \
  --keep_original_extension

python tools/evaluate_public_renders.py \
  --scene hcm0031 \
  --pred_root outputs/test_pose_renders_distorted \
  --layout split_scene
```

## Private Submission

Train one model per private scene:

```bash
for scene_dir in work_undistorted/phase1/private_set1/*; do
  scene=$(basename "$scene_dir")
  python gaussian-splatting/train.py \
    -s "$scene_dir/train" \
    -m "outputs/private_$scene" \
    --iterations 7000 \
    --save_iterations 7000 \
    -r 4 \
    --data_device cpu
done
```

Render and validate:

```bash
python tools/render_test_poses.py \
  --split private_set1 \
  --model_root outputs \
  --model_template "private_{scene}" \
  --output_root outputs/test_pose_renders

python tools/distort_back_renders.py \
  --split private_set1 \
  --render_root outputs/test_pose_renders \
  --output_root submission_round1 \
  --layout scene_only \
  --keep_original_extension

python tools/validate_submission.py \
  --split private_set1 \
  --submission_root submission_round1 \
  --layout scene_only \
  --zip_path /kaggle/working/submission_round1.zip
```
