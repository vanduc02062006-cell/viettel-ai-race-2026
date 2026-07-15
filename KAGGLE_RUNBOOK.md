# Kaggle Runbook

This repository contains the code pipeline for Viettel AI Race 2026 Novel View Synthesis.

Large competition data, trained models, Kaggle handoff zips, outputs, and submissions are intentionally not stored in GitHub.

## Expected Kaggle Input

Upload the corrected Kaggle dataset package separately, for example:

```text
vai_nvs_kaggle_prod_v3.zip
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

## Camera/Distortion Gate

The production package should already contain the regenerated `work_undistorted` data. If it was
built from an older package, regenerate it from the original dataset before training:

```bash
python tools/undistort_simple_radial_dataset.py \
  --data_root VAI_NVS_DATA \
  --out_root work_undistorted \
  --split all \
  --overwrite

python tools/inspect_undistorted_dataset.py --data_root work_undistorted
```

The gate must print `PASS`. If `VAI_NVS_DATA` is also included in the package, run the stronger
pixel-level audit:

```bash
python tools/audit_distortion_roundtrip.py --split private_set1 --max_images 4
```

`HNI0131` and `HNI0265` must report `policy=full_fov` and near-black pixels well below 1%.

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

## Public Production Ablation

Do not train all private scenes until this command completes and its metrics beat the previous run:

```bash
python tools/run_public_ablation.py \
  --scene hcm0031 \
  --tag r1_it30000_aa_dssim020 \
  --iterations 30000 \
  --resolution 1 \
  --lambda_dssim 0.2
```

The metrics are written to `outputs/ablations/r1_it30000_aa_dssim020/metrics.csv`.
Raw renders are lossless PNG; only the final distorted images are encoded as JPEG.

## Private Submission

Train one full-resolution, antialiased model per private scene:

```bash
python tools/train_all_scenes.py \
  --split private_set1 \
  --iterations 30000 \
  --resolution 1 \
  --lambda_dssim 0.2
```

Models are stored as `outputs/prod_private_<scene>`. Every model contains `run_manifest.json`,
and the batch state is recorded in `outputs/training_batch_private_set1.json`.

Render and validate:

```bash
python tools/render_test_poses.py \
  --split private_set1 \
  --model_root outputs \
  --model_template "prod_{short_split}_{scene}" \
  --output_root outputs/production_test_pose_renders \
  --output_format png \
  --antialiasing

python tools/distort_back_renders.py \
  --split private_set1 \
  --render_root outputs/production_test_pose_renders \
  --output_root submission_round1_prod \
  --layout scene_only \
  --keep_original_extension

python tools/validate_submission.py \
  --split private_set1 \
  --submission_root submission_round1_prod \
  --layout scene_only \
  --zip_path /kaggle/working/submission_round1_prod.zip
```

Validation fails if an image is missing, has the wrong shape, or contains more than 5% near-black
pixels. The ZIP writer includes exactly the expected 434 target images and ignores stale files.
