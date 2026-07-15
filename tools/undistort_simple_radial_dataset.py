import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required for undistortion")
    sys.exit(1)

from colmap_io import (
    Camera,
    read_cameras_binary,
    read_images_binary,
    write_cameras_binary,
    write_images_binary,
)


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUPPORTED_CAMERA_MODELS = {"SIMPLE_RADIAL", "SIMPLE_PINHOLE", "PINHOLE"}


def camera_matrix(fx, fy, cx, cy):
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def choose_undistorted_matrix(cam, camera_policy, full_fov_k_threshold):
    f, cx, cy, k = (float(value) for value in cam.params)
    original_k = camera_matrix(f, f, cx, cy)
    dist_coeffs = np.array([k, 0.0, 0.0, 0.0], dtype=np.float64)

    effective_policy = camera_policy
    if camera_policy == "auto":
        effective_policy = "full_fov" if k <= full_fov_k_threshold else "keep_intrinsics"

    if effective_policy == "full_fov":
        undistorted_k, _ = cv2.getOptimalNewCameraMatrix(
            original_k,
            dist_coeffs,
            (cam.width, cam.height),
            1.0,
            (cam.width, cam.height),
            centerPrincipalPoint=False,
        )
    else:
        undistorted_k = original_k.copy()

    return original_k, dist_coeffs, undistorted_k, effective_policy


def intrinsics_dict(matrix):
    return {
        "fx": float(matrix[0, 0]),
        "fy": float(matrix[1, 1]),
        "cx": float(matrix[0, 2]),
        "cy": float(matrix[1, 2]),
    }


def rewrite_test_poses(input_path, output_path, intrinsics):
    with open(input_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        required = {"fx", "fy", "cx", "cy"}
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    for row in rows:
        row["fx"] = f"{intrinsics['fx']:.15g}"
        row["fy"] = f"{intrinsics['fy']:.15g}"
        row["cx"] = f"{intrinsics['cx']:.15g}"
        row["cy"] = f"{intrinsics['cy']:.15g}"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def undistort_dataset(
    data_root,
    out_root,
    split_mode,
    scene_opt,
    overwrite,
    camera_policy,
    full_fov_k_threshold,
    jpeg_quality,
):
    data_root = Path(data_root)
    out_root = Path(out_root)

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root {data_root} not found.")
    if data_root.resolve() == out_root.resolve():
        raise ValueError("--out_root must differ from --data_root; the source dataset is read-only.")

    splits = [split_mode] if split_mode != "all" else ["public_set", "private_set1"]
    log_csv = Path("logs/undistort_dataset_summary.csv")
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for split in splits:
        split_dir = data_root / "phase1" / split
        if not split_dir.exists():
            continue

        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("._"):
                continue
            if scene_opt and scene_dir.name != scene_opt:
                continue

            print(f"\n--- Processing Scene: {scene_dir.name} ({split}) ---")
            in_sparse = scene_dir / "train" / "sparse" / "0"
            in_cameras_bin = in_sparse / "cameras.bin"
            in_images_bin = in_sparse / "images.bin"
            if not in_cameras_bin.exists():
                print(f"WARNING: No cameras.bin for {scene_dir.name}")
                continue

            cameras = read_cameras_binary(in_cameras_bin)
            images = read_images_binary(in_images_bin) if in_images_bin.exists() else {}
            unsupported = {cam.model for cam in cameras.values()} - SUPPORTED_CAMERA_MODELS
            if unsupported:
                raise ValueError(f"Unhandled camera models in {scene_dir.name}: {sorted(unsupported)}")

            out_scene_dir = out_root / "phase1" / split / scene_dir.name
            if out_scene_dir.exists():
                if not overwrite:
                    print(f"Output {out_scene_dir} exists; use --overwrite to regenerate it. Skipping.")
                    continue
                shutil.rmtree(out_scene_dir)

            out_train_images = out_scene_dir / "train" / "images"
            out_sparse = out_scene_dir / "train" / "sparse" / "0"
            out_test = out_scene_dir / "test"
            out_train_images.mkdir(parents=True, exist_ok=True)
            out_sparse.mkdir(parents=True, exist_ok=True)
            out_test.mkdir(parents=True, exist_ok=True)

            out_cameras = {}
            transforms = {}
            old_models = set()
            new_models = set()

            for cam_id, cam in cameras.items():
                old_models.add(cam.model)
                if cam.model == "SIMPLE_RADIAL":
                    original_k, dist_coeffs, undistorted_k, effective_policy = choose_undistorted_matrix(
                        cam, camera_policy, full_fov_k_threshold
                    )
                    transforms[cam_id] = {
                        "original_k": original_k,
                        "dist_coeffs": dist_coeffs,
                        "undistorted_k": undistorted_k,
                        "k": float(cam.params[3]),
                        "policy": effective_policy,
                    }
                    params = np.array(
                        [
                            undistorted_k[0, 0],
                            undistorted_k[1, 1],
                            undistorted_k[0, 2],
                            undistorted_k[1, 2],
                        ],
                        dtype=np.float64,
                    )
                    out_cameras[cam_id] = Camera(cam.id, "PINHOLE", cam.width, cam.height, params)
                    new_models.add("PINHOLE")
                else:
                    out_cameras[cam_id] = cam
                    new_models.add(cam.model)

            radial_ids = sorted(transforms)
            if len(radial_ids) > 1:
                raise ValueError(
                    f"{scene_dir.name} has multiple SIMPLE_RADIAL cameras. "
                    "The test pose file has no camera_id, so this conversion would be ambiguous."
                )

            image_camera_by_name = {image.name: image.camera_id for image in images.values()}
            in_train_images = scene_dir / "train" / "images"
            train_image_count = 0
            for image_path in sorted(in_train_images.iterdir()):
                if image_path.name.startswith("._") or image_path.suffix.lower() not in VALID_IMAGE_EXTS:
                    continue
                camera_id = image_camera_by_name.get(image_path.name)
                if camera_id is None and len(cameras) == 1:
                    camera_id = next(iter(cameras))
                transform = transforms.get(camera_id)
                if transform is None:
                    shutil.copy2(image_path, out_train_images / image_path.name)
                    train_image_count += 1
                    continue

                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not read training image: {image_path}")
                undistorted = cv2.undistort(
                    image,
                    transform["original_k"],
                    transform["dist_coeffs"],
                    None,
                    transform["undistorted_k"],
                )
                output_path = out_train_images / image_path.name
                if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                    cv2.imwrite(
                        str(output_path),
                        undistorted,
                        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
                    )
                else:
                    cv2.imwrite(str(output_path), undistorted)
                train_image_count += 1

            write_cameras_binary(out_cameras, out_sparse / "cameras.bin")
            if images:
                write_images_binary(images, out_sparse / "images.bin")
            for filename in ("points3D.bin", "points3D.ply"):
                source = in_sparse / filename
                if source.exists():
                    shutil.copy2(source, out_sparse / filename)

            input_test_poses = scene_dir / "test" / "test_poses.csv"
            if input_test_poses.exists():
                if radial_ids:
                    transform = transforms[radial_ids[0]]
                    rewrite_test_poses(
                        input_test_poses,
                        out_test / "test_poses.csv",
                        intrinsics_dict(transform["undistorted_k"]),
                    )
                else:
                    shutil.copy2(input_test_poses, out_test / "test_poses.csv")

            if split == "public_set":
                input_test_images = scene_dir / "test" / "images"
                if input_test_images.exists():
                    shutil.copytree(
                        input_test_images,
                        out_test / "images_original_distorted",
                        dirs_exist_ok=True,
                    )

            if radial_ids:
                camera_id = radial_ids[0]
                cam = cameras[camera_id]
                transform = transforms[camera_id]
                original_intrinsics = intrinsics_dict(transform["original_k"])
                undistorted_intrinsics = intrinsics_dict(transform["undistorted_k"])
                metadata = {
                    "schema_version": 2,
                    "camera_id": camera_id,
                    "camera_policy": transform["policy"],
                    "original_camera_model": "SIMPLE_RADIAL",
                    "original_width": cam.width,
                    "original_height": cam.height,
                    "original_intrinsics": original_intrinsics,
                    "distortion": {"k1": transform["k"], "k2": 0.0, "p1": 0.0, "p2": 0.0},
                    "undistorted_camera_model": "PINHOLE",
                    "undistorted_width": cam.width,
                    "undistorted_height": cam.height,
                    "undistorted_intrinsics": undistorted_intrinsics,
                    # Backward-compatible fields used by older scripts.
                    "f": original_intrinsics["fx"],
                    "cx": original_intrinsics["cx"],
                    "cy": original_intrinsics["cy"],
                    "k": transform["k"],
                    "undistorted_params": [
                        undistorted_intrinsics["fx"],
                        undistorted_intrinsics["fy"],
                        undistorted_intrinsics["cx"],
                        undistorted_intrinsics["cy"],
                    ],
                }
                with open(out_scene_dir / "distortion_metadata.json", "w", encoding="utf-8") as file:
                    json.dump(metadata, file, indent=2)

                summary_rows.append(
                    {
                        "split": split,
                        "scene": scene_dir.name,
                        "camera_model_old": "|".join(sorted(old_models)),
                        "camera_model_new": "|".join(sorted(new_models)),
                        "camera_policy": transform["policy"],
                        "train_images": train_image_count,
                        "f": original_intrinsics["fx"],
                        "cx": original_intrinsics["cx"],
                        "cy": original_intrinsics["cy"],
                        "k": transform["k"],
                        "undistorted_fx": undistorted_intrinsics["fx"],
                        "undistorted_fy": undistorted_intrinsics["fy"],
                        "status": "SUCCESS",
                    }
                )

    if summary_rows:
        existing_rows = []
        if log_csv.exists():
            with open(log_csv, "r", newline="", encoding="utf-8") as file:
                existing_rows = list(csv.DictReader(file))
        processed = {(row["split"], row["scene"]) for row in summary_rows}
        combined_rows = [
            row for row in existing_rows if (row.get("split"), row.get("scene")) not in processed
        ] + summary_rows
        combined_rows.sort(key=lambda row: (row.get("split", ""), row.get("scene", "")))
        fieldnames = []
        for row in combined_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(log_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"Saved dataset conversion summary to {log_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="VAI_NVS_DATA")
    parser.add_argument("--out_root", default="work_undistorted")
    parser.add_argument("--split", default="all", choices=["public_set", "private_set1", "all"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--camera_policy",
        default="auto",
        choices=["auto", "keep_intrinsics", "full_fov"],
        help="auto uses full_fov for strong negative radial distortion and keeps intrinsics otherwise.",
    )
    parser.add_argument("--full_fov_k_threshold", type=float, default=-0.05)
    parser.add_argument("--jpeg_quality", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg_quality must be between 1 and 100")
    undistort_dataset(
        args.data_root,
        args.out_root,
        args.split,
        args.scene,
        args.overwrite,
        args.camera_policy,
        args.full_fov_k_threshold,
        args.jpeg_quality,
    )


if __name__ == "__main__":
    main()
