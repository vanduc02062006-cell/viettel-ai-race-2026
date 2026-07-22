import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from colmap_io import read_cameras_binary, read_images_binary


GROUP_SCENES = {
    "C": {"HCM0421", "HCM0539", "HCM0540", "HCM0644"},
    "C1": {"HCM0421", "HCM0540"},
    "C2": {"HCM0539", "HCM0644"},
    "D": {"HCM0674", "bonsai", "chair"},
}
ALL_SCENES = set().union(GROUP_SCENES["C"], GROUP_SCENES["D"])
GROUP_SCENES["ALL"] = ALL_SCENES


def locate_extracted_scene(input_root, scene):
    candidates = []
    for train_dir in input_root.rglob(f"{scene}/train"):
        scene_dir = train_dir.parent
        if (scene_dir / "test" / "test_poses.csv").exists() and (train_dir / "sparse" / "0").exists():
            image_dir = train_dir / "images"
            image_count = sum(
                1 for path in image_dir.iterdir() if path.is_file() and not path.name.startswith("._")
            ) if image_dir.exists() else 0
            candidates.append((image_count, scene_dir))
    candidates.sort(
        key=lambda item: (
            -item[0],
            1 if "work_round2" in str(item[1]) else 0,
            len(str(item[1])),
            str(item[1]),
        )
    )
    if not candidates or candidates[0][0] == 0:
        raise RuntimeError(f"No extracted source with training images for {scene}; found: {candidates}")
    selected_count, selected = candidates[0]
    if len(candidates) > 1:
        print(f"Selected {selected} for {scene} ({selected_count} images); ignored duplicates: {candidates[1:]}")
    return selected


def main():
    parser = argparse.ArgumentParser(description="Prepare one Round 2 Kaggle training group.")
    parser.add_argument("--group", required=True, choices=sorted(GROUP_SCENES))
    parser.add_argument("--input_root", default="/kaggle/input")
    parser.add_argument("--raw_root", default="/kaggle/working/round2_raw")
    parser.add_argument("--output_root", default="work_round2")
    args = parser.parse_args()

    keep_scenes = GROUP_SCENES[args.group]
    input_root = Path(args.input_root)
    raw_root = Path(args.raw_root)
    raw_split = raw_root / "phase1" / "private_set1"
    output_root = Path(args.output_root)

    if raw_root.exists():
        shutil.rmtree(raw_root)
    if output_root.exists():
        shutil.rmtree(output_root)
    raw_split.mkdir(parents=True, exist_ok=True)

    zip_candidates = list(input_root.rglob("02_ROUND2_DATA.zip"))
    if not zip_candidates:
        zip_candidates = list(input_root.rglob("VAI_NVS_DATA_ROUND2.zip"))
    if zip_candidates:
        if len(zip_candidates) != 1:
            raise RuntimeError(f"Expected one Round 2 ZIP, found: {zip_candidates}")
        print(f"Extracting Round 2 data: {zip_candidates[0]}")
        shutil.unpack_archive(str(zip_candidates[0]), str(raw_split))
        for scene_dir in list(raw_split.iterdir()):
            if scene_dir.is_dir() and scene_dir.name not in keep_scenes:
                shutil.rmtree(scene_dir)
    else:
        print("Round 2 ZIP not found; copying the extracted Kaggle dataset")
        for scene in sorted(keep_scenes):
            shutil.copytree(locate_extracted_scene(input_root, scene), raw_split / scene)

    actual_raw = {path.name for path in raw_split.iterdir() if path.is_dir() and path.name in ALL_SCENES}
    if actual_raw != keep_scenes:
        raise RuntimeError(f"Round 2 Group {args.group} mismatch: expected {sorted(keep_scenes)}, got {sorted(actual_raw)}")

    subprocess.run(
        [
            sys.executable,
            "tools/undistort_simple_radial_dataset.py",
            "--data_root",
            str(raw_root),
            "--out_root",
            str(output_root),
            "--split",
            "private_set1",
            "--overwrite",
            "--camera_policy",
            "auto",
            "--jpeg_quality",
            "100",
        ],
        check=True,
    )

    prepared_split = output_root / "phase1" / "private_set1"
    for scene in sorted(keep_scenes):
        scene_train = prepared_split / scene / "train"
        subprocess.run(
            [
                sys.executable,
                "tools/prune_colmap_images_to_existing.py",
                "--scene_train_path",
                str(scene_train),
            ],
            check=True,
        )
        cameras = read_cameras_binary(scene_train / "sparse" / "0" / "cameras.bin")
        camera_models = {camera.model for camera in cameras.values()}
        if not camera_models.issubset({"PINHOLE", "SIMPLE_PINHOLE"}):
            raise RuntimeError(f"{scene} still has unsupported cameras: {sorted(camera_models)}")
        images = read_images_binary(scene_train / "sparse" / "0" / "images.bin")
        image_files = {
            path.name
            for path in (scene_train / "images").iterdir()
            if path.is_file() and not path.name.startswith("._")
        }
        registered_names = {image.name for image in images.values()}
        if registered_names != image_files:
            raise RuntimeError(
                f"{scene} COLMAP/image mismatch after pruning: registered={len(registered_names)}, files={len(image_files)}"
            )
        pose_path = prepared_split / scene / "test" / "test_poses.csv"
        if not pose_path.exists():
            raise RuntimeError(f"Missing test poses for {scene}")
        print(
            f"PREPARED {scene}: train_images={len(image_files)}, registered={len(registered_names)}, "
            f"camera_models={sorted(camera_models)}"
        )

    shutil.rmtree(raw_root)
    free_gib = shutil.disk_usage(output_root.resolve()).free / 1024**3
    print(f"ROUND 2 GROUP {args.group} DATA READY; free disk: {free_gib:.1f} GiB")


if __name__ == "__main__":
    main()
