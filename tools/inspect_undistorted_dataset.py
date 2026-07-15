import argparse
import csv
import sys
import json
from pathlib import Path
import numpy as np
from colmap_io import read_cameras_binary, read_images_binary

def inspect_undistorted(data_root):
    data_root = Path(data_root)
    log_file = Path('logs/undistorted_camera_model_gate.txt')
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    splits = ['public_set', 'private_set1']
    all_pass = True
    failed_reasons = []

    if not data_root.exists():
        print(f"ERROR: {data_root} does not exist.")
        return
        
    for split in splits:
        split_dir = data_root / 'phase1' / split
        if not split_dir.exists():
            continue

        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir() or scene_dir.name.startswith('._') or scene_dir.name in ['__MACOSX', '.DS_Store']:
                continue
            
            # Check required files
            train_images = scene_dir / 'train' / 'images'
            cameras_bin = scene_dir / 'train' / 'sparse' / '0' / 'cameras.bin'
            images_bin = scene_dir / 'train' / 'sparse' / '0' / 'images.bin'
            points3D_bin = scene_dir / 'train' / 'sparse' / '0' / 'points3D.bin'
            test_poses = scene_dir / 'test' / 'test_poses.csv'
            meta_json = scene_dir / 'distortion_metadata.json'

            missing = []
            if not train_images.exists() or not any(train_images.iterdir()):
                missing.append('train/images')
            if not cameras_bin.exists():
                missing.append('cameras.bin')
            if not images_bin.exists():
                missing.append('images.bin')
            if not points3D_bin.exists():
                missing.append('points3D.bin')
            if not test_poses.exists():
                missing.append('test_poses.csv')
            if not meta_json.exists():
                missing.append('distortion_metadata.json')

            if missing:
                failed_reasons.append(f"{scene_dir.name} missing: {', '.join(missing)}")
                all_pass = False

            if cameras_bin.exists():
                try:
                    cameras = read_cameras_binary(cameras_bin)
                    for cam_id, cam in cameras.items():
                        if cam.model not in ['SIMPLE_PINHOLE', 'PINHOLE']:
                            failed_reasons.append(f"{scene_dir.name} cam {cam_id} model is {cam.model}")
                            all_pass = False

                    if meta_json.exists() and test_poses.exists() and len(cameras) == 1:
                        metadata = json.loads(meta_json.read_text(encoding="utf-8"))
                        undistorted = metadata.get("undistorted_intrinsics")
                        if undistorted is None:
                            params = metadata.get("undistorted_params", [])
                            if len(params) == 3:
                                undistorted = {"fx": params[0], "fy": params[0], "cx": params[1], "cy": params[2]}
                        if undistorted:
                            camera = next(iter(cameras.values()))
                            expected = np.array([
                                undistorted["fx"], undistorted["fy"],
                                undistorted["cx"], undistorted["cy"],
                            ], dtype=float)
                            actual = np.array(camera.params, dtype=float)
                            if camera.model == "SIMPLE_PINHOLE":
                                actual = np.array([actual[0], actual[0], actual[1], actual[2]])
                            if not np.allclose(actual, expected, rtol=0, atol=1e-6):
                                failed_reasons.append(f"{scene_dir.name} camera intrinsics disagree with metadata")
                                all_pass = False

                            with open(test_poses, "r", encoding="utf-8") as file:
                                first_pose = next(csv.DictReader(file), None)
                            if first_pose:
                                pose_intrinsics = np.array([
                                    float(first_pose["fx"]), float(first_pose["fy"]),
                                    float(first_pose["cx"]), float(first_pose["cy"]),
                                ])
                                if not np.allclose(pose_intrinsics, expected, rtol=0, atol=1e-6):
                                    failed_reasons.append(f"{scene_dir.name} test-pose intrinsics disagree with metadata")
                                    all_pass = False

                        k = float((metadata.get("distortion") or {}).get("k1", metadata.get("k", 0.0)))
                        if k <= -0.05 and metadata.get("camera_policy") != "full_fov":
                            failed_reasons.append(f"{scene_dir.name} strong negative k does not use full_fov")
                            all_pass = False
                except Exception as e:
                    failed_reasons.append(f"{scene_dir.name} failed to read cameras: {e}")
                    all_pass = False

    with open(log_file, 'w', encoding='utf-8') as f:
        if all_pass:
            f.write("PASS\n")
            print("PASS: All scenes in work_undistorted have valid PINHOLE/SIMPLE_PINHOLE models.")
        else:
            f.write("FAIL\n")
            for reason in failed_reasons:
                f.write(f"{reason}\n")
            print("FAIL: Some scenes failed validation.")
            for reason in failed_reasons:
                print(f"  {reason}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.\\work_undistorted')
    args = parser.parse_args()
    inspect_undistorted(args.data_root)
