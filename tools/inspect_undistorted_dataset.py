import argparse
import sys
import json
from pathlib import Path
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
