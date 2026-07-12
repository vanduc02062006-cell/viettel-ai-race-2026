import os
import sys
import argparse
import json
import csv
import shutil
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required for undistortion")
    sys.exit(1)

from colmap_io import read_cameras_binary, read_images_binary, write_cameras_binary, write_images_binary, Camera

def undistort_dataset(data_root, out_root, split_mode, scene_opt, overwrite):
    data_root = Path(data_root)
    out_root = Path(out_root)
    
    if not data_root.exists():
        print(f"ERROR: Dataset root {data_root} not found.")
        return

    splits = []
    if split_mode in ['public_set', 'all']:
        splits.append('public_set')
    if split_mode in ['private_set1', 'all']:
        splits.append('private_set1')

    log_csv = Path('logs/undistort_dataset_summary.csv')
    log_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for split in splits:
        split_dir = data_root / 'phase1' / split
        if not split_dir.exists():
            continue

        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir() or scene_dir.name.startswith('._') or scene_dir.name in ['__MACOSX', '.DS_Store']:
                continue
            if scene_opt and scene_dir.name != scene_opt:
                continue

            print(f"\n--- Processing Scene: {scene_dir.name} ({split}) ---")
            
            in_sparse = scene_dir / 'train' / 'sparse' / '0'
            in_cameras_bin = in_sparse / 'cameras.bin'
            in_images_bin = in_sparse / 'images.bin'
            in_points3D_bin = in_sparse / 'points3D.bin'
            in_points3D_ply = in_sparse / 'points3D.ply'
            
            if not in_cameras_bin.exists():
                print(f"WARNING: No cameras.bin for {scene_dir.name}")
                continue

            cameras = read_cameras_binary(in_cameras_bin)
            images = {}
            if in_images_bin.exists():
                images = read_images_binary(in_images_bin)

            all_cams_ok = True
            for cam_id, cam in cameras.items():
                if cam.model not in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE', 'PINHOLE']:
                    print(f"ERROR: Unhandled camera model {cam.model} in {scene_dir.name}")
                    all_cams_ok = False
                    break
            if not all_cams_ok:
                continue

            out_scene_dir = out_root / 'phase1' / split / scene_dir.name
            if out_scene_dir.exists():
                if overwrite:
                    print(f"Overwriting {out_scene_dir}")
                else:
                    print(f"Output directory {out_scene_dir} already exists. Use --overwrite if needed. Skipping.")
                    continue
            
            out_train_images = out_scene_dir / 'train' / 'images'
            out_sparse = out_scene_dir / 'train' / 'sparse' / '0'
            out_test = out_scene_dir / 'test'
            out_train_images.mkdir(parents=True, exist_ok=True)
            out_sparse.mkdir(parents=True, exist_ok=True)
            out_test.mkdir(parents=True, exist_ok=True)
            
            out_cameras = {}
            train_image_count = 0

            cam_model_old = set()
            cam_model_new = set()
            f_val, cx_val, cy_val, k_val = None, None, None, None

            for cam_id, cam in cameras.items():
                cam_model_old.add(cam.model)
                if cam.model == 'SIMPLE_RADIAL':
                    f, cx, cy, k = cam.params
                    f_val, cx_val, cy_val, k_val = f, cx, cy, k
                    
                    K = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]], dtype=np.float32)
                    distCoeffs = np.array([k, 0, 0, 0], dtype=np.float32)

                    new_cam = Camera(id=cam.id, model='SIMPLE_PINHOLE', width=cam.width, height=cam.height, params=np.array([f, cx, cy]))
                    out_cameras[cam_id] = new_cam
                    cam_model_new.add('SIMPLE_PINHOLE')

                    distortion_metadata = {
                        "original_camera_model": "SIMPLE_RADIAL",
                        "original_width": cam.width,
                        "original_height": cam.height,
                        "f": float(f),
                        "cx": float(cx),
                        "cy": float(cy),
                        "k": float(k),
                        "undistorted_camera_model": "SIMPLE_PINHOLE",
                        "undistorted_width": cam.width,
                        "undistorted_height": cam.height,
                        "undistorted_params": [float(f), float(cx), float(cy)],
                        "note": "Train images are undistorted. Final renders must be distorted back to original SIMPLE_RADIAL frame before submission."
                    }
                    meta_path = out_scene_dir / 'distortion_metadata.json'
                    with open(meta_path, 'w') as mf:
                        json.dump(distortion_metadata, mf, indent=4)
                    
                elif cam.model in ['SIMPLE_PINHOLE', 'PINHOLE']:
                    out_cameras[cam_id] = cam
                    cam_model_new.add(cam.model)
                    
            if 'SIMPLE_RADIAL' in cam_model_old:
                in_train_images = scene_dir / 'train' / 'images'
                valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                if in_train_images.exists():
                    for f_name in in_train_images.iterdir():
                        if f_name.suffix in valid_exts and not f_name.name.startswith('._'):
                            img = cv2.imread(str(f_name))
                            if img is not None:
                                undistorted_img = cv2.undistort(img, K, distCoeffs)
                                out_path = out_train_images / f_name.name
                                cv2.imwrite(str(out_path), undistorted_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                                train_image_count += 1
            else:
                in_train_images = scene_dir / 'train' / 'images'
                if in_train_images.exists():
                    for f_name in in_train_images.iterdir():
                        if not f_name.name.startswith('._'):
                            shutil.copy2(f_name, out_train_images / f_name.name)
                            train_image_count += 1
            
            write_cameras_binary(out_cameras, out_sparse / 'cameras.bin')
            if images:
                write_images_binary(images, out_sparse / 'images.bin')
            if in_points3D_bin.exists():
                shutil.copy2(in_points3D_bin, out_sparse / 'points3D.bin')
            if in_points3D_ply.exists():
                shutil.copy2(in_points3D_ply, out_sparse / 'points3D.ply')

            in_test_poses = scene_dir / 'test' / 'test_poses.csv'
            if in_test_poses.exists():
                shutil.copy2(in_test_poses, out_test / 'test_poses.csv')

            if split == 'public_set':
                in_test_images = scene_dir / 'test' / 'images'
                if in_test_images.exists():
                    out_test_images_orig = out_test / 'images_original_distorted'
                    shutil.copytree(in_test_images, out_test_images_orig, dirs_exist_ok=True)
            
            status = 'SUCCESS'
            summary_rows.append({
                'split': split,
                'scene': scene_dir.name,
                'camera_model_old': "|".join(cam_model_old),
                'camera_model_new': "|".join(cam_model_new),
                'train_images': train_image_count,
                'f': f_val if f_val else "",
                'cx': cx_val if cx_val else "",
                'cy': cy_val if cy_val else "",
                'k': k_val if k_val else "",
                'status': status
            })

    if summary_rows:
        keys = summary_rows[0].keys()
        with open(log_csv, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary_rows)
        print(f"Saved dataset conversion summary to {log_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.\\VAI_NVS_DATA')
    parser.add_argument('--out_root', default='.\\work_undistorted')
    parser.add_argument('--split', default='all', choices=['public_set', 'private_set1', 'all'])
    parser.add_argument('--scene', default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    undistort_dataset(args.data_root, args.out_root, args.split, args.scene, args.overwrite)
