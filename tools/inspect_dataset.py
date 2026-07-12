import os
import argparse
import csv
from pathlib import Path

def inspect_dataset(data_root, out_csv):
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"ERROR: Dataset root {data_root} does not exist.")
        return

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = []
    
    total_public_scenes = 0
    total_private_scenes = 0
    total_public_train_images = 0
    total_private_train_images = 0
    total_public_test_gt_images = 0
    total_private_target_poses = 0

    splits = ['public_set', 'private_set1']
    valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    for split in splits:
        split_dir = data_root / 'phase1' / split
        if not split_dir.exists():
            print(f"WARNING: Split directory {split_dir} does not exist.")
            continue

        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            if scene_dir.name.startswith('._') or scene_dir.name in ['__MACOSX', '.DS_Store']:
                continue

            scene_name = scene_dir.name
            
            if split == 'public_set':
                total_public_scenes += 1
            else:
                total_private_scenes += 1

            has_readme = (scene_dir / 'README.txt').exists()

            train_img_dir = scene_dir / 'train' / 'images'
            train_image_count = 0
            if train_img_dir.exists():
                for f in train_img_dir.iterdir():
                    if f.suffix in valid_exts and not f.name.startswith('._'):
                        train_image_count += 1
            else:
                print(f"WARNING: Missing train/images for scene {scene_name}")

            if split == 'public_set':
                total_public_train_images += train_image_count
            else:
                total_private_train_images += train_image_count

            test_img_dir = scene_dir / 'test' / 'images'
            public_test_gt_image_count = 0
            if test_img_dir.exists():
                for f in test_img_dir.iterdir():
                    if f.suffix in valid_exts and not f.name.startswith('._'):
                        public_test_gt_image_count += 1
                if split == 'public_set':
                    total_public_test_gt_images += public_test_gt_image_count

            test_poses_csv = scene_dir / 'test' / 'test_poses.csv'
            has_test_poses_csv = test_poses_csv.exists()
            test_pose_count = 0
            test_pose_columns = ""
            if has_test_poses_csv:
                try:
                    with open(test_poses_csv, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header:
                            test_pose_columns = "|".join(header)
                        for _ in reader:
                            test_pose_count += 1
                except Exception as e:
                    print(f"WARNING: Failed to read {test_poses_csv}: {e}")
                
                if split == 'private_set1':
                    total_private_target_poses += test_pose_count
            
            sparse_dir = scene_dir / 'train' / 'sparse' / '0'
            has_sparse_dir = sparse_dir.exists()
            sparse_files = []
            if has_sparse_dir:
                for f in sparse_dir.iterdir():
                    if f.is_file() and not f.name.startswith('._'):
                        sparse_files.append(f.name)
            else:
                print(f"WARNING: Missing train/sparse/0 for scene {scene_name}")

            summary.append({
                'split': split,
                'scene_name': scene_name,
                'scene_path': str(scene_dir),
                'has_readme': has_readme,
                'train_image_count': train_image_count,
                'public_test_gt_image_count': public_test_gt_image_count,
                'has_test_poses_csv': has_test_poses_csv,
                'test_pose_count': test_pose_count,
                'test_pose_columns': test_pose_columns,
                'has_sparse_dir': has_sparse_dir,
                'sparse_files': "|".join(sparse_files)
            })

    if summary:
        keys = summary[0].keys()
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(summary)
        print(f"Saved summary to {out_csv}")
    else:
        print("WARNING: No scenes found.")

    print("\n--- Summary ---")
    print(f"total_public_scenes: {total_public_scenes}")
    print(f"total_private_scenes: {total_private_scenes}")
    print(f"total_public_train_images: {total_public_train_images}")
    print(f"total_private_train_images: {total_private_train_images}")
    print(f"total_public_test_gt_images: {total_public_test_gt_images}")
    print(f"total_private_target_poses: {total_private_target_poses}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.\\VAI_NVS_DATA', help='Dataset root path')
    parser.add_argument('--out_csv', default='logs\\dataset_summary.csv', help='Output CSV path')
    args = parser.parse_args()
    inspect_dataset(args.data_root, args.out_csv)
