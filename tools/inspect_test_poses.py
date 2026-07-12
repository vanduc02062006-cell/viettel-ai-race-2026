import argparse
import csv
from pathlib import Path

def inspect_test_poses(data_root, split_mode, num_rows, out_dir):
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = []
    if split_mode in ['public_set', 'all']:
        splits.append('public_set')
    if split_mode in ['private_set1', 'all']:
        splits.append('private_set1')

    for split in splits:
        split_dir = data_root / 'phase1' / split
        if not split_dir.exists():
            continue

        for scene_dir in split_dir.iterdir():
            if not scene_dir.is_dir() or scene_dir.name.startswith('._') or scene_dir.name in ['__MACOSX', '.DS_Store']:
                continue
            
            test_poses_csv = scene_dir / 'test' / 'test_poses.csv'
            if not test_poses_csv.exists():
                continue

            print(f"\n--- Scene: {scene_dir.name} ({split}) ---")
            
            lines = []
            try:
                with open(test_poses_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header is None:
                        continue
                    
                    rows = list(reader)
                    total_rows = len(rows)
                    
                    print(f"Total rows: {total_rows}")
                    print(f"Columns: {header}")
                    
                    lines.append(header)
                    for i in range(min(num_rows, total_rows)):
                        print(f"Row {i}: {rows[i]}")
                        lines.append(rows[i])

                    # Check required columns
                    required = ['image_name', 'width', 'height']
                    missing = [c for c in required if c not in header]
                    if missing:
                        print(f"WARNING: Missing required columns: {missing}")

                    # Attempt to recognize pose format
                    cols_set = set(header)
                    format_detected = None
                    if {'qw', 'qx', 'qy', 'qz', 'tx', 'ty', 'tz'}.issubset(cols_set):
                        format_detected = "qw,qx,qy,qz,tx,ty,tz"
                    elif {'qx', 'qy', 'qz', 'qw', 'tx', 'ty', 'tz'}.issubset(cols_set):
                        format_detected = "qx,qy,qz,qw,tx,ty,tz"
                    elif {'qvec', 'tvec'}.issubset(cols_set):
                        format_detected = "qvec/tvec"
                    elif {'r11', 'r12', 'r13', 'r21', 'r22', 'r23', 'r31', 'r32', 'r33', 'tx', 'ty', 'tz'}.issubset(cols_set):
                        format_detected = "r11...r33 + tx,ty,tz"
                    elif {'transform_matrix'}.issubset(cols_set):
                        format_detected = "transform_matrix"

                    if format_detected:
                        print(f"Pose format detected: {format_detected}")
                    else:
                        print("WARNING: Could not recognize pose format.")

            except Exception as e:
                print(f"ERROR reading {test_poses_csv}: {e}")
                continue

            # Save sample
            sample_path = out_dir / f"{split}_{scene_dir.name}_head.csv"
            with open(sample_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(lines)
            print(f"Saved sample to {sample_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='.\\VAI_NVS_DATA')
    parser.add_argument('--split', default='all', choices=['public_set', 'private_set1', 'all'])
    parser.add_argument('--num_rows', type=int, default=5)
    parser.add_argument('--out_dir', default='logs\\test_pose_samples')
    args = parser.parse_args()
    inspect_test_poses(args.data_root, args.split, args.num_rows, args.out_dir)
