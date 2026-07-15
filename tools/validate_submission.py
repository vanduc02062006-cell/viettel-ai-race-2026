import argparse
import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np


def read_pose_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "width", "height"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        return list(reader)


def iter_scenes(data_root, split, scene):
    splits = ["public_set", "private_set1"] if split == "all" else [split]
    for split_name in splits:
        split_dir = data_root / "phase1" / split_name
        if not split_dir.exists():
            continue
        for scene_dir in sorted(split_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name.startswith("._"):
                continue
            if scene and scene_dir.name != scene:
                continue
            yield split_name, scene_dir


def candidate_path(root, layout, split_name, scene_name, image_name):
    if layout == "scene_only":
        return root / scene_name / image_name
    if layout == "phase_split_scene":
        return root / "phase1" / split_name / scene_name / image_name
    return root / split_name / scene_name / image_name


def validate(args):
    root = Path(args.submission_root)
    rows = []
    ok_count = 0

    for split_name, scene_dir in iter_scenes(Path(args.data_root), args.split, args.scene):
        poses_path = scene_dir / "test" / "test_poses.csv"
        for pose in read_pose_rows(poses_path):
            image_name = pose["image_name"]
            expected_w = int(float(pose["width"]))
            expected_h = int(float(pose["height"]))
            path = candidate_path(root, args.layout, split_name, scene_dir.name, image_name)
            status = "ok"
            near_black_fraction = ""

            if not path.exists():
                status = "missing"
            else:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if img is None:
                    status = "read_error"
                elif img.shape[1] != expected_w or img.shape[0] != expected_h:
                    status = f"shape_mismatch:{img.shape[1]}x{img.shape[0]}!={expected_w}x{expected_h}"
                else:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    near_black_fraction = float(np.mean(gray < 8))
                    if near_black_fraction > args.max_near_black_fraction:
                        status = f"excess_near_black:{near_black_fraction:.4f}"

            if status == "ok":
                ok_count += 1
            rows.append({
                "split": split_name,
                "scene": scene_dir.name,
                "image_name": image_name,
                "path": str(path),
                "near_black_fraction": near_black_fraction,
                "status": status,
            })

    report_path = Path(args.report_csv)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "scene", "image_name", "path", "near_black_fraction", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    failures = [r for r in rows if r["status"] != "ok"]
    print(f"Expected images: {len(rows)}")
    print(f"Valid images: {ok_count}")
    print(f"Failures: {len(failures)}")
    print(f"Saved validation report to {report_path}")

    if failures:
        for item in failures[:10]:
            print(f"{item['status']}: {item['split']}/{item['scene']}/{item['image_name']}")
        raise SystemExit(1)

    if args.zip_path:
        zip_path = Path(args.zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for row in rows:
                path = Path(row["path"])
                zf.write(path, path.relative_to(root))
        print(f"Created zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate rendered Viettel NVS submission images and optionally zip them.")
    parser.add_argument("--data_root", default="work_undistorted")
    parser.add_argument("--submission_root", default="submission_round1")
    parser.add_argument("--split", default="private_set1", choices=["public_set", "private_set1", "all"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--layout", default="split_scene", choices=["split_scene", "scene_only", "phase_split_scene"])
    parser.add_argument("--report_csv", default="outputs/validation/submission_validation.csv")
    parser.add_argument("--zip_path", default=None)
    parser.add_argument(
        "--max_near_black_fraction",
        type=float,
        default=0.05,
        help="Reject images with excessive invalid black borders (default: 5%%).",
    )
    args = parser.parse_args()
    validate(args)


if __name__ == "__main__":
    main()
