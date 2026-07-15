import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from distort_back_renders import build_distort_back_maps
from evaluate_public_renders import psnr, ssim


def load_mapping(metadata):
    original = metadata.get("original_intrinsics") or {
        "fx": metadata["f"],
        "fy": metadata["f"],
        "cx": metadata["cx"],
        "cy": metadata["cy"],
    }
    undistorted = metadata.get("undistorted_intrinsics") or original
    distortion = metadata.get("distortion") or {"k1": metadata["k"]}
    return build_distort_back_maps(
        width=int(metadata["original_width"]),
        height=int(metadata["original_height"]),
        original_fx=float(original["fx"]),
        original_fy=float(original["fy"]),
        original_cx=float(original["cx"]),
        original_cy=float(original["cy"]),
        k=float(distortion["k1"]),
        undistorted_fx=float(undistorted["fx"]),
        undistorted_fy=float(undistorted["fy"]),
        undistorted_cx=float(undistorted["cx"]),
        undistorted_cy=float(undistorted["cy"]),
    )


def main():
    parser = argparse.ArgumentParser(description="Measure loss introduced by undistort/distort-back alone.")
    parser.add_argument("--original_root", default="VAI_NVS_DATA")
    parser.add_argument("--work_root", default="work_undistorted")
    parser.add_argument("--split", default="private_set1", choices=["public_set", "private_set1"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--max_images", type=int, default=8)
    parser.add_argument("--output_csv", default="outputs/evaluation/distortion_roundtrip.csv")
    args = parser.parse_args()

    original_split = Path(args.original_root) / "phase1" / args.split
    work_split = Path(args.work_root) / "phase1" / args.split
    rows = []
    for scene_dir in sorted(path for path in work_split.iterdir() if path.is_dir()):
        if args.scene and scene_dir.name != args.scene:
            continue
        metadata_path = scene_dir / "distortion_metadata.json"
        if not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        map_x, map_y = load_mapping(metadata)
        files = sorted((scene_dir / "train" / "images").glob("*"))
        files = [path for path in files if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        sample = files[:: max(1, len(files) // args.max_images)][: args.max_images]
        for undistorted_path in sample:
            original_path = original_split / scene_dir.name / "train" / "images" / undistorted_path.name
            original = cv2.imread(str(original_path), cv2.IMREAD_COLOR)
            undistorted = cv2.imread(str(undistorted_path), cv2.IMREAD_COLOR)
            if original is None or undistorted is None:
                raise ValueError(f"Could not read pair: {original_path}, {undistorted_path}")
            recovered = cv2.remap(
                undistorted,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            gray = cv2.cvtColor(recovered, cv2.COLOR_BGR2GRAY)
            rows.append(
                {
                    "scene": scene_dir.name,
                    "image_name": undistorted_path.name,
                    "camera_policy": metadata.get("camera_policy", "legacy"),
                    "near_black_fraction": f"{np.mean(gray < 8):.6f}",
                    "psnr": f"{psnr(recovered, original):.6f}",
                    "ssim": f"{ssim(recovered, original):.6f}",
                }
            )

    if not rows:
        raise SystemExit("No round-trip image pairs found.")
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for scene in sorted({row["scene"] for row in rows}):
        scene_rows = [row for row in rows if row["scene"] == scene]
        print(
            f"{scene}: policy={scene_rows[0]['camera_policy']}, "
            f"PSNR={np.mean([float(row['psnr']) for row in scene_rows]):.3f}, "
            f"SSIM={np.mean([float(row['ssim']) for row in scene_rows]):.4f}, "
            f"near_black={100*np.mean([float(row['near_black_fraction']) for row in scene_rows]):.2f}%"
        )
    print(f"Saved audit: {output_path}")


if __name__ == "__main__":
    main()
