import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def read_pose_names(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "image_name" not in (reader.fieldnames or []):
            raise ValueError(f"{csv_path} is missing image_name")
        return [row["image_name"] for row in reader]


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


def output_scene_dir(output_root, layout, split_name, scene_name):
    if layout == "scene_only":
        return output_root / scene_name
    if layout == "phase_split_scene":
        return output_root / "phase1" / split_name / scene_name
    return output_root / split_name / scene_name


def build_distort_back_maps(width, height, f, cx, cy, k):
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k, 0, 0, 0], dtype=np.float64)

    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    distorted_pixels = np.stack([xs, ys], axis=-1).reshape(-1, 1, 2)

    # For each target distorted pixel, find the corresponding ideal pinhole pixel
    # and sample the undistorted render at that coordinate.
    undistorted_pixels = cv2.undistortPoints(distorted_pixels, K, dist, P=K)
    undistorted_pixels = undistorted_pixels.reshape(height, width, 2).astype(np.float32)
    return undistorted_pixels[..., 0], undistorted_pixels[..., 1]


def distort_image(render_path, output_path, map_x, map_y, jpeg_quality):
    img = cv2.imread(str(render_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read render image: {render_path}")
    distorted = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(output_path), distorted, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    else:
        cv2.imwrite(str(output_path), distorted)


def process_scene(args, split_name, scene_dir):
    scene_name = scene_dir.name
    meta_path = scene_dir / "distortion_metadata.json"
    poses_path = scene_dir / "test" / "test_poses.csv"
    input_dir = Path(args.render_root) / split_name / scene_name
    out_dir = output_scene_dir(Path(args.output_root), args.layout, split_name, scene_name)

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing distortion metadata: {meta_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing render directory: {input_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    width = int(meta["original_width"])
    height = int(meta["original_height"])
    map_x, map_y = build_distort_back_maps(
        width=width,
        height=height,
        f=float(meta["f"]),
        cx=float(meta["cx"]),
        cy=float(meta["cy"]),
        k=float(meta["k"]),
    )

    image_names = read_pose_names(poses_path)
    count = 0
    missing = []
    for image_name in image_names:
        render_path = input_dir / image_name
        if not render_path.exists():
            # Allow renderers that save PNG even when target names are JPG.
            png_fallback = input_dir / f"{Path(image_name).stem}.png"
            render_path = png_fallback if png_fallback.exists() else render_path
        if not render_path.exists():
            missing.append(image_name)
            continue

        output_name = image_name if args.keep_original_extension else f"{Path(image_name).stem}.png"
        distort_image(render_path, out_dir / output_name, map_x, map_y, args.jpeg_quality)
        count += 1

    if missing:
        raise FileNotFoundError(f"{split_name}/{scene_name} missing {len(missing)} renders, first missing: {missing[:5]}")
    return count, out_dir


def main():
    parser = argparse.ArgumentParser(description="Distort undistorted 3DGS renders back to original SIMPLE_RADIAL frame.")
    parser.add_argument("--data_root", default="work_undistorted")
    parser.add_argument("--render_root", default="outputs/test_pose_renders")
    parser.add_argument("--output_root", default="outputs/test_pose_renders_distorted")
    parser.add_argument("--split", default="private_set1", choices=["public_set", "private_set1", "all"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--layout", default="split_scene", choices=["split_scene", "scene_only", "phase_split_scene"])
    parser.add_argument("--keep_original_extension", action="store_true")
    parser.add_argument("--jpeg_quality", type=int, default=95)
    args = parser.parse_args()

    total = 0
    for split_name, scene_dir in iter_scenes(Path(args.data_root), args.split, args.scene):
        count, out_dir = process_scene(args, split_name, scene_dir)
        total += count
        print(f"Saved {count} distorted images to {out_dir}")
    if total == 0:
        raise SystemExit("No images processed. Check --data_root, --render_root, --split, and --scene.")
    print(f"Done. Total distorted images: {total}")


if __name__ == "__main__":
    main()
