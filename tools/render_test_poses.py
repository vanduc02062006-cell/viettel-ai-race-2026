import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def read_pose_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz", "fx", "fy", "width", "height"}
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


def short_split_name(split_name):
    if split_name.startswith("private_"):
        return "private"
    if split_name.startswith("public_"):
        return "public"
    return split_name


def find_model_path(model_root, model_template, split_name, scene_name):
    values = {
        "split": split_name,
        "short_split": short_split_name(split_name),
        "scene": scene_name,
    }
    candidates = [
        model_root / model_template.format(**values),
        model_root / scene_name,
        model_root / f"{split_name}_{scene_name}",
        model_root / f"{values['short_split']}_{scene_name}",
    ]
    for candidate in candidates:
        if (candidate / "point_cloud").exists():
            return candidate
    return candidates[0]


def find_iteration(model_path, iteration):
    point_cloud_dir = model_path / "point_cloud"
    if iteration > 0:
        ply_path = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
        if not ply_path.exists():
            raise FileNotFoundError(f"Missing trained ply: {ply_path}")
        return iteration, ply_path

    iterations = []
    if point_cloud_dir.exists():
        for child in point_cloud_dir.iterdir():
            if child.is_dir() and child.name.startswith("iteration_"):
                try:
                    iterations.append(int(child.name.split("_", 1)[1]))
                except ValueError:
                    pass
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* found under {model_path}")
    best = max(iterations)
    return best, point_cloud_dir / f"iteration_{best}" / "point_cloud.ply"


def tensor_to_uint8_image(tensor):
    arr = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def render_output_path(output_dir, image_name, output_format):
    if output_format == "png":
        return output_dir / f"{Path(image_name).stem}.png"
    return output_dir / image_name


def render_scene(args, split_name, scene_dir):
    repo_dir = Path(args.repo_dir).resolve()
    sys.path.insert(0, str(repo_dir))

    try:
        import torch
        from gaussian_renderer import GaussianModel, render
        from scene.cameras import Camera
    except Exception as exc:
        raise RuntimeError(
            "Cannot import 3DGS CUDA renderer. Run this on a GPU environment after building "
            "diff-gaussian-rasterization and simple-knn."
        ) from exc

    scene_name = scene_dir.name
    if args.model_path:
        if args.scene is None and args.split == "all":
            raise ValueError("--model_path can only be used with a single --scene")
        model_path = Path(args.model_path)
    else:
        model_path = find_model_path(Path(args.model_root), args.model_template, split_name, scene_name)

    loaded_iter, ply_path = find_iteration(model_path, args.iteration)
    poses_path = scene_dir / "test" / "test_poses.csv"
    rows = read_pose_rows(poses_path)
    output_dir = Path(args.output_root) / split_name / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(ply_path), args.use_trained_exposure)
    pipeline = SimpleNamespace(
        convert_SHs_python=args.convert_SHs_python,
        compute_cov3D_python=args.compute_cov3D_python,
        debug=args.debug,
        antialiasing=args.antialiasing,
    )
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"Rendering {split_name}/{scene_name} from {model_path} iteration {loaded_iter}")
    with torch.no_grad():
        for idx, row in enumerate(rows):
            width = int(float(row["width"]))
            height = int(float(row["height"]))
            fx = float(row["fx"])
            fy = float(row["fy"])
            qvec = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
            tvec = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
            image_name = row["image_name"]

            # 3DGS stores R transposed relative to COLMAP world-to-camera qvec.
            R = qvec2rotmat(qvec).T
            camera = Camera(
                resolution=(width, height),
                colmap_id=idx,
                R=R,
                T=tvec,
                FoVx=focal2fov(fx, width),
                FoVy=focal2fov(fy, height),
                depth_params=None,
                image=Image.new("RGB", (width, height), (0, 0, 0)),
                invdepthmap=None,
                image_name=image_name,
                uid=idx,
                data_device="cuda",
            )
            rendering = render(
                camera,
                gaussians,
                pipeline,
                background,
                separate_sh=args.separate_sh,
                use_trained_exp=args.use_trained_exposure,
            )["render"]
            output_path = render_output_path(output_dir, image_name, args.output_format)
            image = tensor_to_uint8_image(rendering)
            if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                image.save(output_path, quality=args.jpeg_quality, subsampling=0)
            else:
                image.save(output_path, compress_level=1)

    return len(rows), output_dir


def main():
    parser = argparse.ArgumentParser(description="Render Viettel NVS test_poses.csv with a trained 3DGS model.")
    parser.add_argument("--data_root", default="work_undistorted")
    parser.add_argument("--repo_dir", default="gaussian-splatting")
    parser.add_argument("--split", default="private_set1", choices=["public_set", "private_set1", "all"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--model_path", default=None, help="Explicit model path for a single scene.")
    parser.add_argument("--model_root", default="outputs")
    parser.add_argument("--model_template", default="{short_split}_{scene}", help="Template under --model_root.")
    parser.add_argument("--output_root", default="outputs/test_pose_renders")
    parser.add_argument(
        "--output_format",
        default="png",
        choices=["png", "original"],
        help="Use PNG by default so the distortion pass reads a lossless intermediate.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=100)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--convert_SHs_python", action="store_true")
    parser.add_argument("--compute_cov3D_python", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--separate_sh", action="store_true")
    parser.add_argument("--use_trained_exposure", action="store_true")
    args = parser.parse_args()

    total = 0
    for split_name, scene_dir in iter_scenes(Path(args.data_root), args.split, args.scene):
        count, output_dir = render_scene(args, split_name, scene_dir)
        total += count
        print(f"Saved {count} renders to {output_dir}")

    if total == 0:
        raise SystemExit("No scenes rendered. Check --data_root, --split, and --scene.")
    print(f"Done. Total rendered images: {total}")


if __name__ == "__main__":
    main()
