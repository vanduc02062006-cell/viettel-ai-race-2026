import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def iter_scenes(data_root, split, selected_scene):
    split_dir = data_root / "phase1" / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    for scene_dir in sorted(split_dir.iterdir()):
        if not scene_dir.is_dir() or scene_dir.name.startswith("._"):
            continue
        if selected_scene and scene_dir.name != selected_scene:
            continue
        yield scene_dir


def model_name(template, split, scene, resolution, iterations):
    short_split = "private" if split.startswith("private_") else "public" if split.startswith("public_") else split
    return template.format(
        split=split,
        short_split=short_split,
        scene=scene,
        resolution=resolution,
        iterations=iterations,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a production 3DGS model for every scene in a split.")
    parser.add_argument("--data_root", default="work_undistorted")
    parser.add_argument("--model_root", default="outputs")
    parser.add_argument("--split", default="private_set1", choices=["public_set", "private_set1"])
    parser.add_argument("--scene", default=None)
    parser.add_argument("--model_template", default="prod_{short_split}_{scene}")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--save_iterations", type=int, nargs="*", default=[15000, 30000])
    parser.add_argument("--no_antialiasing", action="store_true")
    parser.add_argument("--force", action="store_true", help="Train even if the final point cloud already exists.")
    parser.add_argument("--dry_run", action="store_true", help="Write the batch manifest without starting CUDA training.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    scenes = list(iter_scenes(data_root, args.split, args.scene))
    if not scenes:
        raise SystemExit("No matching scenes found.")

    batch = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "antialiasing": not args.no_antialiasing,
        "lambda_dssim": args.lambda_dssim,
        "scenes": [],
    }
    batch_path = model_root / f"training_batch_{args.split}.json"

    for index, scene_dir in enumerate(scenes, start=1):
        name = model_name(
            args.model_template,
            args.split,
            scene_dir.name,
            args.resolution,
            args.iterations,
        )
        model_path = model_root / name
        final_ply = model_path / "point_cloud" / f"iteration_{args.iterations}" / "point_cloud.ply"
        item = {"scene": scene_dir.name, "model_path": str(model_path), "status": "pending"}
        batch["scenes"].append(item)

        if final_ply.exists() and not args.force:
            item["status"] = "skipped_existing"
            print(f"[{index}/{len(scenes)}] Skipping {scene_dir.name}: {final_ply} exists")
            continue

        save_iterations = sorted(
            {value for value in args.save_iterations if 0 < value <= args.iterations} | {args.iterations}
        )
        cmd = [
            sys.executable,
            "tools/train_one_scene.py",
            "--scene_train_path",
            str(scene_dir / "train"),
            "--model_path",
            str(model_path),
            "--iterations",
            str(args.iterations),
            "--resolution",
            str(args.resolution),
            "--data_device",
            args.data_device,
            "--lambda_dssim",
            str(args.lambda_dssim),
            "--save_iterations",
            *[str(value) for value in save_iterations],
        ]
        if not args.no_antialiasing:
            cmd.append("--antialiasing")

        print(f"[{index}/{len(scenes)}] Training {scene_dir.name} -> {model_path}")
        item["command"] = cmd
        if args.dry_run:
            item["status"] = "dry_run"
            continue
        result = subprocess.run(cmd)
        item["returncode"] = result.returncode
        item["status"] = "completed" if result.returncode == 0 else "failed"
        with open(batch_path, "w", encoding="utf-8") as file:
            json.dump(batch, file, indent=2)
        if result.returncode != 0:
            raise SystemExit(f"Training failed for {scene_dir.name}; stopping the batch.")

    batch["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(batch_path, "w", encoding="utf-8") as file:
        json.dump(batch, file, indent=2)
    print(f"Training batch complete. Manifest: {batch_path}")


if __name__ == "__main__":
    main()
