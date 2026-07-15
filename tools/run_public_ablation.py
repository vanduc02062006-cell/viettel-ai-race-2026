import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command):
    print("Running:", " ".join(str(part) for part in command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Train, render, distort and evaluate one public ablation.")
    parser.add_argument("--scene", default="hcm0031")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--lambda_dssim", type=float, default=0.2)
    parser.add_argument("--data_device", default="cpu")
    parser.add_argument("--no_antialiasing", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    model_path = Path("outputs") / "ablations" / args.tag / f"public_{args.scene}"
    raw_root = Path("outputs") / "ablations" / args.tag / "raw"
    distorted_root = Path("outputs") / "ablations" / args.tag / "distorted"
    metrics_path = Path("outputs") / "ablations" / args.tag / "metrics.csv"
    manifest_path = Path("outputs") / "ablations" / args.tag / "ablation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "tag": args.tag,
        "iterations": args.iterations,
        "resolution": args.resolution,
        "lambda_dssim": args.lambda_dssim,
        "antialiasing": not args.no_antialiasing,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train = [
        python,
        "tools/train_one_scene.py",
        "--scene_train_path",
        f"work_undistorted/phase1/public_set/{args.scene}/train",
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
        str(args.iterations),
    ]
    if not args.no_antialiasing:
        train.append("--antialiasing")
    run(train)

    render = [
        python,
        "tools/render_test_poses.py",
        "--split",
        "public_set",
        "--scene",
        args.scene,
        "--model_path",
        str(model_path),
        "--output_root",
        str(raw_root),
        "--output_format",
        "png",
    ]
    if not args.no_antialiasing:
        render.append("--antialiasing")
    run(render)

    run(
        [
            python,
            "tools/distort_back_renders.py",
            "--split",
            "public_set",
            "--scene",
            args.scene,
            "--render_root",
            str(raw_root),
            "--output_root",
            str(distorted_root),
            "--keep_original_extension",
        ]
    )
    run(
        [
            python,
            "tools/evaluate_public_renders.py",
            "--scene",
            args.scene,
            "--pred_root",
            str(distorted_root),
            "--output_csv",
            str(metrics_path),
        ]
    )
    print(f"Ablation complete: {metrics_path}")


if __name__ == "__main__":
    main()
