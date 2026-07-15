import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def train_one_scene(
    scene_train_path,
    model_path,
    iterations,
    resolution,
    data_device,
    repo_dir,
    allow_original_data,
    antialiasing=False,
    lambda_dssim=0.2,
    save_iterations=None,
    port=6009,
):
    scene_train_path = Path(scene_train_path)
    model_path = Path(model_path)
    
    if "VAI_NVS_DATA" in str(scene_train_path.absolute()) and not allow_original_data:
        raise ValueError(
            "Do not train directly on VAI_NVS_DATA; use work_undistorted so the camera model is supported."
        )

    train_script = Path(repo_dir) / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"{train_script} not found.")

    cmd = [
        "python", str(train_script),
        "-s", str(scene_train_path),
        "-m", str(model_path),
        "--iterations", str(iterations),
        "-r", str(resolution),
        "--data_device", str(data_device),
        "--lambda_dssim", str(lambda_dssim),
        "--port", str(port),
    ]
    if antialiasing:
        cmd.append("--antialiasing")
    requested_saves = sorted(set((save_iterations or []) + [iterations]))
    cmd.extend(["--save_iterations", *[str(value) for value in requested_saves]])
    
    print("Running command:")
    print(" ".join(cmd))

    log_file = Path("logs") / f"train_{model_path.name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)
    manifest_path = model_path / "run_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scene_train_path": str(scene_train_path.resolve()),
        "model_path": str(model_path.resolve()),
        "iterations": iterations,
        "resolution": resolution,
        "data_device": data_device,
        "antialiasing": antialiasing,
        "lambda_dssim": lambda_dssim,
        "save_iterations": requested_saves,
        "port": port,
        "command": cmd,
        "status": "running",
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    
    success = False
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        try:
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            if res.returncode != 0:
                manifest["status"] = "failed"
                manifest["returncode"] = res.returncode
                print(f"ERROR: Training failed. See {log_file} for details.")
                with open(log_file, "r") as lf:
                    content = lf.read()
                    if "CUDA out of memory" in content:
                        print("WARNING: OOM detected. Suggestion: reduce resolution, close GPU apps, use stronger GPU.")
                    elif "Saving Gaussians" in content and "Error" in content:
                        print("WARNING: Error while Saving Gaussians. Possibly OOM during save.")
            else:
                manifest["status"] = "completed"
                manifest["returncode"] = 0
                success = True
                print(f"Training completed successfully. Logs saved to {log_file}")
        except Exception as e:
            manifest["status"] = "failed"
            manifest["error"] = str(e)
            print(f"ERROR: Failed to run training: {e}")
        finally:
            with open(manifest_path, "w", encoding="utf-8") as manifest_file:
                json.dump(manifest, manifest_file, indent=2)
    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_train_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--data_device', default='cpu')
    parser.add_argument('--repo_dir', default='gaussian-splatting')
    parser.add_argument('--allow_original_data', action='store_true')
    parser.add_argument('--antialiasing', action='store_true')
    parser.add_argument('--lambda_dssim', type=float, default=0.2)
    parser.add_argument('--save_iterations', type=int, nargs='*', default=[])
    parser.add_argument(
        '--port',
        type=int,
        default=6009,
        help='Network GUI port. Parallel training processes must use different ports.',
    )
    args = parser.parse_args()
    
    succeeded = train_one_scene(
        args.scene_train_path,
        args.model_path,
        args.iterations,
        args.resolution,
        args.data_device,
        args.repo_dir,
        args.allow_original_data,
        args.antialiasing,
        args.lambda_dssim,
        args.save_iterations,
        args.port,
    )
    if not succeeded:
        raise SystemExit(1)
