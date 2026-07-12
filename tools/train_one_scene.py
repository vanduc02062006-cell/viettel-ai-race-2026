import argparse
import subprocess
from pathlib import Path

def train_one_scene(scene_train_path, model_path, iterations, resolution, data_device, repo_dir, allow_original_data):
    scene_train_path = Path(scene_train_path)
    model_path = Path(model_path)
    
    if "VAI_NVS_DATA" in str(scene_train_path.absolute()) and not allow_original_data:
        print("ERROR: Do not train directly on VAI_NVS_DATA! It contains SIMPLE_RADIAL models.")
        print("Use work_undistorted instead, or pass --allow_original_data if absolutely sure.")
        return

    train_script = Path(repo_dir) / "train.py"
    if not train_script.exists():
        print(f"ERROR: {train_script} not found. Please clone gaussian-splatting repository.")
        return

    cmd = [
        "python", str(train_script),
        "-s", str(scene_train_path),
        "-m", str(model_path),
        "--iterations", str(iterations),
        "-r", str(resolution),
        "--data_device", str(data_device)
    ]
    
    print("Running command:")
    print(" ".join(cmd))

    log_file = Path("logs") / f"train_{model_path.name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w") as f:
        f.write(" ".join(cmd) + "\n\n")
        try:
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            if res.returncode != 0:
                print(f"ERROR: Training failed. See {log_file} for details.")
                with open(log_file, "r") as lf:
                    content = lf.read()
                    if "CUDA out of memory" in content:
                        print("WARNING: OOM detected. Suggestion: reduce resolution, close GPU apps, use stronger GPU.")
                    elif "Saving Gaussians" in content and "Error" in content:
                        print("WARNING: Error while Saving Gaussians. Possibly OOM during save.")
            else:
                print(f"Training completed successfully. Logs saved to {log_file}")
        except Exception as e:
            print(f"ERROR: Failed to run training: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_train_path', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--resolution', type=int, default=8)
    parser.add_argument('--data_device', default='cpu')
    parser.add_argument('--repo_dir', default='gaussian-splatting')
    parser.add_argument('--allow_original_data', action='store_true')
    args = parser.parse_args()
    
    train_one_scene(
        args.scene_train_path,
        args.model_path,
        args.iterations,
        args.resolution,
        args.data_device,
        args.repo_dir,
        args.allow_original_data
    )
