import os
import sys
import platform
import subprocess
from pathlib import Path

def check_env():
    log_file = Path("logs/env_check.txt")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_file, "w") as f:
        def lprint(msg):
            print(msg)
            f.write(msg + "\n")

        lprint("--- Environment Check ---")
        lprint(f"Python version: {sys.version}")
        lprint(f"OS: {platform.system()} {platform.release()}")
        lprint(f"Current working directory: {os.getcwd()}")
        
        torch_ok = False
        cuda_ok = False
        nvcc_ok = False
        smi_ok = False
        cl_ok = False

        try:
            import torch
            torch_ok = True
            lprint("torch installed: True")
            lprint(f"torch version: {torch.__version__}")
            lprint(f"torch CUDA version: {torch.version.cuda}")
            cuda_ok = torch.cuda.is_available()
            lprint(f"CUDA available: {cuda_ok}")
            if cuda_ok:
                lprint(f"CUDA device count: {torch.cuda.device_count()}")
                lprint(f"GPU name: {torch.cuda.get_device_name(0)}")
                t = torch.cuda.get_device_properties(0).total_memory
                lprint(f"VRAM total: {t / (1024**3):.2f} GB")
        except ImportError:
            lprint("torch installed: False")
        
        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if res.returncode == 0:
                lprint("\nnvidia-smi output found.")
                smi_ok = True
            else:
                lprint("nvidia-smi error.")
        except FileNotFoundError:
            lprint("nvidia-smi not found.")

        try:
            res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
            if res.returncode == 0:
                lprint("\nnvcc found.")
                nvcc_ok = True
            else:
                lprint("nvcc error.")
        except FileNotFoundError:
            lprint("nvcc not found.")

        if platform.system() == "Windows":
            try:
                res = subprocess.run(["where", "cl"], capture_output=True, text=True)
                if res.returncode == 0:
                    lprint("cl.exe found in PATH.")
                    cl_ok = True
                else:
                    lprint("WARNING: Visual Studio Build Tools C++ environment may not be active")
            except Exception:
                lprint("WARNING: Visual Studio Build Tools C++ environment may not be active")
            
            if "DISTUTILS_USE_SDK" not in os.environ:
                lprint("DISTUTILS_USE_SDK not set. Suggestion: set DISTUTILS_USE_SDK=1")
            else:
                lprint(f"DISTUTILS_USE_SDK={os.environ['DISTUTILS_USE_SDK']}")
        else:
            cl_ok = True # Not required on Linux

        repo_dir = Path("gaussian-splatting")
        if not repo_dir.exists():
            lprint("\ngaussian-splatting repo not found.")
            lprint("Please run: git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive")
        else:
            lprint("\ngaussian-splatting repo found.")
            submodules = [
                "submodules/diff-gaussian-rasterization",
                "submodules/simple-knn"
            ]
            all_subs_ok = True
            for sub in submodules:
                sub_path = repo_dir / sub
                if not sub_path.exists() or not any(sub_path.iterdir()):
                    lprint(f"WARNING: Submodule {sub} is empty or missing.")
                    all_subs_ok = False
            if not all_subs_ok:
                lprint("Please run: cd gaussian-splatting && git submodule update --init --recursive")
                lprint("ERROR: Submodules missing, cannot train.")
        
        gate_file = Path("logs/undistorted_camera_model_gate.txt")
        if gate_file.exists():
            content = gate_file.read_text().strip()
            if content.startswith("FAIL"):
                lprint("WARNING: logs/undistorted_camera_model_gate.txt is FAIL. DO NOT TRAIN.")
            else:
                lprint("logs/undistorted_camera_model_gate.txt is PASS.")
        else:
            lprint("logs/undistorted_camera_model_gate.txt not found.")

        # Check submodules import
        raster_ok = False
        knn_ok = False
        try:
            import diff_gaussian_rasterization
            raster_ok = True
        except ImportError:
            pass
        
        try:
            from simple_knn._C import distCUDA2
            knn_ok = True
        except Exception as e:
            lprint(f"simple_knn import error: {e}")

        lprint("\n--- Final Status ---")
        if not cuda_ok:
            lprint("CUDA_ENV_STATUS = CPU_ONLY")
        else:
            if smi_ok and nvcc_ok and cl_ok:
                if raster_ok and knn_ok:
                    lprint("CUDA_ENV_STATUS = READY")
                else:
                    lprint("CUDA_ENV_STATUS = NEED_BUILD")
            else:
                lprint("CUDA_ENV_STATUS = NEED_BUILD")

if __name__ == "__main__":
    check_env()
