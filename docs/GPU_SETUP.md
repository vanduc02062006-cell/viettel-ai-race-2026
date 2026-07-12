# GPU Setup Guide

## 1. Vì sao local hiện tại không train được
- PyTorch hiện tại là CPU-only.
- `CUDA = false`.
- Submodules (như `diff_gaussian_rasterization`) chưa được build thành công thành CUDA extension.
- Repo 3DGS gốc bắt buộc phải có CUDA extension để train và render.

## 2. Không phải lỗi dataset
- Output trong `work_undistorted` đã được gate `PASS`.
- Camera model hiện tại là `SIMPLE_PINHOLE`/`PINHOLE`.
- `NORMALIZATION_DECISION = NO_POSE_TRANSFORM_NEEDED`.
- Do đó, bộ dataset hoàn toàn tương thích và hợp lệ.

## 3. Windows GPU setup checklist
- NVIDIA GPU hoạt động.
- `nvidia-smi` chạy được và hiển thị thông tin GPU.
- CUDA Toolkit đã được cài đặt và có `nvcc`.
- Visual Studio Build Tools (Desktop development with C++) đã được cài đặt và environment variables (chẳng hạn x64 Native Tools Command Prompt) đã được kích hoạt.
- Python/conda environment cài bản PyTorch có hỗ trợ CUDA build tương ứng.
- Đã clone repo bằng `--recursive` (đã có sẵn trong `gpu_handoff`).
- Chạy `scripts/setup_cuda_extensions_windows.ps1` để build.

## 4. Linux/cloud GPU setup checklist
- `nvidia-smi` chạy được.
- `torch.cuda.is_available()` trả về `True`.
- `nvcc` có sẵn hoặc CUDA toolkit/dev image đã cài đặt.
- Chạy `bash scripts/setup_cuda_extensions_linux.sh` để build extensions.

## 5. Lệnh check
```bash
python tools/check_env.py
```

## 6. Lệnh train debug public trên GPU
```bash
python tools/train_one_scene.py --scene_train_path ".\work_undistorted\phase1\public_set\hcm0031\train" --model_path "outputs\public_hcm0031_undistorted_low" --iterations 1000 --resolution 8 --data_device cpu
```

## 7. Lệnh nếu GPU mạnh hơn
```bash
python tools/train_one_scene.py --scene_train_path ".\work_undistorted\phase1\public_set\hcm0031\train" --model_path "outputs\public_hcm0031_undistorted_r4_it3000" --iterations 3000 --resolution 4 --data_device cpu
```

## 8. Cảnh báo quan trọng
Không chạy train dữ liệu private trước khi:
- `check_env.py` báo `CUDA_ENV_STATUS = READY`.
- Train scene public debug thành công.
- Các script render public, distort_back và evaluate chạy được an toàn.
